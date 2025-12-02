import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from contextlib import nullcontext

# --- Configuration ---
MODEL_NAME = "/media/pc/easystore1/hf_models/gemma3-270m-it"
SLOW_RATIO = 0.8 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class NestedGemmaInference:
    """
    Implements the Active Perception (Test-Time Adaptation) phase of Nested Learning
    without requiring a prior Nested Training run.
    """
    def __init__(self, model_path=MODEL_NAME, slow_ratio=SLOW_RATIO):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Explicitly set pad_token to eos_token for consistent behavior
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model in bfloat16 for better numerical stability
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map=DEVICE
        )
        self.slow_ratio = slow_ratio
        self.fast_parameters = []
        # Use bfloat16 if supported, otherwise no scaler
        self.use_amp = torch.cuda.is_bf16_supported()
        self.scaler = GradScaler('cuda') if (DEVICE == 'cuda' and not self.use_amp) else None
        self._set_fast_slow_weights()
        # Collect reset weights - store in same dtype as model
        self.reset_fast_weights = {name: p.clone().detach().to('cpu') for name, p in self.model.named_parameters() if p.requires_grad}
        
        print("\nNested Inference Engine Ready.")
        print(f"Model dtype: {self.model.dtype}")
        print(f"Active Parameters (Fast): {len(self.fast_parameters)} tensors")
        print(f"Slow Parameters frozen.")
        
    def _set_fast_slow_weights(self):
        """Identifies and sets the Fast (trainable) and Slow (frozen) weights."""
        
        # 1. Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Determine the number of layers to freeze (Slow Weights)
        all_layers = list(self.model.model.layers.children())
        num_layers_to_freeze = int(len(all_layers) * self.slow_ratio)

        # 3. Identify Fast Weights (typically the early layers and embedding)
        # Unfreeze the non-frozen layers
        for layer in all_layers[num_layers_to_freeze:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze the embedding layer (crucial for quick context adaptation)
        for param in self.model.model.embed_tokens.parameters():
            param.requires_grad = True

        # Collect all parameters that are now set to requires_grad=True
        self.fast_parameters = [p for p in self.model.parameters() if p.requires_grad]

    def reset(self):
        """Resets the Fast Weights back to their initial state."""
        # Clear CUDA cache before reset to avoid memory issues
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.reset_fast_weights:
                    try:
                        # Move reset weight to the same device as parameter
                        reset_weight = self.reset_fast_weights[name].to(p.device, dtype=p.dtype)
                        p.copy_(reset_weight)
                    except RuntimeError as e:
                        print(f"Warning: Could not reset parameter {name}: {e}")
                        continue

    def generate(self, prompt: str, learning_rate: float = 5e-6, adapt_steps: int = 2):
        """
        Runs Test-Time Adaptation (Active Perception) followed by generation.

        Args:
            prompt (str): The input prompt containing the adaptation context.
            learning_rate (float): The step size for the gradient update.
            adapt_steps (int): How many optimization steps to run on the prompt.
        """
        
        # Clear any previous CUDA errors
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 1. Tokenize the input prompt and get the attention mask
        tokenized_input = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        tokenized_input = {k: v.to(DEVICE) for k, v in tokenized_input.items()}
        
        input_ids = tokenized_input['input_ids']
        attention_mask = tokenized_input.get('attention_mask', None)
        
        # Create proper labels for causal language modeling
        labels = input_ids.clone()
        
        # Set padding tokens to -100 so they're ignored in loss calculation
        if attention_mask is not None:
            labels[attention_mask == 0] = -100

        # 2. Setup the Optimizer for Fast Weights only
        optimizer = AdamW(
            [{'params': p, 'lr': learning_rate} for p in self.fast_parameters],
            weight_decay=0.01,
            eps=1e-8
        )
        
        # 3. Perform Test-Time Adaptation (Active Perception)
        self.model.train()
        
        successful_adaptation = False
        for step in range(adapt_steps):
            optimizer.zero_grad()
            
            try:
                # Use appropriate autocast based on dtype
                if self.use_amp:
                    with autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model(
                            input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels
                        )
                        loss = outputs.loss
                else:
                    outputs = self.model(
                        input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels
                    )
                    loss = outputs.loss
                
                # Check for invalid loss before backward pass
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0:
                    print(f"Warning: Invalid loss detected at step {step} (loss={loss.item()}), skipping optimization")
                    optimizer.zero_grad()
                    continue
                
                # Backward pass with gradient clipping
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.fast_parameters, max_norm=0.3)
                    
                    # Check for NaN gradients
                    has_nan = False
                    for p in self.fast_parameters:
                        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                            has_nan = True
                            break
                    
                    if has_nan:
                        print(f"Warning: NaN gradients detected at step {step}, skipping")
                        optimizer.zero_grad()
                        continue
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.fast_parameters, max_norm=0.3)
                    
                    # Check for NaN gradients
                    has_nan = False
                    for p in self.fast_parameters:
                        if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                            has_nan = True
                            break
                    
                    if has_nan:
                        print(f"Warning: NaN gradients detected at step {step}, skipping")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                
                print(f"Adaptation Step {step+1}/{adapt_steps} | Loss: {loss.item():.4f}")
                successful_adaptation = True
                
            except RuntimeError as e:
                print(f"Error during adaptation step {step}: {e}")
                optimizer.zero_grad()
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                continue

        if not successful_adaptation:
            print("Warning: No successful adaptation steps completed")
            print("Proceeding with generation using base model weights...")

        # 4. Generate the response using the adapted state
        self.model.eval()
        
        # Clear cache before generation
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        with torch.no_grad():
            try:
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
                
                response = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
                return response
                
            except Exception as e:
                print(f"Generation failed with error: {e}")
                if DEVICE == 'cuda':
                    torch.cuda.empty_cache()
                return "Generation failed due to error. Try with a shorter prompt or different parameters."

# --- Runnable Example ---
if __name__ == "__main__":
    
    # Initialize the engine (loads base Gemma weights)
    engine = NestedGemmaInference()
    
    # Context that requires fast adaptation
    medical_context = (
        "A 'Quark-Chain' Protocol is a new therapy for treating "
        "Type 4 Hyper-Epsilonitis. The primary symptom is rapid change in 'Delta-Tensor' levels. "
        "The treatment stabilizes Delta-Tensors via the Quark-Chain. "
        "What is the core function of the Quark-Chain Protocol?"
    )
    
    print("\n--- User Session 1: Medical Context ---")
    
    response = engine.generate(
        prompt=medical_context,
        learning_rate=5e-6,
        adapt_steps=2
    )
    
    print("\n--- Model Response (Adapted) ---")
    print(response.strip())
    
    # Resetting the fast weights to demonstrate memory is ephemeral
    print("\n--- Resetting model weights ---")
    engine.reset()
    
    print("\n--- User Session 2: After Reset (Should revert to general knowledge) ---")
    
    # Using a similar query after resetting fast weights
    general_query = "What is the Quark-Chain Protocol?"
    
    response_reset = engine.generate(
        prompt=general_query,
        learning_rate=5e-6,
        adapt_steps=2
    )
    
    print("\n--- Model Response (Reset State) ---")
    print(response_reset.strip())