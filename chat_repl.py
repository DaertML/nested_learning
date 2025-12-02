import torch
import os
import json
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict

# ==================== CONFIGURATION VIA ENVIRONMENT VARIABLES ====================
@dataclass
class NestedConfig:
    """Configuration for Nested Learning behavior - all tunable via environment variables"""
    
    # Model settings
    model_path: str = os.getenv("NL_MODEL_PATH", "/media/pc/easystore1/hf_models/gemma3-270m-it")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Fast/Slow weight decomposition
    slow_ratio: float = float(os.getenv("NL_SLOW_RATIO", "0.8"))  # % of layers to freeze
    
    # Adaptation hyperparameters
    learning_rate: float = float(os.getenv("NL_LEARNING_RATE", "5e-6"))
    adapt_steps: int = int(os.getenv("NL_ADAPT_STEPS", "2"))
    max_grad_norm: float = float(os.getenv("NL_MAX_GRAD_NORM", "0.3"))
    weight_decay: float = float(os.getenv("NL_WEIGHT_DECAY", "0.01"))
    
    # Context management (key for Nested Learning)
    context_window: int = int(os.getenv("NL_CONTEXT_WINDOW", "4"))  # How many turns to keep
    adapt_on_every_turn: bool = os.getenv("NL_ADAPT_EVERY_TURN", "true").lower() == "true"
    adapt_on_user_only: bool = os.getenv("NL_ADAPT_USER_ONLY", "false").lower() == "true"
    cumulative_context: bool = os.getenv("NL_CUMULATIVE_CONTEXT", "true").lower() == "true"
    
    # Adaptation frequency control (implementing paper's frequency hierarchy)
    adapt_every_n_turns: int = int(os.getenv("NL_ADAPT_EVERY_N_TURNS", "1"))
    
    # Generation settings
    max_new_tokens: int = int(os.getenv("NL_MAX_NEW_TOKENS", "150"))
    temperature: float = float(os.getenv("NL_TEMPERATURE", "0.7"))
    top_k: int = int(os.getenv("NL_TOP_K", "50"))
    top_p: float = float(os.getenv("NL_TOP_P", "0.9"))
    
    # Session management
    auto_reset_threshold: int = int(os.getenv("NL_AUTO_RESET_THRESHOLD", "20"))  # Reset after N turns
    checkpoint_dir: str = os.getenv("NL_CHECKPOINT_DIR", "./nested_checkpoints")
    auto_save_every_n_turns: int = int(os.getenv("NL_AUTO_SAVE_EVERY_N", "0"))  # 0 = disabled
    
    def print_config(self):
        """Display current configuration"""
        print("\n" + "="*70)
        print("NESTED LEARNING CONFIGURATION")
        print("="*70)
        print(f"Model: {os.path.basename(self.model_path)}")
        print(f"Device: {self.device}")
        print(f"\nWeight Hierarchy:")
        print(f"  Slow Ratio: {self.slow_ratio:.1%} (frozen pretrained layers)")
        print(f"  Fast Ratio: {(1-self.slow_ratio):.1%} (adaptable layers + embeddings)")
        print(f"\nAdaptation Settings:")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Adapt Steps per Turn: {self.adapt_steps}")
        print(f"  Adapt Every N Turns: {self.adapt_every_n_turns}")
        print(f"  Max Gradient Norm: {self.max_grad_norm}")
        print(f"  Weight Decay: {self.weight_decay}")
        print(f"\nContext Management:")
        print(f"  Context Window: {self.context_window} turns")
        print(f"  Cumulative Context: {self.cumulative_context}")
        print(f"  Adapt on User Input Only: {self.adapt_on_user_only}")
        print(f"  Auto-Reset After: {self.auto_reset_threshold} turns")
        print(f"\nCheckpoint Management:")
        print(f"  Checkpoint Directory: {self.checkpoint_dir}")
        print(f"  Auto-Save Every N Turns: {self.auto_save_every_n_turns if self.auto_save_every_n_turns > 0 else 'Disabled'}")
        print(f"\nGeneration:")
        print(f"  Max Tokens: {self.max_new_tokens}")
        print(f"  Temperature: {self.temperature}")
        print(f"  Top-K: {self.top_k}, Top-P: {self.top_p}")
        print("="*70 + "\n")


class NestedChatREPL:
    """
    Chat REPL implementing Nested Learning's Active Perception during conversation.
    
    Key concepts from the paper applied to chat:
    1. Fast Weights adapt continuously to compress conversation context
    2. Slow Weights remain frozen (pretrained knowledge)
    3. Multi-frequency updates: adapt at different rates based on conversation flow
    4. Ephemeral memory: can reset Fast weights to clear conversation-specific adaptations
    """
    
    def __init__(self, config: Optional[NestedConfig] = None):
        self.config = config or NestedConfig()
        self.config.print_config()
        
        # Load tokenizer and model
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
            device_map=self.config.device
        )
        
        # Setup Fast/Slow weight hierarchy
        self.fast_parameters = []
        self.use_amp = torch.cuda.is_bf16_supported()
        self.scaler = GradScaler('cuda') if (self.config.device == 'cuda' and not self.use_amp) else None
        self._set_fast_slow_weights()
        
        # Store initial Fast weights for reset
        self.reset_fast_weights = {
            name: p.clone().detach().to('cpu') 
            for name, p in self.model.named_parameters() 
            if p.requires_grad
        }
        
        # Conversation state
        self.conversation_history: List[Tuple[str, str]] = []  # [(role, content), ...]
        self.turn_count = 0
        self.total_adaptations = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create checkpoint directory
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"✓ Model loaded: {self.model.dtype}")
        print(f"✓ Fast parameters: {len(self.fast_parameters)} tensors")
        print(f"✓ Slow parameters: {sum(1 for p in self.model.parameters() if not p.requires_grad)} tensors (frozen)")
        print(f"✓ Session ID: {self.session_id}")
        print("\nReady for conversation!\n")
        
    def _set_fast_slow_weights(self):
        """
        Implements frequency-based weight decomposition from Nested Learning.
        
        Following the paper's Definition 2 (Update Frequency):
        - Slow Weights (fₐ = 0): Bottom layers frozen at pretrained state
        - Fast Weights (fₐ > 0): Top layers + embeddings that adapt during conversation
        """
        # Freeze all (Slow Weights)
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze top layers (Fast Weights)
        all_layers = list(self.model.model.layers.children())
        num_layers_to_freeze = int(len(all_layers) * self.config.slow_ratio)

        for layer in all_layers[num_layers_to_freeze:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze embeddings for context adaptation
        for param in self.model.model.embed_tokens.parameters():
            param.requires_grad = True

        self.fast_parameters = [p for p in self.model.parameters() if p.requires_grad]

    def _build_context_string(self) -> str:
        """
        Builds context string from conversation history.
        
        Implements the paper's notion of "context flow" - the sequence of events
        that the Fast weights compress into their parameters.
        """
        if not self.conversation_history:
            return ""
        
        # Use sliding window if enabled
        history = self.conversation_history
        if self.config.context_window > 0:
            history = history[-self.config.context_window:]
        
        # Format conversation
        context_parts = []
        for role, content in history:
            if role == "user":
                context_parts.append(f"User: {content}")
            else:
                context_parts.append(f"Assistant: {content}")
        
        return "\n".join(context_parts)
    
    def _should_adapt_this_turn(self) -> bool:
        """
        Determines if Fast weights should adapt on this turn.
        
        Implements adaptive update frequency from the paper's frequency hierarchy.
        """
        # Check if we should skip this turn based on frequency
        if self.turn_count % self.config.adapt_every_n_turns != 0:
            return False
        
        return self.config.adapt_on_every_turn
    
    def _adapt_to_context(self, context: str) -> Tuple[bool, float]:
        """
        Active Perception: Adapt Fast weights to compress context.
        
        From the paper (Section 2.1):
        "Training translates as acquiring effective memory that maps data samples 
        to their Local Surprise Signal (LSS)"
        
        Returns: (success, final_loss)
        """
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Tokenize context
        tokenized = self.tokenizer(
            context,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        tokenized = {k: v.to(self.config.device) for k, v in tokenized.items()}
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized.get('attention_mask', None)
        
        # Create labels for causal LM
        labels = input_ids.clone()
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        
        # Setup optimizer
        optimizer = AdamW(
            [{'params': p, 'lr': self.config.learning_rate} for p in self.fast_parameters],
            weight_decay=self.config.weight_decay,
            eps=1e-8
        )
        
        # Adaptation loop
        self.model.train()
        final_loss = float('inf')
        success = False
        
        for step in range(self.config.adapt_steps):
            optimizer.zero_grad()
            
            try:
                # Forward pass
                if self.use_amp:
                    with autocast('cuda', dtype=torch.bfloat16):
                        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                
                # Validate loss
                if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0:
                    optimizer.zero_grad()
                    continue
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.fast_parameters, max_norm=self.config.max_grad_norm)
                    
                    # Check gradients
                    has_nan = any(
                        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                        for p in self.fast_parameters
                    )
                    
                    if has_nan:
                        optimizer.zero_grad()
                        continue
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.fast_parameters, max_norm=self.config.max_grad_norm)
                    
                    has_nan = any(
                        p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                        for p in self.fast_parameters
                    )
                    
                    if has_nan:
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                
                final_loss = loss.item()
                success = True
                
            except RuntimeError as e:
                optimizer.zero_grad()
                if self.config.device == 'cuda':
                    torch.cuda.empty_cache()
                continue
        
        self.total_adaptations += 1 if success else 0
        return success, final_loss
    
    def _generate_response(self, prompt: str) -> str:
        """Generate response using adapted Fast weights + frozen Slow weights"""
        self.model.eval()
        
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        tokenized = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        tokenized = {k: v.to(self.config.device) for k, v in tokenized.items()}
        
        with torch.no_grad():
            try:
                output_ids = self.model.generate(
                    tokenized['input_ids'],
                    attention_mask=tokenized.get('attention_mask'),
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=True,
                    top_k=self.config.top_k,
                    top_p=self.config.top_p,
                    temperature=self.config.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    num_return_sequences=1
                )
                
                response = self.tokenizer.decode(
                    output_ids[0, tokenized['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
                return response.strip()
                
            except Exception as e:
                if self.config.device == 'cuda':
                    torch.cuda.empty_cache()
                return f"[Generation Error: {str(e)}]"
    
    def reset_fast_weights(self):
        """Reset Fast weights to initial state (clear ephemeral memory)"""
        if self.config.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if name in self.reset_fast_weights:
                    try:
                        reset_weight = self.reset_fast_weights[name].to(p.device, dtype=p.dtype)
                        p.copy_(reset_weight)
                    except RuntimeError:
                        continue
        
        print("✓ Fast weights reset to initial state")
    
    def save_checkpoint(self, checkpoint_name: Optional[str] = None) -> str:
        """
        Save current Fast weights state to disk.
        
        This allows preserving conversation-specific adaptations across sessions,
        implementing the paper's concept of persistent vs. ephemeral memory.
        
        Returns: Path to saved checkpoint
        """
        if checkpoint_name is None:
            checkpoint_name = f"checkpoint_{self.session_id}_turn{self.turn_count}"
        
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save Fast weights (only the adapted parameters)
        fast_weights_state = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fast_weights_state[name] = param.detach().cpu()
        
        weights_path = checkpoint_path / "fast_weights.pt"
        torch.save(fast_weights_state, weights_path)
        
        # Save metadata
        metadata = {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "total_adaptations": self.total_adaptations,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model_path": self.config.model_path,
                "slow_ratio": self.config.slow_ratio,
                "learning_rate": self.config.learning_rate,
                "adapt_steps": self.config.adapt_steps,
            },
            "conversation_length": len(self.conversation_history),
        }
        
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Optionally save conversation history
        history_path = checkpoint_path / "conversation.json"
        with open(history_path, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str, load_history: bool = False) -> bool:
        """
        Load Fast weights from a previous session.
        
        This restores conversation-specific adaptations that are COMPRESSED into the
        Fast weight parameters themselves. According to Nested Learning principles,
        the adapted weights should contain the "compressed context" - they learned
        to map inputs to outputs in a way that reflects the previous conversation.
        
        Key insight from the paper:
        The Fast weights are an "associative memory" that has compressed the context
        flow into its parameters. Loading them should allow the model to generate
        responses that reflect the learned patterns, WITHOUT needing the actual
        conversation text.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            load_history: If True, also restore conversation history for context.
                         If False (default), rely purely on compressed knowledge in weights.
            
        Returns: True if successful
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print(f"✗ Checkpoint not found: {checkpoint_path}")
            return False
        
        try:
            # Load metadata
            metadata_path = checkpoint_path / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify compatibility
            if metadata["config"]["slow_ratio"] != self.config.slow_ratio:
                print(f"⚠ Warning: Checkpoint slow_ratio ({metadata['config']['slow_ratio']}) "
                      f"differs from current ({self.config.slow_ratio})")
            
            # Load Fast weights
            weights_path = checkpoint_path / "fast_weights.pt"
            fast_weights_state = torch.load(weights_path, map_location='cpu')
            
            with torch.no_grad():
                loaded_count = 0
                for name, param in self.model.named_parameters():
                    if name in fast_weights_state:
                        try:
                            loaded_weight = fast_weights_state[name].to(param.device, dtype=param.dtype)
                            param.copy_(loaded_weight)
                            loaded_count += 1
                        except RuntimeError as e:
                            print(f"⚠ Warning: Could not load parameter {name}: {e}")
                            continue
            
            # Restore session state
            self.turn_count = metadata["turn_count"]
            self.total_adaptations = metadata["total_adaptations"]
            
            # Optionally load conversation history (this is NOT the default behavior)
            history_loaded = False
            if load_history:
                history_path = checkpoint_path / "conversation.json"
                if history_path.exists():
                    with open(history_path, 'r') as f:
                        self.conversation_history = json.load(f)
                    # Convert to list of tuples if loaded as list of lists
                    if self.conversation_history and isinstance(self.conversation_history[0], list):
                        self.conversation_history = [tuple(item) for item in self.conversation_history]
                    history_loaded = True
            
            print(f"✓ Checkpoint loaded: {checkpoint_path.name}")
            print(f"  - Loaded {loaded_count} Fast weight tensors")
            print(f"  - Turn count: {self.turn_count}")
            print(f"  - Total adaptations: {self.total_adaptations}")
            if load_history:
                if history_loaded:
                    print(f"  - Conversation history: {len(self.conversation_history)} messages")
                    print(f"  ℹ Mode: Weight + Context (weights contain compressed context + explicit history)")
                else:
                    print(f"  ⚠ History file not found")
                    print(f"  ℹ Mode: Weight-only (weights contain compressed context)")
            else:
                print(f"  ℹ Mode: Weight-only (weights contain compressed context)")
                print(f"  ℹ The model will rely on knowledge compressed into Fast weights")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load checkpoint: {e}")
            return False
    
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = []
        for checkpoint_path in checkpoint_dir.iterdir():
            if checkpoint_path.is_dir():
                metadata_path = checkpoint_path / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        checkpoints.append({
                            "name": checkpoint_path.name,
                            "path": str(checkpoint_path),
                            "timestamp": metadata.get("timestamp", "unknown"),
                            "turn_count": metadata.get("turn_count", 0),
                            "adaptations": metadata.get("total_adaptations", 0),
                        })
                    except Exception:
                        continue
        
        # Sort by timestamp (newest first)
        checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
        return checkpoints
    
    def process_turn(self, user_input: str) -> str:
        """
        Process a single conversation turn with Nested Learning.
        
        Implements the paper's Active Perception cycle:
        1. Add user input to context flow
        2. Optionally adapt Fast weights to compress new context
        3. Generate response using adapted model
        4. Add response to history
        """
        self.turn_count += 1
        
        # Add user input to history
        self.conversation_history.append(("user", user_input))
        
        # Build context for this turn
        if self.config.cumulative_context:
            # Use full conversation as context for adaptation
            context = self._build_context_string()
        else:
            # Use only current user input
            context = f"User: {user_input}"
        
        # Decide whether to adapt on this turn
        should_adapt = self._should_adapt_this_turn()
        
        if should_adapt:
            print(f"  [Adapting Fast weights to context... (turn {self.turn_count})]", end=" ")
            success, loss = self._adapt_to_context(context)
            if success:
                print(f"✓ (loss: {loss:.4f})")
            else:
                print("⚠ (adaptation skipped)")
        
        # Generate response with current (possibly adapted) weights
        prompt = context + "\nAssistant:"
        response = self._generate_response(prompt)
        
        # Add assistant response to history
        self.conversation_history.append(("assistant", response))
        
        # Auto-save check
        if self.config.auto_save_every_n_turns > 0 and self.turn_count % self.config.auto_save_every_n_turns == 0:
            print(f"  [Auto-saving checkpoint at turn {self.turn_count}]")
            self.save_checkpoint()
        
        # Auto-reset check
        if self.config.auto_reset_threshold > 0 and self.turn_count >= self.config.auto_reset_threshold:
            print(f"\n⚠ Auto-reset threshold reached ({self.config.auto_reset_threshold} turns)")
            print("  Resetting Fast weights to prevent overfitting...")
            self.reset_fast_weights()
            self.turn_count = 0
        
        return response
    
    def print_stats(self):
        """Print session statistics"""
        print("\n" + "─"*70)
        print("SESSION STATISTICS")
        print("─"*70)
        print(f"Total Turns: {self.turn_count}")
        print(f"Total Adaptations: {self.total_adaptations}")
        print(f"Conversation Length: {len(self.conversation_history)} messages")
        print(f"Context Window: {min(self.config.context_window, len(self.conversation_history))} turns")
        print("─"*70 + "\n")
    
    def run_repl(self):
        """Run interactive chat REPL"""
        print("="*70)
        print("NESTED LEARNING CHAT REPL")
        print("="*70)
        print("Commands:")
        print("  /reset          - Reset Fast weights (clear conversation adaptations)")
        print("  /save [name]    - Save checkpoint (optional custom name)")
        print("  /load <name>    - Load checkpoint by name")
        print("  /list           - List available checkpoints")
        print("  /stats          - Show session statistics")
        print("  /clear          - Clear conversation history")
        print("  /help           - Show this help")
        print("  /quit           - Exit")
        print("="*70 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    arg = parts[1] if len(parts) > 1 else None
                    
                    if cmd == "/quit" or cmd == "/exit":
                        print("\nGoodbye!")
                        break
                    
                    elif cmd == "/reset":
                        self.reset_fast_weights()
                        self.turn_count = 0
                        print("✓ Session reset\n")
                        continue
                    
                    elif cmd == "/save":
                        checkpoint_name = arg if arg else None
                        self.save_checkpoint(checkpoint_name)
                        print()
                        continue
                    
                    elif cmd == "/load":
                        if not arg:
                            print("Usage: /load <checkpoint_name> [--with-history]")
                            print("  Default: Load weights only (compressed context)")
                            print("  --with-history: Also load conversation text\n")
                            continue
                        
                        # Parse arguments
                        parts = arg.split()
                        checkpoint_name = parts[0]
                        load_history = "--with-history" in parts
                        
                        # Check if it's a full path or just a name
                        if Path(checkpoint_name).exists():
                            checkpoint_path = checkpoint_name
                        else:
                            checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_name
                        
                        self.load_checkpoint(str(checkpoint_path), load_history=load_history)
                        print()
                        continue
                    
                    elif cmd == "/list":
                        checkpoints = self.list_checkpoints()
                        if not checkpoints:
                            print("No checkpoints found.\n")
                        else:
                            print("\nAvailable Checkpoints:")
                            print("─"*70)
                            for i, ckpt in enumerate(checkpoints, 1):
                                print(f"{i}. {ckpt['name']}")
                                print(f"   Timestamp: {ckpt['timestamp']}")
                                print(f"   Turns: {ckpt['turn_count']}, Adaptations: {ckpt['adaptations']}")
                                if i < len(checkpoints):
                                    print()
                            print("─"*70)
                            print("Use: /load <checkpoint_name>\n")
                        continue
                    
                    elif cmd == "/stats":
                        self.print_stats()
                        continue
                    
                    elif cmd == "/clear":
                        self.conversation_history.clear()
                        self.turn_count = 0
                        print("✓ Conversation history cleared\n")
                        continue
                    
                    elif cmd == "/help":
                        print("\nCommands:")
                        print("  /reset       - Reset Fast weights")
                        print("  /save [name] - Save checkpoint")
                        print("  /load <name> - Load checkpoint")
                        print("  /list        - List checkpoints")
                        print("  /stats       - Show statistics")
                        print("  /clear       - Clear history")
                        print("  /quit        - Exit\n")
                        continue
                    
                    else:
                        print(f"Unknown command: {cmd}")
                        print("Type /help for available commands.\n")
                        continue
                
                # Process conversation turn
                response = self.process_turn(user_input)
                print(f"\nAssistant: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit.\n")
                continue
            except Exception as e:
                print(f"\n⚠ Error: {e}\n")
                continue


# ==================== MAIN ====================
if __name__ == "__main__":
    # Create config from environment variables
    config = NestedConfig()
    
    # Initialize and run chat REPL
    chat = NestedChatREPL(config)
    chat.run_repl()
