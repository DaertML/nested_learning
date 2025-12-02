# nested_learning
Tiny implementation of Nested Learning ideas from Google Deepmind on already pretrained LLMs

# Introduction
This is the first glympse of Continual Learning that has happened at DaertML; from the ideas of Google Deepmind about how LLMs should do Test Time Training, in order to modify its weights to the task at hand, and get the information right into the model, instead of using the context window.
Google ideas go in a slightly different path, requiring the model to be trained with certain characteristics prior to inference, and using the Titan architecture. In an attempt to avoid training models from scratch and getting the most ouf of existing LLMs, the implementation from DaertML performs Test Time Training on existing LLMs.

We support the following models at the moment (tested):
- Gemma 3 270M Instruction: ✔️

# Usage
Provide the folder where you Gemma 3 270M Instruction model is, and run the inf-notrain.py script; this script is the first version of the solution. Future code will contain the training script and the inference script for previously finetuned models following the ideas from Google Deepmind.

One can also run the chat_repl.py, and use the commands available, to guide the conversation as the user desires. The REPL supports the possibility of saving the conversation and the trained weights to a local folder, for later loading it, and keeping the conversation and model updates done during the session.

The checkpoint_manager is a helper that manages the saved checkpoints: the trained weights during a session with the model.

# Use cases
DaertML has explored different possible use cases that may be of use, work will be delivered as it is tested, and some surprises will come in the following days. At the moment, the field is likely using Test Time Training for:
- Software Engineering Agents: the needs to imprint the code from the repo in the weights, may benefit from providing the context of the codebase, doing some runs of inner loop training of the model, and then getting better answers than just relying on the context window, or external data sources like RAG.
- Long running conversations: as more context is used in a conversation, the time it takes, as well, as the amount of used memory increases over time; because of that, this approach may help aliviate the context window, by learning the context in a section of the model weights.
- Needle in the Haystack: even though, this use case may better be solved by other mechanisms, the Google Deepmind researchers use this as a benchmark for the solution to compare it with other kinds of LLMs (Mamba, Titans, RNN...)
- Transfer Learning: even though In Context Learning has worked for many years, and is still part of the state of the art in the path towards AGI; there are certain scenarios that need to push gradients down to model weights, in order for the model to better learn about the task at hand. Robotics may improve by using the ideas in this work.

# Limitations
Each LLM is different, it is very likely that if you provide a different model in the script from HuggingFace, the program will not work. The inference script needs adaptation for the different LLM families.
