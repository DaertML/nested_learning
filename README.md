# nested_learning
Tiny implementation of Nested Learning ideas from Google Deepmind on already pretrained LLMs

# Introduction
This is the first glympse of Continual Learning that has happened at DaertML; from the ideas of Google Deepmind about how LLMs should do Test Time Training, in order to modify its weights to the task at hand, and get the information right into the model, instead of using the context window.
Google ideas go in a slightly different path, requiring the model to be trained with certain characteristics prior to inference, and using the Titan architecture. In an attempt to avoid training models from scratch and getting the most ouf of existing LLMs, the implementation from DaertML performs Test Time Training on existing LLMs.

We support the following models at the moment (tested):
- Gemma 3 270M Instruction: ✔️

# Limitations
Each LLM is different, it is very likely that if you provide a different model in the script from HuggingFace, the program will not work. The inference script needs adaptation for the different LLM families.
