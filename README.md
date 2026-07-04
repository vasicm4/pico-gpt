

PicoGPT is a completely transparent, from-scratch implementation of a generative Transformer architecture (nanoGPT) built using pure **NumPy** matrix operations. 

By bypassing heavy deep learning frameworks like PyTorch or TensorFlow for the core engine, this project serves as a fully auditable educational resource and an execution pipeline adaptable to highly constrained edge environments.

---



The codebase is modularly divided into distinct domains to separate the training engine, data stream, and verification tools:

```text
pico-gpt/
│
├── data_handling/       # Tokenization and streaming batch data loaders
├── layers/              # Core architectural layers (Linear, Attention, LayerNorm)
├── oracle/              # PyTorch reference models for gradient verification
├── notebooks/           # Scratchpads for prototyping matrix math
└── main.py              # Main execution and training loop driver
```

Key Features
- Pure NumPy Backpropagation: Every forward and backward pass (analytical derivatives via the chain rule) is mathematically written from scratch.
- Dynamic Dataset Chunking: Avoids memory bloat by streaming the TinyStories dataset, dividing text into local chunk files, and rotating data windows dynamically during training.
- The PyTorch "Oracle": A dedicated reference framework used exclusively to parallel-test NumPy gradient correctness down to the 6th decimal place.
- Blazing Fast Environment: Managed completely via uv for ultra-fast, modern Python package synchronization.


