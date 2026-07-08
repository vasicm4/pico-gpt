# PicoGPT

A completely transparent, from-scratch implementation of a generative Transformer (decoder-only GPT) built using **CuPy** as the core numerical backend - with a **PyTorch** reference model used solely for gradient verification.

By bypassing the heavy deep-learning framework for the training engine, the core forward/backward passes are written as explicit matrix math (analytical derivatives via the chain rule). The result is a fully auditable educational resource and an execution pipeline that can run end-to-end on a single GPU.

---

## Architecture

The model is a decoder-only Transformer with a modern block layout:

| Component            | Choice                                                  |
|----------------------|---------------------------------------------------------|
| Normalization        | RMSNorm (pre-norm)                                      |
| Position encoding    | Rotary Position Embeddings (RoPE)                       |
| Attention            | Causal Grouped-Query Attention (GQA - MHA as a special case) |
| Feed-forward         | SwiGLU                                                  |
| Tokenization         | Character-level over a fixed printable vocab           |
| Loss                 | Cross-entropy (mean over tokens)                        |
| Optimizer            | AdamW with global-norm gradient clipping                |

Default model size used by `main.py`: `d_model=256, n_layers=4, n_heads=8, max_seq_len=128`.

---

## Project layout

```text
pico-gpt/
│
├── data_handling/          # Tokenization and streaming batch data loaders
│   ├── tokenizer.py            # CharacterTokenizer (stoi / itos, encode, decode)
│   ├── batch_loader.py         # PyTorch-style batch loader (used by oracle/)
│   ├── batch_loader_np.py      # Streaming chunked loader for the NumPy engine
│   └── chunker_loader.py
│
├── layers/                 # CuPy/NumPy training engine (forward + backward)
│   ├── normalization.py        # RMSNorm
│   ├── position.py             # RoPE
│   ├── cqa.py                  # Causal Grouped-Query Attention block
│   ├── activation.py           # SwiGLU
│   ├── optimizer.py            # AdamW + global grad-norm clipping
│   ├── oracle.py               # PicoGPTOracle - full model (training + inference)
│   └── runner.py               # Training / eval loop, entropy tracking, dashboard plot
│
├── oracle/                 # PyTorch reference model for gradient verification
│   ├── oracle.py
│   ├── cqa.py
│   ├── normalization.py
│   ├── activation.py
│   ├── position.py
│   └── main.py
│
├── notebooks/              # Scratchpads for prototyping matrix math
│   ├── data_cleaning.ipynb
│   └── training.ipynb
│
├── data_handling/data/     # Streaming chunk files used by the batch loader
│
├── main.py                 # Entry point - `train` (default) or `infer`
├── gradcheck.py            # Finite-difference gradient check vs. analytic grads
├── test_layers.py          # Forward-pass smoke tests for every layer
├── smoke.py                # Quick end-to-end training + generation smoke test
│
├── pico_gpt_oracle.pth           # Pre-trained PyTorch reference weights
├── pico_gpt_oracle_np.npz        # Pre-trained weights in NumPy format
└── training_samples.txt          # Per-step qualitative samples logged during training
```

---

## Key features

- **Pure-matrix backpropagation.** Every layer implements its own `forward` and `backward`. Gradients are derived analytically (chain rule) and verified against finite-difference numerical gradients down to relative error < 1e-5.
- **Causal Grouped-Query Attention.** Q heads share K/V heads in configurable groups; defaults to standard MHA when `n_kv_heads == n_heads`.
- **RMSNorm + RoPE + SwiGLU.** Modern decoder block; RMSNorm scales are foldable into adjacent linear layers for an `forward_inference` path that skips the norm multiply on the hot path.
- **Dynamic dataset chunking.** The training loader streams the TinyStories dataset from local chunk files and rotates the active window every `swap_every_iterations` steps, avoiding the cost of holding the corpus in memory.
- **PyTorch "oracle".** A PyTorch implementation of the same architecture in `oracle/` is used as a ground-truth reference - first to seed a checkpoint, and then as a finite-difference baseline in `gradcheck.py`.
- **First-class evaluation.** Each evaluation step logs train loss, val loss, val perplexity, top-1/top-5 token accuracy, the train–val generalization gap, and the mean next-token generation entropy at multiple temperatures. Results are written to `training_samples.txt` and plotted to `training_eval_dashboard.png`.
- **Modern Python packaging.** Dependencies are pinned in `pyproject.toml` and resolved via `uv` for fast, reproducible installs. Python 3.12.

---

## Quick start

```bash
# 1. Install dependencies (requires uv)
uv sync

# 2. Train from scratch
uv run main.py

# 3. Load the bundled checkpoint and chat
uv run main.py infer
```

Entry point behaviour is selected by the first CLI argument:

| Command              | Effect                                                                 |
|----------------------|------------------------------------------------------------------------|
| `uv run main.py`     | Runs `train()` - full training loop, eval dashboard, saves `pico_gpt_oracle_np.npz` |
| `uv run main.py infer` | Loads `pico_gpt_oracle_np.npz` and starts an interactive prompt loop |

`smoke.py` is a smaller-scale end-to-end check (B=16, T=64, d_model=128, 3 layers) and finishes with a generation sample.

---

## Verification

```bash
uv run gradcheck.py
```

Runs the full model on a small random batch in `float64`, compares analytic gradients (via the chain rule in `layers/oracle.py`) against central finite differences with `EPS=1e-5`. Tolerance is `ATOL=1e-7` or relative error `RTOL=1e-5`. Every layer - `RMSNorm`, `SwiGLU`, `RoPE`, `CausalGQABlock`, and the full `PicoGPTOracle` - must pass for the run to be green.

```bash
uv run test_layers.py
```

Forward-pass shape checks for every layer; useful as a fast smoke test.

---

## Data

Training data is expected under `data_handling/data/` as text files named:

- `train_chunk_*.txt` - training shards
- `tiny_stories_val*.txt` - validation

The bundled `notebooks/data_cleaning.ipynb` shows how the chunks are produced. The character-level vocabulary used by `main.py` covers ASCII letters, digits, common punctuation, and a small set of typographic quotes/dashes - a vocab of ~80 characters.

---

## Outputs

Running `main.py` produces:

| File                          | Content                                                  |
|-------------------------------|----------------------------------------------------------|
| `pico_gpt_oracle_np.npz`      | Final trained weights (NumPy format)                     |
| `training_eval_dashboard.png` | 3-panel plot: loss + perplexity, generalization gap, generation entropy |
| `training_samples.txt`        | Per-step qualitative samples at multiple temperatures   |

---

## Why CuPy instead of NumPy?

`layers/*.py` imports `cupy as np` rather than `numpy as np`. The CuPy source line is left commented at the top of each module - flipping the two lines drops the entire engine back to NumPy with no other code changes. The PyTorch oracle stays as the reference, and `gradcheck.py` continues to validate correctness in either mode.
