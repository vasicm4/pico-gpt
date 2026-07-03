"""
Layers package for PicoGPT.

Provides NumPy implementations of the transformer building blocks:
    - normalization.RMSNorm
    - activation.SwiGLU
    - position.RoPE
    - cqa.CausalGQABlock
    - oracle.PicoGPTOracle

To switch to a GPU‑accelerated backend (e.g., CuPy) simply replace the
NumPy import in each sub‑module with `import cupy as np` or set an
environment variable that triggers the swap.
"""
from .normalization import RMSNorm
from .activation import SwiGLU
from .position import RoPE
from .cqa import CausalGQABlock
from .oracle import PicoGPTOracle

__all__ = [
    "RMSNorm",
    "SwiGLU",
    "RoPE",
    "CausalGQABlock",
    "PicoGPTOracle",
]