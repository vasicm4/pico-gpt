"""RMS Normalization with manual forward/backward (NumPy)."""
import numpy as np
# import cupy as np

class RMSNorm:
    def __init__(self, dimensions: int, epsilon: float = 1e-6, dtype=np.float32):
        self.dim = dimensions
        self.epsilon = epsilon
        self.dtype = dtype
        self.weight = np.ones(dimensions, dtype=dtype)
        # gradient buffers + cache
        self.g = {"weight": np.zeros_like(self.weight)}
        self._cache = None

    # --- parameter plumbing (stable order) ---
    _param_names = ("weight",)

    def params(self):
        return [getattr(self, n) for n in self._param_names]

    def grads(self):
        return [self.g[n] for n in self._param_names]

    def forward(self, x):
        # x: (..., D)
        s = np.mean(np.square(x), axis=-1, keepdims=True) + self.epsilon  # (...,1)
        r = np.sqrt(s)                                                    # rms
        n = x / r                                                         # normalized
        self._cache = (x, r, s, n)
        return n * self.weight

    def backward(self, dy):
        x, r, s, n = self._cache
        D = x.shape[-1]
        # weight grad: sum over every axis except the last
        self.g["weight"] = np.sum(dy * n, axis=tuple(range(dy.ndim - 1)))
        dn = dy * self.weight
        # dx = (1/r) * (dn - x * (sum(dn*x)/(D*s)))
        dot = np.sum(dn * x, axis=-1, keepdims=True)
        dx = (dn - x * (dot / (D * s))) / r
        return dx