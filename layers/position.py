
# import numpy as np
import cupy as np

class RoPE:
    def __init__(self, head_dim: int, max_seq_len: int = 64, theta: float = 10000.0,
                 dtype=np.float32):
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.dtype = dtype

        inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        t = np.arange(max_seq_len, dtype=np.float64)
        freqs = np.outer(t, inv_freq)
        self.cos = np.repeat(np.cos(freqs), 2, axis=1).astype(dtype)
        self.sin = np.repeat(np.sin(freqs), 2, axis=1).astype(dtype)
        self._seq = None  # cache seq_len used in forward


    def params(self):
        return []

    def grads(self):
        return []

    @staticmethod
    def _rotate_half(x):

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return np.stack((-x_odd, x_even), axis=-1).reshape(x.shape)

    @staticmethod
    def _rotate_half_T(g):

        g_even = g[..., 0::2]
        g_odd = g[..., 1::2]
        return np.stack((g_odd, -g_even), axis=-1).reshape(g.shape)

    def _cs(self, seq_len, ndim):
        cos = self.cos[:seq_len, :]
        sin = self.sin[:seq_len, :]
        shape = [1] * (ndim - 2) + [seq_len, self.head_dim]
        return cos.reshape(shape), sin.reshape(shape)

    def forward(self, x):

        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max {self.max_seq_len}")
        self._seq = seq_len
        cos, sin = self._cs(seq_len, x.ndim)
        return x * cos + self._rotate_half(x) * sin

    def backward(self, dy):

        seq_len = self._seq
        cos, sin = self._cs(seq_len, dy.ndim)
        return dy * cos + self._rotate_half_T(dy * sin)