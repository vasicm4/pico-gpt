"""
Rotary Position Embedding (RoPE) implementation using NumPy.

If you want GPU acceleration, replace the import with:
    import cupy as np
and the rest of the code works unchanged.
"""
import numpy as np

class RoPE:
    def __init__(self, head_dim: int, max_seq_len: int = 64, theta: float = 10000.0):
        """
        Parameters
        ----------
        head_dim : int
            Dimensionality of each attention head (must be even).
        max_seq_len : int, default 64
            Maximum sequence length for which positional encodings are pre‑computed.
        theta : float, default 10000.0
            Frequency base as in the original Transformer.
        """
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency terms
        inv_freq = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq)  # shape (head_dim//2,)

        # Precompute cos/sin for each position up to max_seq_len
        t = np.arange(max_seq_len, dtype=np.float32)          # (max_seq_len,)
        freqs = np.outer(t, inv_freq)                         # (max_seq_len, head_dim//2)
        self.register_buffer("cos", np.cos(freqs).repeat(2, axis=1))  # (max_seq_len, head_dim)
        self.register_buffer("sin", np.sin(freqs).repeat(2, axis=1))  # (max_seq_len, head_dim)

    def register_buffer(self, name, tensor):
        """Simple buffer registry – mimics torch.nn.Module.register_buffer."""
        setattr(self, name, tensor)

    def _rotate_half(self, x):
        """
        Rotates the last dimension by splitting into two halves and swapping with sign.
        Input:  (..., dim) where dim is even.
        Output: (..., dim)
        """
        x1 = x[..., ::2]
        # Actually split: x[..., 0::2], x[..., 1::2]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return np.concatenate([-x2, x1], axis=-1)

    def forward(self, x):
        """
        Apply rotary positional embedding to the last dimension of x.

        Parameters
        ----------
        x : np.ndarray
            Shape (batch, seq_len, num_heads, head_dim) or any shape where the
            last dimension is head_dim and the second‑to‑last dimension is seq_len.

        Returns
        -------
        np.ndarray
            Same shape as x, with rotary embedding applied.
        """
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum precomputed length {self.max_seq_len}."
            )
        cos = self.cos[:seq_len, :]          # (seq_len, head_dim)
        sin = self.sin[:seq_len, :]          # (seq_len, head_dim)

        # Reshape for broadcasting: (1, seq_len, 1, head_dim) if needed
        # We assume x.shape = (..., seq_len, head_dim)
        # Add leading dimensions of size 1 for batch/head dims if they exist.
        # Using numpy's broadcasting, we can just align the last two axes.
        # Expand dims to match x's leading dimensions.
        # Determine number of leading dims
        ndim = x.ndim
        # cos/sin shape: (seq_len, head_dim)
        # reshape to (1,)* (ndim-2) + (seq_len, head_dim)
        shape = [1] * (ndim - 2) + [seq_len, self.head_dim]
        cos = cos.reshape(shape)
        sin = sin.reshape(shape)

        return (x * cos) + (self._rotate_half(x) * sin)