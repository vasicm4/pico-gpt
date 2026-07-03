"""
Causal Grouped‑Query Attention (GQA) block implemented with NumPy.

If you want GPU acceleration, replace the import with:
    import cupy as np
and the rest of the code works unchanged.
"""
import numpy as np
from .position import RoPE

def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class CausalGQABlock:
    def __init__(self, d_model: int, n_heads: int, head_dim: int, max_seq_len: int):
        """
        Parameters
        ----------
        d_model : int
            Model dimension (input and output feature size).
        n_heads : int
            Number of query heads.
        head_dim : int
            Dimensionality of each head (must be even for RoPE).
        max_seq_len : int
            Maximum sequence length for the causal mask and RoPE.
        """
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Projection matrices (no bias, following typical LLM practice)
        self.w_q = np.random.randn(d_model, n_heads * head_dim).astype(np.float32) * np.sqrt(2.0 / (d_model + n_heads * head_dim))
        self.w_k = np.random.randn(d_model, n_heads * head_dim).astype(np.float32) * np.sqrt(2.0 / (d_model + n_heads * head_dim))
        self.w_v = np.random.randn(d_model, n_heads * head_dim).astype(np.float32) * np.sqrt(2.0 / (d_model + n_heads * head_dim))
        self.w_o = np.random.randn(n_heads * head_dim, d_model).astype(np.float32) * np.sqrt(2.0 / (n_heads * head_dim + d_model))

        # RoPE instance
        self.rope = RoPE(head_dim, max_seq_len)

        # Pre‑computed causal mask (lower triangular)
        mask = np.tril(np.ones((max_seq_len, max_seq_len), dtype=np.float32))
        # Where mask == 0 we will later set -inf
        self.register_buffer("mask", mask)

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def forward(self, x):
        """
        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (batch, seq_len, d_model)

        Returns
        -------
        np.ndarray
            Output tensor of same shape as input.
        """
        batch, seq_len, _ = x.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")

        # Linear projections
        q = x @ self.w_q.T  # (batch, seq_len, n_heads * head_dim)
        k = x @ self.w_k.T
        v = x @ self.w_v.T

        # Reshape to (batch, seq_len, n_heads, head_dim)
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to q and k
        q = self.rope.forward(q)
        k = self.rope.forward(k)

        # Transpose to (batch, n_heads, seq_len, head_dim) for matmul
        q = q.transpose(0, 2, 1, 3)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot‑product attention
        # scores shape: (batch, n_heads, seq_len, seq_len)
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # Apply causal mask
        mask = self.mask[:seq_len, :seq_len]  # (seq_len, seq_len)
        mask = mask[None, None, :, :]         # (1, 1, seq_len, seq_len)
        scores = np.where(mask == 0, -1e9, scores)  # large negative instead of -inf for stability

        # Softmax over key dimension
        attn_weights = softmax(scores, axis=-1)  # (batch, n_heads, seq_len, seq_len)

        # Weighted sum of values
        context = np.matmul(attn_weights, v)  # (batch, n_heads, seq_len, head_dim)

        # Concatenate heads: (batch, seq_len, n_heads * head_dim)
        context = context.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.n_heads * self.head_dim)

        # Output projection
        output = context @ self.w_o.T  # (batch, seq_len, d_model)
        return output