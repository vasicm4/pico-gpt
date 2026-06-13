import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 64, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, torch.tensor(inv_freq))
        self.register_buffer("cos", torch.cos(freqs).repeat_interleave(2, dim=-1))
        self.register_buffer("sin", torch.sin(freqs).repeat_interleave(2, dim=-1))

    def _rotate_half(self, x):
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        res = torch.stack((-x2, x1), dim=-1)
        return res.flatten(start_dim=-2)

    def forward(self, x):
        seq_len = x.shape[2]
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(1)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)