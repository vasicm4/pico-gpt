import torch
import torch.nn as nn
import torch.nn.functional as F
from position import RoPE


class CausalGQABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, head_dim: int, max_seq_len: int):
        super().__init__()
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # how many Q heads share one KV head
        self.head_dim = head_dim

        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.rope = RoPE(head_dim, max_seq_len)

        self.register_buffer("mask", torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def _repeat_kv(self, x):
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        return x[:, :, None, :, :].expand(B, H, self.n_rep, T, D).reshape(B, self.n_heads, T, D)

    def forward(self, x):
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        attention_weights = F.softmax(scores, dim=-1)

        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(context)
