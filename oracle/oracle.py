from torch import nn
import torch.nn.functional as F
from normalization import RMSNorm
from activation import SwiGLU
from cqa import CausalGQABlock

class PicoGPTOracle(nn.Module):
    def __init__(self, vocab_size=65, d_model=128, n_layers=3, n_heads=4, max_seq_len=64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.head_dim = d_model // n_heads

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "attn_norm": RMSNorm(d_model),
                "attn": CausalGQABlock(d_model, n_heads, self.head_dim, max_seq_len),
                "mlp_norm": RMSNorm(d_model),
                "mlp": SwiGLU(d_model, d_ffn=int(2 * d_model / 3))  # standard SwiGLU scaling
            }))

        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx)

        for layer in self.layers:
            x = x + layer["attn"](layer["attn_norm"](x))
            x = x + layer["mlp"](layer["mlp_norm"](x))

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss