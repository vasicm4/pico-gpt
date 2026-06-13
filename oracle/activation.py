import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * d_ffn, bias=False)
        self.w3 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = torch.chunk(x12, 2, dim=-1)
        return self.w3(F.silu(x1) * x2)