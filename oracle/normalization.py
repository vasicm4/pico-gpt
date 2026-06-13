import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dimensions: int, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.weight = nn.Parameter(torch.ones(dimensions))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)
        return x * rms * self.weight