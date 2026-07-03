"""
RMS Normalization layer.

Drop‑in replacement for torch.nn.RMSNorm.

If you have installed CuPy, simply replace the import
```
import cupy as np   # or: import numpy as np
```
and the rest of the code stays the same.
"""

import numpy as np

class RMSNorm:
    def __init__(self, dimensions: int, epsilon: float = 1e-6):
        self.epsilon = epsilon
        self.weight = np.ones(dimensions, dtype=np.float32)

    def forward(self, x):
        # x shape: (..., dimensions)
        # compute root mean square over the last dimension
        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + self.epsilon)
        return x / rms * self.weight