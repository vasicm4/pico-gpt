"""
SwiGLU activation (gate‑fed linear).

If you prefer a GPU‑accelerated version, replace the numpy import with cupy:
    import cupy as np
The rest of the implementation stays unchanged.

Implementation:
    y = Linear(x) -> split into (x1, x2)
    return Linear2( silu(x1) * x2 )
where silu(x) = x * sigmoid(x)
"""
import numpy as np

class SwiGLU:
    def __init__(self, d_model: int, d_ffn: int):
        """
        Parameters
        ----------
        d_model : int
            Input and output feature dimension.
        d_ffn : int
            Hidden dimension of the inner linear projection
            (the gate+value projection outputs 2 * d_ffn).
        """
        self.d_model = d_model
        self.d_ffn = d_ffn
        # w12 projects from d_model -> 2*d_ffn
        self.w12 = np.random.randn(2 * d_ffn, d_model).astype(np.float32) * np.sqrt(2.0 / (d_model + 2 * d_ffn))
        self.b12 = np.zeros((2 * d_ffn,), dtype=np.float32)
        # w3 projects from d_ffn -> d_model
        self.w3 = np.random.randn(d_model, d_ffn).astype(np.float32) * np.sqrt(2.0 / (d_ffn + d_model))
        self.b3 = np.zeros((d_model,), dtype=np.float32)

    def forward(self, x):
        """
        Parameters
        ----------
        x : np.ndarray
            Input tensor of shape (..., d_model)

        Returns
        -------
        np.ndarray
            Output tensor of shape (..., d_model)
        """
        # Linear projection to gate and value: (..., 2*d_ffn)
        x12 = x @ self.w12.T + self.b12
        gate, value = np.split(x12, 2, axis=-1)  # each (..., d_ffn)

        # SiLU activation: x * sigmoid(x)
        silu_gate = gate * (1.0 / (1.0 + np.exp(-gate)))

        # Element‑wise product
        activated = silu_gate * value              # (..., d_ffn)

        # Final linear projection to d_model
        out = activated @ self.w3.T + self.b3      # (..., d_model)
        return out