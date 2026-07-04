"""SwiGLU activation with manual forward/backward (NumPy).

Matches the PyTorch reference: no biases.
    y = w3( silu(x @ w12.T)[:d_ffn]  *  (x @ w12.T)[d_ffn:] )
with silu(z) = z * sigmoid(z).
Weight orientation mirrors torch.nn.Linear (out, in), applied as x @ W.T.
"""
import numpy as np
# import cupy as np

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class SwiGLU:
    def __init__(self, d_model: int, d_ffn: int, dtype=np.float32):
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dtype = dtype
        self.w12 = (np.random.randn(2 * d_ffn, d_model) *
                    np.sqrt(2.0 / (d_model + 2 * d_ffn))).astype(dtype)
        self.w3 = (np.random.randn(d_model, d_ffn) *
                   np.sqrt(2.0 / (d_ffn + d_model))).astype(dtype)
        self.g = {"w12": np.zeros_like(self.w12), "w3": np.zeros_like(self.w3)}
        self._cache = None

    _param_names = ("w12", "w3")

    def params(self):
        return [getattr(self, n) for n in self._param_names]

    def grads(self):
        return [self.g[n] for n in self._param_names]

    def forward(self, x, w12=None):
        w12 = self.w12 if w12 is None else w12
        # x: (..., d_model) -> flatten leading dims for the matmuls
        lead = x.shape[:-1]
        xf = x.reshape(-1, self.d_model)                 # (N, d_model)
        x12 = xf @ w12.T                                 # (N, 2*d_ffn)
        gate, value = np.split(x12, 2, axis=-1)          # each (N, d_ffn)
        sig = _sigmoid(gate)
        silu = gate * sig                                # (N, d_ffn)
        act = silu * value                               # (N, d_ffn)
        out = act @ self.w3.T                            # (N, d_model)
        self._cache = (xf, gate, sig, value, act, lead)
        return out.reshape(*lead, self.d_model)

    def backward(self, dout):
        xf, gate, sig, value, act, lead = self._cache
        dof = dout.reshape(-1, self.d_model)             # (N, d_model)
        # out = act @ w3.T
        self.g["w3"] = dof.T @ act                       # (d_model, d_ffn)
        dact = dof @ self.w3                             # (N, d_ffn)
        # act = silu * value
        silu = gate * sig
        dsilu = dact * value
        dvalue = dact * silu
        # silu = gate * sigmoid(gate) -> dsilu/dgate = sig*(1 + gate*(1-sig))
        dgate = dsilu * (sig * (1.0 + gate * (1.0 - sig)))
        dx12 = np.concatenate([dgate, dvalue], axis=-1)  # (N, 2*d_ffn)
        # x12 = xf @ w12.T
        self.g["w12"] = dx12.T @ xf                      # (2*d_ffn, d_model)
        dxf = dx12 @ self.w12                            # (N, d_model)
        return dxf.reshape(*lead, self.d_model)