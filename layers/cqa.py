"""Causal (multi-head) attention block with manual forward/backward (NumPy).

Despite the 'GQA' name this is plain MHA (kv heads == q heads), matching the
reference. No biases. RoPE applied to q and k.
"""
import numpy as np
from .position import RoPE
# import cupy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class CausalGQABlock:
    def __init__(self, d_model: int, n_heads: int, head_dim: int, max_seq_len: int,
                 dtype=np.float32):
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        hd = n_heads * head_dim

        def init(a, b):
            return (np.random.randn(a, b) * np.sqrt(2.0 / (a + b))).astype(dtype)

        # stored as (out, in), applied as x @ W.T   (mirrors torch.nn.Linear)
        self.w_q = init(hd, d_model)
        self.w_k = init(hd, d_model)
        self.w_v = init(hd, d_model)
        self.w_o = init(d_model, hd)

        self.rope_q = RoPE(head_dim, max_seq_len, dtype=dtype)
        self.rope_k = RoPE(head_dim, max_seq_len, dtype=dtype)

        self.mask = np.tril(np.ones((max_seq_len, max_seq_len), dtype=bool))
        self.g = {k: np.zeros_like(getattr(self, k))
                  for k in ("w_q", "w_k", "w_v", "w_o")}
        self._cache = None

    _param_names = ("w_q", "w_k", "w_v", "w_o")

    def params(self):
        return [getattr(self, n) for n in self._param_names]

    def grads(self):
        return [self.g[n] for n in self._param_names]

    def forward(self, x):
        B, T, C = x.shape
        H, hd = self.n_heads, self.head_dim
        scale = 1.0 / np.sqrt(hd)

        xf = x.reshape(-1, C)                                  # (B*T, C)
        q = (xf @ self.w_q.T).reshape(B, T, H, hd)
        k = (xf @ self.w_k.T).reshape(B, T, H, hd)
        v = (xf @ self.w_v.T).reshape(B, T, H, hd)

        q = self.rope_q.forward(q)                             # (B,T,H,hd)
        k = self.rope_k.forward(k)

        qh = q.transpose(0, 2, 1, 3)                           # (B,H,T,hd)
        kh = k.transpose(0, 2, 1, 3)
        vh = v.transpose(0, 2, 1, 3)

        scores = np.matmul(qh, kh.transpose(0, 1, 3, 2)) * scale   # (B,H,T,T)
        m = self.mask[:T, :T][None, None]                     # (1,1,T,T) bool
        scores = np.where(m, scores, -1e9)
        attn = softmax(scores, axis=-1)                       # (B,H,T,T)

        ctx = np.matmul(attn, vh)                             # (B,H,T,hd)
        ctx_bt = ctx.transpose(0, 2, 1, 3).reshape(B, T, H * hd)
        out = ctx_bt.reshape(-1, H * hd) @ self.w_o.T         # (B*T, C)
        out = out.reshape(B, T, C)

        self._cache = (xf, qh, kh, vh, attn, ctx_bt, m, scale, B, T, C, H, hd)
        return out

    def backward(self, dout):
        (xf, qh, kh, vh, attn, ctx_bt, m, scale,
         B, T, C, H, hd) = self._cache

        dof = dout.reshape(-1, C)                             # (B*T, C)
        ctx_flat = ctx_bt.reshape(-1, H * hd)
        self.g["w_o"] = dof.T @ ctx_flat                     # (C, H*hd)
        dctx_bt = (dof @ self.w_o).reshape(B, T, H, hd)      # (B,T,H,hd)
        dctx = dctx_bt.transpose(0, 2, 1, 3)                 # (B,H,T,hd)

        # ctx = attn @ v
        dattn = np.matmul(dctx, vh.transpose(0, 1, 3, 2))    # (B,H,T,T)
        dvh = np.matmul(attn.transpose(0, 1, 3, 2), dctx)    # (B,H,T,hd)

        # softmax backward (row-wise)
        s = np.sum(dattn * attn, axis=-1, keepdims=True)
        dscores = attn * (dattn - s)                         # (B,H,T,T)
        dscores = np.where(m, dscores, 0.0)                  # masked entries are constants
        dscores *= scale

        # scores = qh @ kh^T
        dqh = np.matmul(dscores, kh)                         # (B,H,T,hd)
        dkh = np.matmul(dscores.transpose(0, 1, 3, 2), qh)  # (B,H,T,hd)

        # back to (B,T,H,hd)
        dq = dqh.transpose(0, 2, 1, 3)
        dk = dkh.transpose(0, 2, 1, 3)
        dv = dvh.transpose(0, 2, 1, 3)

        # rope backward (q, k)
        dq = self.rope_q.backward(dq)
        dk = self.rope_k.backward(dk)

        # projections: q = xf @ w_q.T
        dq2 = dq.reshape(-1, H * hd)
        dk2 = dk.reshape(-1, H * hd)
        dv2 = dv.reshape(-1, H * hd)
        self.g["w_q"] = dq2.T @ xf
        self.g["w_k"] = dk2.T @ xf
        self.g["w_v"] = dv2.T @ xf
        dxf = dq2 @ self.w_q + dk2 @ self.w_k + dv2 @ self.w_v
        return dxf.reshape(B, T, C)