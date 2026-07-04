# import numpy as np
import cupy as np
from .position import RoPE


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class CausalGQABlock:
    def __init__(self, d_model: int, n_heads: int, head_dim: int, max_seq_len: int,
                 n_kv_heads: int = None, dtype=np.float32):
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # Q heads sharing one KV head
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.dtype = dtype
        qd = n_heads * head_dim
        kvd = n_kv_heads * head_dim

        def init(a, b):
            return (np.random.randn(a, b) * np.sqrt(2.0 / (a + b))).astype(dtype)


        self.w_q = init(qd, d_model)
        self.w_k = init(kvd, d_model)
        self.w_v = init(kvd, d_model)
        self.w_o = init(d_model, qd)

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

    def forward(self, x, w_q=None, w_k=None, w_v=None):



        w_q = self.w_q if w_q is None else w_q
        w_k = self.w_k if w_k is None else w_k
        w_v = self.w_v if w_v is None else w_v

        B, T, C = x.shape
        H, KH, hd = self.n_heads, self.n_kv_heads, self.head_dim
        scale = 1.0 / np.sqrt(hd)

        xf = x.reshape(-1, C)                                  # (B*T, C)
        q = (xf @ w_q.T).reshape(B, T, H, hd)
        k = (xf @ w_k.T).reshape(B, T, KH, hd)
        v = (xf @ w_v.T).reshape(B, T, KH, hd)

        q = self.rope_q.forward(q)                             # (B,T,H,hd)
        k = self.rope_k.forward(k)                             # (B,T,KH,hd)

        qh = q.transpose(0, 2, 1, 3)                           # (B,H,T,hd)
        kh = k.transpose(0, 2, 1, 3)                           # (B,KH,T,hd)
        vh = v.transpose(0, 2, 1, 3)                           # (B,KH,T,hd)




        if self.n_rep > 1:
            kh = np.broadcast_to(kh[:, :, None], (B, KH, self.n_rep, T, hd))
            kh = kh.reshape(B, KH * self.n_rep, T, hd)
            vh = np.broadcast_to(vh[:, :, None], (B, KH, self.n_rep, T, hd))
            vh = vh.reshape(B, KH * self.n_rep, T, hd)

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
        KH, n_rep = self.n_kv_heads, self.n_rep

        dof = dout.reshape(-1, C)                             # (B*T, C)
        ctx_flat = ctx_bt.reshape(-1, H * hd)
        self.g["w_o"] = dof.T @ ctx_flat                     # (C, H*hd)
        dctx_bt = (dof @ self.w_o).reshape(B, T, H, hd)      # (B,T,H,hd)
        dctx = dctx_bt.transpose(0, 2, 1, 3)                 # (B,H,T,hd)


        dattn = np.matmul(dctx, vh.transpose(0, 1, 3, 2))    # (B,H,T,T)
        dvh = np.matmul(attn.transpose(0, 1, 3, 2), dctx)    # (B,H,T,hd)


        s = np.sum(dattn * attn, axis=-1, keepdims=True)
        dscores = attn * (dattn - s)                         # (B,H,T,T)
        dscores = np.where(m, dscores, 0.0)                  # masked entries are constants
        dscores *= scale


        dqh = np.matmul(dscores, kh)                         # (B,H,T,hd)
        dkh = np.matmul(dscores.transpose(0, 1, 3, 2), qh)  # (B,H,T,hd)


        dq = dqh.transpose(0, 2, 1, 3)
        dk = dkh.transpose(0, 2, 1, 3)
        dv = dvh.transpose(0, 2, 1, 3)




        if n_rep > 1:
            dk = dk.reshape(B, T, KH, n_rep, hd).sum(axis=3) # (B,T,KH,hd)
            dv = dv.reshape(B, T, KH, n_rep, hd).sum(axis=3)


        dq = self.rope_q.backward(dq)
        dk = self.rope_k.backward(dk)


        dq2 = dq.reshape(-1, H * hd)
        self.g["w_q"] = dq2.T @ xf
        dxf = dq2 @ self.w_q

        if n_rep > 1:
            dk2 = dk.reshape(-1, KH * hd)
            dv2 = dv.reshape(-1, KH * hd)
            self.g["w_k"] = dk2.T @ xf
            self.g["w_v"] = dv2.T @ xf
            dxf = dxf + dk2 @ self.w_k + dv2 @ self.w_v
        else:

            dk2 = dk.reshape(-1, H * hd)
            dv2 = dv.reshape(-1, H * hd)
            self.g["w_k"] = dk2.T @ xf
            self.g["w_v"] = dv2.T @ xf
            dxf = dxf + dk2 @ self.w_k + dv2 @ self.w_v
        return dxf.reshape(B, T, C)