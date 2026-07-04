"""NumPy decoder-only transformer (PicoGPTOracle) with manual backprop."""
import numpy as np
# import cupy as np
from .normalization import RMSNorm
from .activation import SwiGLU
from .cqa import CausalGQABlock, softmax


def cross_entropy_fwd(logits_flat, targets_flat):
    """logits_flat: (N, V); targets_flat: (N,). Returns (loss, probs)."""
    m = np.max(logits_flat, axis=-1, keepdims=True)
    e = np.exp(logits_flat - m)
    probs = e / np.sum(e, axis=-1, keepdims=True)
    N = logits_flat.shape[0]
    correct = probs[np.arange(N), targets_flat]
    loss = -np.mean(np.log(correct + 1e-12))
    return loss, probs


class PicoGPTOracle:
    def __init__(self, vocab_size=65, d_model=128, n_layers=3, n_heads=4,
                 max_seq_len=64, n_kv_heads=None, dtype=np.float32):
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if n_kv_heads is None:
            n_kv_heads = n_heads
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len
        self.dtype = dtype

        self.token_embedding = (np.random.randn(vocab_size, d_model) *
                                np.sqrt(1.0 / vocab_size)).astype(dtype)
        self.g_emb = np.zeros_like(self.token_embedding)

        d_ffn = int(2 * d_model / 3)
        self.layers = []
        for _ in range(n_layers):
            self.layers.append({
                "attn_norm": RMSNorm(d_model, dtype=dtype),
                "attn": CausalGQABlock(d_model, n_heads, self.head_dim, max_seq_len,
                                        n_kv_heads=n_kv_heads, dtype=dtype),
                "mlp_norm": RMSNorm(d_model, dtype=dtype),
                "mlp": SwiGLU(d_model, d_ffn, dtype=dtype),
            })

        self.final_norm = RMSNorm(d_model, dtype=dtype)
        self.lm_head = (np.random.randn(vocab_size, d_model) *
                        np.sqrt(1.0 / d_model)).astype(dtype)
        self.g_head = np.zeros_like(self.lm_head)
        self._cache = None

    # ---------- parameter aggregation (STABLE order) ----------
    def _modules_in_order(self):
        mods = []
        for L in self.layers:
            mods += [L["attn_norm"], L["attn"], L["mlp_norm"], L["mlp"]]
        mods.append(self.final_norm)
        return mods

    def params(self):
        ps = [self.token_embedding]
        for m in self._modules_in_order():
            ps += m.params()
        ps.append(self.lm_head)
        return ps

    def grads(self):
        gs = [self.g_emb]
        for m in self._modules_in_order():
            gs += m.grads()
        gs.append(self.g_head)
        return gs

    def named_params(self):
        names = ["token_embedding"]
        for i, L in enumerate(self.layers):
            for key in ("attn_norm", "attn", "mlp_norm", "mlp"):
                for pn in L[key]._param_names:
                    names.append(f"layers.{i}.{key}.{pn}")
        for pn in self.final_norm._param_names:
            names.append(f"final_norm.{pn}")
        names.append("lm_head")
        return dict(zip(names, self.params()))

    # ---------- forward ----------
    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError(f"seq_len {T} exceeds max {self.max_seq_len}")
        x = self.token_embedding[idx]                     # (B,T,d)

        for L in self.layers:
            x = x + L["attn"].forward(L["attn_norm"].forward(x))
            x = x + L["mlp"].forward(L["mlp_norm"].forward(x))

        x = self.final_norm.forward(x)
        logits = x @ self.lm_head.T                       # (B,T,V)

        loss = None
        cache = None
        if targets is not None:
            lf = logits.reshape(-1, self.vocab_size)
            tf = targets.reshape(-1)
            loss, probs = cross_entropy_fwd(lf, tf)
            cache = (idx, x, probs, tf, B, T)
        self._cache = cache
        return logits, loss

    # ---------- backward (call after forward with targets) ----------
    def backward(self):
        idx, x, probs, tf, B, T = self._cache
        N = probs.shape[0]
        V, d = self.vocab_size, self.d_model

        # cross-entropy grad wrt logits
        dlogits = probs.copy()
        dlogits[np.arange(N), tf] -= 1.0
        dlogits /= N                                      # (N, V)

        xf = x.reshape(-1, d)                             # (N, d)
        self.g_head = dlogits.T @ xf                      # (V, d)
        dx = (dlogits @ self.lm_head).reshape(B, T, d)    # (B,T,d)

        dx = self.final_norm.backward(dx)

        for L in reversed(self.layers):
            # x = x + mlp(mlp_norm(x))
            d_branch = L["mlp_norm"].backward(L["mlp"].backward(dx))
            dx = dx + d_branch
            # x = x + attn(attn_norm(x))
            d_branch = L["attn_norm"].backward(L["attn"].backward(dx))
            dx = dx + d_branch

        # embedding: scatter-add
        self.g_emb = np.zeros_like(self.token_embedding)
        np.add.at(self.g_emb, idx.reshape(-1), dx.reshape(-1, d))
        return None

    # ---------- inference: fuse each RMSNorm's gamma into the next matmul ----------
    def fuse_norms_for_inference(self):
        """Precompute fused weights so inference skips the separate
        `* gamma` step of every RMSNorm and folds it directly into the
        following linear layer's weight matrix.

        Only the elementwise gamma is foldable -- the x/rms(x) division is
        data-dependent and still has to run per token. This does not touch
        the original trained weights (self.w_q etc.), so training / saving
        / loading are unaffected; it just adds *_fused arrays and a flag.
        Call once after loading weights, before running forward_inference().
        """
        for L in self.layers:
            g_attn = L["attn_norm"].weight            # (d_model,)
            attn = L["attn"]
            attn.w_q_fused = attn.w_q * g_attn         # broadcasts over the input-dim columns
            attn.w_k_fused = attn.w_k * g_attn
            attn.w_v_fused = attn.w_v * g_attn

            g_mlp = L["mlp_norm"].weight
            L["mlp"].w12_fused = L["mlp"].w12 * g_mlp

        self.lm_head_fused = self.lm_head * self.final_norm.weight
        self._fused = True

    def forward_inference(self, idx):
        """Same computation as forward(), but skips every RMSNorm's gamma
        multiply by using the fused weights instead. No targets/loss, no
        cache kept for backward -- inference only. Call
        fuse_norms_for_inference() once beforehand.
        """
        if not getattr(self, "_fused", False):
            self.fuse_norms_for_inference()

        B, T = idx.shape
        if T > self.max_seq_len:
            raise ValueError(f"seq_len {T} exceeds max {self.max_seq_len}")
        x = self.token_embedding[idx]

        for L in self.layers:
            normed = L["attn_norm"].forward(x, apply_weight=False)
            x = x + L["attn"].forward(normed, w_q=L["attn"].w_q_fused,
                                       w_k=L["attn"].w_k_fused, w_v=L["attn"].w_v_fused)
            normed = L["mlp_norm"].forward(x, apply_weight=False)
            x = x + L["mlp"].forward(normed, w12=L["mlp"].w12_fused)

        normed = self.final_norm.forward(x, apply_weight=False)
        logits = normed @ self.lm_head_fused.T
        return logits

    # ---------- inference ----------
    def generate(self, start_tokens, max_new_tokens, temperature=1.0, rng=None):
        rng = rng or np.random
        gen = np.array(start_tokens, dtype=np.int64)
        for _ in range(max_new_tokens):
            ctx = gen[:, -self.max_seq_len:]
            logits, _ = self.forward(ctx)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = softmax(logits, axis=-1)
            nxt = np.array([[rng.choice(self.vocab_size, p=probs[0])]], dtype=np.int64)
            gen = np.concatenate([gen, nxt], axis=1)
        return gen

    # ---------- checkpoint ----------
    def save(self, path):
        np.savez(path, **{k: v for k, v in self.named_params().items()})

    def load(self, path):
        data = np.load(path if path.endswith(".npz") else path + ".npz")
        current = self.named_params()
        for k, v in current.items():
            v[...] = data[k]            # in-place so references stay valid