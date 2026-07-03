"""
NumPy‑based Transformer language model (decoder‑only) mirroring the
PyTorch implementation found in oracle/oracle.py.

All sub‑modules (RMSNorm, SwiGLU, RoPE, CausalGQABlock) are imported from
the local layers package, so you can swap the numpy implementations for
GPU‑accelerated ones (e.g., using cupy) by simply changing the import
at the top of each sub‑module or by setting an environment variable.

Usage
-----
>>> import numpy as np
>>> from layers.oracle import PicoGPTOracle
>>> model = PicoGPTOracle(vocab_size=65, d_model=256, n_layers=4, n_heads=8, max_seq_len=128)
>>> logits, loss = model.forward(idx, targets)   # idx, targets: (batch, seq_len)
"""
import numpy as np
from .normalization import RMSNorm
from .activation import SwiGLU
from .position import RoPE
from .cqa import CausalGQABlock

def cross_entropy(logits, targets):
    """
    Compute cross‑entropy loss for vocabulary prediction.
    logits: (batch*seq_len, vocab_size)
    targets: (batch*seq_len,) with integer indices in [0, vocab_size)
    Returns scalar loss.
    """
    # Subtract max for numerical stability
    logits_max = np.max(logits, axis=-1, keepdims=True)
    logits = logits - logits_max
    exp_logits = np.exp(logits)
    log_sum_exp = np.log(np.sum(exp_logits, axis=-1, keepdims=True))
    # Log probabilities
    log_probs = logits - log_sum_exp  # (batch*seq_len, vocab_size)
    # Gather log prob of the correct class
    batch_size = logits.shape[0]
    correct_log_probs = log_probs[np.arange(batch_size), targets.ravel()]
    loss = -np.mean(correct_log_probs)
    return loss

class PicoGPTOracle:
    def __init__(self, vocab_size: int = 65, d_model: int = 128,
                 n_layers: int = 3, n_heads: int = 4, max_seq_len: int = 64):
        """
        Parameters
        ----------
        vocab_size : int
            Size of the token vocabulary.
        d_model : int
            Embedding dimension and model width.
        n_layers : int
            Number of transformer blocks.
        n_heads : int
            Number of attention heads (must divide d_model).
        max_seq_len : int
            Maximum sequence length for positional encodings.
        """
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        # Token embedding table
        self.token_embedding = np.random.randn(vocab_size, d_model).astype(np.float32) * np.sqrt(1.0 / vocab_size)

        # Transformer blocks
        self.layers = []
        for _ in range(n_layers):
            layer = {
                "attn_norm": RMSNorm(d_model),
                "attn": CausalGQABlock(d_model, n_heads, self.head_dim, max_seq_len),
                "mlp_norm": RMSNorm(d_model),
                "mlp": SwiGLU(d_model, d_ffn=int(2 * d_model / 3))
            }
            self.layers.append(layer)

        # Final norm and language‑model head
        self.final_norm = RMSNorm(d_model)
        # LM weight matrix: shape (vocab_size, d_model)
        self.lm_head = np.random.randn(vocab_size, d_model).astype(np.float32) * np.sqrt(1.0 / d_model)

    def forward(self, idx, targets=None):
        """
        Parameters
        ----------
        idx : np.ndarray
            Integer token IDs of shape (batch, seq_len).
        targets : np.ndarray or None
            If provided, same shape as idx; used to compute next‑token loss.

        Returns
        -------
        logits : np.ndarray
            Shape (batch, seq_len, vocab_size) – raw scores before softmax.
        loss : float or None
            Cross‑entropy loss if targets is supplied, otherwise None.
        """
        batch, seq_len = idx.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds model max_seq_len {self.max_seq_len}")

        # Embedding lookup: (batch, seq_len, d_model)
        x = self.token_embedding[idx]  # uses advanced indexing

        # Transformer blocks
        for layer in self.layers:
            # Pre‑norm attention
            attn_input = layer["attn_norm"].forward(x)
            attn_out = layer["attn"].forward(attn_input)
            x = x + attn_out  # residual connection

            # Pre‑norm MLP
            mlp_input = layer["mlp_norm"].forward(x)
            mlp_out = layer["mlp"].forward(mlp_input)
            x = x + mlp_out   # residual connection

        # Final norm
        x = self.final_norm.forward(x)

        # LM head: (batch, seq_len, vocab_size)
        logits = x @ self.lm_head.T  # (batch, seq_len, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for loss calculation
            logits_flat = logits.reshape(-1, self.vocab_size)          # (batch*seq_len, vocab_size)
            targets_flat = targets.reshape(-1)                         # (batch*seq_len,)
            loss = cross_entropy(logits_flat, targets_flat)

        return logits, loss

    # Optional: method to generate text greedily or with temperature sampling
    def generate(self, start_tokens, max_new_tokens, temperature=1.0):
        """
        Autoregressive generation.

        Parameters
        ----------
        start_tokens : np.ndarray
            Shape (1, L) – token IDs to condition on.
        max_new_tokens : int
            Number of tokens to generate.
        temperature : float, default 1.0
            Sampling temperature; lower => more deterministic.

        Returns
        -------
        np.ndarray
            Shape (1, L + max_new_tokens) – generated token IDs.
        """
        # Ensure model is in eval mode (no dropout etc.)
        generated = start_tokens.copy()
        for _ in range(max_new_tokens):
            # Use only the last max_seq_len tokens as context
            ctx = generated[:, -self.max_seq_len:] if generated.shape[1] > self.max_seq_len else generated
            logits, _ = self.forward(ctx)          # (1, ctx_len, vocab_size)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # (1, vocab_size)
            probs = np.exp(logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True)))
            # Sample from categorical distribution
            next_token = np.random.choice(self.vocab_size, p=probs.ravel())
            generated = np.concatenate([generated, [[next_token]]], axis=1)
        return generated