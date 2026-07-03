"""Finite-difference gradient checks (float64) for every layer + full model."""
from layers.normalization import RMSNorm
from layers.activation import SwiGLU
from layers.position import RoPE
from layers.cqa import CausalGQABlock
from layers.oracle import PicoGPTOracle
import numpy as np
import sys

sys.path.insert(0, "pico_np")


np.random.seed(0)
F64 = np.float64
EPS = 1e-5
RTOL = 1e-5
ATOL = 1e-7  # a gradient this small in absolute terms is 'zero' for FD purposes


def rel_err(a, b):
    # vector-level relative error, robust to individual near-zero coordinates
    return np.max(np.abs(a - b)) / (np.max(np.abs(a)) + np.max(np.abs(b)) + 1e-12)


def passed(a, b):
    return (np.max(np.abs(a - b)) < ATOL) or (rel_err(a, b) < RTOL)


def check_layer(name, layer, x, seed=0):
    """Scalar loss = sum(y * c) for fixed random c. Check dx and all param grads."""
    rng = np.random.RandomState(seed)
    y = layer.forward(x)
    c = rng.randn(*y.shape).astype(F64)

    def loss_from_y(yv):
        return np.sum(yv * c)

    # analytic
    dx = layer.backward(c)
    pgrads = [g.copy() for g in layer.grads()]
    params = layer.params()

    # numeric dx (perturb on copies to avoid in-place/nditer interaction)
    dx_num = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        xp = x.copy();
        xp[i] += EPS;
        lp = loss_from_y(layer.forward(xp))
        xm = x.copy();
        xm[i] -= EPS;
        lm = loss_from_y(layer.forward(xm))
        dx_num[i] = (lp - lm) / (2 * EPS)
    layer.forward(x)  # restore cache to the unperturbed input
    e_dx = rel_err(dx, dx_num)

    # numeric param grads
    worst_p = 0.0
    worst_p_ok = True
    for p, ga in zip(params, pgrads):
        gnum = np.zeros_like(p)
        for i in np.ndindex(p.shape):
            old = p[i]
            p[i] = old + EPS;
            lp = loss_from_y(layer.forward(x))
            p[i] = old - EPS;
            lm = loss_from_y(layer.forward(x))
            p[i] = old
            gnum[i] = (lp - lm) / (2 * EPS)
        layer.forward(x)
        worst_p = max(worst_p, rel_err(ga, gnum))
        worst_p_ok = worst_p_ok and passed(ga, gnum)
    good = passed(dx, dx_num) and worst_p_ok
    status = "OK " if good else "FAIL"
    print(f"[{status}] {name:16s}  dx_relerr={e_dx:.2e}  param_relerr={worst_p:.2e}")
    return good


ok = True

# RMSNorm
rms = RMSNorm(8, dtype=F64)
ok &= check_layer("RMSNorm", rms, np.random.randn(3, 4, 8).astype(F64))

# SwiGLU
swi = SwiGLU(8, 6, dtype=F64)
ok &= check_layer("SwiGLU", swi, np.random.randn(3, 4, 8).astype(F64))

# RoPE (no params)
rope = RoPE(6, max_seq_len=5, dtype=F64)
ok &= check_layer("RoPE", rope, np.random.randn(2, 5, 3, 6).astype(F64))

# Attention
cqa = CausalGQABlock(8, 2, 4, max_seq_len=5, dtype=F64)
ok &= check_layer("CausalAttn", cqa, np.random.randn(2, 5, 8).astype(F64))

# ---- Full model gradient check (loss = cross-entropy) ----
print("\nFull model (cross-entropy loss):")
model = PicoGPTOracle(vocab_size=11, d_model=8, n_layers=2, n_heads=2,
                      max_seq_len=6, dtype=F64)
B, T = 2, 5
idx = np.random.randint(0, 11, size=(B, T))
tgt = np.random.randint(0, 11, size=(B, T))

_, loss = model.forward(idx, tgt)
model.backward()
analytic = {k: g.copy() for k, g in zip(model.named_params().keys(), model.grads())}

named = model.named_params()
worst = 0.0
all_ok = True
# check a subset of coordinates per param (full check is slow); embedding rows that appear
for k, p in named.items():
    gnum = np.zeros_like(p)
    # sample up to 40 random coordinates
    flat_idx = np.random.choice(p.size, size=min(40, p.size), replace=False)
    pf = p.ravel()
    for fi in flat_idx:
        old = pf[fi]
        pf[fi] = old + EPS;
        _, lp = model.forward(idx, tgt)
        pf[fi] = old - EPS;
        _, lm = model.forward(idx, tgt)
        pf[fi] = old
        gnum.ravel()[fi] = (lp - lm) / (2 * EPS)
    ga = analytic[k].ravel()[flat_idx]
    gn = gnum.ravel()[flat_idx]
    e = rel_err(ga, gn)
    worst = max(worst, e)
    tag = "OK " if passed(ga, gn) else "FAIL"
    all_ok = all_ok and passed(ga, gn)
    print(f"[{tag}] {k:26s} relerr={e:.2e}")
ok &= all_ok

print("\nRESULT:", "ALL GRADIENTS CORRECT" if ok else "SOME CHECKS FAILED")
