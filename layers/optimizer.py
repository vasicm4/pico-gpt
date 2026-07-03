"""AdamW with decoupled weight decay + global-norm gradient clipping (NumPy).

Mirrors torch.optim.AdamW(lr=1e-3, weight_decay=0.01) with
torch.nn.utils.clip_grad_norm_(max_norm=1.0): weight decay applies to ALL
parameters (no param-group exclusions), matching the reference training script.
"""
import numpy as np
# import cupy as np

class AdamW:
    def __init__(self, params, grads_fn, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.01, max_grad_norm=1.0):
        # params: list of ndarray references (updated in place)
        # grads_fn: callable returning list of grad ndarrays in the SAME order
        self.params = params
        self.grads_fn = grads_fn
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.max_grad_norm = max_grad_norm
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def clip_grads_(self, grads):
        if self.max_grad_norm is None:
            return 1.0
        total = np.sqrt(sum(float(np.sum(g * g)) for g in grads))
        if total > self.max_grad_norm:
            scale = self.max_grad_norm / (total + 1e-6)
            for g in grads:
                g *= scale
        return total

    def step(self):
        grads = self.grads_fn()
        gnorm = self.clip_grads_(grads)
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(self.params, grads)):
            # decoupled weight decay
            if self.wd:
                p -= self.lr * self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / bc1
            vhat = self.v[i] / bc2
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return gnorm