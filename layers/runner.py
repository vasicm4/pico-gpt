"""Training / generation runner, mirroring the reference main.py Runner
(AdamW lr=1e-3 wd=0.01, grad clip 1.0, periodic eval + sampling)."""
import numpy as np
from .optimizer import AdamW
from layers.cqa import softmax
# import cupy as np

class Runner:
    def __init__(self, model, batch_loader, tokenizer, B, T,
                 max_steps=2000, eval_interval=200, lr=1e-3, weight_decay=0.01,
                 max_grad_norm=1.0):
        self.model = model
        self.batch_loader = batch_loader
        self.tokenizer = tokenizer
        self.B, self.T = B, T
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.opt = AdamW(model.params(), model.grads, lr=lr,
                         weight_decay=weight_decay, max_grad_norm=max_grad_norm)

    # ---- generation (numpy equivalent of the reference generate_text) ----
    def generate_text(self, start_tokens, max_new_tokens, block_size,
                      temperature=1.0, rng=None):
        rng = rng or np.random
        ctx = np.array(start_tokens, dtype=np.int64)
        for _ in range(max_new_tokens):
            ctx_cond = ctx[:, -block_size:]
            logits, _ = self.model.forward(ctx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = softmax(logits, axis=-1)
            nxt = np.array([[rng.choice(self.model.vocab_size, p=probs[0])]],
                           dtype=np.int64)
            ctx = np.concatenate([ctx, nxt], axis=1)
        return ctx

    def _eval_loss(self, split, n=1):
        losses = []
        for _ in range(n):
            x, y = self.batch_loader.get_batch(split, self.B, self.T)
            _, loss = self.model.forward(x, y)
            losses.append(loss)
        return float(np.mean(losses))

    def train(self):
        tok, T = self.tokenizer, self.T
        print("Starting standalone execution loop pipeline...")
        for step in range(self.max_steps + 1):
            if step % self.eval_interval == 0:
                train_loss = self._eval_loss('train')
                val_loss = self._eval_loss('val')
                print(f"Step {step:4d} | Train Loss: {train_loss:.4f} "
                      f"| Val Loss: {val_loss:.4f}")
                start_id = tok.stoi.get("O", 0)
                start = np.array([[start_id]], dtype=np.int64)
                gen = self.generate_text(start, max_new_tokens=40,
                                         block_size=T, temperature=0.7)
                print("--- Sampling Snapshot: ---")
                print(tok.decode(gen[0].tolist()))
                print("--------------------------")

            xb, yb = self.batch_loader.get_batch('train', self.B, T)
            _, loss = self.model.forward(xb, yb)
            self.model.backward()
            self.opt.step()

        print("\nTraining complete! Saving model weights...")
        self.model.save("pico_gpt_oracle_np")
        print("Saved as 'pico_gpt_oracle_np.npz'")