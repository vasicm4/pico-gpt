
import math
from datetime import datetime

import numpy as np
from .optimizer import AdamW
from .cqa import softmax


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
        self.eval_temperatures = (0.7, 1.0)
        self.eval_gen_tokens = 40
        self.eval_seed_char = "O"


    def generate_text(self, start_tokens, max_new_tokens, block_size,
                      temperature=1.0, rng=None, inference=False):
        rng = rng or np.random
        ctx = np.array(start_tokens, dtype=np.int64)
        for _ in range(max_new_tokens):
            ctx_cond = ctx[:, -block_size:]
            logits, _ = self.model.forward_inference(ctx_cond) if inference else self.model.forward(ctx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = softmax(logits, axis=-1)            # (B, V)

            nxt = np.array([rng.choice(self.model.vocab_size, p=probs[b])
                            for b in range(probs.shape[0])], dtype=np.int64)[:, None]
            ctx = np.concatenate([ctx, nxt], axis=1)
        return ctx

    def _eval_loss(self, split, n=1):
        losses = []
        for _ in range(n):
            x, y = self.batch_loader.get_batch(split, self.B, self.T)
            _, loss = self.model.forward(x, y)
            losses.append(float(loss))
        return float(np.mean(losses))

    def _eval_token_accuracy(self, split, n=1):
        correct1, correct5, total = 0, 0, 0
        for _ in range(n):
            x, y = self.batch_loader.get_batch(split, self.B, self.T)
            logits, _ = self.model.forward(x, y)        # (B, T, V)
            B, T, _ = logits.shape
            pred = logits[:, :-1, :]                     # (B, T-1, V)
            tgt  = y[:, :-1]                             # (B, T-1)
            top5 = np.argpartition(-pred, 5, axis=-1)[..., :5]
            correct1 += int(np.sum(np.any(top5[..., :1] == tgt[..., None], axis=-1)))
            correct5 += int(np.sum(np.any(top5 == tgt[..., None], axis=-1)))
            total    += B * (T - 1)
        return correct1 / total, correct5 / total

    def _generation_entropy(self, prompt_tokens, max_new_tokens, temperature, rng=None):
        rng = rng or np.random
        ctx = np.array(prompt_tokens, dtype=np.int64)
        entropies = []
        for _ in range(max_new_tokens):
            ctx_cond = ctx[:, -self.T:]
            logits, _ = self.model.forward(ctx_cond)
            probs = softmax(logits[:, -1, :] / max(temperature, 1e-5), axis=-1)
            entropies.append(float(-np.sum(probs[0] * np.log(probs[0] + 1e-12))))
            nxt = np.array([[rng.choice(self.model.vocab_size, p=probs[0])]], dtype=np.int64)
            ctx = np.concatenate([ctx, nxt], axis=1)
        return float(np.mean(entropies)), ctx

    def train(self):
        tok, T = self.tokenizer, self.T
        history = {
            "step":       [],
            "train_loss": [],
            "val_loss":   [],
            "val_ppl":    [],
            "val_top1":   [],
            "val_top5":   [],
        }
        ent_keys = []
        for T_eval in self.eval_temperatures:
            key = f"gen_ent_{int(round(T_eval * 100)):03d}"
            history[key] = []
            ent_keys.append((T_eval, key))

        sample_log_path = "training_samples.txt"
        sample_log = open(sample_log_path, "w", encoding="utf-8")
        sample_log.write(f"# pico-gpt qualitative samples\n")
        sample_log.write(f"# generated {datetime.now().isoformat(timespec='seconds')}\n")
        sample_log.write(
            f"# temperatures: {', '.join(str(t) for t, _ in ent_keys)}"
            f"  |  prompt: '{self.eval_seed_char}'"
            f"  |  max_new_tokens: {self.eval_gen_tokens}\n\n")
        sample_log.flush()

        start_id = tok.stoi.get(self.eval_seed_char, 0)
        prompt = np.array([[start_id]], dtype=np.int64)

        print("Starting standalone execution loop pipeline...")
        for step in range(self.max_steps + 1):
            if step % self.eval_interval == 0:
                train_loss = self._eval_loss('train')
                val_loss   = self._eval_loss('val')
                top1, top5 = self._eval_token_accuracy('val')
                val_ppl    = math.exp(min(val_loss, 50.0))   # clip — exp(large CE) blows up
                gap        = val_loss - train_loss

                history["step"].append(step)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["val_ppl"].append(val_ppl)
                history["val_top1"].append(top1)
                history["val_top5"].append(top5)

                ent_strs = []
                for T_eval, key in ent_keys:
                    h, gen = self._generation_entropy(prompt,
                                                      self.eval_gen_tokens,
                                                      T_eval)
                    history[key].append(h)
                    ent_strs.append(f"H({T_eval}) {h:.2f}")
                    sample_log.write(
                        f"[step {step:4d}]  T={T_eval}  H={h:.3f}  | "
                        f"{tok.decode(gen[0].tolist())}\n")
                sample_log.write("\n")
                sample_log.flush()

                print(f"Step {step:4d} | Train {train_loss:.4f} | Val {val_loss:.4f} "
                      f"| PPL {val_ppl:7.2f} | Gap {gap:+.4f} "
                      f"| top1 {top1:.3f} top5 {top5:.3f} | "
                      + " ".join(ent_strs))

            xb, yb = self.batch_loader.get_batch('train', self.B, T)
            _, loss = self.model.forward(xb, yb)
            self.model.backward()
            self.opt.step()

        sample_log.close()
        print(f"\nSaved {len(history['step'])} qualitative samples to {sample_log_path}")

        print("\nTraining complete! Saving model weights...")
        self.model.save("pico_gpt_oracle_np")
        print("Saved as 'pico_gpt_oracle_np.npz'")
        return history

    def plot_evaluation(self, history, vocab_size, save_path="training_eval_dashboard.png", show=True):
        import matplotlib.pyplot as plt

        steps = history["step"]
        fig, (ax_loss, ax_gap, ax_ent) = plt.subplots(
            3, 1, figsize=(11, 11), sharex=True,
            gridspec_kw={"hspace": 0.30})

        ax_loss.plot(steps, history["train_loss"], '-',  color='#1f77b4',
                     linewidth=2, label='Train loss (CE)')
        ax_loss.plot(steps, history["val_loss"],   '-',  color='#d62728',
                     linewidth=2, label='Val loss (CE)')
        ax_loss.set_ylabel('Cross-entropy loss (nats)', fontsize=11)
        ax_loss.set_title('Training & validation loss', fontsize=12, fontweight='bold')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend(loc='upper right')

        ax_ppl = ax_loss.twinx()
        ax_ppl.plot(steps, history["val_ppl"], '--', color='#9467bd',
                    linewidth=1.8, label='Val perplexity')
        ax_ppl.set_ylabel('Val perplexity (exp CE)', fontsize=11, color='#9467bd')
        ax_ppl.tick_params(axis='y', labelcolor='#9467bd')
        ax_ppl.legend(loc='lower right')

        gap = np.array(history["val_loss"]) - np.array(history["train_loss"])
        ax_gap.fill_between(steps, 0, gap, where=(gap >= 0),
                            color='#d62728', alpha=0.25, label='Val > Train (overfit)')
        ax_gap.fill_between(steps, 0, gap, where=(gap < 0),
                            color='#2ca02c', alpha=0.25, label='Val < Train (data-shift)')
        ax_gap.plot(steps, gap, 'k-', linewidth=1.5)
        ax_gap.axhline(0, color='black', linewidth=0.7)
        ax_gap.set_ylabel('Val loss - Train loss', fontsize=11)
        ax_gap.set_title('Generalization gap', fontsize=12, fontweight='bold')
        ax_gap.grid(True, alpha=0.3)
        ax_gap.legend(loc='upper left', fontsize=9)

        H_uniform = math.log(vocab_size)
        ax_ent.axhline(H_uniform, color='gray', linestyle=':', linewidth=1,
                       label=f'Uniform log(V={vocab_size}) = {H_uniform:.2f}')
        ent_keys = [k for k in history.keys() if k.startswith("gen_ent_")]
        ent_colors = ['#ff7f0e', '#17becf', '#2ca02c', '#e377c2']
        for i, key in enumerate(sorted(ent_keys)):
            T_eval = int(key[len("gen_ent_"):]) / 100.0
            ax_ent.plot(steps, history[key], '-',
                        color=ent_colors[i % len(ent_colors)],
                        linewidth=2, label=f'T = {T_eval:g}')
        ax_ent.set_xlabel('Training step', fontsize=11)
        ax_ent.set_ylabel('Mean next-token entropy (nats)', fontsize=11)
        ax_ent.set_title(
            f'Generation entropy ({history[ent_keys[0]] and len(history[ent_keys[0]])} steps sampled)',
            fontsize=12, fontweight='bold')
        ax_ent.grid(True, alpha=0.3)
        ax_ent.legend(loc='upper right', fontsize=9)

        fig.suptitle('pico-gpt - evaluation dashboard',
                     fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        print(f"Saved dashboard to {save_path}")

    def print_summary(self, history):
        best_idx = int(np.argmin(history["val_loss"]))
        final_idx = -1
        print("\nFinal Results:")
        print(f"  Final train loss:     {history['train_loss'][final_idx]:.4f}")
        print(f"  Final val loss:       {history['val_loss'][final_idx]:.4f}")
        print(f"  Final val perplexity: {history['val_ppl'][final_idx]:.2f}")
        print(f"  Final val top-1/5:    {history['val_top1'][final_idx]:.3f} / "
              f"{history['val_top5'][final_idx]:.3f}")
        print(f"  Best  val loss:       {history['val_loss'][best_idx]:.4f} "
              f"(at step {history['step'][best_idx]})")
        print(f"  Delta val loss (best->end): "
              f"{history['val_loss'][final_idx] - history['val_loss'][best_idx]:+.4f}")