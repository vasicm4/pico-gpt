import torch
from torch.functional import F
from data_handling.tokenizer import CharacterTokenizer
from data_handling.batch_loader import DynamicBatchLoader
from oracle.oracle import PicoGPTOracle

class Runner:
    def __init__(self, model, batch_loader, max_steps, eval_interval):
        self.eval_interval = eval_interval
        self.max_steps = max_steps
        self.model = model
        self.batch_loader = batch_loader

    @torch.no_grad()
    def generate_text(self, start_tokens, max_new_tokens, block_size, temperature=1.0):
        self.model.eval()
        ctx = start_tokens
        for _ in range(max_new_tokens):
            ctx_cond = ctx[:, -block_size:]
            logits, _ = self.model(ctx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            ctx = torch.cat((ctx, next_token), dim=1)
        self.model.train()
        return ctx

    def train(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = PicoGPTOracle(vocab_size=V, d_model=C, n_layers=L, n_heads=NH, max_seq_len=T)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        print("Starting standalone execution loop pipeline...")
        for step in range(max_steps + 1):
            if step % eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    x_t, y_t = self.batch_loader.get_batch('train', B, T)
                    x_v, y_v = self.batch_loader.get_batch('val', B, T)
                    _, train_loss = model(x_t.to(device), y_t.to(device))
                    _, val_loss = model(x_v.to(device), y_v.to(device))
                    print(f"Step {step:4d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

                    start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
                    generated_indices = self.generate_text(start_context, max_new_tokens=40, block_size=T, temperature=0.8)
                    decoded_sample = tokenizer.decode(generated_indices[0].tolist())
                    print(f"--- Sampling Snapshot: ---\n{decoded_sample}\n--------------------------")

                model.train()

            xb, yb = self.batch_loader.get_batch('train', B, T)
            logits, loss = model(xb.to(device), yb.to(device))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

if __name__ == "__main__":

    tokenizer = CharacterTokenizer()
    B, T = 4, 64  # Batch size and Context Window
    C = 128  # Embedding dimension
    L = 3  # Interlocking transformer blocks
    NH = 4  # Attention heads
    V = tokenizer.vocab_size
    max_steps = 1000
    eval_interval = 200

    batch_loader = DynamicBatchLoader(batch_size=B, block_size=4, chunk_dir="data")
    model = PicoGPTOracle(vocab_size=V, d_model=C, n_layers=L, n_heads=NH, max_seq_len=T)

    runner = Runner(model, batch_loader, max_steps, eval_interval)
    runner.train()