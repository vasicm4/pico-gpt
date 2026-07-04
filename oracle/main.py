import torch
from torch.functional import F
from data_handling.tokenizer import CharacterTokenizer
from data_handling.batch_loader import DynamicBatchLoader
from oracle import PicoGPTOracle


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
        # model = PicoGPTOracle(vocab_size=V, d_model=C, n_layers=L, n_heads=NH, max_seq_len=T)
        model = self.model
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        print("Starting standalone execution loop pipeline...")
        for step in range(self.max_steps + 1):
            if step % self.eval_interval == 0:

                model.eval()
                with torch.no_grad():
                    x_t, y_t = self.batch_loader.get_batch('train', B, T)
                    x_v, y_v = self.batch_loader.get_batch('val', B, T)
                    print(f"DEBUG - Raw Target IDs: {y_t[0][:15].tolist()}")
                    print(f"DEBUG - Decoded Target: {tokenizer.decode(y_t[0][:15].tolist())}")
                    _, train_loss = model(x_t.to(device), y_t.to(device))
                    _, val_loss = model(x_v.to(device), y_v.to(device))
                    print(f"Step {step:4d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

                    start_char = "O"
                    start_id = tokenizer.stoi.get(start_char, 10)
                    start_context = torch.tensor([[start_id]], dtype=torch.long, device=device)
                    generated_indices = self.generate_text(start_context, max_new_tokens=40, block_size=T, temperature=0.7)
                    decoded_sample = tokenizer.decode(generated_indices[0].tolist())
                    print(f"--- Sampling Snapshot: ---\n{decoded_sample}\n--------------------------")

                model.train()

            xb, yb = self.batch_loader.get_batch('train', B, T)
            logits, loss = model(xb.to(device), yb.to(device))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print("\nTraining complete! Saving model weights...")
        torch.save(model.state_dict(), "pico_gpt_oracle.pth")
        print("Model saved successfully as 'pico_gpt_oracle.pth'")

# if __name__ == "__main__":
#
#     tokenizer = CharacterTokenizer(" \n\t0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!;:'\"-—…()[]{}*_&$%/\\“”‘’")
#     B, T = 32, 128
#     C = 256
#     L = 4
#     NH = 8
#     V = tokenizer.vocab_size
#     max_steps = 2000
#     eval_interval = 200
#
#     batch_loader = DynamicBatchLoader(batch_size=B, block_size=T, chunk_dir="./data_handling/data")
#     model = PicoGPTOracle(vocab_size=V, d_model=C, n_layers=L, n_heads=NH, max_seq_len=T)
#
#     runner = Runner(model, batch_loader, max_steps, eval_interval)
#     runner.train()

if __name__ == "__main__":
    vocab_str = " \n\t0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!;:'\"-—…()[]{}*_&$%/\\Code“”‘’"
    tokenizer = CharacterTokenizer(vocab_str)

    B = 32
    T = 128
    C, L, NH = 256, 4, 8
    KV_HEADS = None  # set to an int < NH for GQA; leave None to match the MHA-trained checkpoint
    V = tokenizer.vocab_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if KV_HEADS is not None:
        model = PicoGPTOracle(vocab_size=V, d_model=C, n_layers=L, n_kv_heads=KV_HEADS, n_heads=NH, max_seq_len=T)
    else:
        model = PicoGPTOracle(vocab_size=V, d_model=C, n_layers=L, n_heads=NH, max_seq_len=T)

    print("Loading pre-trained weights from 'pico_gpt_oracle.pth'...")
    model.load_state_dict(torch.load("pico_gpt_oracle.pth", map_location=device), strict=False)
    model.to(device)
    model.eval()

    runner = Runner(model, batch_loader=None, max_steps=0, eval_interval=0)


    prompt = ""
    while prompt != "q":
        prompt = input("Prompt: ")
        print(f"\nFeeding prompt: '{prompt}'")

        encoded_prompt = tokenizer.encode(prompt)

        start_context = torch.tensor([encoded_prompt], dtype=torch.long, device=device)

        generated_indices = runner.generate_text(
            start_tokens=start_context,
            max_new_tokens=250,
            block_size=T,
            temperature=0.65
        )

        completed_story = tokenizer.decode(generated_indices[0].tolist())

        print("\n--- Model Output ---")
        print(completed_story)
        print("--------------------")