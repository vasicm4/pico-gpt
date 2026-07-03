"""Entry point mirroring the reference oracle/main.py, in pure NumPy.

Training uses the SAME hyperparameters as the reference:
    B=32, T=128, C=256, L=4, NH=8, lr=1e-3, wd=0.01, clip=1.0,
    max_steps=2000, eval_interval=200
and the SAME data pipeline (CharacterTokenizer + DynamicBatchLoader over the
TinyStories chunks in ./data_handling/data).
"""
# import numpy as np
import cupy as np
from data_handling.tokenizer import CharacterTokenizer
from data_handling.batch_loader_np import DynamicBatchLoaderNp as DynamicBatchLoader
from layers.oracle import PicoGPTOracle
from layers.runner import Runner

VOCAB_STR = (" \n\t0123456789abcdefghijklmnopqrstuvwxyz"
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!;:'\"-—…()[]{}*_&$%/\\Code“”‘’")


def build(tokenizer, T, C, L, NH):
    return PicoGPTOracle(vocab_size=tokenizer.vocab_size, d_model=C,
                         n_layers=L, n_heads=NH, max_seq_len=T)


def train():
    tokenizer = CharacterTokenizer(VOCAB_STR)
    B, T = 32, 128
    C, L, NH = 256, 4, 8
    loader = DynamicBatchLoader("data_handling/data", B, T,
                                swap_every_iterations=100,
                                char_to_int=tokenizer.stoi)
    model = build(tokenizer, T, C, L, NH)
    runner = Runner(model, loader, tokenizer, B, T,
                    max_steps=2000, eval_interval=200)
    runner.train()


def infer(checkpoint="pico_gpt_oracle_np", temperature=0.65, max_new_tokens=250):
    tokenizer = CharacterTokenizer(VOCAB_STR)
    B, T = 32, 128
    C, L, NH = 256, 4, 8
    model = build(tokenizer, T, C, L, NH)
    print(f"Loading weights from '{checkpoint}.npz'...")
    model.load(checkpoint)
    runner = Runner(model, batch_loader=None, tokenizer=tokenizer, B=B, T=T)

    prompt = ""
    while prompt != "q":
        prompt = input("Prompt: ")
        if prompt == "q":
            break
        ctx = np.array([tokenizer.encode(prompt)], dtype=np.int64)
        out = runner.generate_text(ctx, max_new_tokens=max_new_tokens,
                                   block_size=T, temperature=temperature)
        print("\n--- Model Output ---")
        print(tokenizer.decode(out[0].tolist()))
        print("--------------------")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        infer()
    else:
        train()