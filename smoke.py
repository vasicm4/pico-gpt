
from data_handling.tokenizer import CharacterTokenizer
from data_handling.batch_loader_np import DynamicBatchLoaderNp as DynamicBatchLoader
from layers.oracle import PicoGPTOracle
from layers.runner import Runner
import cupy as np

np.random.seed(0)

vocab_str = (" \n\t0123456789abcdefghijklmnopqrstuvwxyz"
             "ABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!;:'\"-—…()[]{}*_&$%/\\Code“”‘’")
tokenizer = CharacterTokenizer(vocab_str)
V = tokenizer.vocab_size
print("vocab_size:", V)


B, T = 16, 64
C, L, NH = 128, 3, 4
#
# B = 32
# T = 128
# C, L, NH = 256, 4, 8
loader = DynamicBatchLoader("./data_handling/data", B, T,
                            swap_every_iterations=100,
                            char_to_int=tokenizer.stoi, verbose=False)
model = PicoGPTOracle(vocab_size=V, d_model=C, n_layers=L, n_heads=NH,
                      max_seq_len=T)
runner = Runner(model, loader, tokenizer, B, T,
                max_steps=3000, eval_interval=150, lr=1e-3)
runner.train()


print("\n=== reload checkpoint and generate ===")
model2 = PicoGPTOracle(vocab_size=V, d_model=C, n_layers=L, n_heads=NH,
                       max_seq_len=T)
model2.load("pico_gpt_oracle_np")
r2 = Runner(model2, loader, tokenizer, B, T)
prompt = "Once upon a time"
ctx = np.array([tokenizer.encode(prompt)], dtype=np.int64)
out = r2.generate_text(ctx, max_new_tokens=120, block_size=T, temperature=0.6,
                       rng=np.random.RandomState(1))
print(tokenizer.decode(out[0].tolist()))