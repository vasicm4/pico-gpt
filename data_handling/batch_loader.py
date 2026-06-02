import os
import numpy as np

class DynamicBatchLoader:
    def __init__(self, chunk_dir, batch_size, block_size, swap_every_iterations=100):
        self.chunk_dir = chunk_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.swap_every = swap_every_iterations

        self.chunk_files = sorted([
            f for f in os.listdir(chunk_dir) if f.startswith("train_chunk_") and f.endswith(".txt")
        ])
        self.current_chunk_idx = 0
        self.iteration_counter = 0

        self._load_current_chunk()

    def _load_current_chunk(self):
        target_file = os.path.join(self.chunk_dir, self.chunk_files[self.current_chunk_idx])
        print(f"\n[Data Loader] Iteration {self.iteration_counter}: Loading {target_file} into memory...")

        with open(target_file, "r", encoding="utf-8") as f:
            text = f.read()

        self.chars = sorted(list(set(text)))
        self.char_to_int = {ch: i for i, ch in enumerate(self.chars)}
        self.data = np.array([self.char_to_int[c] for c in text], dtype=np.int32)

    def get_batch(self):
        if self.iteration_counter > 0 and self.iteration_counter % self.swap_every == 0:
            self.current_chunk_idx = (self.current_chunk_idx + 1) % len(self.chunk_files)
            self._load_current_chunk()

        self.iteration_counter += 1

        ix = np.random.randint(0, len(self.data) - self.block_size, self.batch_size)

        x = np.stack([self.data[i : i + self.block_size] for i in ix])
        y = np.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])

        return x, y