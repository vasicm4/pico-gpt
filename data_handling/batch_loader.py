import os
import numpy as np
import torch


class DynamicBatchLoader:
    def __init__(self, chunk_dir, batch_size, block_size, swap_every_iterations=100):
        self.chunk_dir = chunk_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.swap_every = swap_every_iterations

        all_files = os.listdir(chunk_dir)
        self.chunk_files = {
            'train': sorted([f for f in all_files if f.startswith("train_chunk_") and f.endswith(".txt")]),
            'val': sorted([f for f in all_files if f.startswith("tiny_stories_val") and f.endswith(".txt")]),
        }

        self.current_chunk_idx = {'train': 0, 'val': 0}
        self.iteration_counter = {'train': 0, 'val': 0}

        self.loaded_data = {'train': None, 'val': None}
        self.char_to_int = None

    def _load_chunk(self, split):
        chunks = self.chunk_files[split]
        if not chunks:
            raise FileNotFoundError(f"No chunk files found in '{self.chunk_dir}' for split '{split}'")

        idx = self.current_chunk_idx[split]
        target_file = os.path.join(self.chunk_dir, chunks[idx])
        print(
            f"\n[Data Loader] Split [{split}] Iteration {self.iteration_counter[split]}: Loading {target_file} into memory...")

        with open(target_file, "r", encoding="utf-8") as f:
            text = f.read()

        if self.char_to_int is None:
            # chars = sorted(list(set(text)))
            chars = sorted(list(set(" \n\t0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!;:'\"-—…()[]{}*_&$%/\\“”‘’")))
            self.char_to_int = {ch: i for i, ch in enumerate(chars)}

        encoded = [self.char_to_int.get(c, 0) for c in text]
        self.loaded_data[split] = np.array(encoded, dtype=np.int32)

    def get_batch(self, split, batch_size, block_size):
        if self.loaded_data[split] is None:
            self._load_chunk(split)

        elif self.iteration_counter[split] > 0 and self.iteration_counter[split] % self.swap_every == 0:
            total_chunks = len(self.chunk_files[split])
            self.current_chunk_idx[split] = (self.current_chunk_idx[split] + 1) % total_chunks
            self._load_chunk(split)

        self.iteration_counter[split] += 1
        data = self.loaded_data[split]

        ix = np.random.randint(0, len(data) - block_size, batch_size)

        x_np = np.stack([data[i: i + block_size] for i in ix])
        y_np = np.stack([data[i + 1: i + block_size + 1] for i in ix])

        return torch.tensor(x_np, dtype=torch.long), torch.tensor(y_np, dtype=torch.long)