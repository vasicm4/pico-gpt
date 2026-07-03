"""NumPy-native DynamicBatchLoader.

Identical semantics to the reference torch loader (chunk swapping, random
offsets, x = data[i:i+T], y = data[i+1:i+T+1]) but returns numpy int64 arrays
instead of torch tensors.

Pass the tokenizer's `stoi` as `char_to_int` so the encoding is guaranteed
consistent with the tokenizer used for decode(); if omitted it falls back to
the reference's hard-coded vocabulary.
"""
import os
import numpy as np

_DEFAULT_VOCAB = (" \n\t0123456789abcdefghijklmnopqrstuvwxyz"
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZ.,?!;:'\"-—…()[]{}*_&$%/\\“”‘’")


class DynamicBatchLoaderNp:
    def __init__(self, chunk_dir, batch_size, block_size,
                 swap_every_iterations=100, char_to_int=None, verbose=True):
        self.chunk_dir = chunk_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.swap_every = swap_every_iterations
        self.verbose = verbose

        all_files = os.listdir(chunk_dir)
        self.chunk_files = {
            'train': sorted(f for f in all_files
                            if f.startswith("train_chunk_") and f.endswith(".txt")),
            'val': sorted(f for f in all_files
                          if f.startswith("tiny_stories_val") and f.endswith(".txt")),
        }
        self.current_chunk_idx = {'train': 0, 'val': 0}
        self.iteration_counter = {'train': 0, 'val': 0}
        self.loaded_data = {'train': None, 'val': None}

        if char_to_int is None:
            chars = sorted(set(_DEFAULT_VOCAB))
            char_to_int = {ch: i for i, ch in enumerate(chars)}
        self.char_to_int = char_to_int

    def _load_chunk(self, split):
        chunks = self.chunk_files[split]
        if not chunks:
            raise FileNotFoundError(
                f"No chunk files in '{self.chunk_dir}' for split '{split}'")
        idx = self.current_chunk_idx[split]
        target = os.path.join(self.chunk_dir, chunks[idx])
        if self.verbose:
            print(f"[Data Loader] {split} iter {self.iteration_counter[split]}: "
                  f"loading {target}")
        with open(target, "r", encoding="utf-8") as f:
            text = f.read()
        encoded = [self.char_to_int.get(c, 0) for c in text]
        self.loaded_data[split] = np.array(encoded, dtype=np.int64)

    def get_batch(self, split, batch_size, block_size):
        if self.loaded_data[split] is None:
            self._load_chunk(split)
        elif (self.iteration_counter[split] > 0 and
              self.iteration_counter[split] % self.swap_every == 0):
            total = len(self.chunk_files[split])
            self.current_chunk_idx[split] = (self.current_chunk_idx[split] + 1) % total
            self._load_chunk(split)

        self.iteration_counter[split] += 1
        data = self.loaded_data[split]

        ix = np.random.randint(0, len(data) - block_size, batch_size)
        x = np.stack([data[i:i + block_size] for i in ix])
        y = np.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.astype(np.int64), y.astype(np.int64)