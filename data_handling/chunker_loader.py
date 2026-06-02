from datasets import load_dataset
import os

class ChunkerLoader:
    def __init__(self, stories_per_chunk=10000, max_chunks=10):
        self.stories_per_chunk = stories_per_chunk
        self.max_chunks = max_chunks

    def create_chunks(self):
        print("Loading TinyStories dataset...")
        dataset = load_dataset("roneneldan/TinyStories", streaming=True)

        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)

        chunk_idx = 1
        story_count = 0
        current_chunk_lines = []

        print(f"Streaming and splitting data into chunks of {self.stories_per_chunk} stories...")

        for item in dataset["train"]:
            story_lines = item["text"].strip().split("\n")
            for line in story_lines:
                if line.strip():
                    current_chunk_lines.append(line)

            story_count += 1

            if story_count >= self.stories_per_chunk:
                chunk_path = os.path.join(output_dir, f"train_chunk_{chunk_idx}.txt")
                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(current_chunk_lines))

                print(f"Saved: {chunk_path} ({len(current_chunk_lines)} raw text lines)")

                chunk_idx += 1
                story_count = 0
                current_chunk_lines = []

                if chunk_idx > self.max_chunks:
                    break

        val_path = "data/tiny_stories_val.txt"
        print(f"Saving validation set to {val_path}...")
        with open(val_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(dataset["validation"]):
                if i >= 2000:
                    break
                f.write(item["text"] + "\n\n")

if __name__ == "__main__":
    chunker_loader = ChunkerLoader(stories_per_chunk=10000, max_chunks=10)
    chunker_loader.create_chunks()