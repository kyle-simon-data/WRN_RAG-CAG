import json
import random
from datasets import load_dataset
import os

OUTPUT_DIR = "data/nq"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nq_100.json")
SAMPLE_SIZE = 100

def main():
    print("Loading nq_open dataset (validation split)...")
    dataset = load_dataset("nq_open", split="validation")

    print("Filtering valid entries...")
    filtered = []
    for i, ex in enumerate(dataset):
        if i % 1000 == 0:
            print(f"Checked {i} entries... found {len(filtered)} valid so far")

        if ex.get("question") and ex.get("answer"):
            filtered.append({
                "id": f"nq_{len(filtered)+1:03}",
                "question": ex["question"],
                "answer": ex["answer"][0]  # Use first answer
            })

    print(f"Found {len(filtered)} valid entries.")

    sample = random.sample(filtered, SAMPLE_SIZE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(sample, f, indent=2)

    print(f"Saved {SAMPLE_SIZE} examples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
