import json
import os

INPUT_PATH = "data/hotpotqa/hotpotqa_100.json"
OUTPUT_DIR = "data/hotpotqa/testdocs"

def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, example in enumerate(data):
        slimmed = {
            "id": example["_id"],
            "question": example["question"],
            "answer": example["answer"],
            "supporting_facts": example["supporting_facts"],
            "context": example["context"]
        }

        out_path = os.path.join(OUTPUT_DIR, f"example_{i:03}.json")
        with open(out_path, "w") as out_file:
            json.dump(slimmed, out_file, indent=2)

    print(f"Saved {len(data)} slimmed testdocs to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
