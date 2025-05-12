import json
import os

INPUT_PATH = "data/nq/nq_100.json"
OUTPUT_DIR = "data/nq/testdocs"

def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, example in enumerate(data):
        structured = {
            "id": example["id"],
            "question": example["question"],
            "answer": example["answer"],
            "context": [], #RAG/CAG will retrieve this from vector store
            "supporting_facts": [] #placeholder for sturctural consistency
        }

        out_path = os.path.join(OUTPUT_DIR, f"example_{i:03}.json")
        with open(out_path, "w") as out_file:
            json.dump(structured, out_file, indent=2)

    print(f"Saved {len(data)} testdocs to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()