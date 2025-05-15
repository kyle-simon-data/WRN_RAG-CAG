import json
import os

INPUT_PATH = "data/triviaqa/triviaqa_100.json"
SUPPORTING_DOCS_DIR = "/home/ubuntu/datasets/triviaqa/evidence/wikipedia"
OUTPUT_DIR = "data/triviaqa/testdocs"

def load_supporting_doc(filename):
    file_path = os.path.join(SUPPORTING_DOCS_DIR, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, example in enumerate(data):
        q = example["Question"]
        ans = (
            example["Answer"].get("NormalizedAliases", []) or
            example["Answer"].get("Aliases", []) or
            [example["Answer"]["Value"]]
        )[0]

        filenames = [ep["Filename"] for ep in example.get("EntityPages", []) if "Filename" in ep]
        if not filenames:
            filenames = [sr.get("Filename") for sr in example.get("SearchResults", []) if sr.get("Filename")]

        context_entries = []
        for fname in filenames[:5]:
            content = load_supporting_doc(fname)
            if content.strip():
                context_entries.append(["", content.strip()])

        if not context_entries:
            context_entries = [["", ""]]

        slimmed = {
            "id": example["QuestionId"],
            "question": q,
            "answer": ans,
            "supporting_facts": [],
            "context": context_entries
        }

        out_path = os.path.join(OUTPUT_DIR, f"example_{i:03}.json")
        with open(out_path, "w") as out_file:
            json.dump(slimmed, out_file, indent=2)

    print(f"Saved {len(data)} testdocs with Wikipedia context to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
