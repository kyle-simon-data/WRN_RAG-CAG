#this lets me view the contents of a HotpotQA example

import json

INPUT_PATH = "data/hotpotqa/hotpotqa_100.json"

def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    #pick the first sample
    sample = data[0]

    print(f"ID: {sample['_id']}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")
    print("\nSupporting Facts (title, sentence_idx):")
    for title, idx in sample["supporting_facts"]:
        print(f"  - {title}, sentence #{idx}")

    print("\nContext Paragraphs:")
    for title, sentences in sample["context"]:
        print(f"\n== {title} ==")
        for i, sentence in enumerate(sentences):
            print(f"[{i}] {sentence}")

if __name__ == "__main__":
    main()