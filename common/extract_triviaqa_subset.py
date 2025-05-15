import json
import random
import os

INPUT_PATH = "/home/ubuntu/datasets/triviaqa/qa/web-dev.json"
OUTPUT_PATH = "/home/ubuntu/WRN_RAG-CAG/data/triviaqa/triviaqa_100.json"
SAMPLE_SIZE = 100

def main():
    with open(INPUT_PATH, "r") as infile:
        data = json.load(infile)["Data"]

    sample = random.sample(data, SAMPLE_SIZE)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as outfile:
        json.dump(sample, outfile, indent=2)

    print(f"Saved {SAMPLE_SIZE} examples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()