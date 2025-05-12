import json
import random

INPUT_PATH = "data/hotpotqa/hotpot_dev_distractor_v1.json"
OUTPUT_PATH = "data/hotpotqa/hotpotqa_100.json"
SAMPLE_SIZE = 100

def main():
    with open(INPUT_PATH, "r") as infile:
        data = json.load(infile)

    sample = random.sample(data, SAMPLE_SIZE)

    with open(OUTPUT_PATH, "w") as outfile:
        json.dump(sample, outfile, indent=2)

    print(f"Saved {SAMPLE_SIZE} examples to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
