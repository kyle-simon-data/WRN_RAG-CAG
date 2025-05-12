#This simply allows me to the structure of the data of a HotpotQA example

import json

with open("data/hotpotqa/hotpotqa_100.json") as f:
    data = json.load(f)

sample = data[0]

print(sample.keys())
