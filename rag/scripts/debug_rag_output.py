import json

INPUT_PATH = "evaluation_results_TQA_RAG.jsonl"
MAX_EXAMPLES = 5

def main():
    shown = 0
    with open(INPUT_PATH, "r") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get("rag_fuzzy_ratio", 100) <= 5:
                    print("\n" + "=" * 80)
                    print(f"Question ID: {obj['question_id']}")
                    print(f"Question:     {obj['question']}")
                    print(f"Ground Truth: {obj['ground_truth_answer']}")
                    print(f"RAG Answer:   {obj['rag_answer']}")
                    print(f"Top Docs:     {obj['rag_top_docs']}")
                    print(f"Fuzzy Score:  {obj['rag_fuzzy_ratio']}")
                    print(f"BLEU:         {obj['rag_bleu']:.4f} | ROUGE-L: {obj['rag_rougeL']:.4f}")
                    shown += 1
            except Exception as e:
                continue
            if shown >= MAX_EXAMPLES:
                break

if __name__ == "__main__":
    main()