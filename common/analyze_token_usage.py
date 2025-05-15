from transformers import AutoTokenizer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cag/cag_pipeline')))
from load_local_documents import load_local_documents

# Initialize tokenizer (for WhiteRabbitNeo)
tokenizer = AutoTokenizer.from_pretrained("WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def analyze_documents(documents):
    total_tokens = 0
    token_stats = []

    for doc in documents:
        tokens = count_tokens(doc.page_content)
        total_tokens += tokens
        token_stats.append({
            "source": doc.metadata.get("source", "unknown"),
            "type": doc.metadata.get("type", "unknown"),
            "tokens": tokens
        })

    print(f"\n[SUMMARY] Total documents: {len(documents)}")
    print(f"[SUMMARY] Total tokens: {total_tokens}\n")
    print(f"{'File':40s} | {'Type':10s} | {'Tokens':>6s}")
    print("-" * 65)
    for entry in sorted(token_stats, key=lambda x: -x['tokens']):
        print(f"{entry['source'][:40]:40s} | {entry['type'][:10]:10s} | {entry['tokens']:6d}")

if __name__ == "__main__":
    docs = load_local_documents()
    analyze_documents(docs)
