from transformers import AutoTokenizer
import sys
import os
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cag/cag_pipeline')))
from load_local_documents import load_local_documents

# Initialize tokenizer (for WhiteRabbitNeo)
tokenizer = AutoTokenizer.from_pretrained("WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def analyze_documents():
    # Call the modified load_local_documents function that now returns document objects
    documents = load_local_documents(return_with_metadata=True)
    
    if not documents:
        print("[ERROR] No documents found or loaded.")
        return
    
    total_tokens = 0
    token_stats = []
    
    # For stats by file type
    type_stats = defaultdict(lambda: {"count": 0, "tokens": 0})
    
    # For histogram data
    token_counts = []
    char_counts = []
    
    for doc in documents:
        tokens = count_tokens(doc.page_content)
        chars = len(doc.page_content)
        doc_type = doc.metadata.get("type", "unknown").lower()
        
        total_tokens += tokens
        token_counts.append(tokens)
        char_counts.append(chars)
        
        # Update type statistics
        type_stats[doc_type]["count"] += 1
        type_stats[doc_type]["tokens"] += tokens
        
        token_stats.append({
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", "unknown"),
            "type": doc_type,
            "tokens": tokens,
            "chars": chars
        })

    # Calculate statistics
    avg_tokens = total_tokens / len(documents) if documents else 0
    avg_chars = sum(char_counts) / len(char_counts) if char_counts else 0
    max_tokens = max(token_counts) if token_counts else 0
    min_tokens = min(token_counts) if token_counts else 0
    
    # Print overall summary
    print("\n" + "="*80)
    print(f"DOCUMENT TOKEN ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total documents: {len(documents)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average tokens per chunk: {avg_tokens:.2f}")
    print(f"Min/Max tokens per chunk: {min_tokens}/{max_tokens}")
    print(f"Average characters per chunk: {avg_chars:.2f}")
    print("="*80 + "\n")
    
    # Print file type breakdown
    print("TOKEN USAGE BY FILE TYPE")
    print("-"*60)
    print(f"{'File Type':10s} | {'Chunks':>8s} | {'Total Tokens':>12s} | {'Avg Tokens':>10s}")
    print("-"*60)
    
    for doc_type, stats in sorted(type_stats.items()):
        avg = stats["tokens"] / stats["count"] if stats["count"] > 0 else 0
        print(f"{doc_type:10s} | {stats['count']:8d} | {stats['tokens']:12,d} | {avg:10.2f}")
    
    print("\n" + "="*80)
    print("TOP CHUNKS BY TOKEN COUNT")
    print("="*80)
    print(f"{'File':30s} | {'Type':6s} | {'Chunk #':7s} | {'Tokens':>7s} | {'Chars':>7s}")
    print("-"*80)
    
    # Sort by token count (descending)
    for entry in sorted(token_stats, key=lambda x: -x['tokens'])[:20]:  # Show top 20
        print(f"{entry['source'][:30]:30s} | {entry['type']:6s} | {entry['chunk_index']:7} | {entry['tokens']:7d} | {entry['chars']:7d}")
    
    print("\n[NOTE] Showing only the top 20 chunks by token count")
    
    # Optional: Generate histogram of token distribution
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(token_counts, bins=30, alpha=0.7)
        plt.title("Distribution of Tokens per Chunk")
        plt.xlabel("Token Count")
        plt.ylabel("Number of Chunks")
        plt.grid(True, alpha=0.3)
        
        # Add vertical line for average
        plt.axvline(x=avg_tokens, color='r', linestyle='--', label=f'Average: {avg_tokens:.1f}')
        plt.legend()
        
        # Save the histogram
        plt.savefig("token_distribution.png")
        print(f"\n[INFO] Token distribution histogram saved to 'token_distribution.png'")
    except Exception as e:
        print(f"\n[WARNING] Could not generate histogram: {e}")

if __name__ == "__main__":
    analyze_documents()
"""
from transformers import AutoTokenizer
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../cag/cag_pipeline')))
from load_local_documents import load_local_documents

# Initialize tokenizer (for WhiteRabbitNeo)
tokenizer = AutoTokenizer.from_pretrained("WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def analyze_documents():
    # Modify load_local_documents to return document objects with metadata
    documents_with_metadata = []
    
    # Call the modified load_local_documents function that now returns document objects
    documents = load_local_documents(return_with_metadata=True)
    
    total_tokens = 0
    token_stats = []
    
    chunk_sizes = []
    
    for doc in documents:
        tokens = count_tokens(doc.page_content)
        total_tokens += tokens
        chunk_sizes.append(len(doc.page_content))
        
        token_stats.append({
            "source": doc.metadata.get("source", "unknown"),
            "chunk_index": doc.metadata.get("chunk_index", "unknown"),
            "tokens": tokens,
            "chars": len(doc.page_content)
        })

    # Calculate statistics
    avg_tokens = total_tokens / len(documents) if documents else 0
    avg_chars = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
    
    print(f"\n[SUMMARY] Total documents: {len(documents)}")
    print(f"[SUMMARY] Total tokens: {total_tokens}")
    print(f"[SUMMARY] Average tokens per chunk: {avg_tokens:.2f}")
    print(f"[SUMMARY] Average characters per chunk: {avg_chars:.2f}\n")
    
    print(f"{'File':30s} | {'Chunk #':7s} | {'Tokens':>7s} | {'Chars':>7s}")
    print("-" * 60)
    
    # Sort by token count (descending)
    for entry in sorted(token_stats, key=lambda x: -x['tokens'])[:20]:  # Show top 20
        print(f"{entry['source'][:30]:30s} | {entry['chunk_index']:7} | {entry['tokens']:7d} | {entry['chars']:7d}")
    
    print("\n[NOTE] Showing only the top 20 chunks by token count")

if __name__ == "__main__":
    analyze_documents()
"""