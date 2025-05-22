import csv
import time
import argparse
from datetime import datetime

# Import rag1 components
from rag1.vector.vector_store import VectorStore, load_vectors as seed_store
from rag1.rag1_pipeline.query_handler import run_query as rag1_run_query

# Import RAG components
from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import torch
# Import the updated RAG function
from rag2.scripts.rag_generate_2f import generate_rag_answer, load_rag_components

# Test queries
QUERIES = [
    "Are there any vulnerabilities for the Wordpress Smart Product Review plugin? If so, how do they work?",
    "What is an example of a SQL injection vulnerability",
    "I use BuddyBoss Platform and I'm interested in keeping my wordpress account secure. What steps should I take to secure my site?",
    "I have LibreOffice.  Describe any security issues that I need to be aware of.",
    "I'm interested in testing the security of our Wordpress service.  What recommendations do you have for attack vectors?",
    "Our system has recently been attacked. It seems as though the attacker then escalated privileges but we can't find the attack vector. Any suggestions?",
    "What is the danger associated with the new vulnerability in Novel-Plus?",
    "I heard that UrbanGO had a critical vulnerability. Explain the significance and ways to secure against attacks.",
    "I'm a red-teamer interested in attacking our Flynax Bridge service. What can I do?",
    "As a pentester, my target has a Netgear AX1600 Router. What options do I have for attack vectors?",
]

def run_rag1_benchmark(output_file="rag1_benchmark_results.csv", relevance_threshold=0.6):
    """Run benchmarks for the rag1 system and save results to CSV."""
    print(f"Starting rag1 benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize the cache
    cache = CacheStore()
    seed_cache(cache)
    
    # Prepare CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'response', 'documents', 'documents_used', 'processing_time', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each query
        for i, query in enumerate(QUERIES, 1):
            print(f"[RAG1] Processing query {i}/{len(QUERIES)}: {query[:40]}...")
            
            # Time the operation
            start_time = time.time()
            
            # Run the query with aligned function
            result = rag1_run_query(cache, query, debug=True, relevance_threshold=relevance_threshold)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare documents info with distance scores
            documents_info = []
            for passage, score in zip(result.get("context_passages", []), 
                                     result.get("context_scores", [])):
                doc_preview = passage[:150] + "..." if len(passage) > 150 else passage
                documents_info.append(f"[Distance: {score:.4f}] {doc_preview}")
            
            # Write results to CSV
            writer.writerow({
                'query': query,
                'response': result["final_response"],  # Use final response with citations
                'documents': "\n".join(documents_info) if documents_info else "No documents used",
                'documents_used': result["documents_used"],
                'processing_time': f"{processing_time:.2f}s",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Print progress
            print(f"  Completed in {processing_time:.2f}s with {result['documents_used']} documents used")
            
            # Optional: add a small delay between queries
            time.sleep(0.5)
    
    print(f"RAG1 benchmark completed. Results saved to {output_file}")
    return output_file

def run_rag_benchmark(output_file="rag_benchmark_results.csv", relevance_threshold=0.6):
    """Run benchmarks for the RAG system and save results to CSV."""
    print(f"Starting RAG benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize RAG components
    tokenizer, model, embedding_model, collection = load_rag_components()
    
    # Prepare CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'response', 'documents', 'documents_used', 'processing_time', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each query
        for i, query in enumerate(QUERIES, 1):
            print(f"[RAG] Processing query {i}/{len(QUERIES)}: {query[:40]}...")
            
            # Time the operation
            start_time = time.time()
            
            # Generate answer with the updated RAG function
            result = generate_rag_answer(query, tokenizer, model, embedding_model, collection, 
                                        relevance_threshold=relevance_threshold, debug=True)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare documents info
            documents_info = []
            for doc, score in zip(result.get("context_passages", []), 
                                 result.get("context_scores", [])):
                doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
                documents_info.append(f"[Distance: {score:.4f}] {doc_preview}")
            
            # Write results to CSV
            writer.writerow({
                'query': query,
                'response': result["final_response"],
                'documents': "\n".join(documents_info) if documents_info else "No documents used",
                'documents_used': result["documents_used"],
                'processing_time': f"{processing_time:.2f}s",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Print progress
            print(f"  Completed in {processing_time:.2f}s with {result['documents_used']} documents used")
            
            # Optional: add a small delay between queries
            time.sleep(0.5)
    
    print(f"RAG benchmark completed. Results saved to {output_file}")
    return output_file

def generate_comparison_report(rag1_file, rag_file, output_file="comparison_report.csv"):
    """Generate a comparison report between RAG1 and RAG2 results."""
    print(f"Generating comparison report...")
    
    # Read RAG1 results
    RAG1_results = {}
    with open(rag1_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rag1_results[row['query']] = row
    
    # Read RAG results
    rag_results = {}
    with open(rag_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rag_results[row['query']] = row
    
    # Create comparison CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'rag1_time', 'rag_time', 'time_diff', 
                     'rag1_doc_count', 'rag_doc_count', 'doc_count_diff',
                     'rag1_response_length', 'rag_response_length', 'response_length_diff']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for query in QUERIES:
            rag1_row = rag1_results.get(query, {})
            rag_row = rag_results.get(query, {})
            
            if not rag1_row or not rag_row:
                continue
                
            # Extract time
            rag1_time = float(rag1_row.get('processing_time', '0s').replace('s', ''))
            rag_time = float(rag_row.get('processing_time', '0s').replace('s', ''))
            
            # Count documents
            rag1_doc_count = int(rag1_row.get('documents_used', '0'))
            rag_doc_count = int(rag_row.get('documents_used', '0'))
            
            # Calculate response lengths
            rag1_response_length = len(rag1_row.get('response', ''))
            rag_response_length = len(rag_row.get('response', ''))
            
            writer.writerow({
                'query': query,
                'rag1_time': f"{rag1_time:.2f}s",
                'rag_time': f"{rag_time:.2f}s",
                'time_diff': f"{rag_time - rag1_time:.2f}s",
                'rag1_doc_count': rag1_doc_count,
                'rag_doc_count': rag_doc_count, 
                'doc_count_diff': rag_doc_count - rag1_doc_count,
                'rag1_response_length': rag1_response_length,
                'rag_response_length': rag_response_length,
                'response_length_diff': rag_response_length - rag1_response_length
            })
    
    print(f"Comparison report generated: {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark RAG1 and RAG2 systems')
    parser.add_argument('--rag1-only', action='store_true', help='Run only RAG1 benchmark')
    parser.add_argument('--rag2-only', action='store_true', help='Run only RAG2 benchmark')
    parser.add_argument('--relevance', type=float, default=0.6, help='Relevance threshold (default: 0.6)')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    rag1_file = f"rag1_benchmark_{timestamp}.csv"
    rag_file = f"rag_benchmark_{timestamp}.csv"
    comparison_file = f"comparison_report_{timestamp}.csv"
    
    if args.rag1_only:
        rag1_file = run_rag1_benchmark(rag1_file, relevance_threshold=args.relevance)
    elif args.rag_only:
        rag_file = run_rag_benchmark(rag_file, relevance_threshold=args.relevance)
    else:
        # Run both benchmarks
        print(f"Running both RAG1 and RAG2 benchmarks with relevance threshold: {args.relevance}...")
        rag1_file = run_rag1_benchmark(rag1_file, relevance_threshold=args.relevance)
        rag_file = run_rag_benchmark(rag_file, relevance_threshold=args.relevance)
        generate_comparison_report(rag1_file, rag_file, comparison_file)
        
        print("\nBenchmark Summary:")
        print(f"- RAG1 results: {rag1_file}")
        print(f"- RAG results: {rag_file}")
        print(f"- Comparison: {comparison_file}")
        print(f"- Relevance threshold used: {args.relevance}")