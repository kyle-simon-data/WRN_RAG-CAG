

import csv
import time
import argparse
from datetime import datetime

# Import CAG components
from cag.cache.cache_store import CacheStore, load_cache as seed_cache
from cag.cag_pipeline.query_handler_old import run_query as cag_run_query

# Import RAG components
from transformers import AutoTokenizer, AutoModelForCausalLM
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import torch
# Import the generate_rag_answer function from your rag module
from rag2.scripts.rag_generate_2f import generate_rag_answer, load_rag_components

# Test queries - replace these with your own 10 queries
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

def run_cag_benchmark(output_file="cag_benchmark_results.csv"):
    """Run benchmarks for the CAG system and save results to CSV."""
    print(f"Starting CAG benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize the cache
    cache = CacheStore()
    seed_cache(cache)
    
    # Prepare CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'response', 'documents', 'processing_time', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each query
        for i, query in enumerate(QUERIES, 1):
            print(f"[CAG] Processing query {i}/{len(QUERIES)}: {query[:40]}...")
            
            # Time the operation
            start_time = time.time()
            
            # Run the query
            result = cag_run_query(cache, query, debug=True, relevance_threshold=0.3)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare documents info - include both text and relevance score
            documents_info = []
            for passage, score in zip(result.get("context_passages", []), 
                                     result.get("context_scores", [])):
                # Truncate document text for readability if needed
                doc_preview = passage[:150] + "..." if len(passage) > 150 else passage
                documents_info.append(f"[Score: {score:.4f}] {doc_preview}")
            
            # Write results to CSV
            writer.writerow({
                'query': query,
                'response': result["model_response"],
                'documents': "\n".join(documents_info) if documents_info else "No documents used",
                'processing_time': f"{processing_time:.2f}s",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Print progress
            print(f"  Completed in {processing_time:.2f}s")
            
            # Optional: add a small delay between queries
            time.sleep(0.5)
    
    print(f"CAG benchmark completed. Results saved to {output_file}")
    return output_file

def run_rag_benchmark(output_file="rag_benchmark_results.csv"):
    """Run benchmarks for the RAG system and save results to CSV."""
    print(f"Starting RAG benchmark at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize RAG components
    tokenizer, model, embedding_model, collection = load_rag_components()
    
    # Prepare CSV file
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'response', 'documents', 'processing_time', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each query
        for i, query in enumerate(QUERIES, 1):
            print(f"[RAG] Processing query {i}/{len(QUERIES)}: {query[:40]}...")
            
            # Time the operation
            start_time = time.time()
            
            # We need to capture documents being used, so we'll modify the approach
            # First, get query embedding
            query_embedding = embedding_model.embed_query(query)
            
            # Get results
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Get documents and distances
            retrieved_docs = results['documents'][0]
            retrieved_metadata = results['metadatas'][0]
            distances = results['distances'][0]
            
            # Apply relevance threshold as in the generate_rag_answer function
            RELEVANCE_THRESHOLD = 0.3
            filtered = [
                (doc, meta, dist) for doc, meta, dist in 
                zip(retrieved_docs, retrieved_metadata, distances)
                if dist <= RELEVANCE_THRESHOLD
            ]
            
            # Prepare documents info
            documents_info = []
            for i, (doc, meta, dist) in enumerate(filtered):
                source = meta.get('source', f'Doc {i+1}')
                doc_preview = doc[:150] + "..." if len(doc) > 150 else doc
                documents_info.append(f"[Distance: {dist:.4f}] {doc_preview} (Source: {source})")
            
            # Generate the answer
            response = generate_rag_answer(query, tokenizer, model, embedding_model, collection)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Write results to CSV
            writer.writerow({
                'query': query,
                'response': response,
                'documents': "\n".join(documents_info) if documents_info else "No documents used",
                'processing_time': f"{processing_time:.2f}s",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Print progress
            print(f"  Completed in {processing_time:.2f}s")
            
            # Optional: add a small delay between queries
            time.sleep(0.5)
    
    print(f"RAG benchmark completed. Results saved to {output_file}")
    return output_file

def generate_comparison_report(cag_file, rag_file, output_file="comparison_report.csv"):
    """Generate a comparison report between CAG and RAG results."""
    print(f"Generating comparison report...")
    
    # Read CAG results
    cag_results = {}
    with open(cag_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cag_results[row['query']] = row
    
    # Read RAG results
    rag_results = {}
    with open(rag_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rag_results[row['query']] = row
    
    # Create comparison CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'cag_time', 'rag_time', 'time_diff', 
                     'cag_doc_count', 'rag_doc_count', 'cag_response_length', 
                     'rag_response_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for query in QUERIES:
            cag_row = cag_results.get(query, {})
            rag_row = rag_results.get(query, {})
            
            if not cag_row or not rag_row:
                continue
                
            # Extract time (convert from string like "12.34s" to float)
            cag_time = float(cag_row.get('processing_time', '0s').replace('s', ''))
            rag_time = float(rag_row.get('processing_time', '0s').replace('s', ''))
            
            # Count documents
            cag_doc_count = cag_row.get('documents', '').count('[Score:')
            rag_doc_count = rag_row.get('documents', '').count('[Distance:')
            
            # Calculate response lengths
            cag_response_length = len(cag_row.get('response', ''))
            rag_response_length = len(rag_row.get('response', ''))
            
            writer.writerow({
                'query': query,
                'cag_time': f"{cag_time:.2f}s",
                'rag_time': f"{rag_time:.2f}s",
                'time_diff': f"{rag_time - cag_time:.2f}s",
                'cag_doc_count': cag_doc_count,
                'rag_doc_count': rag_doc_count,
                'cag_response_length': cag_response_length,
                'rag_response_length': rag_response_length
            })
    
    print(f"Comparison report generated: {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark CAG and RAG systems')
    parser.add_argument('--cag-only', action='store_true', help='Run only CAG benchmark')
    parser.add_argument('--rag-only', action='store_true', help='Run only RAG benchmark')
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cag_file = f"cag_benchmark_{timestamp}.csv"
    rag_file = f"rag_benchmark_{timestamp}.csv"
    comparison_file = f"comparison_report_{timestamp}.csv"
    
    if args.cag_only:
        cag_file = run_cag_benchmark(cag_file)
    elif args.rag_only:
        rag_file = run_rag_benchmark(rag_file)
    else:
        # Run both benchmarks
        print("Running both CAG and RAG benchmarks...")
        cag_file = run_cag_benchmark(cag_file)
        rag_file = run_rag_benchmark(rag_file)
        generate_comparison_report(cag_file, rag_file, comparison_file)
        
        print("\nBenchmark Summary:")
        print(f"- CAG results: {cag_file}")
        print(f"- RAG results: {rag_file}")
        print(f"- Comparison: {comparison_file}")
