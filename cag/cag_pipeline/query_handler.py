from cag.cag_pipeline.cag_generate import generate_cag_response
from cag.cache.cache_store import CacheStore, load_cache as seed_cache

def load_cache() -> CacheStore:
    # Initializes an empty cache
    return CacheStore()

def build_prompt(retrieved_docs: list, query: str) -> str:
    """
    Build a prompt by combining retrieved documents and user query,
    using the same format as the RAG pipeline.
    
    Args:
        retrieved_docs (list): List of document texts.
        query (str): User's input

    Returns:
        str: Assembled prompt.
    """
    # Handle empty document case exactly like RAG
    if not retrieved_docs:
        return f"<|user|>\n{query}\n<|assistant|>"
    
    # Start with user query
    prompt = f"<|user|>\n{query}\n"
    
    # Add context section with documents
    prompt += "<|context|>\n"
    for idx, doc in enumerate(retrieved_docs):
        prompt += f"[{idx+1}] {doc}\n"
    
    # Add assistant marker to signal response (previously commented out)
    prompt += "<|assistant|>"
    
    return prompt

def run_query(cache: CacheStore, query: str, top_k: int=5, relevance_threshold: float=0.6, max_new_tokens: int=256, debug: bool=False) -> dict:
    """
    Handles the users' query: search the cache, filter by relevance, pass to LLM, and generate a response.
    Aligned with RAG pipeline behavior for consistent comparison.
    
    Args:
        cache (CacheStore): The cache store containing the documents
        query (str): User's question
        top_k (int): Maximum number of documents to retrieve
        relevance_threshold (float): Minimum similarity score for documents to be included
        max_new_tokens (int): Maximum new tokens to generate (matching RAG parameter)
        debug (bool): Whether to print debug information
        
    Returns:
        dict: Result containing query, context, prompt, and response
    """
    # Searching cache
    results = cache.search(query, top_k=top_k)
    
    # Filter results by relevance threshold - same approach as RAG
    filtered_results = [(text, score) for text, score in results if (1 - score) <= relevance_threshold]
    
    # Debug: print relevance scores before filtering
    if debug:
        print("\n[DEBUG] Document Relevance Scores:")
        for i, (text, score) in enumerate(results):
            distance = 1 - score
            status = "✓ Used" if distance <= relevance_threshold else "✗ Filtered out"
            print(f"Document {i+1}: Similarity {score:.4f} (Distance {distance:.4f}) - {status}")
            print(f"  Preview: {text[:50]}...")
        print("\n[END DEBUG]\n")
    
    # Handle empty results exactly like RAG
    if not filtered_results:
        if debug:
            print("\n[DEBUG] No documents met the relevance threshold. Proceeding without context.")
        
        prompt = build_prompt([], query)
        warning = "Note: No highly relevant documents were found for this query. Response is generated without retrieved context."
    else:
        # Extract document text from filtered results
        docs_only = [text for text, _ in filtered_results]
        prompt = build_prompt(docs_only, query)
        warning = ""
    
    # Debug: print the assembled prompt
    if debug:
        print("\n[DEBUG] Assembled Prompt:\n")
        print(prompt)
        print("\n[END DEBUG]\n")

    # Generate response with model
    model_response = generate_cag_response(prompt, max_new_tokens=max_new_tokens)

    # Build citations exactly like RAG does
    if filtered_results:
        citation_lines = []
        for i, (doc, score) in enumerate(filtered_results):
            distance = 1 - score
            # In RAG, this would use metadata - we'll just use Doc ID
            source = f"Doc {i+1}"
            citation_lines.append(f"[{i+1}] Source: {source} (distance={distance:.2f})")
        
        citations = "\n\n" + "\n".join(citation_lines)
        final_response = f"{model_response}{citations}"
    else:
        # Add warning for no documents (match RAG behavior)
        final_response = f"{warning}\n\n{model_response}"
    
    return {
        "query": query,
        "context_passages": [text for text, _ in filtered_results] if filtered_results else [],
        "context_scores": [1-score for _, score in filtered_results] if filtered_results else [],  # Convert to distances to match RAG
        "assembled_prompt": prompt,
        "model_response": model_response,  # Raw model response
        "final_response": final_response,  # Response with citations/warning
        "all_retrieved": [(text[:100] + "...", 1-score) for text, score in results],  # Convert to distances
        "used_threshold": relevance_threshold,
        "documents_used": len(filtered_results)
    }

# Testing with CLI input
if __name__ == "__main__":
    cache = CacheStore()      # Create instance
    seed_cache(cache)         # Load docs
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        
        result = run_query(cache, query, debug=True)
        
        print("\nGenerated Answer:\n")
        print(result["final_response"])  # Use the response with citations
        
        print("\nContext Documents Used:", result["documents_used"])
        if result["documents_used"] > 0:
            print("\nDocument Details:")
            for idx, (doc, score) in enumerate(zip(result["context_passages"], result["context_scores"]), 1):
                print(f"{idx}. Distance: {score:.4f} | Preview: {doc[:100]}...")