### handles loading the CacheStore and managing user query flow.

from cag.cag_pipeline.cag_generate import generate_cag_response
#from cag.cache.cache_store import CacheStore
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
    # Start with user query
    prompt = f"<|user|>\n{query}\n"
    
    # Add context section
    prompt += "<|context|>\n"
    for idx, doc in enumerate(retrieved_docs):
        prompt += f"[{idx+1}] {doc}\n"
    
    # Add assistant marker to signal response
    prompt #+= "<|assistant|>"
    
    return prompt

"""
def build_prompt(retrieved_docs: list, query: str) -> str:
    #Build a simple prompt by combining retrieved documents and user query.
    #Args:
    #    retrieved_docs (list): List of document texts.
    #    query (str): User's input

    #Returns:
    #    str: Assembled prompt.

    prompt = "Context Documents:\n"
    for idx, doc in enumerate(retrieved_docs):
        prompt += f"{idx+1}. {doc}\n"

    #prompt += f"\nUser Question:\n{query}\n\nAnswer:"
    #Trying to force a response from LLM by adding a new line (\n)
    prompt += f"\nUser Question:\n{query}\n\nAnswer:\n"
    return prompt
"""    
def run_query(cache: CacheStore, query: str, top_k: int=5, relevance_threshold: float=0.3, debug: bool=False) -> dict:
    """
    Handles the users' query: search the cache, filter by relevance, pass to LLM, and generate a response.
    
    Args:
        cache (CacheStore): The cache store containing the documents
        query (str): User's question
        top_k (int): Maximum number of documents to retrieve
        relevance_threshold (float): Minimum similarity score for documents to be included
        debug (bool): Whether to print debug information
        
    Returns:
        dict: Result containing query, context, prompt, and response
    """
    # Searching cache
    results = cache.search(query, top_k=top_k)
    
    # Filter results by relevance threshold (similar to RAG pipeline)
    filtered_results = [(text, score) for text, score in results if score >= relevance_threshold]
    
    # If no documents meet the threshold, handle appropriately
    if not filtered_results:
        if debug:
            print("\n[DEBUG] No documents met the relevance threshold. Proceeding without context.")
        
        # Option 1: Use an empty context prompt
        prompt = build_prompt([], query)
        
        # Option 2 (alternative): Fall back to a few documents anyway
        # prompt = build_prompt([text for text, _ in results[:2]], query)
    else:
        # Extract just the document text from filtered results
        docs_only = [text for text, _ in filtered_results]
        prompt = build_prompt(docs_only, query)
    
    # Debug: print the assembled prompt before generation
    if debug:
        print("\n[DEBUG] Assembled Prompt:\n")
        print(prompt)
        print("\n[END DEBUG]\n")
        
        # Also print the relevance scores of all retrieved documents
        print("\n[DEBUG] Document Relevance Scores:")
        for i, (text, score) in enumerate(results):
            status = "✓ Used" if score >= relevance_threshold else "✗ Filtered out"
            print(f"Document {i+1}: Score {score:.4f} - {status}")
            print(f"  Preview: {text[:50]}...")
        print("\n[END DEBUG]\n")

    # Pass this prompt to the LLM for actual response/answer generation
    model_response = generate_cag_response(prompt)

    return {
        "query": query,
        "context_passages": [text for text, _ in filtered_results] if filtered_results else [],
        "context_scores": [score for _, score in filtered_results] if filtered_results else [],
        "assembled_prompt": prompt,
        "model_response": model_response,
        "all_retrieved": [(text[:100] + "...", score) for text, score in results],
        "used_threshold": relevance_threshold
    }
"""
def run_query(cache: CacheStore, query: str, top_k: int=3, debug: bool=False) -> dict:
    
    ###Handles the users' query: search the cache, pass to LLM, and generate a response.

    #searching cache
    results = cache.search(query, top_k=top_k)

    #extract just the document text (currently going to ingore similarity scores)
   #docs_only = [text for text, metadata, score in results]
    docs_only = [text for text, score in results]

    #buidl prompt
    prompt = build_prompt(docs_only, query)

    # Debug: print the assembled prompt before generation
    if debug:
        print("\n[DEBUG] Assembled Prompt:\n")
        print(prompt)
        print("\n[END DEBUG]\n")

    #Pass this prompt to the LLM for actual reponse/answer generation
    model_response = generate_cag_response(prompt)

    return {
        "query": query,
        "context_passages": docs_only,
        "assembled_prompt": prompt,
        "model_response": model_response
    }
"""
    
#testing with CLI input
if __name__ == "__main__":
    cache = CacheStore()      #create instance
    seed_cache(cache)       #load docs
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        result = run_query(cache, query, debug=True)
        print("\nGenerated Answer:\n")
        print(result["model_response"])

        print("\nContext Documents Used:\n")
        for idx, doc in enumerate(result["context_passages"], 1):
            print(f"{idx}. {doc[:100]}...") #show first 100 chars of each