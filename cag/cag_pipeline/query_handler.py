### handles loading the CacheStore and managing user query flow.

from cag.cag_pipeline.cag_generate import generate_cag_response
from cag.cache.cache_store import CacheStore

def load_cache() -> CacheStore:
    
    #Initializes and seeds the CacheStore with dummy data
    #This will later be replaced with documents pulled from s3 bucket

    cache = CacheStore()

    #current dummy docs
    documents = [
        "The National Vulnerability Database tracks cybersecurity vulnerabilities.",
        "Zero-day exploits are attacks on vulnerabilities before the vendor knows about them.",
        "The CIA triad stands for Confidentiality, Integrity, and Availability.",
        "WhiteRabbitNeo-7B is a 7-billion parameter language model."
    ]
    cache.add_documents(documents)

    return cache

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

def run_query(cache: CacheStore, query: str, top_k: int=3, debug: bool=False) -> dict:
    
    ###Handles the users' query: search the cache, pass to LLM, and generate a response.

    #searching cache
    results = cache.search(query, top_k=top_k)

    #extract just the document text (currently going to ingore similarity scores)
    docs_only = [doc for doc, score in results]

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

#testing with CLI input
if __name__ == "__main__":
    cache = load_cache()
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