from cag.cache.cache_store import CacheStore, load_cache

def main():
    # Step 1: Initialize CacheStore
    cache = CacheStore()

    # Step 2: Load documents from downloads/ into the cache
    load_cache(cache)

    # Step 3: Print basic test output
    print(f"[TEST] CacheStore now contains {len(cache)} documents.")

    # Step 4: Test a simple search
    query = "cybersecurity vulnerabilities"
    results = cache.search(query, top_k=3)
    
    print(f"[TEST] Top 3 search results for query '{query}':")
    for doc, score in results:
        print(f" - Score: {score:.4f}, Content Preview: {doc[:100]}...")

if __name__ == "__main__":
    main()
