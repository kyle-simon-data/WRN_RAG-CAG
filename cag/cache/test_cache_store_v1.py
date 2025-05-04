from cag.cache.cache_store import CacheStore

cache = CacheStore()

# Add some dummy documents
documents = [
    "The National Vulnerability Database tracks CVEs.",
    "Zero-day vulnerabilities are exploited before they are known.",
    "Cybersecurity involves both prevention and response.",
    "WhiteRabbitNeo-7B is a powerful language model."
]
cache.add_documents(documents)

# Search
results = cache.search("What is a zero-day exploit?", top_k=2)
for doc, score in results:
    print(f"Score: {score:.4f} - Doc: {doc}")
