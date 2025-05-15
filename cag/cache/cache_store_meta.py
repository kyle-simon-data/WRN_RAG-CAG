import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from cag.cag_pipeline.load_local_documents_meta import load_local_documents

class CacheStore:
    def __init__(self, embedder_model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedder_model_name)
        self.documents = []  # List of (text, metadata) tuples
        self.document_embeddings = None

    def add_documents(self, docs: List[Document]):
        texts = [doc.page_content for doc in docs]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)

        if self.document_embeddings is None:
            self.document_embeddings = embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, embeddings])

        self.documents.extend([(doc.page_content, doc.metadata) for doc in docs])

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, dict, float]]:
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)
        similarity_scores = np.dot(self.document_embeddings, query_embedding.T).flatten()
        top_indices = similarity_scores.argsort()[-top_k:][::-1]

        results = [
            (self.documents[idx][0], self.documents[idx][1], similarity_scores[idx])
            for idx in top_indices
        ]
        return results

    
    def __len__(self):
        return len(self.documents)
    
def load_cache(cache: CacheStore):
    #Load docs from local downloads directory into CacheStore
    print("[INFO] Loading local documents into CacheStore...")
    documents = load_local_documents()
    cache.add_documents(documents)
    print(f"[INFO] CacheStore seeded with {len(documents)} documents.")

if __name__ == "__main__":
    print("[INFO] Initializing cache and loading documents...")
    cache = CacheStore()
    docs = load_local_documents()
    cache.add_documents(docs)
    print(f"[INFO] Loaded {len(docs)} documents into cache.")

    # Optional: test a query
    query = "When was Sean Lennon born?"
    results = cache.search(query, top_k=3)
    for i, (text, meta, score) in enumerate(results):
        print(f"\n[RESULT {i+1}]")
        print(f"Score: {score:.4f}")
        print(f"Text: {text}")
        print(f"Metadata: {meta}")
