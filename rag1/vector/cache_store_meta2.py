import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from cag.cag_pipeline.load_local_documents2 import load_local_documents

class CacheStore:
    def __init__(self, embedder_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the CacheStore.
        
        Args:
            embedder_model_name (str): Hugging Face model name for embedding.
        """

        self.embedder = SentenceTransformer(embedder_model_name)
        self.documents = []         #List of raw text chunks
        self.document_embeddings = None      #Numpy array of embeddings

    def add_documents(self, docs: List[str], metadata_list: List[dict] = None):
        """
        Embed and store new documents with optional metadata.
        
        Args:
            docs (List[str]): List of text chunks to embed and store
            metadata_list (List[dict], optional): List of metadata dicts. Must match length of docs if provided.
        """
        if metadata_list is None:
            metadata_list = [{} for _ in docs]  # Default to empty metadata

        assert len(docs) == len(metadata_list), "Each document must have corresponding metadata."

        embeddings = self.embedder.encode(docs, normalize_embeddings=True)

        if self.document_embeddings is None:
            self.document_embeddings = embeddings
        else:
            self.document_embeddings = np.vstack([self.document_embeddings, embeddings])

        self.documents.extend(zip(docs, metadata_list))


    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, dict, float]]:
        """
        Search for the top-k most similar documents to a query.
        
        Returns:
            List of tuples: (document_text, metadata, similarity_score)
        """
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
    documents, metadata = load_local_documents()
    cache.add_documents(documents, metadata)
    print(f"[INFO] CacheStore seeded with {len(documents)} documents.")