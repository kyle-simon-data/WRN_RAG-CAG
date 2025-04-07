from sentence_transformers import SentenceTransformer

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_query_embedding(query):
    """Encode a query into its embedding."""
    return embedding_model.encode(query, convert_to_tensor=True)
