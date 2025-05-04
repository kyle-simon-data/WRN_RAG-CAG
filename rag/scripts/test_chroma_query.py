import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings

# Setup paths and embeddings
CHROMA_DIR = 'vectorstore'
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Conecty to Chroma
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("cyberbot-knowledgebase")

#Sample Query
query_text = "What is cybersecurity risk management?"
query_embedding = embedding_model.embed_query(query_text)

#Query Chroma
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=['documents', 'metadatas']
)

#Display results
for i, doc in enumerate(results['documents'][0]):
    metadata = results['metadatas'][0][i]
    print(f"\nResult {i+1}:")
    print(doc)