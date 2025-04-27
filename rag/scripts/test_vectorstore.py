from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the vector store
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# Search query
query = "What does UMBC focus on in cybersecurity?"
results = vectorstore.similarity_search(query, k=2)

print("Search Results:")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:\n{doc.page_content}")