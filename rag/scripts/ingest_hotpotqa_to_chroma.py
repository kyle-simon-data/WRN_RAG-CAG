import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from cag.cag_pipeline.load_local_documents_meta import load_local_documents

# Constants
CHROMA_DIR = 'vectorstore'
COLLECTION_NAME = 'cyberbot-knowledgebase'
#EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Step 1: Load HotpotQA sentence-level documents
documents = load_local_documents()
print(f"[INFO] Loaded {len(documents)} HotpotQA sentence-level documents")

# Step 2: Initialize embedding model and Chroma client
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

#Force-delete the old collection
try:
    chroma_client.delete_collection(COLLECTION_NAME)
    print(f"[INFO] Deleted existing collection '{COLLECTION_NAME}'")
except:
    print(f"[INFO] No existing collection '{COLLECTION_NAME}' found. Creating new one.")

collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Step 3: Embed and add to Chroma
texts = [doc.page_content for doc in documents]
metadatas = [doc.metadata for doc in documents]
embeddings = embedding_model.embed_documents(texts)

collection.add(
    embeddings=embeddings,
    documents=texts,
    metadatas=metadatas,
    ids=[f"hotpotqa-{i}" for i in range(len(texts))]
)

print(f"[INFO] Successfully indexed {len(texts)} HotpotQA chunks into Chroma.")
