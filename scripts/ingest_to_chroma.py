import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# Define paths
DOCUMENT_DIR = 'data/downloads'
CHROMA_DIR = 'vectorstore'

# Initialize embedding model (replace if different)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Create or get Chroma collection
collection = chroma_client.get_or_create_collection("cyberbot-knowledgebase")

# Supported loaders based on file type
def load_document(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    return loader.load()

# Process and embed documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

for filename in os.listdir(DOCUMENT_DIR):
    file_path = os.path.join(DOCUMENT_DIR, filename)
    docs = load_document(file_path)
    splits = text_splitter.split_documents(docs)
    
    texts = [split.page_content for split in splits]
    embeddings = embedding_model.embed_documents(texts)
    
    # Add to Chroma collection
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": filename}] * len(texts),
        ids=[f"{filename}-{i}" for i in range(len(texts))]
    )

    print(f"Processed and indexed {filename}")

print("All documents ingested successfully.")
