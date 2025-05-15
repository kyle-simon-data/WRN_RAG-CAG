import os
import json
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Define paths
DOCUMENT_DIR = 'data/downloads'
CHROMA_DIR = 'vectorstore'

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Chroma client and collection
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection("cyberbot-knowledgebase")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def load_documents_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    filename = os.path.basename(file_path)
    documents = []

    try:
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()

        elif ext in {'.txt', '.md', '.sh'}:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()

        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for title, sentences in data.get("context", []):
                for idx, sentence in enumerate(sentences):
                    documents.append(
                        Document(
                            page_content=sentence.strip(),
                            metadata={
                                "source": filename,
                                "type": "json",
                                "title": title,
                                "sentence_id": idx
                            }
                        )
                    )
        else:
            print(f"[SKIP] Unsupported file type: {filename}")

    except Exception as e:
        print(f"[ERROR] Failed to load {filename}: {e}")

    return documents

# Ingest loop
for filename in os.listdir(DOCUMENT_DIR):
    file_path = os.path.join(DOCUMENT_DIR, filename)
    if not os.path.isfile(file_path):
        continue

    docs = load_documents_from_file(file_path)
    if not docs:
        print(f"[SKIP] No content loaded from {filename}")
        continue

    splits = text_splitter.split_documents(docs)
    texts = [split.page_content for split in splits]
    metadatas = [split.metadata for split in splits]

    embeddings = embedding_model.embed_documents(texts)
    if not embeddings:
        print(f"[ERROR] Embedding failed for {filename}")
        continue

    ids = [f"{filename}-{i}" for i in range(len(texts))]

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    print(f"[SUCCESS] Ingested {filename} ({len(texts)} chunks)")

print("[COMPLETE] All supported documents ingested.")

"""
import os
import chromadb
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

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
"""