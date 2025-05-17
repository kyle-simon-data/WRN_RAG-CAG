"""
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
"""

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

# Initialize embedding model with explicit parameters to match SentenceTransformer
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True}
)

# Reinitialize ChromaDB collection with explicit settings
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

# Optionally reset collection to ensure consistency
try:
    chroma_client.delete_collection("cyberbot-knowledgebase")
except:
    pass  # Collection might not exist yet

# Create collection with explicit distance metric
collection = chroma_client.create_collection(
    name="cyberbot-knowledgebase",
    metadata={"hnsw:space": "cosine"}
)

# Text splitter (already matches CAG)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


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
            
            # Handle CVE JSON structure - only extract descriptions
            if "descriptions" in data:
                for idx, description in enumerate(data["descriptions"]):
                    documents.append(
                        Document(
                            page_content=description.strip(),
                            metadata={
                                "source": filename,
                                "type": "json",
                                "content_type": "description",
                                "item_id": idx
                            }
                        )
                    )
            
            # For backward compatibility, still try the old format
            elif "context" in data:
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
        import traceback
        traceback.print_exc()

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