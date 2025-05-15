import os
import json
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Config
TESTDOCS_DIR = "data/triviaqa/testdocs"
CHROMA_DIR = "vectorstore"
COLLECTION_NAME = "cyberbot-knowledgebase"

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

doc_count = 0

for filename in os.listdir(TESTDOCS_DIR):
    if not filename.endswith(".json"):
        continue

    file_path = os.path.join(TESTDOCS_DIR, filename)
    with open(file_path, "r") as f:
        data = json.load(f)

    context_blocks = data.get("context", [])
    for idx, (title, passage) in enumerate(context_blocks):
        if not passage.strip():
            continue

        # Wrap as LangChain Document
        doc = Document(page_content=passage, metadata={"source_file": filename, "index": idx})
        splits = text_splitter.split_documents([doc])
        texts = [split.page_content for split in splits]
        embeddings = embedding_model.embed_documents(texts)

        ids = [f"{data['id']}_{idx}_{i}" for i in range(len(texts))]

        # Add to collection
        collection.add(
            documents=texts,
            metadatas=[doc.metadata] * len(texts),
            ids=ids
        )
        doc_count += len(texts)

print(f" Ingested {doc_count} context chunks from TriviaQA testdocs.")