import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    print("📁 Loading documents from ./data/")
    loader = DirectoryLoader("v1_local_ingest/data/", glob="**/*.txt")
    docs = loader.load()

    print(f"📄 Loaded {len(docs)} document(s). Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks.")

    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    print("📦 Saving vector store to ./vectorstore/")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local("vectorstore")

    print("✅ Done! Vector store saved to 'vectorstore/'.")

if __name__ == "__main__":
    main()
