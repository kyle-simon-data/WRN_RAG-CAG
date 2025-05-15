# rag_generate.py

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Configuration
CHROMA_DIR = 'vectorstore'
COLLECTION_NAME = 'cyberbot-knowledgebase'
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_rag_components():
    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    model = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    return tokenizer, model, embedding_model, collection


def generate_rag_answer(user_query: str, tokenizer, model, embedding_model, collection, k: int = 5, max_new_tokens: int = 256) -> str:
    query_embedding = embedding_model.embed_query(user_query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )

    retrieved_docs = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]
    distances = results['distances'][0]
    print("Distances:", distances)

    RELEVANCE_THRESHOLD = 0.3
    filtered = [
        (doc, meta) for doc, meta, dist in zip(retrieved_docs, retrieved_metadata, distances)
            if dist <= RELEVANCE_THRESHOLD
    ]

    if not filtered:
        context = ""
        warning = "Note: No highly relevant documents were found for this query. Response is generated without retrieved context."
        prompt = f"<|user|>\n{user_query}\n<|assistant|>"
    else:
        context_lines = []
        citation_lines = []
        for i, (doc, meta, dist) in enumerate(filtered):
            source = meta.get('source', f'Doc {i+1}')
            context_lines.append(f"[{i+1}] {doc}")
            citation_lines.append(f"[{i+1}] Source: {source} (distance={dist:.2f})")

        context = "\n".join(context_lines)
        citations = "\n\n" + "\n".join(citation_lines)
        warning = ""
        prompt = f"<|user|>\n{user_query}\n<|context|>\n{context}\n<|assistant|>"

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    response = decoded_output.split("<|assistant|>")[-1].strip()

    # Return warning first if no context was used
    return f"{warning}\n\n{response}" if warning else f"{response}{citations}"


# Optional CLI access
if __name__ == "__main__":
    tokenizer, model, embedding_model, collection = load_rag_components()
    while True:
        query = input("Enter your question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        response = generate_rag_answer(query, tokenizer, model, embedding_model, collection)
        print("\n[Response]:", response, "\n")