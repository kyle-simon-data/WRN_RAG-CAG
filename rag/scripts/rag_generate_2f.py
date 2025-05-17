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
    embedding_model = HuggingFaceEmbeddings(
       model_name=EMBED_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}  # Add explicit normalization
    )
    #embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    return tokenizer, model, embedding_model, collection


def generate_rag_answer(user_query: str, tokenizer, model, embedding_model, collection, k: int = 5, max_new_tokens: int = 512, relevance_threshold: float = 0.6, debug: bool = False) -> dict:
    """
    Generate answers using the RAG pipeline, now returning a dictionary with all components
    similar to the CAG pipeline for better comparison.
    
    Args:
        user_query: The user's question
        tokenizer: The model tokenizer
        model: The language model
        embedding_model: The embedding model
        collection: The Chroma collection
        k: Number of documents to retrieve
        max_new_tokens: Maximum tokens to generate
        relevance_threshold: Threshold for filtering documents
        debug: Whether to print debug information
        
    Returns:
        dict: Results dictionary with components matching CAG output
    """
    # Get query embedding and search
    query_embedding = embedding_model.embed_query(user_query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )

    # Extract results
    retrieved_docs = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]
    distances = results['distances'][0]
    
    if debug:
        print("Distances:", distances)
        print("\n[DEBUG] Document Relevance Scores:")
        for i, (doc, meta, dist) in enumerate(zip(retrieved_docs, retrieved_metadata, distances)):
            status = "✓ Used" if dist <= relevance_threshold else "✗ Filtered out"
            print(f"Document {i+1}: Distance {dist:.4f} - {status}")
            print(f"  Preview: {doc[:50]}...")
        print("\n[END DEBUG]\n")

    # Filter by relevance threshold
    filtered = [
        (doc, meta, dist) for doc, meta, dist in zip(retrieved_docs, retrieved_metadata, distances)
            if dist <= relevance_threshold
    ]

    # Build prompt
    if not filtered:
        if debug:
            print("\n[DEBUG] No documents met the relevance threshold. Proceeding without context.")
        prompt = f"<|user|>\n{user_query}\n<|assistant|>"
        warning = "Note: No highly relevant documents were found for this query. Response is generated without retrieved context."
        context = ""
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

    if debug:
        print("\n[DEBUG] Assembled Prompt:\n")
        print(prompt)
        print("\n[END DEBUG]\n")

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id
    )

    # Extract response
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    model_response = decoded_output.split("<|assistant|>")[-1].strip()

    # Combine response with citations/warning
    if filtered:
        final_response = f"{model_response}{citations}"
    else:
        final_response = f"{warning}\n\n{model_response}"

    # Return a dictionary with all components (similar to CAG)
    return {
        "query": user_query,
        "context_passages": [doc for doc, _, _ in filtered] if filtered else [],
        "context_scores": [dist for _, _, dist in filtered] if filtered else [],
        "assembled_prompt": prompt,
        "model_response": model_response,  # Raw model response
        "final_response": final_response,  # Response with citations/warning
        "all_retrieved": [(doc[:100] + "...", dist) for doc, _, dist in zip(retrieved_docs, retrieved_metadata, distances)],
        "used_threshold": relevance_threshold,
        "documents_used": len(filtered)
    }


# Optional CLI access
if __name__ == "__main__":
    tokenizer, model, embedding_model, collection = load_rag_components()
    while True:
        query = input("Enter your question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        
        result = generate_rag_answer(query, tokenizer, model, embedding_model, collection, debug=True)
        
        print("\n[Response]:", result["final_response"], "\n")
        
        print("Context Documents Used:", result["documents_used"])
        if result["documents_used"] > 0:
            print("\nDocument Details:")
            for idx, (doc, score) in enumerate(zip(result["context_passages"], result["context_scores"]), 1):
                print(f"{idx}. Distance: {score:.4f} | Preview: {doc[:100]}...")
"""
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


def generate_rag_answer(user_query: str, tokenizer, model, embedding_model, collection, k: int = 5, max_new_tokens: int = 256, relevance_threshold: float = 0.5) -> str:
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

    # Now using relevance_threshold parameter to match CAG pipeline
    filtered = [
        (doc, meta, dist) for doc, meta, dist in zip(retrieved_docs, retrieved_metadata, distances)
            if dist <= relevance_threshold
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
"""