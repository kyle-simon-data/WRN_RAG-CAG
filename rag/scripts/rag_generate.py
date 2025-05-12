import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#Configuration Block
CHROMA_DIR = 'vectorstore'
COLLECTION_NAME = 'cyberbot-knowledgebase'
#EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_Name = "WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Load WRN and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_Name)
model = AutoModelForCausalLM.from_pretrained(LLM_Name).to(DEVICE)

#Load Embedding and Vector Store
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)

#RAG Function
def generate_rag_response(user_query: str, k: int=3, max_new_tokens: int = 256):
    RELEVANCE_THRESHOLD = 0.3

    # Embed query and retrieve
    query_embedding = embedding_model.embed_query(user_query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )

    #Analyze results
    retrieved_docs = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]
    distances = results['distances'][0]

    """ 
    # Filter by distance threshold
    max_dist = 0.5
    filtered = [
        (doc, meta, dist)
        for doc, meta, dist in zip(retrieved_docs, retrieved_metadata, distances)
        if dist <= max_dist
    ]

    if not filtered:
        print(f"[DEBUG] No documents within distance threshold ({max_dist})")
        use_context = False
        retrieved_docs = []
        retrieved_metadata = []
        distances = []
    else:
        retrieved_docs, retrieved_metadata, distances = zip(*filtered)
    """

    #Debug print
    print("\n[DEBUG] Retrieved Documents and Distances:")
    for idx, (doc, dist) in enumerate(zip(retrieved_docs, distances)):
        print(f"{idx + 1}. [DIST: {dist:.4f}] {doc[:100]}...")  

    #Check if the closest match is relevant
    use_context = distances and min(distances) < RELEVANCE_THRESHOLD

    #Combine document text with source
    sources = set()
    for meta in retrieved_metadata:
        source = meta.get("source") or meta.get("file_path") or "unknown"
        sources.add(source)

    if use_context:
        context_blocks = []
        for doc, meta in zip(retrieved_docs, retrieved_metadata):
            source = meta.get("source") or meta.get("file_path") or "unknown"
            sources.add(source)
            context_blocks.append(f"[Source: {source}]\n{doc}")

        # Format prompt exactly like CAG
        prompt = "Context Documents:\n"
        for idx, doc in enumerate(retrieved_docs):
            prompt += f"{idx + 1}. {doc}\n"

        prompt += (
            f"\nUser Question:\n{user_query}\n\n"
            f"Answer the question using only the context above. Respond with a single sentence that directly answers the question.\n\n"
            f"Answer:\n"
        )

        print("\n[DEBUG] Assembled Prompt:\n")
        print(prompt)
        print("\n[END DEBUG]\n")


    else:
        context = None
        prompt = f"Answer the following question truthfully and concisely. If unsure, say so.\n\nQuestion: {user_query}\nAnswer:"

    #Tokenize and Generate
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text[len(prompt):].strip(), sources #"strip" removes the prompt from the response

    # Trim hallucinated junk after answer
    # Remove hallucinated continuations
    for stop_token in ["##", "Question:", "\n\n"]:
        if stop_token in response:
            response = response.split(stop_token)[0].strip()
            break


    return response, sources

#Main Test
if __name__ == "__main__":
    user_question = input("Enter your question: ")
    response, sources = generate_rag_response(user_question)
    print("\nGenerated Answer:\n", response)
    print("\nSources used:")
    for source in sources:
        print(f" - {source}")