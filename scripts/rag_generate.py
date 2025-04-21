import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#Configuration Block
CHROMA_DIR = 'vectorstore'
COLLECTION_NAME = 'cyberbot-knowledgebase'
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
    # Embed query and retrieve
    query_embedding = embedding_model.embed_query(user_query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['documents', 'metadatas']
    )
    retrieved_docs = results['documents'][0]
    retrieved_metadata = results['metadatas'][0]

    #Combine document text with source
    context_blocks = []
    sources = set()

    for doc, meta in zip(retrieved_docs, retrieved_metadata):
        source = meta.get("source", "unknown")
        sources.add(source)
        context_blocks.append(f"[Source: {source}]\n{doc}")

    #Format Prompt
    context = "\n---\n".join(retrieved_docs)
    prompt = f"Use the following context to answer the question using the provided context. Be specific and do not invent information.\n\nContext:\n{context}\n\nQuestion: {user_query}\nAnswer:"

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

#Main Test
if __name__ == "__main__":
    user_question = input("Enter your question: ")
    response, sources = generate_rag_response(user_question)
    print("\nGenerated Answer:\n", response)
    print("\nSources used:")
    for source in sources:
        print(f" - {source}")