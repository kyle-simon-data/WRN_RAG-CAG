from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline


# Load WhiteRabbitNeo model
chat_model = pipeline("text-generation", model="WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a")

def generate_response(prompt):
    response = chat_model(prompt, max_length=300, temperature=0.5)
    return response[0]['generated_text']

# Test it out
if __name__ == "__main__":
    user_input = "What is what is CVE-2025=1234?"
    print(generate_response(user_input))