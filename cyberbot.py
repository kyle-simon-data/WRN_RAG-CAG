from transformers import pipeline

# Load WhiteRabbitNeo model
chat_model = pipeline("text-generation", model="WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a")

def generate_response(prompt):
    response = chat_model(prompt, max_length=300, temperature=0.5)
    return response[0]['generated_text']

# Test it out
if __name__ == "__main__":
    user_input = "Explain the CVE-2024-1234 vulnerability."
    print(generate_response(user_input))
