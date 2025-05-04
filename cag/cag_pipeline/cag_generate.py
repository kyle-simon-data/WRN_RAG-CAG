from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#Configuration Block
LLM_NAME = "WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Load WhiteRabbitNeo model and tockenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)

#CAG Response Function
def generate_cag_response(assembled_prompt: str, max_new_tokens: int = 256):
    #Tokenize & Generate
    inputs = tokenizer(assembled_prompt, return_tensors="pt", padding=True).to(DEVICE)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    #Remove the prompt from the output to return only the new generated text
    return output_text[len(assembled_prompt):].strip()

#Main Test
if __name__ == "__main__":
    print("Cache-Augmented Generation (CAG) System")
    user_prompt = input("Paste the full assembled prompt: ")
    response = generate_cag_response(user_prompt)
    print("\nGenerated Answer:\n", response)