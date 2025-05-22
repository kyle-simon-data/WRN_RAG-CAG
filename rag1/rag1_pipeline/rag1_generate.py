from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Configuration Block
LLM_NAME = "WhiteRabbitNeo/WhiteRabbitNeo-7B-v1.5a"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load WhiteRabbitNeo model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_NAME).to(DEVICE)

# RAG1 Response Function
def generate_rag1_response(assembled_prompt: str, max_new_tokens: int = 256):
    # Tokenize & Generate
    inputs = tokenizer(assembled_prompt, return_tensors="pt", padding=True).to(DEVICE)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Remove the prompt from the output to return only the new generated text
    generated_text = output_text[len(assembled_prompt):].strip()
    
    # Clean the output by removing the special tags
    clean_text = clean_model_output(generated_text)
    
    return clean_text

def clean_model_output(text: str) -> str:
    """
    Clean the model's output by removing special tags and extracting only the actual response.
    
    Args:
        text (str): Raw text from the model
        
    Returns:
        str: Cleaned response text
    """
    # Define the patterns to match and remove
    patterns = [
        r'### Actual Response <\|response\|> (.+?)(### |$)',
        r'### Contextual Response <\|response\|> (.+?)(### |$)',
        r'<\|response\|> (.+?)(<\||$)',
        r'### Actual Completion <\|completion\|> (.+?)(### |$)',
        r'### Contextual Completion <\|completion\|> (.+?)(### |$)'
    ]
    
    # Try each pattern to extract meaningful content
    for pattern in patterns:
        matches = re.search(pattern, text, re.DOTALL)
        if matches:
            return matches.group(1).strip()
    
    # If no pattern matches, remove all tag blocks entirely
    cleaned = re.sub(r'### [^#]+?<\|[^#]+?\|>[^#]+?(?=### |$)', '', text)
    
    # If still contains tags, do a simpler cleanup
    if '<|' in cleaned or '###' in cleaned:
        # Remove all tags of the form <|tag|>
        cleaned = re.sub(r'<\|[^|]+\|>', '', cleaned)
        # Remove all ### headers
        cleaned = re.sub(r'### [^#\n]+', '', cleaned)
    
    return cleaned.strip()

# Main Test
if __name__ == "__main__":
    print("Retrieval-Augmented Generation (RAG1) System")
    user_prompt = input("Paste the full assembled prompt: ")
    response = generate_rag1_response(user_prompt)
    print("\nGenerated Answer:\n", response)
