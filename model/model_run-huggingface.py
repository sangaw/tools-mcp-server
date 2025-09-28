import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- Configuration ---
# 1. Replace 'path/to/local_model_directory' with the actual path to the folder 
#    containing the model files (e.g., 'config.json', 'tokenizer.json', etc.).
# local_model_directory = "path/to/local_model_directory" 
# local_model_directory = r"C:\Users\Sandeep\.lmstudio\models\Mungert\Mistral-7B-Instruct-v0.2-GGUF\Mistral-7B-Instruct-v0.2-q4_k_s.gguf"
local_model_directory = r"C:\Users\Sandeep\.lmstudio\models\Mungert\Mistral-7B-Instruct-v0.2-GGUF"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---------------------

print(f"Using device: {device}")
print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(local_model_directory)
model = AutoModelForCausalLM.from_pretrained(
    local_model_directory,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" # Automatically handles device placement
)
model.to(device)

# Set up the text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1 # 0 for first GPU, -1 for CPU
)

# Your prompt
prompt = "Explain why the sky is blue."

# For instruction-tuned models, wrap your prompt in the instruction format:
full_prompt = f"<s>[INST] {prompt} [/INST]"

print(f"\n--- Prompt ---\n{prompt}\n")

# Run inference
result = generator(
    full_prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    return_full_text=False # Only return the generated response
)

# Extract and print the response text
response_text = result[0]['generated_text']
print("--- Response ---\n")
print(response_text)
print("\n----------------\n")