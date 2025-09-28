
from ctransformers import AutoModelForCausalLM

# --- Configuration ---
# 1. Replace 'path/to/your/mistral-7b-instruct-v0.1.Q4_K_M.gguf' with the actual path to your downloaded GGUF file.
# 2. 'mistral' is the model type.
# 3. 'Q4_K_M' is a common quantization type. Adjust if your file name is different.
#model_path = "path/to/your/mistral-7b-instruct-v0.1.Q4_K_M.gguf" 
model_path = r"C:\Users\Sandeep\.lmstudio\models\Mungert\Mistral-7B-Instruct-v0.2-GGUF\Mistral-7B-Instruct-v0.2-q4_k_s.gguf"

model_type = "mistral" 
# Set gpu_layers=0 to run purely on the CPU. Increase for GPU acceleration if available.
gpu_layers = 0 
# ---------------------

print("Loading model...")
llm = AutoModelForCausalLM.from_pretrained(
    model_path,
    model_type=model_type,
    gpu_layers=gpu_layers
)

# Your prompt
prompt = "Write a short, simple poem about a rainy day."

print(f"\n--- Prompt ---\n{prompt}\n")

# Run inference
print("--- Response ---\n")
# Using stream=True allows the model's output to print token-by-token,
# which is good for user experience, but you can set it to False if preferred.
for token in llm(prompt, stream=True):
    print(token, end="", flush=True)

print("\n----------------\n")