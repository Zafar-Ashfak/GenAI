from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("1. Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_id)

print("2. Loading model...")

model = AutoModelForCausalLM.from_pretrained(model_id)

print("3. Model loaded successfully!")

prompt = "What is Computer Vision?"

inputs = tokenizer(prompt, return_tensors="pt")

print("4. Generating response...")

outputs = model.generate(
    **inputs,
    max_new_tokens=30
)

print("5. Response:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))