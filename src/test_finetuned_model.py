# src/test_finetuned_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PeftModel, PeftConfig

# Load base model and LoRA adapter
base_model_name = "google/flan-t5-small"
lora_model_path = "./output/finetuned-lora"

# Load tokenizer and model with LoRA adapter
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

# Example prompts
questions = [
    "What are your operating hours?",
    "How do I reset my password?",
    "Do you offer international shipping?"
]

# Inference loop
for q in questions:
    input_ids = tokenizer(q, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=64)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Q: {q}\nA: {answer}\n{'-'*40}")