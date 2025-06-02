from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel

base_model_name = "google/flan-t5-small"
lora_weights_path = "./output/finetuned-lora"
save_combined_path = "./output/combined-flan-t5-small-lora"

print("Loading base model...")
base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
print("Loading LoRA adapter weights...")
model = PeftModel.from_pretrained(base_model, lora_weights_path)

print("Merging LoRA adapter weights into base model...")
model = model.merge_and_unload()

print(f"Saving combined model to {save_combined_path} ...")
model.save_pretrained(save_combined_path)

print("Loading tokenizer and saving it as well...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.save_pretrained(save_combined_path)

print("Done!")