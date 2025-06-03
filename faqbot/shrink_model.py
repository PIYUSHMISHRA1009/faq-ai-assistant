from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import shutil
import os

SOURCE_DIR = "./output/combined-flan-t5-small-lora"
TARGET_DIR = "./output/shrunk-model"

def shrink_model():
    print(f"Loading model from {SOURCE_DIR}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(SOURCE_DIR)

    print("Converting to float16...")
    model = model.half()

    print(f"Saving to {TARGET_DIR}...")
    model.save_pretrained(TARGET_DIR)

    print("Copying tokenizer files...")
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "spiece.model", "special_tokens_map.json"]
    os.makedirs(TARGET_DIR, exist_ok=True)

    for file in tokenizer_files:
        src_path = os.path.join(SOURCE_DIR, file)
        if os.path.exists(src_path):
            shutil.copy(src_path, TARGET_DIR)
        else:
            print(f"⚠️ Tokenizer file missing: {file}")

    print("✅ Shrinking complete!")

if __name__ == "__main__":
    shrink_model()
