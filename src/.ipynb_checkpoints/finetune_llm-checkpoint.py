import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType

def load_data():
    import json
    with open("data/faq_dataset.json") as f:
        data = json.load(f)
    return Dataset.from_dict({
        "prompt": [d["question"] for d in data],
        "response": [d["answer"] for d in data]
    })

def preprocess(batch, tokenizer):
    inputs = [f"Question: {q}" for q in batch["prompt"]]
    targets = batch["response"]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)

    # Replace pad token id's in labels with -100 so loss ignores padding
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq]
        for labels_seq in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    model_name = "google/flan-t5-small"
    print(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Prepare PEFT LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    print("Loading dataset...")
    dataset = load_data()
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(lambda x: preprocess(x, tokenizer), batched=True, remove_columns=["prompt", "response"])

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=3e-4,
        save_total_limit=2,
        load_best_model_at_end=False,
        # Do NOT use fp16 on Mac M1/M2 (no CUDA GPU)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Saving fine-tuned model and tokenizer...")
    model.save_pretrained("./output/finetuned-lora")
    tokenizer.save_pretrained("./output/finetuned-lora")
    print("Fine-tuning complete.")

if __name__ == "__main__":
    main()
