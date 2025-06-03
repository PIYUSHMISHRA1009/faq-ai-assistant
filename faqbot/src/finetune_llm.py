import torch
import wandb
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)
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
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # PEFT LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)

    # Load and preprocess dataset
    dataset = load_data()
    tokenized_dataset = dataset.map(lambda x: preprocess(x, tokenizer), batched=True)

    # Split for evaluation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=4,
        num_train_epochs=5,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-4,
        save_total_limit=2,
        load_best_model_at_end=False,
        report_to="wandb",                  # Enable wandb logging
        run_name="faqbot-finetune-v1"       # Customize run name
    )

    # Initialize wandb
    wandb.init(project="faqbot-llm", name="finetune-flan-t5-small")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Evaluate and print perplexity
    eval_results = trainer.evaluate()
    print(f"Eval Loss: {eval_results['eval_loss']:.4f}")
    print(f"Perplexity: {torch.exp(torch.tensor(eval_results['eval_loss'])):.2f}")

    # Save model & tokenizer
    model.save_pretrained("./output/finetuned-lora")
    tokenizer.save_pretrained("./output/finetuned-lora")

if __name__ == "__main__":
    main()
