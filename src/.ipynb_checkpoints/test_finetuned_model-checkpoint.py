from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer(path="./output/finetuned-lora"):
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return tokenizer, model

def generate_answer(question, tokenizer, model, max_length=128):
    inputs = tokenizer(f"Question: {question}", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    tokenizer, model = load_model_and_tokenizer()
    # Example questions to test
    test_questions = [
        "What are your operating hours?",
        "How do I reset my password?",
        "Can I get a refund if I'm not satisfied?",
    ]

    for q in test_questions:
        print(f"Q: {q}")
        print(f"A: {generate_answer(q, tokenizer, model)}\n")
