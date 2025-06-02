import json
from datetime import datetime
import os

LOG_PATH = "./logs/chat_log.jsonl"
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_qa(question: str, answer: str):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer": answer
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
