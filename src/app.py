from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

app = FastAPI()

# Enable CORS (optional since frontend and backend served together)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend files under /static
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Serve index.html at root "/"
@app.get("/")
async def read_index():
    return FileResponse("frontend/static/index.html")

# Load LoRA-adapted model (adjust paths if needed)
base_model = "google/flan-t5-small"
lora_path = "./output/finetuned-lora"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, lora_path)
model.eval()

class Question(BaseModel):
    question: str

@app.post("/generate")
async def generate_answer(q: Question):
    inputs = tokenizer(q.question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer}
