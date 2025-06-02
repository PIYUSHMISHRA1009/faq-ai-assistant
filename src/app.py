from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

app = FastAPI()

# Allow all CORS for dev; restrict in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from ./frontend (adjust path as needed)
frontend_dir = os.path.join(os.path.dirname(__file__), "../frontend")
static_dir = os.path.join(frontend_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Serve root HTML
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

# Load the fine-tuned model
MODEL_DIR = os.getenv("MODEL_DIR", "./output/shrunk-model")  # or use your original full model

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)
model.eval()

# Pydantic input model
class Question(BaseModel):
    question: str

# POST endpoint for generation
@app.post("/generate")
async def generate_answer(payload: Question):
    inputs = tokenizer(payload.question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": answer}
