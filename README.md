````markdown
# 🤖 FAQBot | LoRA-Fine-Tuned FLAN-T5 Small

> 🚀 **Live Demo:** [Click here to try it on Hugging Face Spaces](https://huggingface.co/spaces/Piyush0001/faqbot-lora)

---

> ### 📌 Project Overview
> FAQBot is a lightweight chatbot fine-tuned on a domain-specific FAQ dataset using **LoRA (Low-Rank Adaptation)** on `flan-t5-small`.  
> It's designed to be efficient, responsive, and deployable under cloud memory constraints — ideal for real-world inference at scale.

---

> ### ✅ Key Highlights
> - Fine-tuned using Hugging Face 🤗 Transformers + PEFT + LoRA  
> - Model merged and **shrunk** for real-time inference  
> - FastAPI backend with async inference API  
> - Gradio UI frontend for interactive experience  
> - Fully containerized (Docker)  
> - Deployed on Hugging Face Spaces (100% free & fast!)

---

## 📖 What’s Inside?

```text
├── data/                       # FAQ dataset
├── output/                     # Merged, shrunk model
├── src/
│   ├── app.py                  # FastAPI or Gradio backend
│   ├── finetune_llm.py         # LoRA training script
│   ├── merge_lora.py           # Merge adapter weights
│   └── shrink_model.py         # Shrink model for deployment
├── frontend/static/            # HTML/CSS/JS UI (if used)
├── Dockerfile                  # Backend container
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── render.yaml                 # Optional Render deployment config
````

---

## 📅 Phase-wise Project Journey

> ### 🧹 Phase 1: Data Preparation
>
> * Cleaned and structured the FAQ dataset.
> * Converted to Q-A pairs in seq2seq format.
> * Split into train/validation using scripts & notebooks.

> ### 🧠 Phase 2: Fine-Tuning with LoRA
>
> * Used `flan-t5-small` base model.
> * Applied LoRA adapters via 🤗 PEFT.
> * Efficient fine-tuning with reduced trainable parameters.
> * Tracked experiments using Weights & Biases.

> ### 🧬 Phase 3: Model Merging & Shrinking
>
> * Merged LoRA adapters into the base model.
> * Shrunk model size from \~950MB to \~300MB.
> * Saved and tested model for CPU inference.

> ### ⚙️ Phase 4: Backend API (FastAPI)
>
> * Developed async POST `/generate` endpoint.
> * Loaded tokenizer/model from Hugging Face.
> * Handled request/response formats.

> ### 💬 Phase 5: Frontend UI (Gradio or Vanilla JS)
>
> * Created interactive UI to test chatbot.
> * Integrated with FastAPI or Gradio backend.
> * UI deployed via Hugging Face Spaces.

> ### 📦 Phase 6: Deployment
>
> * Created Dockerfile for containerization.
> * Added `render.yaml` for optional Render deploy.
> * Final deployment done via Hugging Face Spaces (Gradio).
> * No GPU or heavy RAM required!

---

## 💡 Why LoRA?

> ✅ LoRA fine-tuning drastically reduces compute requirements
> ✅ Retains model performance on small data
> ✅ Ideal for edge/cloud deployments

---

## 🚀 Run Locally

### 🛠️ Setup

```bash
git clone https://github.com/Piyush0001/faqbot.git
cd faqbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### ▶️ Launch

```bash
uvicorn src.app:app --reload
```

Open `http://127.0.0.1:8000` in your browser
Or run `app.py` with Gradio interface for immediate UI testing.

---

## 📦 Docker Support

```bash
# Build image
docker build -t faqbot .

# Run container
docker run -p 8000:8000 faqbot
```

---

## 🌍 Technologies Used

| Tool/Tech                 | Purpose                 |
| ------------------------- | ----------------------- |
| Python 3.10               | Core language           |
| Hugging Face Transformers | Model & tokenizer       |
| PEFT + LoRA               | Lightweight fine-tuning |
| FastAPI / Gradio          | API + Web interface     |
| Docker                    | Deployment and testing  |
| Git LFS                   | Managing large files    |
| Hugging Face Spaces       | Final deployment        |

---

## 🧪 Challenges Overcome

> 🧠 Reduced model memory from >900MB to <350MB
> 🔧 Debugged `render.com` crashes due to RAM overuse
> ⚡ Migrated from FastAPI to Gradio for lighter cloud deployment
> 🚛 Used Git LFS for handling large model files cleanly

---

## 🌱 Future Improvements

* [ ] Add conversational context memory (multi-turn chat)
* [ ] Improve UI with React or Next.js
* [ ] Integrate Pinecone or FAISS for RAG-based QA
* [ ] Add support for file uploads (PDF → FAQ)

---

## 🤝 Recruiter Notes

> ✅ Hands-on NLP project using LoRA
> ✅ Fully built, tested, and deployed end-to-end
> ✅ Covers fine-tuning, optimization, API dev, and cloud deploy
> ✅ Strong understanding of model serving and size trade-offs
> ✅ Open to collaboration or extensions for new domains!

---

## 📬 Contact Me

**👨‍💻 Piyush Kumar Mishra**
📧 [mishra.piyush827@gmail.com](mailto:mishra.piyush827@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/piyushkmishra)

---

Thank you for visiting the FAQBot repository.
Feel free to ⭐ the repo or [try the chatbot live](https://huggingface.co/spaces/Piyush0001/faqbot-lora)!

````