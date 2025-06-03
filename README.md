
````markdown

FAQBot is a custom-built chatbot fine-tuned on a domain-specific FAQ dataset using **LoRA (Low-Rank Adaptation)** tuning applied to the Flan-T5-small transformer model. This approach significantly reduces the number of trainable parameters while maintaining high accuracy, making the model lightweight and efficient for real-time deployment.

The project includes the entire pipeline — from data preparation, model fine-tuning, merging and shrinking the model, to serving it via a FastAPI backend and an interactive React-based frontend UI. The solution is containerized for cloud deployment (Render.com), keeping resource constraints in mind.

---

## Why This Project?

In many real-world applications, fine-tuning large language models (LLMs) fully is resource-intensive and time-consuming. LoRA offers a parameter-efficient fine-tuning method that drastically reduces computation while retaining task-specific performance.

This project demonstrates:

- How to fine-tune transformer models efficiently using LoRA.
- Techniques to merge LoRA adapters and shrink models for deployment.
- Serving ML models with asynchronous APIs.
- Building interactive frontends communicating with backend ML services.
- Practical deployment of ML projects under cloud resource constraints.

---

## What You Will Find Here

- Dataset used for fine-tuning (FAQ dataset).
- Scripts and notebooks for data preprocessing.
- Training scripts implementing LoRA fine-tuning.
- Post-training utilities to merge and compress models.
- FastAPI backend serving the model for inference.
- Responsive frontend UI for chatbot interaction.
- Docker and Render deployment configurations.
- Detailed documentation and examples.

---

## Phase-wise Detailed Journey & Learnings

### Phase 1: Data Preparation & Exploration

- **Objective:** Understand and preprocess the FAQ dataset.
- **Activities:**
  - Loaded and explored JSON structured FAQ pairs.
  - Cleaned and formatted data suitable for model input.
  - Split data into training and validation sets.
- **Learnings:**
  - Data pipeline design for NLP.
  - Importance of clean, well-structured datasets for fine-tuning.
  - Using Jupyter notebooks for rapid prototyping.

---

### Phase 2: Model Fine-Tuning with LoRA

- **Objective:** Fine-tune the Flan-T5-small model on FAQ data efficiently.
- **Activities:**
  - Set up Hugging Face Transformers and PEFT (Parameter-Efficient Fine-Tuning).
  - Integrated LoRA adapters to limit trainable parameters.
  - Configured training scripts (`src/finetune_llm.py`) with hyperparameter tuning.
  - Tracked experiments with Weights & Biases (W&B).
- **Learnings:**
  - Concepts and benefits of LoRA in NLP.
  - Transformer architecture and tokenization specifics.
  - Hands-on training loop customization and logging.

---

### Phase 3: Model Merging & Shrinking for Deployment

- **Objective:** Prepare the fine-tuned model for efficient deployment.
- **Activities:**
  - Merged LoRA adapter weights into the base model using `merge_lora.py`.
  - Applied model shrinking (`shrink_model.py`) to reduce model size and memory footprint.
  - Saved and organized checkpoints and final model artifacts.
- **Learnings:**
  - Model checkpoint management.
  - Techniques to optimize model size without losing accuracy.
  - Handling large model files and serialization formats.

---

### Phase 4: Backend API Development

- **Objective:** Build a scalable and efficient backend API to serve the model.
- **Activities:**
  - Developed FastAPI server (`src/app.py`) with async endpoints.
  - Loaded the merged, shrunk model and tokenizer.
  - Implemented request/response handling with concurrency support.
  - Added logging and error handling.
- **Learnings:**
  - Building asynchronous APIs in Python.
  - Efficient serving of transformer models.
  - Structuring ML services for real-time interaction.

---

### Phase 5: Frontend Development

- **Objective:** Create a user-friendly interface for chatbot interaction.
- **Activities:**
  - Built a responsive chat UI in plain HTML, CSS, and JavaScript.
  - Integrated frontend with backend API using fetch/XHR calls.
  - Managed chat history and UI state.
  - Tested for usability and responsiveness.
- **Learnings:**
  - Frontend-backend integration for ML apps.
  - UX considerations in chatbot design.
  - Basic JS DOM manipulation and asynchronous networking.

---

### Phase 6: Containerization & Deployment

- **Objective:** Deploy the FAQBot application on Render with resource limits.
- **Activities:**
  - Created Dockerfile to containerize backend and frontend.
  - Wrote `render.yaml` config for deployment automation.
  - Set up Git LFS to manage large model files.
  - Deployed to Render.com’s free tier considering storage and memory limits.
- **Learnings:**
  - Containerization with Docker for ML apps.
  - Managing large files in Git using LFS.
  - Deploying ML services on cloud platforms under constraints.
  - Troubleshooting network and resource issues in deployment.

---

## Installation & Usage

### Clone & Setup

```bash
git clone https://github.com/yourusername/faqbot.git
cd faqbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

### Run Locally

```bash
uvicorn src.app:app --reload
```

Open `frontend/static/index.html` in a browser to chat with the bot.

### Deploy to Render

* Push repo with Git LFS enabled.
* Connect GitHub repo to Render.com.
* Use `render.yaml` for service configuration.
* Deploy and access live app via Render URL.

---

## Project Structure

```
├── data/                       # FAQ dataset
├── output/                     # Model checkpoints and artifacts
├── src/                        # Source code: training, merging, serving
├── frontend/static/            # Frontend chat UI
├── Dockerfile                  # Container image definition
├── render.yaml                 # Render deployment config
├── requirements.txt            # Python dependencies
├── shrink_model.py             # Model size optimization utility
├── README.md                   # This file
└── notebooks/                  # Jupyter notebooks for data prep
```

---

## Technologies Used

* **Python 3.9+**
* **FastAPI** for backend API
* **Hugging Face Transformers & PEFT** for LoRA fine-tuning
* **Weights & Biases** for experiment tracking
* **Git LFS** for large file management
* **Docker** for containerization
* **Render.com** for deployment
* **Vanilla JavaScript & HTML/CSS** for frontend UI

---

## Challenges & How I Overcame Them

* Managing large model files on GitHub using Git LFS.
* Reducing model size to fit cloud deployment limits.
* Handling asynchronous request processing in FastAPI.
* Debugging deployment errors with limited cloud resources.
* Ensuring smooth frontend-backend communication.

---

## Future Improvements

* Add user authentication and session management.
* Improve chatbot conversational context handling.
* Integrate caching for faster responses.
* Implement multi-turn dialogue capability.
* Upgrade frontend to React for better UX and maintainability.

---

## Contact & Contributions

I’m open to feedback, questions, and collaboration!
Reach out at: mishra.piyush827@gmail.com

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

Thank you for checking out FAQBot!
Looking forward to your feedback and contributions.
# faq-ai-assistant
