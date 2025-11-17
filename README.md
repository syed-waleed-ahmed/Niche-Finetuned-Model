# 🧠 Niche Fine-Tuned Open Source Model  
### Fine-tuning TinyLlama with LoRA to build a FastAPI expert assistant

This project demonstrates how to fine-tune a small open-source LLM on a **very specific niche domain**.  
In this case, the niche is **FastAPI**, and we train a TinyLlama model to behave like a FastAPI expert that answers framework-specific questions with accurate explanations and code examples.

The project includes:
- A custom dataset (JSONL) of FastAPI questions & expert answers  
- LoRA fine-tuning using `transformers` + `peft`  
- A complete training pipeline (local + Colab compatible)  
- An inference script and interactive CLI  
- Clean modular project structure for easy reuse

---

## 🚀 Features

✔ Fine-tunes **TinyLlama-1.1B-Chat** using **LoRA**  
✔ Custom **FastAPI Q&A dataset**  
✔ Works on **Google Colab GPU**  
✔ Modular Python design (`src/config`, `dataset`, `train_lora`, `inference`)  
✔ Simple CLI to chat with the fine-tuned model  
✔ Loads & runs the model locally after training  

---

## 📂 Project Structure

```text
niche_finetuned_model/
│
├── data/
│ ├── fastapi_qa_train.jsonl
│ └── fastapi_qa_eval.jsonl
│
├── src/
│ ├── config.py # Model paths & hyperparameters
│ ├── dataset.py # Loads + tokenizes JSONL dataset
│ ├── train_lora.py # LoRA training script
│ ├── inference.py # Load + generate answers
│ └── init.py
│
├── main.py # CLI interface
├── requirements.txt
└── README.md
```

## 🧪 Training the Model (LoRA Fine-Tuning)

- You can train locally or in Colab.
- Run training: python -m src.train_lora

This will:

- Load TinyLlama
- Load the FastAPI dataset
- Apply LoRA adapters
- Train for a few epochs
- Save the fine-tuned model to: outputs/fastapi_tinyllama_lora/

## 💬 Using the Fine-Tuned Model

- Run the interactive CLI: python main.py
- Example usage:

You: How do I define a POST endpoint in FastAPI?
Assistant: Use @app.post and a Pydantic model...

## 🔮 Future Enhancements

- Expand dataset with hundreds more Q&A samples
- Add RAG support for external FastAPI docs
- Package final model for HF Hub
- Create a Streamlit UI for the niche assistant
- Add quantized inference (GGUF / GPTQ) for faster local use

## Author

SYED WALEED AHMED