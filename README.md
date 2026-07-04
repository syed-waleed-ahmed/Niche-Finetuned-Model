# Niche Fine-Tuned Model
### Production-oriented FastAPI assistant built with TinyLlama and LoRA

This repository demonstrates an end-to-end workflow for fine-tuning a small open-source LLM on a focused domain and exposing it as a standalone service.

The target domain is **FastAPI**. The resulting model answers framework-specific questions with code examples, health checks, and a reusable API surface suitable for production deployment.

The project includes:
- A custom JSONL dataset of FastAPI questions and expert answers
- LoRA fine-tuning using `transformers` and `peft`
- A reproducible training pipeline that works locally or in Colab
- A cached inference layer so the model is loaded once and reused
- A FastAPI service with health and readiness endpoints
- An interactive CLI for local experimentation
- A modular project structure that is easy to deploy and extend

---

## 🚀 Features

✔ Fine-tunes **TinyLlama-1.1B-Chat** using **LoRA**
✔ Custom **FastAPI Q&A dataset**
✔ Works on **Google Colab GPU**
✔ Modular Python design (`src/config`, `dataset`, `train_lora`, `inference`, `api`)
✔ Simple CLI for local chat
✔ FastAPI app with `/health`, `/ready`, and `/generate` endpoints
✔ Cached model loading for repeated inference calls

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
│ ├── config.py # Model paths and hyperparameters
│ ├── dataset.py # Loads and tokenizes the JSONL dataset
│ ├── train_lora.py # LoRA training script
│ ├── inference.py # Loads and generates answers
│ ├── api.py # FastAPI service layer
│ ├── cli.py # Interactive terminal chat
│ └── __init__.py
│
├── main.py # CLI and service entrypoint
├── requirements.txt
└── README.md
```

## Architecture

The application is intentionally split into a small set of responsibilities:

1. `src/dataset.py` prepares training examples and applies the shared prompt format.
2. `src/train_lora.py` fine-tunes the base model with LoRA and saves the adapter artifacts.
3. `src/inference.py` loads the saved model once and reuses it for repeated requests.
4. `src/api.py` exposes the inference service over HTTP with explicit request validation.
5. `main.py` can run either the CLI or the API server.

This makes the repository straightforward to explain in an HLD review: model training, model serving, and client interaction are separated and independently replaceable.

## Getting Started

### 1. Install dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

### 2. Train the adapter

```bash
python -m src.train_lora
```

### 3. Start the service

```bash
python main.py --serve
```

The interactive CLI is available with `python main.py`.

## Training Workflow

The training command loads TinyLlama, formats the FastAPI dataset, applies LoRA adapters, runs a short supervised fine-tuning pass, and saves the resulting adapter artifacts to `outputs/fastapi_tinyllama_lora/`.

## Runtime Usage

- CLI mode: `python main.py`
- API mode: `python main.py --serve`
- API docs: `http://localhost:8000/docs`

Example interaction:

```text
You: How do I define a POST endpoint in FastAPI?
Assistant: Use @app.post and a Pydantic model...
```

## API Endpoints

- `GET /health` - basic liveness check
- `GET /ready` - verifies the model can be loaded
- `POST /generate` - returns a model answer for a question

Example request:

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I define a POST endpoint in FastAPI?","max_new_tokens":200}'
```

## Production Notes

- Model loading is cached so repeat requests do not reload weights.
- Training masks padding tokens so loss is computed only on real tokens.
- Health endpoints make it easier to wire the app into Docker, Kubernetes, or a load balancer.
- The service uses explicit request validation and bounded generation parameters.
- The repository is structured so the API layer, training job, and CLI can evolve independently.

## Repository Conventions

- Keep shared prompt logic in `src/dataset.py` so training and inference stay aligned.
- Keep service settings in `src/config.py` so deployment changes stay centralized.
- Treat generated model artifacts under `outputs/` as build outputs rather than source code.

## Future Enhancements

- Expand dataset with hundreds more Q&A samples
- Add RAG support for external FastAPI docs
- Package final model for HF Hub
- Create a Streamlit UI for the niche assistant
- Add quantized inference (GGUF / GPTQ) for faster local use
- Add rate limiting and auth for exposed deployments
- Add observability hooks for request latency and token counts

## Support

This repository is intended as a self-contained portfolio project and reference implementation for a small model-serving workflow.

## Author

Created by Syed Waleed Ahmed
