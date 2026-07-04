# Niche Fine-Tuned Model
### Production-oriented FastAPI assistant built with TinyLlama and LoRA

This repository demonstrates an end-to-end workflow for fine-tuning a focused open-source language model and exposing it as a standalone service.

The domain is FastAPI. The resulting model answers framework-specific questions with code examples, health checks, and a reusable API surface suitable for deployment.

## Overview

The project includes:
- A custom JSONL dataset of FastAPI questions and expert answers
- LoRA fine-tuning using `transformers` and `peft`
- A reproducible training pipeline that works locally or in Colab
- A cached inference layer so the model is loaded once and reused
- A FastAPI service with health and readiness endpoints
- An interactive CLI for local experimentation
- A modular project structure that is easy to deploy and extend

## Project Structure

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

The application is split into a small set of responsibilities:

1. `src/dataset.py` prepares training examples and applies the shared prompt format.
2. `src/train_lora.py` fine-tunes the base model with LoRA and saves the adapter artifacts.
3. `src/inference.py` loads the saved model once and reuses it for repeated requests.
4. `src/api.py` exposes the inference service over HTTP with explicit request validation.
5. `main.py` runs either the CLI or the API server.

This separation keeps the repository easy to explain in an HLD review and makes the training, serving, and client layers independently replaceable.

## Getting Started

### Install dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

### Configure the environment

Copy [.env.example](.env.example) to `.env` and adjust values for your machine or deployment target. The application loads environment variables automatically at startup.

Common settings:

- `API_HOST` and `API_PORT` control the HTTP service binding.
- `MODEL_MAX_NEW_TOKENS`, `MODEL_TEMPERATURE`, `MODEL_TOP_P`, and `MODEL_REPETITION_PENALTY` control generation behavior.
- Training values such as `BATCH_SIZE`, `NUM_EPOCHS`, and `LEARNING_RATE` are documented in `.env.example` for reproducibility.

### Train the adapter

```bash
python -m src.train_lora
```

### Start the service

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
- Configuration is loaded from the environment so production secrets and deployment-specific settings stay out of source control.

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

## Security Notes

- Environment-specific values should remain in `.env` or deployment secrets, not in source control.
- The service validates request sizes and generation bounds at the API layer.
- The model is loaded once and reused to reduce startup churn and repeated disk I/O.

## Author

Created by Syed Waleed Ahmed
