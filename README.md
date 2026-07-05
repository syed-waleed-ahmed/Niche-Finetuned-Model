# FastAPI Niche Assistant

A **production-oriented, LoRA fine-tuned TinyLlama** model that answers FastAPI
questions, served as a standalone HTTP microservice with health probes, structured
logging, typed configuration, tests, and containerization.

![CI](https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> This repository is a portfolio-grade reference implementation of a small
> model-serving workflow: fine-tune a focused open-source model, then expose it as
> a clean, testable, deployable service. See **[ARCHITECTURE.md](ARCHITECTURE.md)**
> for the full high-level design.

## Why this project

Most "fine-tune a model" repos stop at a training notebook. This one is built like
a service you would actually run:

- **Correct model serving.** The on-disk artifact is a LoRA *adapter*, not a full
  model — so the base model is loaded and the adapter is applied via `peft`
  (a common bug that breaks naive implementations). If no adapter exists, the base
  model is served with a warning (graceful degradation).
- **Native chat template.** Training and inference share one prompt format built
  from the tokenizer's own chat template, preserving the base model's alignment.
- **Completion-only training.** Loss is masked to the answer tokens, the correct
  objective for instruction tuning.
- **Operable.** Typed config, `/health` + `/ready` probes, request IDs, structured
  JSON logs, optional API-key auth, bounded generation parameters.
- **Tested & CI'd.** The full HTTP surface is tested with a fake engine (no model
  download), and GitHub Actions runs lint + tests on 3.10 and 3.11.

## Architecture at a glance

```
Client ──HTTP──▶ FastAPI (api.py)
                   │  middleware: request-id + timing · CORS · API-key auth
                   │  routes: /health · /ready · /generate
                   ▼
              AssistantEngine (inference.py)
                   │  load once (base model + LoRA adapter) · generate (locked)
                   ▼
              torch + transformers + peft

Offline:  data/*.jsonl ─▶ tokenize+mask (data.py) ─▶ LoRA train (training.py) ─▶ outputs/ adapter
```

Full component responsibilities, request lifecycle, concurrency model, scaling
strategy, and the decision log are in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## Project structure

```text
niche_finetuned_model/
├── src/fastapi_assistant/
│   ├── config.py          # Typed settings (pydantic-settings)
│   ├── logging_config.py  # Human / JSON structured logging
│   ├── prompts.py         # Shared chat format (training == inference)
│   ├── data.py            # JSONL loading + completion-only tokenization
│   ├── training.py        # LoRA fine-tuning entrypoint
│   ├── inference.py        # AssistantEngine: load once, generate safely
│   ├── schemas.py         # Pydantic request/response contracts
│   ├── api.py             # FastAPI app factory, routes, middleware
│   ├── cli.py             # Interactive terminal chat
│   └── __main__.py        # `python -m fastapi_assistant`
├── data/                  # FastAPI Q&A dataset (JSONL)
├── tests/                 # Pytest suite (no model/network needed)
├── main.py                # Convenience entrypoint (`python main.py`)
├── Dockerfile             # Container image
├── pyproject.toml         # Packaging, deps, ruff + pytest config
├── ARCHITECTURE.md        # High-level design
└── requirements.txt
```

## Quickstart

Requires Python 3.10 or 3.11.

```bash
# 1. Clone and create an isolated environment
git clone https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model.git
cd niche_finetuned_model
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate

# 2. Install (runtime + dev tools)
pip install -e ".[dev]"

# 3. Configure (optional)
cp .env.example .env      # then edit as needed

# 4. Fine-tune the LoRA adapter (downloads the base model on first run)
python -m fastapi_assistant --train      # or: make train

# 5. Serve the API
python -m fastapi_assistant --serve      # or: make serve
#    Interactive CLI instead:
python -m fastapi_assistant              # or: make cli
```

`python main.py [--serve|--train]` works too, without installing the package.

Open http://localhost:8000/docs for interactive Swagger UI.

> **Note:** `outputs/` (the trained adapter) is a build artifact and is
> git-ignored. Run `make train` to (re)generate it. The base model
> (`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, ~2.2 GB) is downloaded from Hugging Face
> on first training/serving.

## Run with Docker

```bash
docker build -t fastapi-assistant:latest .
docker run --rm -p 8000:8000 \
  -v "$PWD/outputs:/app/outputs" \      # mount a trained adapter
  -v hf-cache:/models/hf \              # cache base weights across runs
  fastapi-assistant:latest
```

The container runs as a non-root user, emits JSON logs, exposes a `HEALTHCHECK`,
and loads the model lazily so it becomes healthy quickly.

## API reference

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Liveness: process is up (does not load the model). |
| `GET` | `/ready` | Readiness: ensures the model can be loaded → 503 if not. |
| `POST` | `/generate` | Generate an answer to a FastAPI question. |
| `GET` | `/docs` | Swagger UI. `GET /openapi.json` for the schema. |

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"question":"How do I define a POST endpoint in FastAPI?","max_new_tokens":200}'
```

```json
{
  "question": "How do I define a POST endpoint in FastAPI?",
  "answer": "Use @app.post with a Pydantic model ...",
  "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "uses_adapter": true
}
```

If `API_KEY` is set, include `-H "X-API-Key: <key>"` on `/generate`.

## Configuration

All settings are environment variables (see `.env.example`), validated at startup.

| Variable | Default | Description |
| --- | --- | --- |
| `BASE_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Hugging Face base model. |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | HTTP bind address. |
| `API_KEY` | *(unset)* | If set, `/generate` requires `X-API-Key`. |
| `CORS_ALLOW_ORIGINS` | `["*"]` | JSON list of allowed origins. |
| `WARMUP_ON_STARTUP` | `true` | Load model at boot vs. on first request. |
| `LOG_LEVEL` / `LOG_JSON` | `INFO` / `false` | Logging verbosity / JSON output. |
| `MODEL_MAX_NEW_TOKENS` | `256` | Default generation length cap. |
| `MODEL_TEMPERATURE` | `0.4` | Sampling temperature (`0` = greedy). |
| `MODEL_TOP_P` | `0.9` | Nucleus sampling. |
| `MODEL_REPETITION_PENALTY` | `1.05` | Repetition penalty. |
| `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `MAX_SEQ_LENGTH`, `LORA_*` | see `.env.example` | Training hyperparameters. |
| `DATA_DIR` / `OUTPUT_DIR` | `./data` / `./outputs/...` | Overridable paths. |

## Development

```bash
make test    # pytest
make lint    # ruff check
make fmt     # ruff format
```

The test suite injects a fake engine via the `create_app` factory, so it runs in
under a second with no model download or network access. CI runs the same on
Python 3.10 and 3.11.

## Training details

`python -m fastapi_assistant --train` loads TinyLlama, attaches LoRA adapters
(`q_proj`, `v_proj`), tokenizes the FastAPI Q&A dataset with **completion-only
masking** and **dynamic padding**, fine-tunes with the HF `Trainer`, and saves the
adapter to `OUTPUT_DIR`. Hyperparameters are configurable via the environment.

## Roadmap

- Expand the dataset with hundreds more curated Q&A pairs.
- Optional RAG over the official FastAPI docs.
- Push the adapter to the Hugging Face Hub as a versioned release.
- Quantized inference (GGUF / GPTQ) for faster local use.
- Swap `AssistantEngine` for a vLLM/TGI backend behind the same HTTP contract.
- Per-request rate limiting and token/latency metrics (Prometheus).

## License

[MIT](LICENSE) © Syed Waleed Ahmed
