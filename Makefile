# Convenience targets. On Windows without `make`, run the underlying commands
# directly (e.g. `python -m pytest`).
PY ?= python
CLI ?= $(PY) -m fastapi_assistant

.PHONY: help install dev train serve cli eval eval-rag data-validate rag-index lock test lint fmt \
        worker-test docker-build docker-run compose-up clean

help:
	@echo "Targets:"
	@echo "  dev            Install with dev + training extras"
	@echo "  train          Fine-tune the LoRA adapter (GPU recommended; see kaggle/)"
	@echo "  serve          Run the HTTP API"
	@echo "  cli            Interactive chat"
	@echo "  eval           Score the configured model on the eval set (gated vs eval/baseline.json)"
	@echo "  eval-rag       Same, with retrieved documentation in the prompt"
	@echo "  data-validate  Dataset schema/code/duplicate/leakage checks"
	@echo "  rag-index      Build the local documentation index"
	@echo "  lock           Regenerate the hash-pinned serving lock file"
	@echo "  test / lint / fmt / worker-test"
	@echo "  docker-build / docker-run / compose-up"

install:
	$(PY) -m pip install -e ".[serve]"

dev:
	$(PY) -m pip install -e ".[dev,train,postgres]"

train:
	$(CLI) train

serve:
	$(CLI) serve

cli:
	$(CLI) chat

eval:
	$(CLI) eval --report eval/report.json --markdown eval/report.md --baseline eval/baseline.json

eval-rag:
	$(CLI) eval --rag --report eval/report-rag.json --markdown eval/report-rag.md

data-validate:
	$(CLI) data validate

rag-index:
	$(CLI) rag build --target local

lock:
	$(PY) -m uv pip compile pyproject.toml --extra serve --universal --python-version 3.11 \
		--generate-hashes --no-header -o requirements/serve.txt

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

fmt:
	$(PY) -m ruff format src tests

worker-test:
	cd deploy/cloudflare-worker && npm ci && npm run typecheck && npm test

docker-build:
	docker build -t fastapi-assistant:latest .

docker-run:
	docker run --rm -p 8080:8080 -v hf-cache:/models/hf fastapi-assistant:latest

compose-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build .cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
