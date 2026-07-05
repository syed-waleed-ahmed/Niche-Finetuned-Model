# Convenience targets. On Windows without `make`, run the underlying commands
# directly (e.g. `python -m pytest`).
PY ?= python

.PHONY: help install dev train serve cli test lint fmt docker-build docker-run clean

help:
	@echo "Targets:"
	@echo "  install       Install the package (runtime deps)"
	@echo "  dev           Install with dev extras (pytest, ruff, httpx)"
	@echo "  train         Fine-tune the LoRA adapter"
	@echo "  serve         Run the HTTP API"
	@echo "  cli           Run the interactive CLI"
	@echo "  test          Run the test suite"
	@echo "  lint          Lint with ruff"
	@echo "  fmt           Format with ruff"
	@echo "  docker-build  Build the container image"
	@echo "  docker-run    Run the container image"
	@echo "  clean         Remove caches and build outputs"

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

train:
	$(PY) -m fastapi_assistant --train

serve:
	$(PY) -m fastapi_assistant --serve

cli:
	$(PY) -m fastapi_assistant

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

fmt:
	$(PY) -m ruff format src tests

docker-build:
	docker build -t fastapi-assistant:latest .

docker-run:
	docker run --rm -p 8000:8000 fastapi-assistant:latest

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
