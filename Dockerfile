# syntax=docker/dockerfile:1.7
#
# Production image: FastAPI service + llama.cpp (quantized GGUF on CPU).
# No torch, no compiler in the final image. Builds for linux/amd64 and linux/arm64
# (Oracle Ampere) from the same file.
#
#   docker build -t fastapi-assistant .
#   docker build --build-arg BAKE_MODEL=true -t fastapi-assistant:baked .   # weights inside
#
# Model weights are pulled from the Hugging Face Hub at the pinned revision on first
# start unless BAKE_MODEL=true. Configure the release with the env vars printed by
# `fastapi-assistant export env model-release.json`.

ARG PYTHON_VERSION=3.11

# ---- builder: compile wheels (llama-cpp-python needs a C/C++ toolchain) -------------
FROM python:${PYTHON_VERSION}-slim AS builder
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Portable CPU build: no -march=native, so an image built on a CI runner does not
    # crash with "illegal instruction" on a different host CPU.
    CMAKE_ARGS="-DGGML_NATIVE=OFF" \
    FORCE_CMAKE=1
WORKDIR /build
COPY requirements/serve.txt ./requirements/serve.txt
# Hash-checked download of every pinned dependency, compiled to wheels.
RUN pip wheel --require-hashes --wheel-dir /wheels -r requirements/serve.txt
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /wheels .

# ---- runtime ------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
ARG BAKE_MODEL=false
ARG BAKE_RAG_INDEX=true
ARG GGUF_REPO_ID=TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
ARG GGUF_FILENAME=tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
ARG GGUF_REVISION=52e7645ba7c309695bec7ac98f4f005b139cf465
ARG GGUF_SHA256=9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    EMBEDDING_CACHE_DIR=/models/fastembed \
    RAG_INDEX_DIR=/models/rag_index \
    RAG_ENABLED=false \
    API_HOST=0.0.0.0 \
    API_PORT=8080 \
    LOG_JSON=true \
    ENGINE_BACKEND=llamacpp \
    GGUF_REPO_ID=${GGUF_REPO_ID} \
    GGUF_FILENAME=${GGUF_FILENAME} \
    GGUF_REVISION=${GGUF_REVISION} \
    GGUF_SHA256=${GGUF_SHA256} \
    LLAMACPP_CHAT_FORMAT=zephyr \
    FINETUNED_WEIGHTS=false \
    MODEL_VERSION=base-tinyllama-q4km \
    WARMUP_ON_STARTUP=true

# Apply Debian security updates the base image does not have yet (the release gate
# fails on fixable HIGH CVEs). libgomp1: OpenMP runtime for llama.cpp's CPU backend.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links /wheels /wheels/*.whl && rm -rf /wheels

RUN useradd --create-home --uid 10001 appuser && mkdir -p /models

# Optional: bake the documentation index (about 70 MB with the embedding model).
# Retrieval stays opt-in per request (use_retrieval=true, or RAG_ENABLED=true): on
# the base model it raised keyword recall but lowered code validity (see eval/).
RUN if [ "$BAKE_RAG_INDEX" = "true" ]; then \
        fastapi-assistant rag build --target local --workdir /tmp/fastapi-docs \
        && rm -rf /tmp/fastapi-docs; \
    fi
# Optional: bake the GGUF weights (about 0.7 GB) for faster cold starts.
RUN if [ "$BAKE_MODEL" = "true" ]; then \
        python -c "from huggingface_hub import hf_hub_download as d; d('${GGUF_REPO_ID}', '${GGUF_FILENAME}', revision='${GGUF_REVISION}')"; \
    fi
# Packaging tools are build-time only. Removing them drops their CVEs (pip, and the
# wheel and jaraco.context copies vendored in setuptools) from the shipped image.
RUN python -m pip uninstall -y pip setuptools wheel
RUN chown -R appuser:appuser /models
USER appuser
WORKDIR /home/appuser

EXPOSE 8080

# Liveness only; /ready gates traffic until the model is loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen(f'http://127.0.0.1:{os.environ.get(\"API_PORT\",\"8080\")}/health', timeout=4).status == 200 else 1)"

CMD ["fastapi-assistant", "serve"]
