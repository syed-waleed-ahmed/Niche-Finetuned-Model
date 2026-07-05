# syntax=docker/dockerfile:1
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/models/hf \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_JSON=true \
    # The model is loaded on first request rather than at startup so the
    # container becomes healthy quickly; set to "true" to warm on boot.
    WARMUP_ON_STARTUP=false

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source (installed via PYTHONPATH, no build step needed).
COPY src ./src
COPY data ./data

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models/hf /app/outputs \
    && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

# Liveness check hitting the /health endpoint (no model load required).
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "-m", "fastapi_assistant", "--serve"]
