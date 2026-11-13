"""Centralized, validated application configuration.

All settings are read from the environment (optionally via a local ``.env`` file)
and validated once at process startup. Import :func:`get_settings` rather than
reading ``os.environ`` directly so that configuration stays typed, bounded, and
easy to override in tests and deployments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: src/fastapi_assistant/config.py -> parents[2] == project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

EngineBackend = Literal["transformers", "llamacpp", "remote"]
RagBackend = Literal["local", "pgvector"]


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Field names map to upper-cased environment variables (case-insensitive), e.g.
    ``api_port`` <- ``API_PORT`` and ``model_temperature`` <- ``MODEL_TEMPERATURE``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow field names such as ``model_temperature`` without clashing with
        # Pydantic's reserved ``model_`` namespace.
        protected_namespaces=(),
        populate_by_name=True,
    )

    # --- Model identity ---------------------------------------------------
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    model_version: str = Field(
        default="dev",
        description="Release label reported in responses, metrics, and usage records.",
    )

    # --- Inference backend -------------------------------------------------
    engine_backend: EngineBackend = Field(
        default="transformers",
        description=(
            "transformers: base model + LoRA adapter via torch/peft (dev, GPU). "
            "llamacpp: quantized GGUF on CPU (production default). "
            "remote: an OpenAI-compatible server such as vLLM, TGI, or llama.cpp server."
        ),
    )
    finetuned_weights: bool = Field(
        default=True,
        description="Whether GGUF/remote weights include the fine-tune (reported as uses_adapter).",
    )

    # LoRA adapter (transformers backend). Pin a revision in production.
    use_adapter: bool = True
    adapter_repo_id: str | None = None
    adapter_revision: str | None = None

    # GGUF (llamacpp backend): a local file, or a file in a Hugging Face repo.
    gguf_path: Path | None = None
    gguf_repo_id: str | None = None
    gguf_filename: str = "*Q4_K_M.gguf"
    gguf_revision: str | None = None
    gguf_sha256: str | None = Field(
        default=None,
        description="Expected SHA-256 of the GGUF file; the load fails on a mismatch.",
    )
    llamacpp_n_ctx: int = Field(default=2048, ge=256, le=32768)
    llamacpp_n_threads: int | None = Field(default=None, ge=1)
    llamacpp_chat_format: str | None = Field(
        default=None,
        description="Override the chat template embedded in the GGUF (e.g. 'zephyr').",
    )

    # Remote OpenAI-compatible backend.
    remote_base_url: str | None = None
    remote_model: str | None = None
    remote_api_key: str | None = None
    remote_timeout_s: float = Field(default=120.0, gt=0)

    # --- HTTP service -----------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        # PORT is injected by Cloud Run, Hugging Face Spaces, Render, and similar.
        validation_alias=AliasChoices("API_PORT", "PORT", "api_port"),
    )
    api_key: str | None = Field(
        default=None,
        description="Single static key. With DATABASE_URL set, keys are also read from Postgres.",
    )
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    warmup_on_startup: bool = True
    origin_shared_secret: str | None = Field(
        default=None,
        description="If set, every non-probe request must carry X-Origin-Secret (sent by the edge).",
    )
    metrics_token: str | None = Field(
        default=None, description="If set, GET /metrics requires 'Authorization: Bearer <token>'."
    )
    max_chat_turns: int = Field(default=6, ge=1, le=50)
    max_input_chars: int = Field(default=6000, ge=100, le=100_000)
    max_request_bytes: int = Field(
        default=65_536,
        ge=0,
        le=10_000_000,
        description="Largest accepted request body (HTTP 413 above it); 0 disables the cap.",
    )

    # --- Admission control (backpressure) ----------------------------------
    max_concurrency: int = Field(default=1, ge=1, le=256)
    max_queue: int = Field(default=8, ge=0, le=10_000)
    queue_timeout_s: float = Field(default=30.0, gt=0)
    generation_timeout_s: float = Field(default=120.0, gt=0)

    # --- Persistence ------------------------------------------------------
    database_url: str | None = Field(
        default=None, description="Postgres URL (e.g. Neon). Enables per-user keys and usage."
    )
    db_pool_max_size: int = Field(default=4, ge=1, le=100)
    key_cache_ttl_s: float = Field(default=60.0, ge=0)
    store_transcripts: bool = Field(
        default=False, description="Persist question/answer text with usage events."
    )

    # --- Observability ----------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False
    enable_metrics: bool = True
    sentry_dsn: str | None = None
    sentry_environment: str = "production"
    sentry_traces_sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Retrieval-augmented generation ----------------------------------
    rag_enabled: bool = False
    rag_backend: RagBackend = "local"
    rag_index_dir: Path = PROJECT_ROOT / "outputs" / "rag_index"
    rag_index_version: str = Field(
        default="default", description="Row namespace in the pgvector doc_chunks table."
    )
    rag_top_k: int = Field(default=3, ge=1, le=20)
    rag_min_score: float = Field(default=0.35, ge=-1.0, le=1.0)
    rag_max_context_chars: int = Field(default=1500, ge=100, le=20_000)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path | None = Field(
        default=None, description="Where fastembed stores the ONNX model (bake into images)."
    )

    # --- Generation defaults (bounded) -----------------------------------
    model_max_new_tokens: int = Field(default=256, ge=1, le=2048)
    model_temperature: float = Field(default=0.4, ge=0.0, le=2.0)
    model_top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    model_repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)

    # --- Training hyperparameters ----------------------------------------
    batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    num_epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    max_seq_length: int = Field(default=512, ge=16, le=4096)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    seed: int = 42

    # --- Filesystem paths (override in containers/deployments) ------------
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs" / "fastapi_tinyllama_lora"

    @field_validator("gguf_sha256", mode="before")
    @classmethod
    def _check_sha256(cls, value):
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        digest = str(value).strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("GGUF_SHA256 must be 64 hexadecimal characters")
        return digest

    @field_validator("database_url", "sentry_dsn", "api_key", mode="before")
    @classmethod
    def _unset_placeholders(cls, value):
        """Treat placeholder values as "not configured".

        Cloud Run mounts every secret it references, so an optional secret that
        the owner has not provided yet holds a placeholder ("disabled").
        """
        if isinstance(value, str) and value.strip().lower() in {"", "disabled", "none", "null"}:
            return None
        return value

    @property
    def train_path(self) -> Path:
        return self.data_dir / "fastapi_qa_train.jsonl"

    @property
    def eval_path(self) -> Path:
        return self.data_dir / "fastapi_qa_eval.jsonl"

    @property
    def has_trained_adapter(self) -> bool:
        return (self.output_dir / "adapter_config.json").is_file()

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_key or self.database_url)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
