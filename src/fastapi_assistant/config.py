"""Centralized, validated application configuration.

All settings are read from the environment (optionally via a local ``.env`` file)
and validated once at process startup. Import :func:`get_settings` rather than
reading ``os.environ`` directly so that configuration stays typed, bounded, and
easy to override in tests and deployments.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: src/fastapi_assistant/config.py -> parents[2] == project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    )

    # --- Base model -------------------------------------------------------
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # --- HTTP service -----------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_key: str | None = Field(
        default=None,
        description="If set, /generate requires a matching X-API-Key header.",
    )
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    warmup_on_startup: bool = True

    # --- Logging ----------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

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
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])

    # --- Filesystem paths (override in containers/deployments) ------------
    data_dir: Path = PROJECT_ROOT / "data"
    output_dir: Path = PROJECT_ROOT / "outputs" / "fastapi_tinyllama_lora"

    @property
    def train_path(self) -> Path:
        return self.data_dir / "fastapi_qa_train.jsonl"

    @property
    def eval_path(self) -> Path:
        return self.data_dir / "fastapi_qa_eval.jsonl"

    @property
    def has_trained_adapter(self) -> bool:
        return (self.output_dir / "adapter_config.json").is_file()


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""
    return Settings()
