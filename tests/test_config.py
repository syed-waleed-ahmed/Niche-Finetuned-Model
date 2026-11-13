"""Tests for typed configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fastapi_assistant.config import Settings


def test_defaults():
    settings = Settings(_env_file=None)
    assert settings.base_model_name.startswith("TinyLlama/")
    assert settings.api_port == 8000
    assert settings.model_temperature == pytest.approx(0.4)
    assert settings.lora_target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert settings.engine_backend == "transformers"
    assert settings.auth_enabled is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("API_PORT", "9001")
    monkeypatch.setenv("MODEL_MAX_NEW_TOKENS", "128")
    monkeypatch.setenv("API_KEY", "secret")
    settings = Settings(_env_file=None)
    assert settings.api_port == 9001
    assert settings.model_max_new_tokens == 128
    assert settings.api_key == "secret"


def test_out_of_range_rejected(monkeypatch):
    monkeypatch.setenv("MODEL_TOP_P", "5")  # must be <= 1
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_derived_paths():
    settings = Settings(_env_file=None)
    assert settings.train_path.name == "fastapi_qa_train.jsonl"
    assert settings.eval_path.name == "fastapi_qa_eval.jsonl"
    assert settings.train_path.parent == settings.data_dir


def test_sentry_placeholder_does_not_initialize(monkeypatch):
    """The Cloud Run spec always mounts sentry-dsn; a non-URL value keeps Sentry off."""
    import sentry_sdk

    from fastapi_assistant.api import _init_sentry

    calls = []
    monkeypatch.setattr(sentry_sdk, "init", lambda **kw: calls.append(kw))
    for value in (None, "", "disabled"):
        _init_sentry(Settings(_env_file=None, sentry_dsn=value))
    assert calls == []
    _init_sentry(Settings(_env_file=None, sentry_dsn="https://key@o1.ingest.sentry.io/1"))
    assert len(calls) == 1


def test_optional_secret_placeholders_mean_unset():
    for value in ("disabled", "", "None", " null "):
        settings = Settings(_env_file=None, database_url=value, sentry_dsn=value, api_key=value)
        assert settings.database_url is None
        assert settings.sentry_dsn is None
        assert settings.api_key is None
        assert settings.auth_enabled is False
    real = Settings(_env_file=None, database_url="postgresql://u:p@h/db")
    assert real.database_url == "postgresql://u:p@h/db"
