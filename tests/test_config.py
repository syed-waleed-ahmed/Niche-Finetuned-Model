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
    assert settings.lora_target_modules == ["q_proj", "v_proj"]


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
