"""transformers 4.x/5.x compatibility helpers."""

from __future__ import annotations

from fastapi_assistant import _compat


def test_kwargs_follow_installed_major_version(monkeypatch):
    monkeypatch.setattr(_compat, "_major_version", lambda: 5)
    assert _compat.dtype_kwargs("fp16") == {"dtype": "fp16"}
    assert _compat.warmup_kwargs(0.03) == {"warmup_steps": 0.03}
    monkeypatch.setattr(_compat, "_major_version", lambda: 4)
    assert _compat.dtype_kwargs("fp16") == {"torch_dtype": "fp16"}
    assert _compat.warmup_kwargs(0.03) == {"warmup_ratio": 0.03}


def test_training_arguments_accept_warmup_kwarg(tmp_path):
    from transformers import TrainingArguments

    args = TrainingArguments(
        output_dir=str(tmp_path), report_to="none", **_compat.warmup_kwargs(0.03)
    )
    assert args is not None
