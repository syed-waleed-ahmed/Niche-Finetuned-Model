"""Publishing a finished training run from CI (no Hugging Face token on Kaggle)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fastapi_assistant.export import publish_training_outputs
from fastapi_assistant.hub import sha256_file


def _kernel_output(root: Path, *, published: bool = False) -> Path:
    out = root / "out"
    (out / "adapter").mkdir(parents=True)
    (out / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (out / "adapter" / "adapter_model.safetensors").write_bytes(b"lora")
    (out / "adapter" / "training_args.bin").write_bytes(b"pickle")  # must not be uploaded
    (out / "gguf").mkdir()
    (out / "gguf" / "model-Q4_K_M.gguf").write_bytes(b"gguf-weights")
    (out / "eval-finetuned.json").write_text(
        json.dumps({"summary": {"score": 0.538, "items": 108}}), encoding="utf-8"
    )
    manifest = {
        "model_version": "v2026.11.13-abc1234",
        "created_at": "2026-11-13T18:43:49+00:00",
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "adapter": None,
        "gguf": {"repo_id": "me/x", "filename": "f", "revision": "r"} if published else None,
        "eval": {"score": 0.538},
        "finetuned": True,
    }
    (out / "model-release.json").write_text(json.dumps(manifest), encoding="utf-8")
    return out


class FakePush:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, folder, repo_id, *, tag, allow_patterns=None, token=None, private=False):
        self.calls.append(
            {"folder": Path(folder).name, "repo": repo_id, "tag": tag, "patterns": allow_patterns}
        )
        return "commit-sha"


def test_publishes_both_artifacts_and_completes_the_manifest(tmp_path):
    out = _kernel_output(tmp_path)
    push = FakePush()
    manifest = publish_training_outputs(out, "someone", token="hf_x", push=push)

    assert [(c["folder"], c["repo"], c["tag"]) for c in push.calls] == [
        ("adapter", "someone/fastapi-assistant-lora", "v2026.11.13-abc1234"),
        ("gguf", "someone/fastapi-assistant-gguf", "v2026.11.13-abc1234"),
    ]
    adapter_patterns = push.calls[0]["patterns"]
    assert "adapter_model.safetensors" in adapter_patterns
    assert "training_args.bin" not in adapter_patterns
    assert push.calls[1]["patterns"][0] == "model-Q4_K_M.gguf"

    gguf = out / "gguf" / "model-Q4_K_M.gguf"
    assert manifest["gguf"] == {
        "repo_id": "someone/fastapi-assistant-gguf",
        "filename": "model-Q4_K_M.gguf",
        "revision": "v2026.11.13-abc1234",
        "sha256": sha256_file(gguf),
    }
    assert manifest["adapter"]["repo_id"] == "someone/fastapi-assistant-lora"
    assert manifest["finetuned"] is True
    assert manifest["created_at"] == "2026-11-13T18:43:49+00:00"
    # Model cards and Apache-2.0 notices are written before the upload.
    assert (out / "gguf" / "NOTICE").is_file()
    assert "Apache License 2.0" in (out / "adapter" / "README.md").read_text(encoding="utf-8")


def test_already_published_manifest_is_left_alone(tmp_path):
    out = _kernel_output(tmp_path, published=True)
    push = FakePush()
    manifest = publish_training_outputs(out, "someone", push=push)
    assert push.calls == []
    assert manifest["gguf"]["repo_id"] == "me/x"


def test_missing_artifacts_fail_clearly(tmp_path):
    out = _kernel_output(tmp_path)
    (out / "gguf" / "model-Q4_K_M.gguf").unlink()
    with pytest.raises(FileNotFoundError, match="GGUF"):
        publish_training_outputs(out, "someone", push=FakePush())


def test_cli_writes_the_completed_manifest_when_already_published(tmp_path):
    out = _kernel_output(tmp_path, published=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fastapi_assistant",
            "export",
            "publish-outputs",
            str(out),
            "--namespace",
            "someone",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads((out / "model-release.json").read_text(encoding="utf-8"))["gguf"]["repo_id"]
