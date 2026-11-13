"""Model release tooling: merge the adapter, write a model card, publish to the Hub.

Release flow (normally run on a Kaggle GPU by ``kaggle/pipeline.py``):

1. ``train`` writes the LoRA adapter to ``OUTPUT_DIR``.
2. ``eval`` scores it; the gate must pass.
3. ``merge_adapter`` folds the adapter into the base weights (needed for GGUF).
4. ``scripts/export_gguf.sh`` converts and quantizes the merged model.
5. ``push_folder`` uploads the adapter and the GGUF to versioned Hub repos and tags
   the commit, and ``release_manifest`` records the exact revisions to deploy.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .hub import resolve_adapter_dir

log = logging.getLogger(__name__)


def merge_adapter(settings: Settings, out_dir: Path) -> Path:
    """Load base + adapter, merge LoRA weights into the base, save a full model."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from ._compat import dtype_kwargs

    adapter_dir = resolve_adapter_dir(settings)
    if adapter_dir is None:
        raise FileNotFoundError("No adapter to merge; train one or set ADAPTER_REPO_ID")
    log.info("Merging %s into %s", adapter_dir, settings.base_model_name)
    # fp16 halves the artifact; GGUF conversion re-quantizes anyway.
    base = AutoModelForCausalLM.from_pretrained(
        settings.base_model_name, **dtype_kwargs(torch.float16)
    )
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer_src = (
        adapter_dir
        if (adapter_dir / "tokenizer_config.json").is_file()
        else settings.base_model_name
    )
    AutoTokenizer.from_pretrained(str(tokenizer_src)).save_pretrained(str(out_dir))
    log.info("Merged model written to %s", out_dir)
    return out_dir


def write_model_card(
    folder: Path,
    *,
    settings: Settings,
    version: str,
    kind: str,
    eval_report: dict | None = None,
) -> Path:
    """Write a README.md model card with provenance and evaluation results."""
    summary = (eval_report or {}).get("summary", {})
    lines = [
        "---",
        "license: apache-2.0",
        f"base_model: {settings.base_model_name}",
        "library_name: " + ("peft" if kind == "adapter" else "gguf"),
        "tags: [fastapi, python, code, lora]",
        "---",
        "",
        f"# FastAPI Assistant {kind} ({version})",
        "",
        f"LoRA fine-tune of `{settings.base_model_name}` that answers questions about the",
        "FastAPI web framework. Trained with completion-only loss on a curated Q&A set.",
        "",
        "## Training",
        "",
        f"- LoRA r={settings.lora_r}, alpha={settings.lora_alpha}, dropout={settings.lora_dropout}",
        f"- Target modules: {', '.join(settings.lora_target_modules)}",
        f"- Epochs={settings.num_epochs}, lr={settings.learning_rate}, max_seq_length={settings.max_seq_length}",
        "",
    ]
    if summary:
        lines += [
            "## Evaluation (greedy decoding, held-out set)",
            "",
            "| metric | value |",
            "| --- | --- |",
            *[f"| {k} | {v} |" for k, v in summary.items() if v is not None],
            "",
        ]
    lines += [
        "## Intended use",
        "",
        "Small, fast FastAPI help. Verify generated code; the model can be wrong.",
        "",
    ]
    path = folder / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def push_folder(
    folder: Path,
    repo_id: str,
    *,
    tag: str,
    private: bool = False,
    allow_patterns: list[str] | None = None,
    token: str | None = None,
) -> str:
    """Upload ``folder`` to ``repo_id`` and tag the resulting commit. Returns the commit sha."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        folder_path=str(folder),
        allow_patterns=allow_patterns,
        commit_message=f"Release {tag}",
    )
    api.create_tag(repo_id, tag=tag, revision=commit.oid, exist_ok=False)
    log.info("Pushed %s to %s@%s (%s)", folder, repo_id, tag, commit.oid)
    return commit.oid


def release_manifest(
    *,
    version: str,
    base_model: str,
    adapter_repo: str | None,
    gguf_repo: str | None,
    gguf_filename: str | None,
    eval_report: dict | None,
    gguf_sha256: str | None = None,
) -> dict:
    """The document committed as ``model-release.json`` to deploy a release.

    Revisions are pinned to the release tag so a deployment is reproducible and a
    rollback is a one-line revert.
    """
    return {
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_model": base_model,
        "adapter": {"repo_id": adapter_repo, "revision": version} if adapter_repo else None,
        "gguf": (
            {
                "repo_id": gguf_repo,
                "filename": gguf_filename,
                "revision": version,
                **({"sha256": gguf_sha256} if gguf_sha256 else {}),
            }
            if gguf_repo
            else None
        ),
        "eval": (eval_report or {}).get("summary"),
    }


def manifest_env(manifest: dict) -> dict[str, str]:
    """Translate a release manifest into the service's environment variables."""
    env = {"MODEL_VERSION": manifest["model_version"], "BASE_MODEL_NAME": manifest["base_model"]}
    if manifest.get("gguf"):
        env.update(
            {
                "ENGINE_BACKEND": "llamacpp",
                "GGUF_REPO_ID": manifest["gguf"]["repo_id"],
                "GGUF_FILENAME": manifest["gguf"]["filename"],
                "GGUF_REVISION": manifest["gguf"]["revision"],
                "FINETUNED_WEIGHTS": str(manifest.get("finetuned", True)).lower(),
            }
        )
        if manifest["gguf"].get("sha256"):
            env["GGUF_SHA256"] = manifest["gguf"]["sha256"]
        if manifest.get("chat_format"):
            env["LLAMACPP_CHAT_FORMAT"] = manifest["chat_format"]
    if manifest.get("adapter"):
        env.update(
            {
                "ADAPTER_REPO_ID": manifest["adapter"]["repo_id"],
                "ADAPTER_REVISION": manifest["adapter"]["revision"],
            }
        )
    return env


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
