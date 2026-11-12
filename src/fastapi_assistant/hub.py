"""Resolve model artifacts from local disk or the Hugging Face Hub.

Weights are build artifacts, never source: the LoRA adapter and the GGUF export
live in versioned Hugging Face repos, and a deployment pins a revision (tag or
commit). Local paths remain supported for development.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from pathlib import Path

from .config import Settings

log = logging.getLogger(__name__)

ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "tokenizer*",
    "special_tokens_map.json",
    "chat_template.jinja",
]

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def latest_checkpoint_with_adapter(output_dir: Path) -> Path | None:
    """Return the highest-numbered ``checkpoint-N`` folder that holds an adapter."""
    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        match = _CHECKPOINT_RE.match(child.name)
        if match and (child / "adapter_config.json").is_file():
            candidates.append((int(match.group(1)), child))
    return max(candidates)[1] if candidates else None


def resolve_adapter_dir(settings: Settings) -> Path | None:
    """Locate the LoRA adapter to apply, or ``None`` to serve the base model.

    Order: Hub repo (``ADAPTER_REPO_ID`` at ``ADAPTER_REVISION``), then the final
    adapter in ``OUTPUT_DIR``, then the newest training checkpoint in it.
    """
    if not settings.use_adapter:
        return None
    if settings.adapter_repo_id:
        from huggingface_hub import snapshot_download

        if not settings.adapter_revision:
            log.warning(
                "ADAPTER_REVISION is not pinned; using the latest revision of %s",
                settings.adapter_repo_id,
            )
        path = snapshot_download(
            repo_id=settings.adapter_repo_id,
            revision=settings.adapter_revision,
            allow_patterns=ADAPTER_FILES,
        )
        return Path(path)
    if settings.has_trained_adapter:
        return settings.output_dir
    checkpoint = latest_checkpoint_with_adapter(settings.output_dir)
    if checkpoint is not None:
        log.warning(
            "No final adapter in %s; falling back to training checkpoint %s",
            settings.output_dir,
            checkpoint.name,
        )
    return checkpoint


def resolve_gguf_path(settings: Settings) -> Path:
    """Locate the GGUF file for the llama.cpp backend (downloading it if needed)."""
    if settings.gguf_path:
        if not settings.gguf_path.is_file():
            raise FileNotFoundError(f"GGUF_PATH does not exist: {settings.gguf_path}")
        return settings.gguf_path
    if not settings.gguf_repo_id:
        raise ValueError("Set GGUF_PATH or GGUF_REPO_ID for the llamacpp backend")

    from huggingface_hub import HfApi, hf_hub_download

    filename = settings.gguf_filename
    if any(ch in filename for ch in "*?["):
        files = HfApi().list_repo_files(settings.gguf_repo_id, revision=settings.gguf_revision)
        matches = sorted(f for f in files if fnmatch.fnmatch(f, filename))
        if not matches:
            raise FileNotFoundError(
                f"No file matching {filename!r} in {settings.gguf_repo_id}@{settings.gguf_revision}"
            )
        filename = matches[0]
    log.info("Fetching %s from %s@%s", filename, settings.gguf_repo_id, settings.gguf_revision)
    return Path(
        hf_hub_download(
            repo_id=settings.gguf_repo_id,
            filename=filename,
            revision=settings.gguf_revision,
        )
    )
