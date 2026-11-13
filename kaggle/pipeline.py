"""Kaggle GPU training pipeline: train -> evaluate -> gate -> merge -> GGUF -> publish.

Runs as a Kaggle "script" kernel on a free GPU (T4/P100, about 30 GPU hours a
week). It is pushed and started by .github/workflows/train.yml, which rewrites the
CONFIG block below; it can also be pasted into a Kaggle notebook and run by hand.

Secrets (Kaggle: Add-ons -> Secrets, attach to this kernel once):
  HF_TOKEN  Hugging Face token with write access. Without it the pipeline still
            trains and evaluates, and leaves the artifacts in /kaggle/working.

Outputs in /kaggle/working:
  model-release.json   pinned release manifest to commit (see model-release.json)
  eval-base.json       base model score on the held-out set
  eval-finetuned.json  fine-tuned score (the gate compares these two)
  adapter/, gguf/      the artifacts (also pushed to the Hub when HF_TOKEN is set)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG = {
    "repo": "https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model.git",
    "ref": "main",
    "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "hf_namespace": "",  # your Hugging Face user or org; empty disables publishing
    "num_epochs": "3",
    "learning_rate": "0.0002",
    "lora_r": "16",
    "quant": "Q4_K_M",
    # The fine-tune must beat the base model's composite score by at least this much.
    # Greedy decoding makes the eval deterministic, so a tie means "no better": the
    # margin is small but strictly positive.
    "min_improvement": "0.005",
}
# __CONFIG_OVERRIDES__ (train.yml inserts CONFIG.update({...}) here)

WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd() / "kaggle-out"
SRC = Path("/kaggle/tmp/repo") if Path("/kaggle").exists() else WORK / "repo"


def sh(*args: str, env: dict | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True, env={**os.environ, **(env or {})})


def secret(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name]
    try:
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:
        return None


def main() -> int:
    WORK.mkdir(parents=True, exist_ok=True)
    if not SRC.exists():
        # `git clone --branch` rejects a commit SHA, and the Train workflow pins the
        # exact commit it was started from. init + fetch accepts a SHA, a branch or a tag.
        SRC.mkdir(parents=True)
        sh("git", "-C", str(SRC), "init", "--quiet")
        sh("git", "-C", str(SRC), "remote", "add", "origin", CONFIG["repo"])
        sh("git", "-C", str(SRC), "fetch", "--depth", "1", "origin", CONFIG["ref"])
        sh("git", "-C", str(SRC), "checkout", "--quiet", "FETCH_HEAD")
    os.chdir(SRC)
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    version = f"v{dt.date.today():%Y.%m.%d}-{sha}"
    # Kaggle images ship CUDA torch; the train extra keeps it (torch>=2.2 is satisfied).
    sh(sys.executable, "-m", "pip", "install", "--quiet", "-e", ".[train]")

    adapter_dir = WORK / "adapter"
    env = {
        "BASE_MODEL_NAME": CONFIG["base_model"],
        "OUTPUT_DIR": str(adapter_dir),
        "NUM_EPOCHS": CONFIG["num_epochs"],
        "LEARNING_RATE": CONFIG["learning_rate"],
        "LORA_R": CONFIG["lora_r"],
        "LORA_ALPHA": str(int(CONFIG["lora_r"]) * 2),
        "BATCH_SIZE": "8",
        "GRADIENT_ACCUMULATION_STEPS": "2",
        "ENGINE_BACKEND": "transformers",
        "MODEL_VERSION": version,
        "LOG_LEVEL": "INFO",
    }
    cli = [sys.executable, "-m", "fastapi_assistant"]

    sh(*cli, "data", "validate", env=env)
    sh(*cli, "train", env=env)

    base_report, ft_report = WORK / "eval-base.json", WORK / "eval-finetuned.json"
    sh(*cli, "eval", "--no-adapter", "--report", str(base_report), env=env)
    gate = subprocess.run(
        [
            *cli,
            "eval",
            "--report",
            str(ft_report),
            "--markdown",
            str(WORK / "eval-finetuned.md"),
            "--baseline",
            str(base_report),
            # Negative regression budget == required improvement over the base model.
            "--max-regression",
            str(-float(CONFIG["min_improvement"])),
        ],
        env={**os.environ, **env},
        check=False,
    )
    if gate.returncode != 0:
        print("Fine-tuned model did not beat the base model; not publishing.", flush=True)
        return 1

    merged_dir, gguf_dir = WORK / "merged", WORK / "gguf"
    sh(*cli, "export", "merge", "--out", str(merged_dir), env=env)
    sh("bash", "scripts/export_gguf.sh", str(merged_dir), str(gguf_dir), CONFIG["quant"])
    gguf_file = f"model-{CONFIG['quant']}.gguf"
    from fastapi_assistant.hub import sha256_file

    gguf_sha256 = sha256_file(gguf_dir / gguf_file)

    report = json.loads(ft_report.read_text(encoding="utf-8"))
    namespace = CONFIG["hf_namespace"]
    token = secret("HF_TOKEN")
    adapter_repo = gguf_repo = None
    if namespace and token:
        os.environ["HF_TOKEN"] = token
        adapter_repo = f"{namespace}/fastapi-assistant-lora"
        gguf_repo = f"{namespace}/fastapi-assistant-gguf"
        for folder, repo, kind in ((adapter_dir, adapter_repo, "adapter"), (gguf_dir, gguf_repo, "gguf")):
            sh(
                *cli, "export", "push", "--folder", str(folder), "--repo", repo, "--tag", version,
                "--kind", kind, "--eval-report", str(ft_report), env=env,
            )
    else:
        print("HF_TOKEN or hf_namespace missing: artifacts left in /kaggle/working only.")

    from fastapi_assistant.export import release_manifest

    manifest = release_manifest(
        version=version,
        base_model=CONFIG["base_model"],
        adapter_repo=adapter_repo,
        gguf_repo=gguf_repo,
        gguf_filename=gguf_file,
        gguf_sha256=gguf_sha256,
        eval_report=report,
    )
    manifest["finetuned"] = True
    (WORK / "model-release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
