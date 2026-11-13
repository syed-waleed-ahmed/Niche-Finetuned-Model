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
from .prompts import STOP_SEQUENCES, SYSTEM_PROMPT

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


PROJECT_URL = "https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model"
# The upstream project's live demo and documentation, linked from published model
# cards. A fork that publishes its own releases should point these at its own.
DEMO_URL = "https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev"
DOCS_URL = "https://syed-waleed-ahmed.github.io/Niche-Finetuned-Model/"

_CARD_METRICS = (
    ("score", "Composite score"),
    ("keyword_recall", "Keyword recall"),
    ("code_valid", "Code validity"),
    ("rouge_l", "ROUGE-L"),
)
# How each eval backend ran the model, for the card. Latency is left out of the card:
# it depends on the machine the gate ran on.
_CARD_BACKENDS = {
    "llamacpp": "llama.cpp, with this quantized GGUF, as in production",
    "transformers": (
        "transformers, with the unquantized model on the training GPU; the quantized GGUF"
        " is scored again with llama.cpp before it is deployed"
    ),
}
_EXAMPLE_QUESTION = "How do I return a 201 status code from a POST endpoint?"


def _metric(value: object) -> str:
    if value is None:
        return "not measured"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _card_evaluation(eval_report: dict | None, base_report: dict | None) -> list[str]:
    summary = (eval_report or {}).get("summary") or {}
    if not summary:
        return []
    base = (base_report or {}).get("summary") or {}
    backend = ((eval_report or {}).get("meta") or {}).get("backend", "unknown")
    where = _CARD_BACKENDS.get(backend, f"the `{backend}` backend")
    header, rule = ("| Metric | This release |", "| --- | --- |")
    if base:
        header, rule = header + " Base model |", rule + " --- |"
    rows = [
        f"| {label} | {_metric(summary.get(key))} |"
        + (f" {_metric(base.get(key))} |" if base else "")
        for key, label in _CARD_METRICS
        if key in summary
    ]
    return [
        "## Evaluation",
        "",
        f"The release gate answers {summary.get('items', 'the')} held-out FastAPI questions"
        " with greedy decoding and scores each answer for keyword recall (the API names a"
        " correct answer has to mention), code validity (its Python blocks parse) and"
        " ROUGE-L against a reference answer. The composite score is their mean per"
        f" question. Scored with {where}.",
        "",
        header,
        rule,
        *rows,
        "",
        "The metrics are automatic and lexical: they reward naming the right APIs and"
        " code that parses, not correctness. The"
        f" [model card in the source repository]({PROJECT_URL}/blob/main/MODEL_CARD.md)"
        " has the scores of both release gates, results per category and examples of wrong"
        " answers.",
        "",
    ]


def _card_usage(kind: str, repo_id: str, version: str, base_model: str, folder: Path) -> list[str]:
    system = json.dumps(SYSTEM_PROMPT)
    question = json.dumps(_EXAMPLE_QUESTION)
    stop = json.dumps(list(STOP_SEQUENCES))
    if kind == "gguf":
        ggufs = sorted(p.name for p in folder.glob("*.gguf"))
        filename = ggufs[0] if ggufs else "model-Q4_K_M.gguf"
        code = [
            "from huggingface_hub import hf_hub_download",
            "from llama_cpp import Llama",
            "",
            "path = hf_hub_download(",
            f'    "{repo_id}", "{filename}", revision="{version}"',
            ")",
            "# llama.cpp applies the chat template stored in the GGUF.",
            "llm = Llama(model_path=path, n_ctx=2048, verbose=False)",
            "answer = llm.create_chat_completion(",
            "    messages=[",
            f'        {{"role": "system", "content": {system}}},',
            f'        {{"role": "user", "content": {question}}},',
            "    ],",
            "    max_tokens=256,",
            "    temperature=0.0,",
            f"    stop={stop},",
            ")",
            'print(answer["choices"][0]["message"]["content"])',
        ]
        intro = (
            "With [llama-cpp-python](https://github.com/abetlen/llama-cpp-python); a CPU is"
            " enough (about 0.7 GB of RAM for the weights):"
        )
    else:
        code = [
            "from peft import PeftModel",
            "from transformers import AutoModelForCausalLM, AutoTokenizer",
            "",
            f'tokenizer = AutoTokenizer.from_pretrained("{base_model}")',
            f'model = AutoModelForCausalLM.from_pretrained("{base_model}")',
            f'model = PeftModel.from_pretrained(model, "{repo_id}", revision="{version}")',
            "messages = [",
            f'    {{"role": "system", "content": {system}}},',
            f'    {{"role": "user", "content": {question}}},',
            "]",
            "inputs = tokenizer.apply_chat_template(",
            '    messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"',
            ")",
            "output = model.generate(**inputs, max_new_tokens=256, do_sample=False)",
            'text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)',
            f"for marker in {stop}:  # see Prompt format",
            "    text = text.split(marker)[0]",
            "print(text.strip())",
        ]
        intro = "With `transformers` and `peft`, on top of the base model:"
    return [
        "## Usage",
        "",
        intro,
        "",
        "```python",
        *code,
        "```",
        "",
        "To serve it as an HTTP API with streaming, retrieval over the FastAPI documentation",
        f"and an OpenAI-compatible endpoint, use the [source repository]({PROJECT_URL}).",
        "",
    ]


def write_model_card(
    folder: Path,
    *,
    settings: Settings,
    version: str,
    kind: str,
    eval_report: dict | None = None,
    base_report: dict | None = None,
    repo_id: str | None = None,
) -> Path:
    """Write a README.md model card with provenance, usage and evaluation results.

    ``eval_report`` scores this release; ``base_report``, when given, scores the base
    model on the same set and adds a comparison column. ``repo_id`` is the Hub
    repository the card is published to, for the usage example.
    """
    base_model = settings.base_model_name
    repo_id = repo_id or f"<namespace>/fastapi-assistant-{'lora' if kind == 'adapter' else 'gguf'}"
    title = "LoRA adapter" if kind == "adapter" else "GGUF"
    library = "peft" if kind == "adapter" else "gguf"
    extra_tags = "lora, peft" if kind == "adapter" else "lora, gguf, llama-cpp"
    modification = (
        "a LoRA adapter trained on top of it"
        if kind == "adapter"
        else "the base weights with a LoRA fine-tune merged in, converted to GGUF and quantized"
    )
    what = (
        "This repository holds the LoRA adapter; load it on top of the base model."
        if kind == "adapter"
        else "This repository holds the fine-tune merged into the base model and quantized to"
        " a single GGUF file for llama.cpp."
    )
    markers = ", ".join(f"`{m}`" for m in STOP_SEQUENCES)
    lines = [
        "---",
        "language:",
        "  - en",
        "license: apache-2.0",
        f"base_model: {base_model}",
        f"library_name: {library}",
        "pipeline_tag: text-generation",
        f"tags: [fastapi, python, code, {extra_tags}]",
        "---",
        "",
        f"# FastAPI Assistant: {title} ({version})",
        "",
        f"A LoRA fine-tune of [`{base_model}`](https://huggingface.co/{base_model}) that"
        " answers questions about the [FastAPI](https://fastapi.tiangolo.com/) web"
        f" framework. {what}",
        "",
        f"- [Live demo]({DEMO_URL}) (no key needed)",
        f"- [Source code, training pipeline and service]({PROJECT_URL})",
        f"- [Documentation]({DOCS_URL})",
        f"- [Full model card]({PROJECT_URL}/blob/main/MODEL_CARD.md)",
        "",
        f"The tag `{version}` is the release; pin it with `revision=` to get exactly these"
        " weights.",
        "",
        *_card_usage(kind, repo_id, version, base_model, folder),
        "## Prompt format",
        "",
        "Use the system prompt the model was trained with, in the chat template of the base"
        " model (stored in the GGUF and in the base model's tokenizer):",
        "",
        "```text",
        SYSTEM_PROMPT,
        "```",
        "",
        f"The template's role markers ({markers}) are plain text to this model, not special"
        " tokens. When it does not end its answer with the end-of-sequence token, it runs on"
        " into an invented next turn, so stop generation at any of them, as the example does.",
        "",
        *_card_evaluation(eval_report, base_report),
        "## Training",
        "",
        "- LoRA with completion-only loss (only answer tokens are trained), using the same"
        " chat template in training, evaluation and serving",
        f"- LoRA r={settings.lora_r}, alpha={settings.lora_alpha}, dropout={settings.lora_dropout};"
        f" target modules: {', '.join(settings.lora_target_modules)}",
        f"- Epochs={settings.num_epochs}, lr={settings.learning_rate},"
        f" max_seq_length={settings.max_seq_length}",
        f"- Data: the curated FastAPI question-answer pairs in [`data/`]({PROJECT_URL}/tree/main/data);"
        " the held-out evaluation set is never used for training or checkpoint selection",
        "",
        "## Intended use and limitations",
        "",
        "Short answers to FastAPI questions, each with a minimal code example, for developers"
        " who can check them.",
        "",
        "- At 1.1B parameters it is often wrong, including in code that parses: it invents"
        " parameters and APIs and states them confidently. Check answers against the"
        " FastAPI documentation.",
        "- Generated code is untrusted. Review it, and run it only in a sandbox.",
        "- FastAPI and English only, with a 2,048-token context window.",
        "",
        "## License and attribution",
        "",
        f"Derived from `{base_model}`, released by its authors under the",
        "Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). This repository",
        f"contains modified work: {modification}. It is distributed under the same",
        "license; see NOTICE.",
        "",
    ]
    path = folder / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    (folder / "NOTICE").write_text(
        f"FastAPI Assistant {kind} {version}\n\n"
        f"This work is derived from {base_model}, licensed under the Apache\n"
        "License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).\n\n"
        f"Modifications: {modification}, by the FastAPI Assistant project\n"
        f"({PROJECT_URL}).\n",
        encoding="utf-8",
    )
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


def publish_training_outputs(
    out_dir: Path,
    namespace: str,
    *,
    token: str | None = None,
    push=push_folder,
) -> dict:
    """Publish a finished training run's adapter and GGUF; return the completed manifest.

    The Kaggle kernel writes ``model-release.json``, the adapter folder, the GGUF and
    the eval reports into its output. When the kernel could not upload them itself
    (no Hugging Face token on Kaggle), CI downloads that output and calls this: both
    folders get a model card and NOTICE, are pushed under the run's version tag, and
    the manifest gains the repository names and the GGUF checksum. A manifest that
    already names a GGUF repository is returned unchanged.
    """
    from .hub import ADAPTER_FILES, sha256_file

    manifest = load_manifest(out_dir / "model-release.json")
    if (manifest.get("gguf") or {}).get("repo_id"):
        return manifest

    adapter_config = next(iter(sorted(out_dir.rglob("adapter_config.json"))), None)
    ggufs = sorted(out_dir.rglob("model-*.gguf"))
    if adapter_config is None or not ggufs:
        raise FileNotFoundError(
            f"{out_dir} has no adapter (adapter_config.json) or GGUF (model-*.gguf) to publish"
        )
    adapter_dir, gguf = adapter_config.parent, ggufs[0]
    report, base_report = (
        json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        for path in (out_dir / "eval-finetuned.json", out_dir / "eval-base.json")
    )

    version = manifest["model_version"]
    settings = Settings(base_model_name=manifest["base_model"])
    adapter_repo = f"{namespace}/fastapi-assistant-lora"
    gguf_repo = f"{namespace}/fastapi-assistant-gguf"
    card_files = ["README.md", "NOTICE"]
    for folder, repo, kind, patterns in (
        (adapter_dir, adapter_repo, "adapter", [*ADAPTER_FILES, *card_files]),
        (gguf.parent, gguf_repo, "gguf", [gguf.name, *card_files]),
    ):
        write_model_card(
            folder,
            settings=settings,
            version=version,
            kind=kind,
            eval_report=report,
            base_report=base_report,
            repo_id=repo,
        )
        push(folder, repo, tag=version, allow_patterns=patterns, token=token)

    published = release_manifest(
        version=version,
        base_model=manifest["base_model"],
        adapter_repo=adapter_repo,
        gguf_repo=gguf_repo,
        gguf_filename=gguf.name,
        eval_report=report,
        gguf_sha256=sha256_file(gguf),
    )
    published["created_at"] = manifest.get("created_at", published["created_at"])
    published["finetuned"] = True
    return published
