"""Draft new training data from the official FastAPI docs with Claude.

This scales the dataset beyond what is practical to write by hand while keeping it
grounded: every draft is generated from one documentation chunk, constrained to a
JSON schema, then filtered locally (code must parse, length limits, required
identifiers present, no duplicates of existing rows). Drafts are written to a
separate file for human review and only enter the dataset through
``fastapi-assistant data merge``.

Requires ``pip install -e ".[datagen]"`` and Anthropic credentials
(``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile). Each chunk is one API
request, so ``--pages 20 --per-chunk 3`` costs 20 requests.
"""

from __future__ import annotations

import ast
import json
import logging
import random
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .config import get_settings
from .datasets_tools import read_rows, row_key
from .docs_corpus import (
    DEFAULT_FASTAPI_REF,
    DocChunk,
    chunk_pages,
    download_fastapi_repo,
    load_doc_pages,
)
from .evaluation import extract_code_blocks

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"
MAX_OUTPUT_CHARS = 1400

SYSTEM_PROMPT = """You write training data for a small language model that answers questions \
about the FastAPI Python web framework. You will be given one section of the official \
FastAPI documentation. Write question-and-answer pairs that a developer would plausibly ask \
and that this section fully answers.

Answer style (the model learns it exactly):
- A one or two sentence lead, then one minimal runnable example in a ```python fence, then at \
most two short sentences on the key point. No headings, no emoji, no filler.
- At most 1400 characters per answer; aim for 350-900.
- Modern idioms only: FastAPI 0.110+, Pydantic v2 (model_dump, ConfigDict, field_validator), \
Annotated[...] for Query/Path/Depends, `str | None`, lifespan instead of on_event.
- Only state what the documentation section supports. Do not invent parameters or APIs.

Questions:
- Vary phrasing (how do I, why does, what is the difference, can I) and difficulty.
- About a third of the pairs should put a short buggy or incomplete code snippet in `input` and \
ask to fix or explain it; the rest leave `input` empty.

For each pair also give `must_include`: 2-4 short API identifiers (for example "Depends", \
"status_code=201") that any correct answer must contain and that your answer contains."""


class DraftPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    input: str
    output: str
    must_include: list[str]


class DraftBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[DraftPair] = Field(description="The question-answer pairs.")


# Written out explicitly (no $ref) so the request schema is self-contained.
DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "input": {"type": "string"},
                    "output": {"type": "string"},
                    "must_include": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["instruction", "input", "output", "must_include"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


def check_draft(pair: DraftPair) -> str | None:
    """Return a rejection reason, or ``None`` if the draft passes local checks."""
    if not pair.instruction.strip() or not pair.output.strip():
        return "empty field"
    if len(pair.output) > MAX_OUTPUT_CHARS:
        return f"output too long ({len(pair.output)} chars)"
    blocks = extract_code_blocks(pair.output)
    for block in blocks:
        try:
            ast.parse(block)
        except SyntaxError as exc:
            return f"code does not parse: {exc.msg}"
    if not pair.must_include:
        return "no must_include identifiers"
    lowered = pair.output.lower()
    missing = [k for k in pair.must_include if k.lower() not in lowered]
    if missing:
        return f"answer lacks its own must_include {missing}"
    return None


def _api_errors() -> tuple[tuple[type[BaseException], ...], tuple[type[BaseException], ...]]:
    """(fatal, skippable) SDK exception types; empty when the SDK is not installed.

    The SDK is optional ([datagen] extra); tests drive this module with a fake client.
    """
    try:
        import anthropic
    except ImportError:
        return (), ()
    fatal = (
        anthropic.BadRequestError,  # configuration problems: stop instead of burning
        anthropic.AuthenticationError,  # through every chunk
        anthropic.PermissionDeniedError,
        anthropic.RateLimitError,  # the SDK already retried with backoff; rerun later
    )
    return fatal, (anthropic.APIStatusError, anthropic.APIConnectionError)


def _request(client, model: str, chunk: DocChunk, per_chunk: int, effort: str) -> DraftBatch | None:
    fatal, skippable = _api_errors()
    user = (
        f"Documentation page: {chunk.url}\nSection: {chunk.heading}\n\n"
        f"<documentation>\n{chunk.text}\n</documentation>\n\n"
        f"Write {per_chunk} question-answer pairs grounded in this section."
    )
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            # The instructions are identical for every chunk, so cache them.
            system=[
                {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": DRAFT_SCHEMA},
            },
            # If the model declines, the API retries the request on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except fatal:
        raise
    except skippable as exc:
        log.warning("Skipping %s: %s", chunk.url, exc)
        return None

    if response.stop_reason == "refusal":
        log.warning("Request declined for %s; skipping", chunk.url)
        return None
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        log.warning("No text block for %s (stop_reason=%s)", chunk.url, response.stop_reason)
        return None
    return DraftBatch.model_validate_json(text)


def generate_dataset(
    *,
    out_path: Path,
    n_chunks: int,
    per_chunk: int = 3,
    ref: str | None = None,
    model: str | None = None,
    seed: int = 0,
    effort: str = "medium",
    client=None,
    chunks: list[DocChunk] | None = None,
) -> int:
    """Generate, filter, and write draft rows. Returns the number written."""
    settings = get_settings()
    if chunks is None:
        repo = download_fastapi_repo(ref or DEFAULT_FASTAPI_REF, Path(".cache/fastapi-docs"))
        chunks = chunk_pages(load_doc_pages(repo), max_chars=2500, overlap=0)
    # Prefer sections with code: they make the most useful training pairs.
    with_code = [c for c in chunks if "```" in c.text]
    pool = with_code if len(with_code) >= n_chunks else chunks
    sample = random.Random(seed).sample(pool, min(n_chunks, len(pool)))

    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=4)

    existing = set()
    for path in (settings.train_path, settings.eval_path):
        if path.exists():
            existing.update(row_key(r) for r in read_rows(path))

    written = 0
    rejected: dict[str, int] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8", newline="\n") as fh:
        for i, chunk in enumerate(sample, start=1):
            batch = _request(client, model or DEFAULT_MODEL, chunk, per_chunk, effort)
            if batch is None:
                continue
            for pair in batch.items:
                reason = check_draft(pair)
                key = row_key(pair.model_dump())
                if reason is None and key in existing:
                    reason = "duplicate of an existing row"
                if reason:
                    rejected[reason.split(" (")[0]] = rejected.get(reason.split(" (")[0], 0) + 1
                    continue
                existing.add(key)
                row = {
                    **pair.model_dump(),
                    "source_url": chunk.url,
                    "source_heading": chunk.heading,
                }
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            log.info("[%d/%d] %s -> %d kept so far", i, len(sample), chunk.url, written)
    if rejected:
        log.info("Rejected drafts: %s", rejected)
    return written
