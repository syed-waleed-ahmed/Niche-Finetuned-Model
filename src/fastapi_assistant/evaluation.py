"""Offline evaluation harness and release gate.

Every eval item has a reference answer and ``must_include`` identifiers (API names
any correct answer has to mention). For each item the model's answer is scored on:

* ``keyword_recall`` — fraction of ``must_include`` strings present (case-insensitive).
* ``code_valid`` — every fenced Python block parses (only scored when the reference
  contains code); optionally ``code_runs`` executes each block in a subprocess.
* ``rouge_l`` — ROUGE-L F1 against the reference (lexical overlap).

The composite ``score`` is the mean of keyword recall, code validity, and ROUGE-L
over the items where each applies. :func:`gate` compares a report with a baseline
and fails on regressions, which is what CI and the training pipeline use to decide
whether a new model may be released.
"""

from __future__ import annotations

import ast
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .engines.base import Engine, GenerationParams
from .prompts import build_messages
from .retrieval import Retriever, format_context

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[^\sA-Za-z0-9_]")


@dataclass
class EvalItem:
    id: str
    instruction: str
    output: str
    input: str = ""
    category: str = "general"
    must_include: list[str] = field(default_factory=list)


@dataclass
class ItemResult:
    id: str
    category: str
    keyword_recall: float | None
    code_valid: float | None
    code_runs: float | None
    rouge_l: float
    score: float
    latency_s: float
    answer: str
    missing: list[str]


def load_eval_items(path: Path, limit: int | None = None) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            items.append(
                EvalItem(
                    id=row.get("id") or f"item-{i:04d}",
                    instruction=row["instruction"],
                    output=row["output"],
                    input=row.get("input") or "",
                    category=row.get("category") or "general",
                    must_include=list(row.get("must_include") or []),
                )
            )
    return items[:limit] if limit else items


# -- metrics -----------------------------------------------------------------------
def extract_code_blocks(text: str) -> list[str]:
    return [block for block in _CODE_BLOCK.findall(text) if block.strip()]


def keyword_recall(answer: str, keywords: Sequence[str]) -> tuple[float | None, list[str]]:
    if not keywords:
        return None, []
    lowered = answer.lower()
    missing = [k for k in keywords if k.lower() not in lowered]
    return 1.0 - len(missing) / len(keywords), missing


def code_validity(answer: str) -> float:
    blocks = extract_code_blocks(answer)
    if not blocks:
        return 0.0
    ok = 0
    for block in blocks:
        try:
            ast.parse(block)
            ok += 1
        except SyntaxError:
            pass
    return ok / len(blocks)


def code_execution(answer: str, timeout_s: float = 15.0) -> float:
    """Fraction of code blocks that execute without error in a subprocess.

    This runs model-generated code. Only enable it in a disposable environment
    (a CI runner or container), never on a machine with secrets.
    """
    blocks = extract_code_blocks(answer)
    if not blocks:
        return 0.0
    ok = 0
    for block in blocks:
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(block, encoding="utf-8")
            try:
                proc = subprocess.run(  # noqa: S603 - deliberate, see docstring
                    [sys.executable, "-I", str(script)],
                    cwd=tmp,
                    capture_output=True,
                    timeout=timeout_s,
                    check=False,
                )
                ok += proc.returncode == 0
            except subprocess.TimeoutExpired:
                pass
    return ok / len(blocks)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1 over code-aware tokens (identifiers, numbers, punctuation)."""
    a, b = _tokens(candidate), _tokens(reference)
    if not a or not b:
        return 0.0
    # LCS length with O(min(n, m)) memory.
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(_tokens(candidate)), lcs / len(_tokens(reference))
    return 2 * precision * recall / (precision + recall)


def score_answer(item: EvalItem, answer: str, *, execute_code: bool = False) -> ItemResult:
    recall, missing = keyword_recall(answer, item.must_include)
    reference_has_code = bool(extract_code_blocks(item.output))
    valid = code_validity(answer) if reference_has_code else None
    runs = code_execution(answer) if (execute_code and reference_has_code) else None
    rl = rouge_l(answer, item.output)
    parts = [v for v in (recall, valid, rl) if v is not None]
    return ItemResult(
        id=item.id,
        category=item.category,
        keyword_recall=recall,
        code_valid=valid,
        code_runs=runs,
        rouge_l=rl,
        score=sum(parts) / len(parts),
        latency_s=0.0,
        answer=answer,
        missing=missing,
    )


# -- running -------------------------------------------------------------------------
def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return round(statistics.fmean(present), 4) if present else None


def summarize(results: list[ItemResult], meta: dict) -> dict:
    categories: dict[str, list[ItemResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)
    latencies = sorted(r.latency_s for r in results)
    return {
        "meta": meta,
        "summary": {
            "items": len(results),
            "score": _mean([r.score for r in results]),
            "keyword_recall": _mean([r.keyword_recall for r in results]),
            "code_valid": _mean([r.code_valid for r in results]),
            "code_runs": _mean([r.code_runs for r in results]),
            "rouge_l": _mean([r.rouge_l for r in results]),
            "latency_p50_s": round(latencies[len(latencies) // 2], 3) if latencies else None,
            "latency_p95_s": (
                round(latencies[max(0, math.ceil(len(latencies) * 0.95) - 1)], 3)
                if latencies
                else None
            ),
        },
        "by_category": {
            name: {"items": len(rs), "score": _mean([r.score for r in rs])}
            for name, rs in sorted(categories.items())
        },
        "items": [asdict(r) for r in results],
    }


def run_eval(
    engine: Engine,
    items: list[EvalItem],
    *,
    max_new_tokens: int = 256,
    execute_code: bool = False,
    meta: dict | None = None,
    progress: Callable[[int, int, ItemResult], None] | None = None,
    retriever: Retriever | None = None,
    rag_top_k: int = 3,
    rag_max_chars: int = 1500,
) -> dict:
    """Generate an answer for every item with greedy decoding and score it.

    With ``retriever`` set, retrieved documentation is attached exactly as the
    service does at serving time, so the effect of RAG is measured, not assumed.
    """
    params = GenerationParams(max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
    engine.load()
    results: list[ItemResult] = []
    for i, item in enumerate(items, start=1):
        started = time.perf_counter()
        context = item.input or None
        if retriever is not None:
            docs = format_context(retriever.retrieve(item.instruction, rag_top_k), rag_max_chars)
            context = "\n\n".join(part for part in (item.input, docs) if part) or None
        generated = engine.generate(build_messages(item.instruction, context), params)
        result = score_answer(item, generated.text, execute_code=execute_code)
        result.latency_s = round(time.perf_counter() - started, 3)
        results.append(result)
        if progress:
            progress(i, len(items), result)
    return summarize(
        results,
        {
            "backend": engine.backend,
            "model": engine.model_id,
            "uses_adapter": engine.uses_adapter,
            "max_new_tokens": max_new_tokens,
            "retrieval": retriever is not None,
            **(meta or {}),
        },
    )


# -- gating --------------------------------------------------------------------------
@dataclass
class GateResult:
    passed: bool
    reasons: list[str]


def gate(
    report: dict,
    baseline: dict | None = None,
    *,
    max_regression: float = 0.02,
    min_score: float | None = None,
) -> GateResult:
    """Decide whether ``report`` is good enough to release.

    Fails if the composite score is below ``min_score``, or more than
    ``max_regression`` below the baseline's score.
    """
    reasons: list[str] = []
    score = report["summary"]["score"] or 0.0
    if min_score is not None and score < min_score:
        reasons.append(f"score {score:.4f} is below the minimum {min_score:.4f}")
    if baseline is not None:
        base = baseline["summary"]["score"] or 0.0
        if score < base - max_regression:
            reasons.append(
                f"score {score:.4f} regressed more than {max_regression:.4f} "
                f"from baseline {base:.4f}"
            )
    return GateResult(passed=not reasons, reasons=reasons)


def render_markdown(report: dict, baseline: dict | None = None) -> str:
    s = report["summary"]
    lines = [
        "| metric | value | baseline |",
        "| --- | --- | --- |",
    ]
    b = (baseline or {}).get("summary", {})
    for key in (
        "score",
        "keyword_recall",
        "code_valid",
        "code_runs",
        "rouge_l",
        "latency_p50_s",
        "latency_p95_s",
    ):
        lines.append(f"| {key} | {s.get(key)} | {b.get(key, '-')} |")
    lines += ["", "| category | items | score |", "| --- | --- | --- |"]
    for name, cat in report["by_category"].items():
        lines.append(f"| {name} | {cat['items']} | {cat['score']} |")
    return "\n".join(lines)
