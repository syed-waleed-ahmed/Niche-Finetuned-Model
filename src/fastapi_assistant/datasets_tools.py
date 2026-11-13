"""Dataset quality tooling: validation, leakage checks, statistics, shard merging.

A fine-tune is only as good as its data, and an eval set is only meaningful if it
does not leak into training. ``validate`` enforces both:

* schema and length limits for every row;
* every fenced Python block parses;
* eval rows carry ``must_include`` identifiers that their own reference contains;
* no exact duplicates, and no near-duplicate question or answer shared between
  train and eval (contamination).
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .evaluation import extract_code_blocks

MAX_OUTPUT_CHARS = 2000
SOFT_OUTPUT_CHARS = 1400
_WORD = re.compile(r"[a-z0-9_]+")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i in is it my of on or should "
    "the this to use using what when where which why with you your me fastapi".split()
)


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def row_key(row: dict) -> str:
    return normalize(row.get("instruction", "")) + "||" + normalize(row.get("input", ""))


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS)


def shingles(text: str, n: int = 5) -> frozenset[str]:
    words = _WORD.findall(text.lower())
    if len(words) < n:
        return frozenset([" ".join(words)]) if words else frozenset()
    return frozenset(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def near_duplicate_pairs(
    left: list[frozenset[str]],
    right: list[frozenset[str]],
    threshold: float,
    *,
    same: bool = False,
    max_df: float = 0.05,
) -> list[tuple[int, int, float]]:
    """Pairs (i, j, similarity) with Jaccard >= threshold.

    Uses an inverted index over rare features so the work is roughly linear in the
    corpus size instead of quadratic.
    """
    index: dict[str, list[int]] = defaultdict(list)
    for j, feats in enumerate(right):
        for f in feats:
            index[f].append(j)
    df_cap = max(5, int(len(right) * max_df))
    pairs: list[tuple[int, int, float]] = []
    for i, feats in enumerate(left):
        candidates: set[int] = set()
        for f in feats:
            postings = index.get(f, ())
            if len(postings) <= df_cap:
                candidates.update(postings)
        for j in candidates:
            if same and j <= i:
                continue
            sim = jaccard(feats, right[j])
            if sim >= threshold:
                pairs.append((i, j, round(sim, 3)))
    return pairs


def read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: invalid JSON ({exc})") from exc
    return rows


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        lines = [f"rows: {self.counts}"]
        lines += [f"ERROR   {e}" for e in self.errors]
        lines += [f"warning {w}" for w in self.warnings]
        lines.append("OK" if self.ok else f"FAILED with {len(self.errors)} error(s)")
        return "\n".join(lines)


def _check_row(row: dict, where: str, is_eval: bool, report: ValidationReport) -> None:
    for name in ("instruction", "input", "output"):
        if not isinstance(row.get(name, None), str):
            report.errors.append(f"{where}: field {name!r} must be a string")
            return
    if not row["instruction"].strip() or not row["output"].strip():
        report.errors.append(f"{where}: empty instruction or output")
    length = len(row["output"])
    if length > MAX_OUTPUT_CHARS:
        report.errors.append(f"{where}: output is {length} chars (max {MAX_OUTPUT_CHARS})")
    elif length > SOFT_OUTPUT_CHARS:
        report.warnings.append(f"{where}: output is {length} chars (target <= {SOFT_OUTPUT_CHARS})")
    for k, block in enumerate(extract_code_blocks(row["output"])):
        if "```" in block:
            continue
        try:
            ast.parse(block)
        except SyntaxError as exc:
            report.errors.append(
                f"{where}: code block {k} does not parse ({exc.msg}, line {exc.lineno})"
            )
    if is_eval:
        keywords = row.get("must_include")
        if not isinstance(keywords, list) or not keywords:
            report.errors.append(f"{where}: eval rows need a non-empty must_include list")
        else:
            lowered = row["output"].lower()
            for kw in keywords:
                if kw.lower() not in lowered:
                    report.errors.append(f"{where}: reference answer lacks must_include {kw!r}")


def validate(
    train_path: Path, eval_path: Path, *, near_dup_threshold: float = 0.7
) -> ValidationReport:
    report = ValidationReport()
    train, evals = read_rows(train_path), read_rows(eval_path)
    report.counts = {"train": len(train), "eval": len(evals)}

    for i, row in enumerate(train):
        _check_row(row, f"train[{i}]", False, report)
    ids = Counter(row.get("id") for row in evals)
    for i, row in enumerate(evals):
        _check_row(row, f"eval[{i}]", True, report)
        if not row.get("id"):
            report.errors.append(f"eval[{i}]: missing id")
        elif ids[row["id"]] > 1:
            report.errors.append(f"eval[{i}]: duplicate id {row['id']!r}")

    # Exact duplicates.
    for name, rows in (("train", train), ("eval", evals)):
        seen: dict[str, int] = {}
        for i, row in enumerate(rows):
            key = row_key(row)
            if key in seen:
                report.errors.append(f"{name}[{i}] duplicates {name}[{seen[key]}]")
            else:
                seen[key] = i
    train_keys: dict[str, int] = {}
    for i, r in enumerate(train):
        train_keys.setdefault(row_key(r), i)
    for i, row in enumerate(evals):
        if row_key(row) in train_keys:
            report.errors.append(f"eval[{i}] also appears in train[{train_keys[row_key(row)]}]")

    # Near-duplicate leakage between eval and train (question or answer).
    q_train = [content_words(r.get("instruction", "") + " " + r.get("input", "")) for r in train]
    q_eval = [content_words(r.get("instruction", "") + " " + r.get("input", "")) for r in evals]
    for i, j, sim in near_duplicate_pairs(q_eval, q_train, near_dup_threshold):
        report.errors.append(
            f"eval[{i}] question is a near-duplicate of train[{j}] (jaccard {sim}): "
            f"{evals[i]['instruction']!r} ~ {train[j]['instruction']!r}"
        )
    a_train = [shingles(r.get("output", "")) for r in train]
    a_eval = [shingles(r.get("output", "")) for r in evals]
    for i, j, sim in near_duplicate_pairs(a_eval, a_train, 0.6):
        report.errors.append(f"eval[{i}] answer overlaps train[{j}] answer (jaccard {sim})")

    # Near-duplicates inside train are wasteful but not fatal.
    for i, j, sim in near_duplicate_pairs(q_train, q_train, 0.9, same=True):
        report.warnings.append(f"train[{i}] and train[{j}] questions are near-duplicates ({sim})")
    return report


def stats(train_path: Path, eval_path: Path) -> dict:
    def _describe(rows: list[dict]) -> dict:
        outputs = [len(r.get("output", "")) for r in rows]
        return {
            "rows": len(rows),
            "with_context": sum(1 for r in rows if r.get("input")),
            "with_code": sum(1 for r in rows if extract_code_blocks(r.get("output", ""))),
            "output_chars_mean": round(sum(outputs) / len(outputs)) if outputs else 0,
            "output_chars_max": max(outputs, default=0),
        }

    evals = read_rows(eval_path)
    return {
        "train": _describe(read_rows(train_path)),
        "eval": {
            **_describe(evals),
            "categories": dict(Counter(r.get("category", "general") for r in evals).most_common()),
        },
    }


def merge_shards(target: Path, shards: list[Path], *, split: str) -> tuple[int, int]:
    """Append rows from ``shards`` to ``target``, skipping exact duplicates."""
    existing = read_rows(target) if target.exists() else []
    keys = {row_key(r) for r in existing}
    ids = {r.get("id") for r in existing}
    added: list[dict] = []
    skipped = 0
    for shard in shards:
        for row in read_rows(shard):
            key = row_key(row)
            if key in keys:
                skipped += 1
                continue
            keys.add(key)
            if split == "train":
                row = {k: row.get(k, "") for k in ("instruction", "input", "output")}
            else:
                if not row.get("id") or row["id"] in ids:
                    row["id"] = f"{shard.stem}-{len(existing) + len(added):04d}"
                ids.add(row["id"])
                row = {
                    "id": row["id"],
                    "category": row.get("category", "general"),
                    "instruction": row["instruction"],
                    "input": row.get("input", ""),
                    "output": row["output"],
                    "must_include": row.get("must_include", []),
                }
            added.append(row)
    with target.open("a", encoding="utf-8", newline="\n") as fh:
        for row in added:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(added), skipped
