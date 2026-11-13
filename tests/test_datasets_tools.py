"""Dataset validation, leakage detection, and shard merging.

Also validates the real dataset shipped in data/, so CI fails if a bad row lands.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi_assistant.config import Settings
from fastapi_assistant.datasets_tools import (
    jaccard,
    merge_shards,
    near_duplicate_pairs,
    shingles,
    stats,
    validate,
)

CODE = "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n```"


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _train(q: str, out: str = CODE) -> dict:
    return {"instruction": q, "input": "", "output": out}


def _eval(i: str, q: str, out: str = CODE, keys=("FastAPI",)) -> dict:
    return {
        "id": i,
        "category": "c",
        "instruction": q,
        "input": "",
        "output": out,
        "must_include": list(keys),
    }


def test_shipped_dataset_is_valid():
    settings = Settings(_env_file=None)
    report = validate(settings.train_path, settings.eval_path)
    assert report.ok, report.render()


def test_valid_dataset_passes(tmp_path):
    train = _write(tmp_path / "t.jsonl", [_train("How do I create an app object?")])
    evals = _write(
        tmp_path / "e.jsonl",
        [
            _eval(
                "e1",
                "Which class handles websocket routing?",
                "Use `WebSocket` from fastapi.",
                ["WebSocket"],
            )
        ],
    )
    assert validate(train, evals).ok


def test_detects_schema_and_code_errors(tmp_path):
    train = _write(
        tmp_path / "t.jsonl",
        [
            {"instruction": "x", "output": "y"},  # missing input
            _train("broken code?", "```python\ndef f(:\n```"),
            _train("too long?", "a" * 2500),
        ],
    )
    evals = _write(
        tmp_path / "e.jsonl",
        [_eval("e1", "q", "no keyword here", ["Depends"]), {**_eval("", "q2"), "must_include": []}],
    )
    report = validate(train, evals)
    text = "\n".join(report.errors)
    assert "field 'input' must be a string" in text
    assert "does not parse" in text
    assert "2500 chars" in text
    assert "lacks must_include 'Depends'" in text
    assert "non-empty must_include" in text
    assert "missing id" in text


def test_detects_duplicates_and_leakage(tmp_path):
    q = "How do I add a startup hook with lifespan events in my application?"
    train = _write(
        tmp_path / "t.jsonl",
        [
            _train(q),
            _train(q),
            _train("Totally different topic about uploads?", "Use UploadFile " * 30),
        ],
    )
    evals = _write(
        tmp_path / "e.jsonl",
        [
            _eval("e1", q),
            _eval("e2", "How do I add a startup hook with lifespan events in my app?"),
            _eval("e3", "Unrelated question on cookies?", "Use UploadFile " * 30, ["UploadFile"]),
        ],
    )
    report = validate(train, evals)
    text = "\n".join(report.errors)
    assert "train[1] duplicates train[0]" in text
    assert "eval[0] also appears in train[0]" in text
    assert "eval[1] question is a near-duplicate" in text
    assert "eval[2] answer overlaps train[2]" in text


def test_near_duplicate_pairs_and_jaccard():
    a = [frozenset({"x", "y", "z"}), frozenset({"p"})]
    b = [frozenset({"x", "y", "z", "w"}), frozenset({"q"})]
    assert near_duplicate_pairs(a, b, 0.7) == [(0, 0, 0.75)]
    assert jaccard(frozenset(), frozenset({"a"})) == 0.0
    assert shingles("one two", n=5) == frozenset({"one two"})


def test_merge_shards(tmp_path):
    target = _write(tmp_path / "train.jsonl", [_train("existing?")])
    shard = _write(
        tmp_path / "shard.jsonl",
        [_train("existing?"), {**_train("new one?"), "source_url": "u"}, _train("new one?")],
    )
    added, skipped = merge_shards(target, [shard], split="train")
    assert (added, skipped) == (1, 2)
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert rows[-1] == {"instruction": "new one?", "input": "", "output": CODE}

    eval_target = _write(tmp_path / "eval.jsonl", [_eval("dup", "first?")])
    eval_shard = _write(tmp_path / "es.jsonl", [_eval("dup", "second?")])
    merge_shards(eval_target, [eval_shard], split="eval")
    ids = [json.loads(line)["id"] for line in eval_target.read_text(encoding="utf-8").splitlines()]
    assert len(set(ids)) == 2


def test_stats(tmp_path):
    train = _write(tmp_path / "t.jsonl", [_train("a?"), {**_train("b?", "prose"), "input": "ctx"}])
    evals = _write(tmp_path / "e.jsonl", [_eval("e", "c?")])
    s = stats(train, evals)
    assert (
        s["train"]["rows"] == 2 and s["train"]["with_context"] == 1 and s["train"]["with_code"] == 1
    )
    assert s["eval"]["categories"] == {"c": 1}
