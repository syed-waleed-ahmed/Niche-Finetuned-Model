"""Evaluation metrics, harness, and release gate."""

from __future__ import annotations

import json

import pytest

from fastapi_assistant.evaluation import (
    EvalItem,
    code_validity,
    extract_code_blocks,
    gate,
    keyword_recall,
    load_eval_items,
    render_markdown,
    rouge_l,
    run_eval,
    score_answer,
)
from tests.conftest import FakeEngine

REFERENCE = (
    "Raise `HTTPException`:\n\n```python\nfrom fastapi import FastAPI, HTTPException\n\n"
    "app = FastAPI()\n\n@app.get('/items/{item_id}')\ndef read(item_id: int):\n"
    "    raise HTTPException(status_code=404, detail='Not found')\n```\n"
)


def test_extract_code_blocks():
    assert len(extract_code_blocks(REFERENCE)) == 1
    assert extract_code_blocks("no code") == []
    assert extract_code_blocks("```py\nx = 1\n```\n```\ny = 2\n```") == ["x = 1\n", "y = 2\n"]


def test_keyword_recall_is_case_insensitive():
    recall, missing = keyword_recall(
        "use httpexception with STATUS_CODE=404", ["HTTPException", "status_code=404", "Depends"]
    )
    assert recall == pytest.approx(2 / 3)
    assert missing == ["Depends"]
    assert keyword_recall("anything", []) == (None, [])


def test_code_validity():
    assert code_validity(REFERENCE) == 1.0
    assert code_validity("```python\ndef broken(:\n```") == 0.0
    assert code_validity("prose only") == 0.0


def test_rouge_l_bounds():
    assert rouge_l(REFERENCE, REFERENCE) == pytest.approx(1.0)
    assert rouge_l("completely unrelated words", REFERENCE) < 0.1
    assert rouge_l("", REFERENCE) == 0.0
    partial = rouge_l("raise HTTPException(status_code=404)", REFERENCE)
    assert 0.0 < partial < 1.0


def test_score_answer_perfect_and_poor():
    item = EvalItem(
        id="x", instruction="q", output=REFERENCE, must_include=["HTTPException", "status_code=404"]
    )
    perfect = score_answer(item, REFERENCE)
    assert perfect.score == pytest.approx(1.0)
    poor = score_answer(item, "I am not sure.")
    assert poor.score < 0.2
    assert poor.code_valid == 0.0
    assert poor.missing == ["HTTPException", "status_code=404"]


def test_load_and_run_eval(tmp_path):
    path = tmp_path / "eval.jsonl"
    rows = [
        {
            "id": "a",
            "category": "errors",
            "instruction": "q1",
            "input": "",
            "output": "echo: q1",
            "must_include": ["echo"],
        },
        {"instruction": "q2", "output": "something else", "must_include": ["missing"]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    items = load_eval_items(path)
    assert [i.id for i in items] == ["a", "item-0001"]
    assert load_eval_items(path, limit=1)[0].id == "a"

    seen = []
    report = run_eval(FakeEngine(), items, progress=lambda i, n, r: seen.append(i))
    assert seen == [1, 2]
    assert report["summary"]["items"] == 2
    assert report["items"][0]["score"] == pytest.approx(1.0)
    assert report["by_category"]["errors"]["items"] == 1
    assert report["meta"]["backend"] == "fake"
    assert "| score |" in render_markdown(report)


def test_eval_uses_greedy_decoding(tmp_path):
    engine = FakeEngine()
    run_eval(engine, [EvalItem(id="a", instruction="q", output="r")])
    assert engine.calls[0]["params"].temperature == 0.0


def _report(score: float) -> dict:
    return {"summary": {"score": score}}


def test_gate():
    assert gate(_report(0.5)).passed
    assert not gate(_report(0.5), min_score=0.6).passed
    assert gate(_report(0.49), _report(0.5), max_regression=0.02).passed
    failed = gate(_report(0.40), _report(0.5), max_regression=0.02)
    assert not failed.passed and "regressed" in failed.reasons[0]


def test_eval_with_retrieval_attaches_docs_like_serving():
    from fastapi_assistant.retrieval import Passage

    class OneDoc:
        def retrieve(self, query, k):
            return [Passage("https://d/x/", "Status codes", "Use status_code=201.", 0.9)]

    engine = FakeEngine()
    item = EvalItem(id="a", instruction="q?", output="r", input="code ctx")
    report = run_eval(engine, [item], retriever=OneDoc())
    user = engine.calls[0]["messages"][-1]["content"]
    assert user == "q?\n\nContext:\ncode ctx\n\n[1] Status codes\nUse status_code=201."
    assert report["meta"]["retrieval"] is True
