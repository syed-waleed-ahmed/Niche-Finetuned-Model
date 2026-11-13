"""Request limits, context-window budgeting, error typing, headers, and new metrics."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from fastapi_assistant.api import create_app
from fastapi_assistant.config import Settings
from fastapi_assistant.engines.base import (
    MIN_COMPLETION_TOKENS,
    GenerationParams,
    Message,
    PromptTooLongError,
)
from fastapi_assistant.engines.llamacpp_engine import LlamaCppEngine, _context_errors

from .conftest import FakeEngine


class WindowedEngine(FakeEngine):
    """FakeEngine with a context window; one token per whitespace-separated word."""

    def __init__(self, window: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.window = window

    def context_window(self) -> int:
        return self.window

    def count_prompt_tokens(self, messages: Sequence[Message]) -> int:
        return sum(len(m["content"].split()) for m in messages)


def _client(make_settings, engine, **overrides) -> TestClient:
    return TestClient(create_app(settings=make_settings(**overrides), engine=engine))


# -- context window ---------------------------------------------------------------


def test_fit_clamps_answer_budget_to_remaining_context():
    engine = WindowedEngine(window=100)
    params = engine.fit([{"role": "user", "content": "word " * 60}], GenerationParams(256))
    assert params.max_new_tokens == 40


def test_fit_rejects_prompt_without_room_for_an_answer():
    engine = WindowedEngine(window=100)
    words = 100 - MIN_COMPLETION_TOKENS + 1
    with pytest.raises(PromptTooLongError) as info:
        engine.fit([{"role": "user", "content": "w " * words}], GenerationParams(16))
    assert info.value.prompt_tokens == words
    assert info.value.context_window == 100


def test_fit_is_a_no_op_when_the_window_is_unknown():
    params = GenerationParams(256)
    assert FakeEngine().fit([{"role": "user", "content": "x " * 10_000}], params) is params


def test_generate_prompt_too_long_is_400_not_500(make_settings):
    engine = WindowedEngine(window=64)
    with _client(make_settings, engine) as c:
        r = c.post("/generate", json={"question": "tok " * 200})
    assert r.status_code == 400
    assert "context window" in r.json()["detail"]
    assert engine.calls == []  # never reached the model


def test_chat_prompt_too_long_uses_openai_error_shape(make_settings):
    with _client(make_settings, WindowedEngine(window=64)) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "tok " * 200}]},
        )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["code"] == "context_length_exceeded"


def test_streaming_prompt_too_long_fails_before_the_stream_starts(make_settings):
    with _client(make_settings, WindowedEngine(window=64)) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "tok " * 200}], "stream": True},
        )
        native = c.post("/generate", json={"question": "tok " * 200, "stream": True})
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("application/json")
    assert native.status_code == 400


def test_clamped_budget_reaches_the_engine(make_settings):
    engine = WindowedEngine(window=120)
    with _client(make_settings, engine) as c:
        r = c.post("/generate", json={"question": "tok " * 50, "max_new_tokens": 500})
    assert r.status_code == 200
    # build_messages adds a system prompt; whatever remains is the budget.
    assert engine.calls[0]["params"].max_new_tokens < 120 - 50


def test_prompt_rejections_are_counted(make_settings):
    with _client(make_settings, WindowedEngine(window=64)) as c:
        c.post("/generate", json={"question": "tok " * 200})
        body = c.get("/metrics").text
    assert "assistant_prompt_too_long_total 1.0" in body


def test_llamacpp_context_overflow_is_translated():
    def chunks():
        yield {"choices": [{}]}
        raise ValueError("Requested tokens (2300) exceed context window of 2048")

    gen = _context_errors(chunks(), 2048)
    next(gen)
    with pytest.raises(PromptTooLongError) as info:
        next(gen)
    assert (info.value.prompt_tokens, info.value.context_window) == (2300, 2048)


class _FakeLlama:
    """Stands in for llama_cpp.Llama: records the call, streams two chunks."""

    n_tokens = 12

    def __init__(self):
        self.kwargs = {}

    def create_chat_completion(self, **kwargs):
        self.kwargs = kwargs
        yield {"choices": [{"delta": {"content": "Use status_code=201."}}]}
        yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}


def test_llamacpp_stops_at_the_template_role_markers(monkeypatch):
    # The fine-tuned model sometimes skips end-of-sequence and writes an invented
    # "<|user|>" turn; the engine must ask llama.cpp to stop there.
    engine = LlamaCppEngine(Settings(_env_file=None, engine_backend="llamacpp"))
    fake = _FakeLlama()
    monkeypatch.setattr(engine, "_load_impl", lambda: setattr(engine, "_llm", fake))
    events = list(engine.stream([{"role": "user", "content": "q"}], GenerationParams()))
    assert "".join(e.delta for e in events) == "Use status_code=201."
    assert fake.kwargs["stop"] == ["<|user|>", "<|system|>", "<|assistant|>"]
    assert events[-1].finish_reason == "stop"


def test_llamacpp_other_value_errors_pass_through():
    def chunks():
        raise ValueError("something else")
        yield  # pragma: no cover

    with pytest.raises(ValueError, match="something else"):
        list(_context_errors(chunks(), 2048))


# -- body size limit -------------------------------------------------------------


def test_declared_oversized_body_is_413(make_settings):
    with _client(make_settings, FakeEngine(), max_request_bytes=1024) as c:
        r = c.post(
            "/generate",
            content=json.dumps({"question": "hi", "pad": "a" * 5000}),
            headers={"content-type": "application/json"},
        )
    assert r.status_code == 413
    assert "larger than 1024 bytes" in r.json()["detail"]


def test_chunked_oversized_body_is_413(make_settings):
    def body():
        yield b'{"question": "hi", "pad": "'
        for _ in range(10):
            yield b"a" * 512
        yield b'"}'

    with _client(make_settings, FakeEngine(), max_request_bytes=1024) as c:
        r = c.post(
            "/v1/chat/completions", content=body(), headers={"content-type": "application/json"}
        )
    assert r.status_code == 413
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_body_limit_runs_before_authentication(make_settings):
    app = create_app(
        settings=make_settings(max_request_bytes=1024, api_key="secret-key"), engine=FakeEngine()
    )
    with TestClient(app) as c:
        r = c.post(
            "/generate",
            content=json.dumps({"question": "hi", "pad": "a" * 5000}),
            headers={"content-type": "application/json"},
        )
    assert r.status_code == 413


def test_bodies_under_the_limit_are_accepted(make_settings):
    with _client(make_settings, FakeEngine(), max_request_bytes=1024) as c:
        r = c.post("/generate", json={"question": "hi"})
    assert r.status_code == 200


def test_body_limit_can_be_disabled(make_settings):
    with _client(make_settings, FakeEngine(), max_request_bytes=0) as c:
        r = c.post("/generate", json={"question": "hi", "pad": "a" * 200_000})
    assert r.status_code == 200


# -- error typing and headers ------------------------------------------------------


def test_server_errors_are_typed_server_error(make_settings):
    with _client(make_settings, FakeEngine(fail_generate=True)) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "server_error"


def test_security_headers_on_every_response(make_settings):
    with _client(make_settings, FakeEngine(), origin_shared_secret="s3cret") as c:
        ok = c.get("/health")
        blocked = c.post("/generate", json={"question": "hi"})
    for r in (ok, blocked):
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "no-referrer"
    assert blocked.status_code == 403


# -- metrics ----------------------------------------------------------------------


def test_queue_wait_and_ttft_are_recorded(client):
    client.post("/generate", json={"question": "hello", "stream": True})
    body = client.get("/metrics").text
    assert "assistant_queue_wait_seconds_count 1.0" in body
    assert 'assistant_time_to_first_token_seconds_count{endpoint="/generate"} 1.0' in body


# -- model integrity -----------------------------------------------------------------


def test_sha256_verification_accepts_matching_file(tmp_path):
    from fastapi_assistant.hub import sha256_file, verify_sha256

    f = tmp_path / "model.gguf"
    f.write_bytes(b"weights")
    verify_sha256(f, sha256_file(f))  # no exception
    verify_sha256(f, None)  # unset: skipped


def test_sha256_mismatch_fails_closed(tmp_path):
    from fastapi_assistant.hub import verify_sha256

    f = tmp_path / "model.gguf"
    f.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_sha256(f, "0" * 64)


def test_local_gguf_path_is_verified(tmp_path, make_settings):
    from fastapi_assistant.hub import resolve_gguf_path

    f = tmp_path / "model.gguf"
    f.write_bytes(b"weights")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        resolve_gguf_path(make_settings(gguf_path=f, gguf_sha256="a" * 64))


@pytest.mark.parametrize("bad", ["abc", "z" * 64, "a" * 63])
def test_gguf_sha256_setting_is_validated(make_settings, bad):
    with pytest.raises(ValueError):
        make_settings(gguf_sha256=bad)


def test_gguf_sha256_setting_is_normalised(make_settings):
    assert make_settings(gguf_sha256="A" * 64).gguf_sha256 == "a" * 64
    assert make_settings(gguf_sha256="").gguf_sha256 is None


def test_shipped_manifest_pins_a_checksum():
    import json as _json
    from pathlib import Path

    from fastapi_assistant.export import manifest_env

    manifest = _json.loads(Path("model-release.json").read_text(encoding="utf-8"))
    env = manifest_env(manifest)
    assert len(env["GGUF_SHA256"]) == 64


# -- training never sees the held-out eval set ---------------------------------------


def test_checkpoint_selection_uses_a_slice_of_the_training_file(tmp_path, make_settings):
    pytest.importorskip("datasets")
    from fastapi_assistant.data import load_raw_dataset

    train, held_out = tmp_path / "fastapi_qa_train.jsonl", tmp_path / "fastapi_qa_eval.jsonl"
    train.write_text(
        "\n".join(
            json.dumps({"instruction": f"train q{i}", "input": "", "output": f"a{i}"})
            for i in range(40)
        ),
        encoding="utf-8",
    )
    held_out.write_text(
        json.dumps({"instruction": "HELD OUT", "input": "", "output": "x"}), encoding="utf-8"
    )
    settings = make_settings(data_dir=tmp_path)
    ds = load_raw_dataset(settings)
    assert set(ds) == {"train", "validation"}
    assert len(ds["train"]) + len(ds["validation"]) == 40
    seen = {row["instruction"] for split in ds.values() for row in split}
    assert "HELD OUT" not in seen
    assert seen == {f"train q{i}" for i in range(40)}


def test_validation_can_be_disabled(tmp_path, make_settings):
    pytest.importorskip("datasets")
    from fastapi_assistant.data import load_raw_dataset

    (tmp_path / "fastapi_qa_train.jsonl").write_text(
        "\n".join(
            json.dumps({"instruction": f"q{i}", "input": "", "output": "a"}) for i in range(20)
        ),
        encoding="utf-8",
    )
    settings = make_settings(data_dir=tmp_path, validation_fraction=0.0)
    assert set(load_raw_dataset(settings)) == {"train"}


# -- configuration and logging safety ------------------------------------------------


@pytest.mark.parametrize("placeholder", ["disabled", "None", "null", "change-me"])
def test_api_key_placeholder_is_rejected_not_silently_disabled(make_settings, placeholder):
    with pytest.raises(ValueError, match="placeholder"):
        make_settings(api_key=placeholder)


def test_empty_api_key_means_unset(make_settings):
    assert make_settings(api_key="").api_key is None


def test_open_origin_logs_a_warning(make_settings, caplog):
    with caplog.at_level("WARNING", logger="fastapi_assistant.api"):
        create_app(settings=make_settings(), engine=FakeEngine())
    assert any("Authentication is DISABLED" in r.getMessage() for r in caplog.records)


def test_json_logs_carry_cloud_logging_severity():
    import logging

    from fastapi_assistant.logging_config import JsonFormatter

    record = logging.LogRecord("x", logging.ERROR, __file__, 1, "boom", None, None)
    payload = json.loads(JsonFormatter().format(record))
    assert payload["severity"] == "ERROR"
    assert payload["level"] == "ERROR"


@pytest.mark.parametrize("kind", ["adapter", "gguf"])
def test_published_model_card_carries_apache_attribution(tmp_path, make_settings, kind):
    from fastapi_assistant.export import write_model_card

    card = write_model_card(tmp_path, settings=make_settings(), version="v1", kind=kind)
    text = card.read_text(encoding="utf-8")
    assert "license: apache-2.0" in text
    assert "Apache License 2.0" in text and "modified work" in text
    notice = (tmp_path / "NOTICE").read_text(encoding="utf-8")
    assert "TinyLlama/TinyLlama-1.1B-Chat-v1.0" in notice
    assert "Modifications:" in notice


@pytest.mark.parametrize("kind", ["adapter", "gguf"])
def test_published_model_card_shows_usage_comparison_and_limits(tmp_path, make_settings, kind):
    from fastapi_assistant.export import write_model_card
    from fastapi_assistant.prompts import STOP_SEQUENCES, SYSTEM_PROMPT

    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"")
    report = {
        "meta": {"backend": "transformers"},
        "summary": {"items": 108, "score": 0.538, "code_valid": 0.8365, "code_runs": None},
    }
    base = {"summary": {"items": 108, "score": 0.3834, "code_valid": 0.4}}
    card = write_model_card(
        tmp_path,
        settings=make_settings(),
        version="v2026.11.13-abc",
        kind=kind,
        eval_report=report,
        base_report=base,
        repo_id="someone/fastapi-assistant-x",
    ).read_text(encoding="utf-8")

    # The usage example pins this release and stops at the template's role markers.
    assert '"someone/fastapi-assistant-x"' in card and 'revision="v2026.11.13-abc"' in card
    assert ("model-Q4_K_M.gguf" in card) == (kind == "gguf")
    assert ("PeftModel.from_pretrained" in card) == (kind == "adapter")
    assert SYSTEM_PROMPT in card
    assert all(marker in card for marker in STOP_SEQUENCES)
    # Both models, scored on the same set, side by side; unmeasured metrics are left out.
    assert "| Metric | This release | Base model |" in card
    assert "| Composite score | 0.538 | 0.383 |" in card
    assert "code_runs" not in card and "Code runs" not in card
    assert "unquantized model on the training GPU" in card
    assert "Latency" not in card
    assert "often wrong" in card and "/blob/main/MODEL_CARD.md" in card


def test_published_model_card_without_reports_has_no_evaluation(tmp_path, make_settings):
    from fastapi_assistant.export import write_model_card

    card = write_model_card(tmp_path, settings=make_settings(), version="v1", kind="gguf")
    text = card.read_text(encoding="utf-8")
    assert "## Evaluation" not in text
    assert "## Usage" in text and "<namespace>/fastapi-assistant-gguf" in text


def test_version_is_consistent_across_package_metadata():
    import re
    from pathlib import Path

    import fastapi_assistant

    root = Path(__file__).resolve().parent.parent
    # tomllib is 3.11+, and CI also runs 3.10, so read the field with a regex.
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    assert fastapi_assistant.__version__ == version
    assert re.search(rf"^version: {re.escape(version)}$", citation, re.M)
    assert f"## [{version}]" in (root / "CHANGELOG.md").read_text(encoding="utf-8")
