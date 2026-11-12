"""Tests for the OpenAI-compatible /v1 API."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from fastapi_assistant.api import create_app
from fastapi_assistant.prompts import SYSTEM_PROMPT
from fastapi_assistant.storage import MemoryStore
from tests.conftest import FakeEngine


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"][0]["id"] == "fastapi-assistant"


def test_chat_completion(client, fake_engine):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "anything",
            "messages": [
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "What is Depends?"},
            ],
            "max_tokens": 20,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {
        "role": "assistant",
        "content": "echo: What is Depends?",
    }
    assert body["choices"][0]["finish_reason"] == "stop"
    assert (
        body["usage"]["total_tokens"]
        == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )
    # The client system prompt is replaced by the trained one.
    messages = fake_engine.calls[0]["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert len([m for m in messages if m["role"] == "system"]) == 1
    assert fake_engine.calls[0]["params"].max_new_tokens == 20


def test_chat_completion_stream(client):
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hello there"}], "stream": True},
    ) as resp:
        assert resp.status_code == 200
        lines = [line for line in resp.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert content == "echo: hello there"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_chat_must_end_with_user(client):
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        },
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_chat_validation_errors_use_openai_shape(client):
    resp = client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_chat_trims_history(make_settings):
    engine = FakeEngine()
    app = create_app(settings=make_settings(max_chat_turns=3), engine=engine)
    history = []
    for i in range(5):
        history += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    history.append({"role": "user", "content": "final"})
    with TestClient(app) as client:
        assert client.post("/v1/chat/completions", json={"messages": history}).status_code == 200
    sent = engine.calls[0]["messages"]
    assert sent[1]["role"] == "user"  # never starts with an assistant turn
    assert sent[-1]["content"] == "final"
    assert len(sent) <= 4


def test_openai_auth_error_shape(make_settings, fake_engine):
    app = create_app(settings=make_settings(api_key="k"), engine=fake_engine)
    with TestClient(app) as client:
        resp = client.get("/v1/models")
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"
    assert resp.headers["www-authenticate"] == "Bearer"


def test_feedback_recorded(make_settings, fake_engine):
    store = MemoryStore()
    key, _ = store.create_key("alice")
    app = create_app(settings=make_settings(), engine=fake_engine, store=store)
    with TestClient(app) as client:
        resp = client.post(
            "/v1/feedback",
            json={"request_id": "r1", "rating": -1, "comment": "wrong import"},
            headers={"Authorization": f"Bearer {key}"},
        )
    assert resp.status_code == 200
    assert store.feedback[0].rating == -1
    assert store.feedback[0].key_id == 1


def test_feedback_rejects_bad_rating(client):
    assert client.post("/v1/feedback", json={"request_id": "r", "rating": 5}).status_code == 400
