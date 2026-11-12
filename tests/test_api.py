"""HTTP API tests for the native endpoints, using a fake engine."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from fastapi_assistant.api import create_app
from tests.conftest import FakeEngine


def _sse_events(text: str) -> list[tuple[str | None, str]]:
    events = []
    for block in text.strip().split("\n\n"):
        name, data = None, ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        events.append((name, data))
    return events


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["uses_adapter"] is True
    assert body["backend"] == "fake"


def test_ready_ok(client):
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


def test_ready_reports_unready_model(make_settings):
    engine = FakeEngine(loaded=False, fail_load=True)
    app = create_app(settings=make_settings(), engine=engine)
    with TestClient(app) as client:
        resp = client.get("/ready")
    assert resp.status_code == 503
    assert "not ready" in resp.json()["detail"].lower()


def test_ready_turns_ready_after_background_load(make_settings):
    engine = FakeEngine(loaded=False)
    app = create_app(settings=make_settings(), engine=engine)
    with TestClient(app) as client:
        client.get("/ready")  # kicks off the background load
        for _ in range(50):
            if client.get("/ready").status_code == 200:
                break
        assert client.get("/ready").status_code == 200


def test_generate_returns_answer(client):
    resp = client.post("/generate", json={"question": "How do I define a POST endpoint?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "echo: How do I define a POST endpoint?"
    assert body["question"] == "How do I define a POST endpoint?"
    assert body["uses_adapter"] is True
    assert body["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 7
    assert body["request_id"] == resp.headers["x-request-id"]
    assert "x-process-time-ms" in resp.headers


def test_request_id_is_propagated(client):
    resp = client.post("/generate", json={"question": "hi"}, headers={"X-Request-ID": "abc123"})
    assert resp.headers["x-request-id"] == "abc123"
    assert resp.json()["request_id"] == "abc123"


def test_generate_forwards_generation_params(make_settings):
    engine = FakeEngine()
    app = create_app(settings=make_settings(), engine=engine)
    with TestClient(app) as client:
        client.post(
            "/generate",
            json={"question": "hi", "max_new_tokens": 32, "temperature": 0.1, "top_p": 0.5},
        )
    params = engine.calls[0]["params"]
    assert params.max_new_tokens == 32
    assert params.temperature == 0.1
    assert params.top_p == 0.5


def test_generate_uses_shared_prompt_format(client, fake_engine):
    client.post("/generate", json={"question": "  spaced  "})
    messages = fake_engine.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"] == "spaced"


def test_generate_stream_sse(client):
    with client.stream("POST", "/generate", json={"question": "stream me", "stream": True}) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        text = "".join(resp.iter_text())
    events = _sse_events(text)
    tokens = "".join(json.loads(d)["delta"] for name, d in events if name == "token")
    assert tokens == "echo: stream me"
    name, data = events[-1]
    assert name == "done"
    done = json.loads(data)
    assert done["finish_reason"] == "stop"
    assert done["usage"]["completion_tokens"] == 3


def test_generate_stream_reports_errors_in_band(make_settings):
    app = create_app(settings=make_settings(), engine=FakeEngine(fail_generate=True))
    with TestClient(app) as client:
        resp = client.post("/generate", json={"question": "x", "stream": True})
    assert resp.status_code == 200
    assert _sse_events(resp.text)[-1][0] == "error"


def test_generate_failure_returns_500_without_leaking(make_settings):
    app = create_app(settings=make_settings(), engine=FakeEngine(fail_generate=True))
    with TestClient(app) as client:
        resp = client.post("/generate", json={"question": "x"})
    assert resp.status_code == 500
    assert resp.json() == {"detail": "Generation failed"}


def test_generate_model_not_ready_returns_503(make_settings):
    app = create_app(settings=make_settings(), engine=FakeEngine(loaded=False, fail_load=True))
    with TestClient(app) as client:
        resp = client.post("/generate", json={"question": "x"})
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "30"


def test_generate_validation_error(client):
    resp = client.post("/generate", json={"question": ""})  # violates min_length
    assert resp.status_code == 422


def test_generate_rejects_out_of_range_tokens(client):
    resp = client.post("/generate", json={"question": "hi", "max_new_tokens": 100000})
    assert resp.status_code == 422


def test_api_key_required_when_configured(make_settings, fake_engine):
    app = create_app(settings=make_settings(api_key="s3cret"), engine=fake_engine)
    with TestClient(app) as client:
        assert client.post("/generate", json={"question": "hi"}).status_code == 401
        assert (
            client.post(
                "/generate", json={"question": "hi"}, headers={"X-API-Key": "nope"}
            ).status_code
            == 401
        )
        ok = client.post("/generate", json={"question": "hi"}, headers={"X-API-Key": "s3cret"})
        assert ok.status_code == 200
        bearer = client.post(
            "/generate", json={"question": "hi"}, headers={"Authorization": "Bearer s3cret"}
        )
        assert bearer.status_code == 200
        # Health remains open even when an API key is configured.
        assert client.get("/health").status_code == 200


def test_origin_secret_blocks_direct_access(make_settings, fake_engine):
    app = create_app(settings=make_settings(origin_shared_secret="edge-only"), engine=fake_engine)
    with TestClient(app) as client:
        assert client.post("/generate", json={"question": "hi"}).status_code == 403
        assert client.get("/docs").status_code == 403
        ok = client.post(
            "/generate", json={"question": "hi"}, headers={"X-Origin-Secret": "edge-only"}
        )
        assert ok.status_code == 200
        # Platform probes stay reachable.
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200


def test_cors_does_not_allow_credentials(client):
    resp = client.options(
        "/generate",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert resp.headers.get("access-control-allow-credentials") != "true"


def test_metrics_endpoint(client):
    client.post("/generate", json={"question": "hi"})
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert 'assistant_http_requests_total{method="POST",route="/generate",status="200"}' in text
    assert "assistant_completion_tokens_total 2.0" in text
    assert "assistant_model_loaded 1.0" in text
    assert "assistant_queue_depth 0.0" in text


def test_metrics_token(make_settings, fake_engine):
    app = create_app(settings=make_settings(metrics_token="m3trics"), engine=fake_engine)
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 401
        ok = client.get("/metrics", headers={"Authorization": "Bearer m3trics"})
        assert ok.status_code == 200


def test_metrics_can_be_disabled(make_settings, fake_engine):
    app = create_app(settings=make_settings(enable_metrics=False), engine=fake_engine)
    with TestClient(app) as client:
        assert client.get("/metrics").status_code == 404
