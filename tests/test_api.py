"""HTTP API tests using a fake engine (no model or network required)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fastapi_assistant.api import create_app
from tests.conftest import FakeEngine


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["uses_adapter"] is True


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


def test_generate_returns_answer(client):
    resp = client.post("/generate", json={"question": "How do I define a POST endpoint?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "echo: How do I define a POST endpoint?"
    assert body["question"] == "How do I define a POST endpoint?"
    assert body["uses_adapter"] is True
    assert "x-request-id" in resp.headers
    assert "x-process-time-ms" in resp.headers


def test_generate_forwards_generation_params(make_settings):
    engine = FakeEngine()
    app = create_app(settings=make_settings(), engine=engine)
    with TestClient(app) as client:
        client.post(
            "/generate",
            json={"question": "hi", "max_new_tokens": 32, "temperature": 0.1, "top_p": 0.5},
        )
    assert engine.calls[0]["max_new_tokens"] == 32
    assert engine.calls[0]["temperature"] == 0.1
    assert engine.calls[0]["top_p"] == 0.5


def test_generate_validation_error(client):
    resp = client.post("/generate", json={"question": ""})  # violates min_length
    assert resp.status_code == 422


def test_generate_rejects_out_of_range_tokens(client):
    resp = client.post("/generate", json={"question": "hi", "max_new_tokens": 100000})
    assert resp.status_code == 422


def test_api_key_required_when_configured(make_settings, fake_engine):
    app = create_app(settings=make_settings(api_key="s3cret"), engine=fake_engine)
    with TestClient(app) as client:
        # Missing key -> 401
        assert client.post("/generate", json={"question": "hi"}).status_code == 401
        # Wrong key -> 401
        assert (
            client.post(
                "/generate", json={"question": "hi"}, headers={"X-API-Key": "nope"}
            ).status_code
            == 401
        )
        # Correct key -> 200
        ok = client.post(
            "/generate", json={"question": "hi"}, headers={"X-API-Key": "s3cret"}
        )
        assert ok.status_code == 200

    # Health remains open even when an API key is configured.
    app2 = create_app(settings=make_settings(api_key="s3cret"), engine=fake_engine)
    with TestClient(app2) as client:
        assert client.get("/health").status_code == 200
