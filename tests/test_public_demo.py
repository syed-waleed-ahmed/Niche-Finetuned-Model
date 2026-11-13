"""Public demo mode: keyless requests are anonymous, keys that are sent are still checked."""

from __future__ import annotations

from fastapi.testclient import TestClient

from fastapi_assistant.api import create_app
from fastapi_assistant.auth import Authenticator, AuthError

from .conftest import FakeEngine


def _client(make_settings, **overrides) -> TestClient:
    return TestClient(create_app(settings=make_settings(**overrides), engine=FakeEngine()))


def test_keyless_request_is_rejected_by_default(make_settings):
    with _client(make_settings, api_key="owner-key") as c:
        assert c.post("/generate", json={"question": "hi"}).status_code == 401
        assert c.get("/health").json()["access"] == "key"


def test_public_demo_accepts_keyless_requests(make_settings):
    with _client(make_settings, api_key="owner-key", public_demo=True) as c:
        r = c.post("/generate", json={"question": "hi"})
        assert r.status_code == 200
        assert r.json()["answer"].startswith("echo: ")
        chat = c.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
        )
        assert chat.status_code == 200
        assert c.get("/health").json()["access"] == "public"


def test_public_demo_still_rejects_a_wrong_key(make_settings):
    with _client(make_settings, api_key="owner-key", public_demo=True) as c:
        r = c.post("/generate", json={"question": "hi"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_public_demo_still_accepts_the_owner_key(make_settings):
    with _client(make_settings, api_key="owner-key", public_demo=True) as c:
        r = c.post("/generate", json={"question": "hi"}, headers={"X-API-Key": "owner-key"})
    assert r.status_code == 200


def test_health_reports_public_when_no_auth_is_configured(make_settings):
    with _client(make_settings) as c:
        assert c.get("/health").json()["access"] == "public"


def test_authenticator_anonymous_only_without_a_key():
    auth = Authenticator("owner-key", None, allow_anonymous=True)
    assert auth.authenticate(None).name == "anonymous"
    assert auth.authenticate("owner-key").name == "static-key"
    try:
        auth.authenticate("nope")
    except AuthError:
        pass
    else:  # pragma: no cover
        raise AssertionError("a wrong key must be rejected even in public demo mode")
