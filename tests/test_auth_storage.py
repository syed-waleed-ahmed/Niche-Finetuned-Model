"""Per-user API keys and usage recording (in-memory store)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fastapi_assistant.api import create_app
from fastapi_assistant.auth import Authenticator, AuthError, extract_api_key
from fastapi_assistant.storage import ANONYMOUS, MemoryStore, hash_api_key
from tests.conftest import FakeEngine


def test_extract_api_key():
    assert extract_api_key("k1", None) == "k1"
    assert extract_api_key(None, "Bearer k2") == "k2"
    assert extract_api_key(None, "bearer  k3 ") == "k3"
    assert extract_api_key(None, "Basic abc") is None
    assert extract_api_key(None, None) is None


def test_keys_are_hashed_and_revocable():
    store = MemoryStore()
    key, record = store.create_key("bob")
    assert key.startswith("fa_") and len(key) > 40
    assert record.prefix == key[:9]
    assert store.lookup_key(hash_api_key(key)).name == "bob"
    assert store.revoke_key(record.prefix) == 1
    assert store.lookup_key(hash_api_key(key)) is None


def test_authenticator_modes():
    assert Authenticator(None, None).authenticate(None) is ANONYMOUS
    static = Authenticator("s", None)
    assert static.authenticate("s").name == "static-key"
    with pytest.raises(AuthError):
        static.authenticate("x")
    with pytest.raises(AuthError):
        static.authenticate(None)


def test_authenticator_caches_lookups():
    calls = []

    class CountingStore(MemoryStore):
        def lookup_key(self, key_hash):
            calls.append(key_hash)
            return super().lookup_key(key_hash)

    store = CountingStore()
    key, _ = store.create_key("c")
    auth = Authenticator(None, store, cache_ttl_s=60)
    for _ in range(5):
        assert auth.authenticate(key).name == "c"
    assert len(calls) == 1
    for _ in range(3):
        with pytest.raises(AuthError):
            auth.authenticate("fa_unknown")
    assert len(calls) == 2  # negative result cached too


def test_db_keys_and_usage_recording(make_settings):
    store = MemoryStore()
    key, record = store.create_key("carol")
    app = create_app(settings=make_settings(), engine=FakeEngine(), store=store)
    with TestClient(app) as client:
        assert client.post("/generate", json={"question": "hi"}).status_code == 401
        resp = client.post("/generate", json={"question": "hi"}, headers={"X-API-Key": key})
        assert resp.status_code == 200
    event = store.usage[0]
    assert event.key_id == record.id
    assert event.endpoint == "/generate"
    assert event.completion_tokens == 2
    assert event.request_id == resp.headers["x-request-id"]
    # Transcripts are not persisted unless explicitly enabled.
    assert event.question is None and event.answer is None


def test_transcripts_opt_in(make_settings):
    store = MemoryStore()
    key, _ = store.create_key("dan")
    app = create_app(
        settings=make_settings(store_transcripts=True), engine=FakeEngine(), store=store
    )
    with TestClient(app) as client:
        client.post("/generate", json={"question": "hi"}, headers={"X-API-Key": key})
    assert store.usage[0].question == "hi"
    assert store.usage[0].answer == "echo: hi"
