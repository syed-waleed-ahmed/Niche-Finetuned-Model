"""Postgres store and pgvector retrieval against a real database.

Skipped unless TEST_DATABASE_URL points at a disposable Postgres with pgvector
(CI starts one as a service container). Never point this at production data:
it drops and recreates the schema.
"""

from __future__ import annotations

import os

import pytest

DB_URL = os.environ.get("TEST_DATABASE_URL")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not DB_URL, reason="TEST_DATABASE_URL is not set"),
]


@pytest.fixture
def store():
    import psycopg

    from fastapi_assistant.storage.postgres import PostgresStore

    with psycopg.connect(DB_URL, autocommit=True) as conn:
        conn.execute(
            "DROP TABLE IF EXISTS feedback, usage_events, api_keys, doc_chunks, schema_migrations CASCADE"
        )
    s = PostgresStore(DB_URL)
    assert s.migrate() == ["0001_init", "0002_doc_chunks"]
    assert s.migrate() == []  # idempotent
    yield s
    s.close()


def test_keys_usage_feedback(store):
    from fastapi_assistant.storage import Feedback, UsageEvent, hash_api_key

    key, record = store.create_key("ci")
    assert store.ping()
    assert store.lookup_key(hash_api_key(key)).key_id == record.id
    assert [r.name for r in store.list_keys()] == ["ci"]
    store.record_usage(
        UsageEvent(
            request_id="r1",
            endpoint="/generate",
            status_code=200,
            latency_ms=12.5,
            model_version="test",
            key_id=record.id,
            prompt_tokens=3,
            completion_tokens=4,
            finish_reason="stop",
        )
    )
    store.record_feedback(Feedback(request_id="r1", rating=1, key_id=record.id))
    assert store.revoke_key(record.prefix) == 1
    assert store.lookup_key(hash_api_key(key)) is None


def test_pgvector_retrieval(store):
    from fastapi_assistant.docs_corpus import DocChunk
    from fastapi_assistant.retrieval import PgVectorRetriever, build_pgvector_index
    from tests.test_retrieval import HashEmbedder

    class Embedder384(HashEmbedder):
        dim = 384

    chunks = [
        DocChunk("https://d/cors/", "CORS", "CORSMiddleware allow_origins cross origin"),
        DocChunk("https://d/ws/", "WebSockets", "websocket accept receive_text"),
    ]
    assert build_pgvector_index(chunks, Embedder384(), DB_URL, "t1") == 2
    retriever = PgVectorRetriever(DB_URL, Embedder384(), "t1")
    top = retriever.retrieve("cors origins middleware", 1)
    assert top[0].url == "https://d/cors/"
