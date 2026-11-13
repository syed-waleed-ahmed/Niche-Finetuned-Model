-- Retrieval index over the FastAPI documentation (pgvector).
-- The vector dimension matches the default embedding model (BAAI/bge-small-en-v1.5).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS doc_chunks (
    id             BIGSERIAL PRIMARY KEY,
    index_version  TEXT        NOT NULL,
    url            TEXT        NOT NULL,
    heading        TEXT        NOT NULL,
    content        TEXT        NOT NULL,
    embedding      vector(384) NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS doc_chunks_version_idx ON doc_chunks (index_version);
CREATE INDEX IF NOT EXISTS doc_chunks_embedding_idx
    ON doc_chunks USING hnsw (embedding vector_cosine_ops);
