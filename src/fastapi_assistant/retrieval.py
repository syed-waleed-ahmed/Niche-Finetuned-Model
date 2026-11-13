"""Retrieval-augmented generation over the FastAPI documentation.

A 1B model cannot memorize an API surface. Retrieval puts the relevant
documentation into the prompt at answer time, through the same ``context`` slot the
training format already uses (see :func:`fastapi_assistant.prompts.build_messages`).

Two interchangeable index backends:

* ``local`` — a NumPy matrix shipped with the model (no database; good for the
  free tier and for a single replica).
* ``pgvector`` — the ``doc_chunks`` table in Postgres (shared by all replicas).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import Settings
from .docs_corpus import DocChunk

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Passage:
    url: str
    heading: str
    text: str
    score: float


class Embedder(Protocol):
    dim: int

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[Passage]: ...


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class FastEmbedEmbedder:
    """ONNX sentence embeddings via ``fastembed`` (no torch needed)."""

    def __init__(self, model_name: str, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding

        kwargs: dict[str, Any] = {"cache_dir": str(cache_dir)} if cache_dir else {}
        self._model = TextEmbedding(model_name=model_name, **kwargs)
        self.dim = int(self.embed_query("probe").shape[-1])

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        vectors = np.asarray(list(self._model.passage_embed(texts)), dtype=np.float32)
        return _normalize(vectors)

    def embed_query(self, text: str) -> np.ndarray:
        vector = np.asarray(next(iter(self._model.query_embed([text]))), dtype=np.float32)
        return _normalize(vector)


class LocalIndexRetriever:
    """Brute-force cosine search over an in-memory matrix (fine to ~100k chunks)."""

    def __init__(self, index_dir: Path, embedder: Embedder, min_score: float = 0.0) -> None:
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        self._embeddings = np.load(index_dir / "embeddings.npy")
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as fh:
            self._chunks = [DocChunk(**json.loads(line)) for line in fh if line.strip()]
        if len(self._chunks) != self._embeddings.shape[0]:
            raise ValueError("RAG index is corrupt: chunk and embedding counts differ")
        if meta.get("dim") != embedder.dim:
            raise ValueError(
                f"RAG index dim {meta.get('dim')} does not match embedder dim {embedder.dim}"
            )
        self._embedder = embedder
        self._min_score = min_score
        self.meta = meta

    def retrieve(self, query: str, k: int) -> list[Passage]:
        scores = self._embeddings @ self._embedder.embed_query(query)
        top = np.argsort(-scores)[:k]
        return [
            Passage(**asdict(self._chunks[i]), score=float(scores[i]))
            for i in top
            if scores[i] >= self._min_score
        ]


class PgVectorRetriever:
    def __init__(
        self, database_url: str, embedder: Embedder, index_version: str, min_score: float = 0.0
    ) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(database_url, min_size=0, max_size=2, open=False)
        self._opened = False
        self._embedder = embedder
        self._version = index_version
        self._min_score = min_score

    def retrieve(self, query: str, k: int) -> list[Passage]:
        if not self._opened:
            self._pool.open(wait=False)
            self._opened = True
        vector = "[" + ",".join(f"{x:.6f}" for x in self._embedder.embed_query(query)) + "]"
        with self._pool.connection(timeout=10.0) as conn:
            rows = conn.execute(
                "SELECT url, heading, content, 1 - (embedding <=> %s::vector) AS score"
                " FROM doc_chunks WHERE index_version = %s"
                " ORDER BY embedding <=> %s::vector LIMIT %s",
                (vector, self._version, vector, k),
            ).fetchall()
        return [
            Passage(url=r[0], heading=r[1], text=r[2], score=float(r[3]))
            for r in rows
            if r[3] >= self._min_score
        ]


def format_context(passages: list[Passage], max_chars: int) -> str:
    """Render passages into the prompt's context block within a character budget."""
    parts: list[str] = []
    used = 0
    for i, passage in enumerate(passages, start=1):
        block = f"[{i}] {passage.heading}\n{passage.text}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining < 200:
                break
            block = block[:remaining].rsplit("\n", 1)[0]
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


# -- index building -------------------------------------------------------------
def build_local_index(
    chunks: list[DocChunk], embedder: Embedder, out_dir: Path, meta: dict
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = embedder.embed_passages([f"{c.heading}\n{c.text}" for c in chunks])
    np.save(out_dir / "embeddings.npy", embeddings.astype(np.float32))
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    (out_dir / "meta.json").write_text(
        json.dumps({**meta, "dim": embedder.dim, "count": len(chunks)}, indent=2), encoding="utf-8"
    )
    return out_dir


def build_pgvector_index(
    chunks: list[DocChunk], embedder: Embedder, database_url: str, index_version: str
) -> int:
    import psycopg

    embeddings = embedder.embed_passages([f"{c.heading}\n{c.text}" for c in chunks])
    with psycopg.connect(database_url, autocommit=False) as conn:
        conn.execute("DELETE FROM doc_chunks WHERE index_version = %s", (index_version,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO doc_chunks (index_version, url, heading, content, embedding)"
                " VALUES (%s, %s, %s, %s, %s::vector)",
                [
                    (
                        index_version,
                        c.url,
                        c.heading,
                        c.text,
                        "[" + ",".join(f"{x:.6f}" for x in e) + "]",
                    )
                    for c, e in zip(chunks, embeddings, strict=True)
                ],
            )
        conn.commit()
    return len(chunks)


def create_retriever(settings: Settings, embedder: Embedder | None = None) -> Retriever | None:
    if not settings.rag_enabled:
        return None
    embedder = embedder or FastEmbedEmbedder(settings.embedding_model, settings.embedding_cache_dir)
    if settings.rag_backend == "pgvector":
        if not settings.database_url:
            raise ValueError("RAG_BACKEND=pgvector requires DATABASE_URL")
        return PgVectorRetriever(
            settings.database_url, embedder, settings.rag_index_version, settings.rag_min_score
        )
    return LocalIndexRetriever(settings.rag_index_dir, embedder, settings.rag_min_score)
