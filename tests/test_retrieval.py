"""Docs corpus processing, retrieval, and RAG wiring (fake embedder, no downloads)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

import numpy as np
from fastapi.testclient import TestClient

from fastapi_assistant.api import create_app
from fastapi_assistant.docs_corpus import (
    DocChunk,
    DocPage,
    chunk_pages,
    clean_markdown,
    load_doc_pages,
    page_url,
)
from fastapi_assistant.retrieval import (
    LocalIndexRetriever,
    Passage,
    build_local_index,
    format_context,
)
from tests.conftest import FakeEngine


class HashEmbedder:
    """Deterministic bag-of-words embedder for tests."""

    dim = 64

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def _fake_repo(tmp_path: Path) -> Path:
    docs = tmp_path / "docs" / "en" / "docs" / "tutorial"
    docs.mkdir(parents=True)
    (tmp_path / "docs_src" / "cors").mkdir(parents=True)
    (tmp_path / "docs_src" / "cors" / "tutorial001.py").write_text(
        "from fastapi.middleware.cors import CORSMiddleware\n", encoding="utf-8"
    )
    (docs / "cors.md").write_text(
        "# CORS { #cors }\n\nIntro text about cross origin resource sharing in browsers. "
        * 3
        + "\n\n## Use CORSMiddleware\n\n<!-- hidden -->Configure it:\n\n"
        "{* ../../docs_src/cors/tutorial001.py hl[2] *}\n\n/// tip | Remember\n\nOrder matters.\n\n///\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "en" / "docs" / "index.md").write_text(
        "# Home\n\n" + "FastAPI home page. " * 20, encoding="utf-8"
    )
    return tmp_path


def test_clean_markdown_resolves_includes(tmp_path):
    repo = _fake_repo(tmp_path)
    page = load_doc_pages(repo)[1]
    assert page.url == "https://fastapi.tiangolo.com/tutorial/cors/"
    assert "```python\nfrom fastapi.middleware.cors import CORSMiddleware" in page.markdown
    assert "<!--" not in page.markdown and "{*" not in page.markdown and "///" not in page.markdown
    assert "Remember:" in page.markdown
    assert page.title == "CORS"


def test_page_url():
    assert page_url(PurePosixPath("index.md")) == "https://fastapi.tiangolo.com/"
    assert page_url(PurePosixPath("tutorial/index.md")) == "https://fastapi.tiangolo.com/tutorial/"
    assert (
        page_url(PurePosixPath("advanced/events.md"))
        == "https://fastapi.tiangolo.com/advanced/events/"
    )


def test_chunking_keeps_heading_path_and_code_fences():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(80)) + "\n```"
    page = DocPage(
        url="u",
        title="T",
        markdown="# T\n\nintro " * 20 + "\n\n## Section\n\n" + ("para " * 60 + "\n\n") * 6 + code,
    )
    chunks = chunk_pages([page], max_chars=500, overlap=50)
    assert len(chunks) > 3
    assert any(c.heading == "T > Section" for c in chunks)
    fenced = [c for c in chunks if "```python" in c.text]
    assert fenced and all(c.text.count("```") % 2 == 0 for c in fenced)


def test_clean_markdown_without_repo_drops_includes():
    assert clean_markdown("a {* ../../docs_src/x.py *} b") == "a  b"


def _index(tmp_path: Path) -> Path:
    chunks = [
        DocChunk(
            "https://d/cors/",
            "CORS",
            "Use CORSMiddleware with allow_origins for cross origin requests.",
        ),
        DocChunk(
            "https://d/deps/",
            "Dependencies",
            "Declare dependencies with Depends and yield for cleanup.",
        ),
        DocChunk(
            "https://d/ws/",
            "WebSockets",
            "Accept websocket connections with websocket accept and receive_text.",
        ),
    ]
    return build_local_index(chunks, HashEmbedder(), tmp_path / "idx", {"fastapi_ref": "test"})


def test_local_index_retrieval(tmp_path):
    retriever = LocalIndexRetriever(_index(tmp_path), HashEmbedder())
    top = retriever.retrieve("how do I configure cors middleware origins", k=2)
    assert top[0].url == "https://d/cors/"
    assert top[0].score >= top[1].score
    assert retriever.meta["count"] == 3


def test_min_score_filters(tmp_path):
    retriever = LocalIndexRetriever(_index(tmp_path), HashEmbedder(), min_score=0.99)
    assert retriever.retrieve("zzz unrelated", k=3) == []


def test_format_context_budget():
    passages = [Passage("u", f"H{i}", "word " * 100, 0.9) for i in range(5)]
    ctx = format_context(passages, max_chars=1200)
    assert ctx.startswith("[1] H0")
    assert len(ctx) <= 1300


def test_generate_with_retrieval(make_settings, tmp_path):
    engine = FakeEngine()
    retriever = LocalIndexRetriever(_index(tmp_path), HashEmbedder())
    app = create_app(settings=make_settings(rag_enabled=True), engine=engine, retriever=retriever)
    with TestClient(app) as client:
        body = client.post(
            "/generate", json={"question": "configure CORS middleware origins"}
        ).json()
        assert body["sources"][0]["url"] == "https://d/cors/"
        user_turn = engine.calls[0]["messages"][-1]["content"]
        assert user_turn.startswith("configure CORS middleware origins\n\nContext:\n[1] CORS")
        # Retrieval can be turned off per request.
        off = client.post("/generate", json={"question": "cors", "use_retrieval": False}).json()
        assert off["sources"] == []


def test_retrieval_failure_degrades(make_settings):
    class Broken:
        def retrieve(self, query, k):
            raise RuntimeError("db down")

    engine = FakeEngine()
    app = create_app(settings=make_settings(rag_enabled=True), engine=engine, retriever=Broken())
    with TestClient(app) as client:
        resp = client.post("/generate", json={"question": "hi"})
    assert resp.status_code == 200
    assert resp.json()["sources"] == []


def test_index_loads_for_opt_in_even_when_rag_disabled(make_settings, tmp_path, monkeypatch):
    """RAG_ENABLED=false only sets the default; an available index still serves opt-ins."""
    import fastapi_assistant.api as api_module

    index_dir = _index(tmp_path)
    seen = {}

    def fake_create(settings):
        seen["rag_enabled"] = settings.rag_enabled
        return LocalIndexRetriever(index_dir, HashEmbedder())

    monkeypatch.setattr(api_module, "create_retriever", fake_create)
    engine = FakeEngine()
    app = create_app(
        settings=make_settings(rag_enabled=False, rag_index_dir=index_dir), engine=engine
    )
    with TestClient(app) as client:
        default = client.post("/generate", json={"question": "cors middleware origins"}).json()
        opted = client.post(
            "/generate", json={"question": "cors middleware origins", "use_retrieval": True}
        ).json()
    assert seen == {"rag_enabled": True}
    assert default["sources"] == []
    assert opted["sources"][0]["url"] == "https://d/cors/"
