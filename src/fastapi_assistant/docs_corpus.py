"""Load and chunk the official FastAPI documentation.

The docs are Markdown in the FastAPI repository (``docs/en/docs``) with code
examples pulled in from ``docs_src`` by include directives. This module downloads
a pinned release, resolves those includes so each page is self-contained, and
splits pages into heading-scoped chunks. It feeds both the retrieval index and the
synthetic dataset generator, so both see the same source of truth.
"""

from __future__ import annotations

import io
import logging
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)

DOCS_BASE_URL = "https://fastapi.tiangolo.com/"
DEFAULT_FASTAPI_REF = "0.162.5"

# {* ../../docs_src/first_steps/tutorial001.py hl[3] *}  and  {!../../docs_src/x.py!}
_INCLUDE_NEW = re.compile(r"\{\*\s*(?P<path>[^\s*]+)[^*]*\*\}")
_INCLUDE_OLD = re.compile(r"\{!\s*(?P<path>[^\s!]+)\s*!\}")
_ADMONITION = re.compile(r"^///\s*(\w+)?\s*(?:\|\s*(.*))?$", re.MULTILINE)
_HTML_TAG = re.compile(r"</?(?:abbr|span|div|img|a|br|details|summary)\b[^>]*>", re.IGNORECASE)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# mkdocstrings directives on the reference pages ("::: fastapi.Depends", optionally
# followed by an indented options block) render the API reference on the website. As
# text they carry no content, and the model copied them into answers.
_MKDOCSTRINGS = re.compile(r"^:::[ \t]+\S[^\n]*(?:\n[ \t]+[^\n]*)*", re.MULTILINE)
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*(?:\{[^}]*\})?\s*$", re.MULTILINE)


@dataclass(frozen=True)
class DocPage:
    url: str
    title: str
    markdown: str


@dataclass(frozen=True)
class DocChunk:
    url: str
    heading: str
    text: str


def download_fastapi_repo(ref: str, dest: Path) -> Path:
    """Download and safely extract the docs and examples of a FastAPI release.

    Only ``docs/en/docs`` and ``docs_src`` are extracted. Returns the repository root.
    """
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/fastapi/fastapi/tar.gz/refs/tags/{ref}"
    log.info("Downloading %s", url)
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https URL
        payload = resp.read()
    root_name = None
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        members = []
        for member in tar.getmembers():
            parts = PurePosixPath(member.name).parts
            if not parts:
                continue
            root_name = root_name or parts[0]
            rel = parts[1:]
            keep = rel[:3] == ("docs", "en", "docs") or rel[:1] == ("docs_src",)
            if keep and (member.isfile() or member.isdir()):
                members.append(member)
        tar.extractall(dest, members=members, filter="data")
    if root_name is None:
        raise RuntimeError(f"Empty archive from {url}")
    return dest / root_name


def _resolve_include(raw_path: str, repo_root: Path) -> str | None:
    marker = "docs_src/"
    idx = raw_path.find(marker)
    if idx < 0:
        return None
    target = repo_root / "docs_src" / raw_path[idx + len(marker) :]
    try:
        return target.read_text(encoding="utf-8").rstrip()
    except OSError:
        return None


def clean_markdown(markdown: str, repo_root: Path | None = None) -> str:
    """Resolve code includes and strip site-only markup."""

    def _sub(match: re.Match[str]) -> str:
        code = _resolve_include(match.group("path"), repo_root) if repo_root else None
        return f"```python\n{code}\n```" if code else ""

    text = _HTML_COMMENT.sub("", markdown)
    text = _MKDOCSTRINGS.sub("", text)
    text = _INCLUDE_NEW.sub(_sub, text)
    # Old-style includes sit inside an existing code fence; insert the code only.
    text = _INCLUDE_OLD.sub(
        lambda m: (_resolve_include(m.group("path"), repo_root) or "") if repo_root else "",
        text,
    )
    text = _ADMONITION.sub(
        lambda m: f"{(m.group(2) or m.group(1) or '').strip()}:" if m.group(1) else "", text
    )
    text = _HTML_TAG.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_url(rel_path: PurePosixPath) -> str:
    stem = rel_path.with_suffix("")
    if stem.name == "index":
        stem = stem.parent
    path = "" if str(stem) in ("", ".") else f"{stem.as_posix()}/"
    return DOCS_BASE_URL + path


def load_doc_pages(repo_root: Path) -> list[DocPage]:
    docs_dir = repo_root / "docs" / "en" / "docs"
    pages: list[DocPage] = []
    for md in sorted(docs_dir.rglob("*.md")):
        rel = PurePosixPath(md.relative_to(docs_dir).as_posix())
        if rel.parts and rel.parts[0] in {
            "release-notes.md",
            "management.md",
            "management-tasks.md",
        }:
            continue
        text = clean_markdown(md.read_text(encoding="utf-8"), repo_root)
        if len(text) < 200:
            continue
        title_match = _HEADING.search(text)
        title = title_match.group(2).strip() if title_match else rel.stem
        pages.append(DocPage(url=page_url(rel), title=title, markdown=text))
    return pages


def _split_sections(page: DocPage) -> list[tuple[str, str]]:
    """Split a page into (heading path, body) sections at ## / ### headings."""
    sections: list[tuple[str, str]] = []
    matches = list(_HEADING.finditer(page.markdown))
    if not matches:
        return [(page.title, page.markdown)]
    stack: list[tuple[int, str]] = []
    preamble = page.markdown[: matches[0].start()].strip()
    if preamble:
        sections.append((page.title, preamble))
    for i, match in enumerate(matches):
        level, title = len(match.group(1)), match.group(2).strip()
        stack = [(lvl, t) for lvl, t in stack if lvl < level] + [(level, title)]
        end = matches[i + 1].start() if i + 1 < len(matches) else len(page.markdown)
        body = page.markdown[match.end() : end].strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))
    return sections


def _window(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text at paragraph boundaries into windows of at most ``max_chars``.

    Code fences are kept intact even if that exceeds the budget.
    """
    blocks: list[str] = []
    buf: list[str] = []
    in_fence = False
    for para in text.split("\n\n"):
        fences = para.count("```")
        if in_fence:
            buf[-1] = buf[-1] + "\n\n" + para
            in_fence = fences % 2 == 0
            continue
        buf.append(para)
        in_fence = fences % 2 == 1
    blocks = buf
    windows: list[str] = []
    current = ""
    for block in blocks:
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= max_chars or not current:
            current = candidate
            continue
        windows.append(current)
        tail = current[-overlap:] if overlap else ""
        current = f"{tail}\n\n{block}" if tail and len(tail) + len(block) <= max_chars else block
    if current:
        windows.append(current)
    return windows


def chunk_pages(pages: list[DocPage], max_chars: int = 1200, overlap: int = 150) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    for page in pages:
        for heading, body in _split_sections(page):
            for text in _window(body, max_chars, overlap):
                if len(text) >= 80:
                    chunks.append(DocChunk(url=page.url, heading=heading, text=text.strip()))
    return chunks
