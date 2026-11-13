"""MkDocs hook: keep links that leave the documentation site working.

The site is built from ``docs/`` plus small mirror pages under ``docs/project/``
whose only content is a pymdownx.snippets include of a root-level file, for
example ``--8<-- "CONTRIBUTING.md"``. Those root files are written for GitHub, so
their relative links are repository paths (``docs/architecture/...``, ``LICENSE``,
``.github/...``), and a few pages under ``docs/`` link out of it as well
(``../../README.md``).

For every relative link and image in a rendered page, this hook works out the
repository path it points at, then:

* if that path is part of the site (a page, an asset, or a root file that a
  mirror page shows), the link is rewritten relative to the current page, so
  MkDocs resolves and validates it as usual;
* otherwise the link points at the file on GitHub
  (``<repo_url>/blob/<branch>/<path>``, or ``tree`` for a directory), and a
  warning is logged if the path does not exist in the repository either, so
  ``mkdocs build --strict`` still fails on a genuinely broken link.

For mirror pages it also drops the included file's YAML front matter (the model
card carries Hugging Face metadata) and points "Edit this page" at the root file.

Markdown sources are never modified: GitHub renders every file as before. Snippet
paths are taken as relative to the repository root, the directory of mkdocs.yml,
which is where mkdocs runs (the snippets ``base_path`` is ".").
"""

from __future__ import annotations

import logging
import os
import posixpath
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

import yaml
from markdown import Extension
from markdown.preprocessors import Preprocessor
from markdown.treeprocessors import Treeprocessor
from markdown.util import AMP_SUBSTITUTE
from mkdocs.utils.meta import get_data

if TYPE_CHECKING:
    from xml.etree.ElementTree import Element

    from markdown import Markdown
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page

# A logger under "mkdocs." so that `mkdocs build --strict` counts its warnings.
log = logging.getLogger("mkdocs.hooks.links")

# A page whose whole body is one snippet include: --8<-- "PATH" (single quotes too).
SNIPPET_ONLY_RE = re.compile(r"""\A-+8<-+[ \t]+(["'])(?P<path>[^"'\n]+)\1[ \t]*\Z""")
# YAML front matter at the very top of an included file (same pattern MkDocs uses).
FRONT_MATTER_RE = re.compile(r"\A-{3}[ \t]*\n(.*?\n)(?:\.{3}|-{3})[ \t]*(?:\n|\Z)", re.DOTALL)
# The branch in an edit_uri such as "edit/main/docs/".
EDIT_URI_RE = re.compile(r"^(?:edit|blob|tree)/(?P<branch>[^/]+)/")


@dataclass
class _State:
    repo_root: str = ""
    docs_prefix: str = ""
    repo_url: str = ""
    branch: str = "main"
    edit_base: str = ""
    # Repository path (POSIX, relative to the root) -> src_uri of the site file showing it.
    site_paths: dict[str, str] = field(default_factory=dict)
    # src_uri of a mirror page -> repository path of the root file it includes.
    mirrors: dict[str, str] = field(default_factory=dict)
    # The page being rendered: set in on_page_markdown, cleared in on_page_content.
    page: Page | None = None


_state = _State()


# ---- MkDocs events ------------------------------------------------------------------


def on_config(config: MkDocsConfig) -> MkDocsConfig:
    global _state
    repo_root = os.path.dirname(os.path.abspath(config.config_file_path))
    docs_prefix = os.path.relpath(config.docs_dir, repo_root).replace(os.sep, "/")
    branch_match = EDIT_URI_RE.match(config.edit_uri or "")
    branch = branch_match["branch"] if branch_match else "main"
    repo_url = (config.repo_url or "").rstrip("/")
    _state = _State(
        repo_root=repo_root,
        docs_prefix="" if docs_prefix == "." else docs_prefix,
        repo_url=repo_url,
        branch=branch,
        edit_base=f"{repo_url}/edit/{branch}/" if config.edit_uri and repo_url else "",
    )
    config.markdown_extensions.append(_RepoLinksExtension())
    return config


def on_files(files: Files, config: MkDocsConfig) -> Files:
    for file in files:
        if file.inclusion.is_excluded():
            continue
        _state.site_paths[_repo_path(file.src_uri)] = file.src_uri
        if file.is_documentation_page():
            body, _meta = get_data(file.content_string)
            match = SNIPPET_ONLY_RE.match(body.strip())
            if match:
                included = posixpath.normpath(match["path"].split(":", 1)[0].strip())
                _state.mirrors[file.src_uri] = included
                _state.site_paths[included] = file.src_uri
    return files


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    _state.page = page
    included = _state.mirrors.get(page.file.src_uri)
    if included and _state.edit_base:
        page.edit_url = urljoin(_state.edit_base, quote(included))
    return markdown


def on_page_content(html: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    _state.page = None
    return html


# ---- Markdown extension ----------------------------------------------------------------


class _RepoLinksExtension(Extension):
    def extendMarkdown(self, md: Markdown) -> None:  # noqa: N802 (Markdown API name)
        # After pymdownx.snippets (32), before normalize_whitespace (30).
        md.preprocessors.register(_IncludedFrontMatterStripper(md), "repo_links_front_matter", 31)
        # After the inline patterns (20) create the links, before MkDocs' relpath (0).
        md.treeprocessors.register(_LinkRewriter(md), "repo_links", 1)


class _IncludedFrontMatterStripper(Preprocessor):
    def run(self, lines: list[str]) -> list[str]:
        page = _state.page
        if page is None or page.file.src_uri not in _state.mirrors:
            return lines
        text = "\n".join(lines).lstrip("\n")
        match = FRONT_MATTER_RE.match(text)
        if not match:
            return lines
        try:
            data = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            return lines
        if not isinstance(data, dict):
            return lines
        return text[match.end() :].lstrip("\n").split("\n")


class _LinkRewriter(Treeprocessor):
    def run(self, root: Element) -> None:
        page = _state.page
        if page is None:
            return
        for element in root.iter():
            attribute = {"a": "href", "img": "src"}.get(element.tag)
            if attribute is None:
                continue
            url = element.get(attribute)
            if url:
                new_url = _rewrite(url, page)
                if new_url != url:
                    element.set(attribute, new_url)


# ---- Link resolution -------------------------------------------------------------------


def _repo_path(src_uri: str) -> str:
    """Repository path of a file given by its path inside docs/."""
    return posixpath.join(_state.docs_prefix, src_uri) if _state.docs_prefix else src_uri


def _rewrite(url: str, page: Page) -> str:
    scheme, netloc, path, query, fragment = urlsplit(url)
    if scheme or netloc or not path or path.startswith(("/", "\\")) or AMP_SUBSTITUTE in url:
        return url  # external, absolute, same-page anchor, or an obfuscated e-mail address

    src_uri = page.file.src_uri
    page_dir = posixpath.dirname(src_uri)
    # Relative links in a mirror page were written relative to the included root file.
    included = _state.mirrors.get(src_uri)
    base = posixpath.dirname(included) if included else posixpath.dirname(_repo_path(src_uri))
    target = posixpath.normpath(posixpath.join(base, unquote(path)))
    if target == ".." or target.startswith("../"):
        return url  # outside the repository: let MkDocs report it

    site_paths = _state.site_paths
    site_uri = site_paths.get(target) or site_paths.get(posixpath.join(target, "index.md"))
    if site_uri is not None:
        if posixpath.normpath(posixpath.join(page_dir, unquote(path))) == site_uri:
            return url  # already resolves inside the site
        relative = posixpath.relpath(site_uri, page_dir or ".")
        return urlunsplit(("", "", quote(relative), query, fragment))

    absolute = os.path.join(_state.repo_root, *target.split("/"))
    kind = "tree" if os.path.isdir(absolute) else "blob"
    if not os.path.exists(absolute):
        log.warning(
            f"Doc file '{src_uri}' contains a link '{url}', but '{target}' exists neither "
            "in the site nor in the repository."
        )
    github = f"{_state.repo_url}/{kind}/{_state.branch}/{quote(target)}"
    return github + (f"?{query}" if query else "") + (f"#{fragment}" if fragment else "")
