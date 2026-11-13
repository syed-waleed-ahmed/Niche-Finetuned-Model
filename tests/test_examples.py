"""Run every file in examples/ against a real HTTP server (fake engine).

The examples are documentation that users copy, so they are tested like code: each one
runs as a subprocess against uvicorn on a free port, exactly as a user would run it.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from fastapi_assistant.api import create_app

from .conftest import FakeEngine

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
API_KEY = "fa_example_test_key"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server_url(tmp_path_factory):
    from fastapi_assistant.config import Settings

    settings = Settings(
        _env_file=None,
        warmup_on_startup=False,
        api_key=API_KEY,
        database_url=None,
        rag_index_dir=tmp_path_factory.mktemp("no_index"),
    )
    app = create_app(settings=settings, engine=FakeEngine())
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.1)
    else:  # pragma: no cover
        pytest.fail("example server did not start")
    yield url
    server.should_exit = True
    thread.join(timeout=10)


def _run(cmd: list[str], url: str, key: str | None = API_KEY) -> subprocess.CompletedProcess:
    env = {**os.environ, "ASSISTANT_URL": url}
    env.pop("ASSISTANT_API_KEY", None)
    if key is not None:
        env["ASSISTANT_API_KEY"] = key
    # Decode as UTF-8 on every platform: the default on Windows is the ANSI code page.
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, encoding="utf-8", timeout=120
    )


def test_openai_sdk_example(server_url):
    pytest.importorskip("openai")
    result = _run([sys.executable, str(EXAMPLES / "openai_sdk.py"), "path", "params?"], server_url)
    assert result.returncode == 0, result.stderr
    assert "models: ['fastapi-assistant']" in result.stdout
    assert "--- blocking ---" in result.stdout
    assert "echo: path params?" in result.stdout
    # The streamed answer is printed token by token after the blocking one.
    assert result.stdout.count("echo: path params?") == 2


def test_native_stream_example(server_url):
    result = _run(
        [sys.executable, str(EXAMPLES / "native_stream.py"), "what", "is", "Depends?"], server_url
    )
    assert result.returncode == 0, result.stderr
    assert "echo: what is Depends?" in result.stdout
    assert "[finish=stop" in result.stdout


def test_examples_refuse_to_run_without_a_key(server_url):
    names = ["native_stream.py"] + (["openai_sdk.py"] if _has_openai() else [])
    for name in names:
        result = _run([sys.executable, str(EXAMPLES / name)], server_url, key=None)
        assert result.returncode == 2
        assert "ASSISTANT_API_KEY" in result.stderr


def test_examples_report_auth_errors(server_url):
    result = _run([sys.executable, str(EXAMPLES / "native_stream.py")], server_url, key="wrong")
    assert result.returncode == 1
    assert "HTTP 401" in result.stderr


@pytest.mark.skipif(
    sys.platform == "win32" or not (shutil.which("bash") and shutil.which("curl")),
    reason="needs a POSIX bash and curl",
)
def test_curl_example(server_url):
    result = _run(["bash", str(EXAMPLES / "curl.sh")], server_url)
    assert result.returncode == 0, result.stderr
    assert '"status":"ok"' in result.stdout
    assert "event: token" in result.stdout
    assert "data: [DONE]" in result.stdout
    assert "HTTP 401" in result.stdout


def _node() -> str | None:
    """Path to a Node.js with a built-in fetch (18 or later), or None."""
    node = shutil.which("node")
    if not node:
        return None
    try:
        version = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=30)
        major = int(version.stdout.strip().lstrip("v").split(".")[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return node if major >= 18 else None


needs_node = pytest.mark.skipif(_node() is None, reason="needs Node.js 18 or later")
FETCH_STREAM = str(EXAMPLES / "fetch_stream.mjs")


@needs_node
def test_fetch_stream_example(server_url):
    result = _run([_node(), FETCH_STREAM, "what", "is", "Depends?"], server_url)
    assert result.returncode == 0, result.stderr
    assert "echo: what is Depends?" in result.stdout
    assert "[finish=stop]" in result.stdout


@needs_node
def test_fetch_stream_example_explains_a_rejected_or_missing_key(server_url):
    wrong = _run([_node(), FETCH_STREAM], server_url, key="wrong")
    assert wrong.returncode == 1
    assert "HTTP 401" in wrong.stderr and "rejected the API key" in wrong.stderr
    # The key is optional (a public demo needs none); this server requires one.
    missing = _run([_node(), FETCH_STREAM], server_url, key=None)
    assert missing.returncode == 1
    assert "HTTP 401" in missing.stderr and "Set ASSISTANT_API_KEY" in missing.stderr


class _Stub(http.server.BaseHTTPRequestHandler):
    """Answers every POST with the status and body the test chose."""

    status = 200
    headers_out: dict[str, str] = {}
    chunks: list[bytes] = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(self.status)
        for name, value in self.headers_out.items():
            self.send_header(name, value)
        self.end_headers()
        for chunk in self.chunks:  # one TCP write per chunk, to split frames on purpose
            self.wfile.write(chunk)
            self.wfile.flush()
            time.sleep(0.05)

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture
def stub_url():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


@needs_node
@pytest.mark.parametrize(
    ("status", "headers", "expected"),
    [
        (413, {}, "too large"),
        (429, {"Retry-After": "7"}, "Try again in 7 s"),
        (503, {"Retry-After": "30"}, "Try again in 30 s"),
    ],
)
def test_fetch_stream_example_explains_limits(stub_url, status, headers, expected):
    _Stub.status, _Stub.headers_out, _Stub.chunks = status, headers, [b'{"error": "x"}']
    result = _run([_node(), FETCH_STREAM], stub_url)
    assert result.returncode == 1
    assert f"HTTP {status}" in result.stderr and expected in result.stderr


@needs_node
def test_fetch_stream_example_parses_frames_split_across_chunks(stub_url):
    def frame(content: str) -> bytes:
        chunk = {"choices": [{"delta": {"content": content}}]}
        return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()

    cafe = frame("café ")  # the two UTF-8 bytes of the accented letter get split
    cut = cafe.index("é".encode()) + 1
    _Stub.status, _Stub.headers_out = 200, {"Content-Type": "text/event-stream"}
    _Stub.chunks = [
        # A frame cut in the middle of its JSON, a character cut between its bytes, and
        # one frame that ends with CRLF line endings.
        b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\ndata: {"choices":[{"de',
        b'lta":{"content":"Hello "}}]}\n\n' + cafe[:cut],
        cafe[cut:] + frame("ok").replace(b"\n\n", b"\r\n\r\n"),
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n',
    ]
    result = _run([_node(), FETCH_STREAM], stub_url)
    assert result.returncode == 0, result.stderr
    assert "Hello café ok" in result.stdout
    assert "[finish=stop]" in result.stdout


@needs_node
def test_fetch_stream_example_reports_an_error_frame(stub_url):
    _Stub.status, _Stub.headers_out = 200, {"Content-Type": "text/event-stream"}
    _Stub.chunks = [b'data: {"error": {"message": "Generation failed"}}\n\ndata: [DONE]\n\n']
    result = _run([_node(), FETCH_STREAM], stub_url)
    assert result.returncode == 1
    assert "error: Generation failed" in result.stderr


def _has_openai() -> bool:
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True
