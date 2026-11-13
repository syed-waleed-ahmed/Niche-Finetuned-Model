"""Run every file in examples/ against a real HTTP server (fake engine).

The examples are documentation that users copy, so they are tested like code: each one
runs as a subprocess against uvicorn on a free port, exactly as a user would run it.
"""

from __future__ import annotations

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
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)


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


def _has_openai() -> bool:
    try:
        import openai  # noqa: F401
    except ImportError:
        return False
    return True
