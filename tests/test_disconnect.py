"""A non-streaming request whose caller disconnects must stop generating.

The scenario needs a real disconnect, so these tests run uvicorn on a free port and
talk to it over a raw socket that is closed while the generation is running.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator, Sequence

import httpx
import pytest
import uvicorn

from fastapi_assistant.api import create_app
from fastapi_assistant.config import Settings
from fastapi_assistant.engines.base import (
    Deadline,
    FinishReason,
    GenerationParams,
    Message,
    StreamEvent,
)
from fastapi_assistant.engines.remote_engine import RemoteEngine
from fastapi_assistant.service import _cancel_on_disconnect
from tests.conftest import FakeEngine

TOKEN_INTERVAL_S = 0.02


class SlowEngine(FakeEngine):
    """Emits one token every 20 ms, up to ``max_new_tokens``, and honours ``cancel``."""

    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.finished = threading.Event()
        self.finish: FinishReason | None = None
        self.emitted = 0

    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        deadline = Deadline(params.timeout_s, cancel)
        self.emitted = 0
        self.finish = None
        self.finished.clear()
        self.started.set()
        finish: FinishReason = "length"
        for _ in range(params.max_new_tokens):
            if deadline.expired():
                finish = deadline.reason or "cancelled"
                break
            time.sleep(TOKEN_INTERVAL_S)
            self.emitted += 1
            yield StreamEvent(delta="tok ")
        self.finish = finish
        self.finished.set()
        yield StreamEvent(finish_reason=finish, prompt_tokens=5, completion_tokens=self.emitted)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def slow_server(tmp_path):
    """A real server with one generation slot and no queue: a leaked slot means 429."""
    engine = SlowEngine()
    settings = Settings(
        _env_file=None,
        warmup_on_startup=False,
        api_key=None,
        database_url=None,
        rag_index_dir=tmp_path / "no_index",
        max_concurrency=1,
        max_queue=0,
    )
    app = create_app(settings=settings, engine=engine)
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
        pytest.fail("test server did not start")
    yield url, port, engine, app
    server.should_exit = True
    thread.join(timeout=10)


def _post_and_hang_up(port: int, path: str, payload: dict, engine: SlowEngine) -> None:
    """Send a request, wait until the engine is generating, then drop the connection."""
    body = json.dumps(payload).encode()
    head = (
        f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode()
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(head + body)
        assert engine.started.wait(5), "the generation never started"
        time.sleep(5 * TOKEN_INTERVAL_S)  # let it produce a few tokens first


@pytest.mark.parametrize(
    ("path", "payload", "follow_up"),
    [
        (
            "/generate",
            {"question": "q", "max_new_tokens": 1000},
            {"question": "q", "max_new_tokens": 3},
        ),
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "q"}], "max_tokens": 1000},
            {"messages": [{"role": "user", "content": "q"}], "max_tokens": 3},
        ),
    ],
)
def test_disconnect_cancels_a_non_streaming_generation(slow_server, path, payload, follow_up):
    url, port, engine, app = slow_server
    # 1000 tokens at 20 ms would hold the only slot for 20 seconds.
    _post_and_hang_up(port, path, payload, engine)

    assert engine.finished.wait(5), "the generation kept running after the caller left"
    assert engine.finish == "cancelled"
    assert 0 < engine.emitted < 200

    # The slot is free again: with max_queue=0 a leaked slot would answer 429.
    deadline = time.monotonic() + 5
    while app.state.service.admission.inflight and time.monotonic() < deadline:
        time.sleep(0.02)
    assert app.state.service.admission.inflight == 0
    resp = httpx.post(f"{url}{path}", json=follow_up, timeout=10)
    assert resp.status_code == 200

    metrics = httpx.get(f"{url}/metrics", timeout=5).text
    assert 'assistant_generation_finish_total{reason="cancelled"} 1.0' in metrics


def test_a_connected_caller_is_not_cancelled(slow_server):
    url, _, engine, _ = slow_server
    # The watch runs for the whole generation and must not fire while the caller waits.
    resp = httpx.post(f"{url}/generate", json={"question": "q", "max_new_tokens": 40}, timeout=20)
    assert resp.status_code == 200
    assert resp.json()["finish_reason"] == "length"
    assert engine.finish == "length" and engine.emitted == 40


def test_a_failing_disconnect_watch_does_not_cancel():
    async def scenario() -> bool:
        async def broken() -> None:
            raise RuntimeError("receive channel closed")

        cancel = threading.Event()
        await _cancel_on_disconnect(broken, cancel)
        return cancel.is_set()

    assert asyncio.run(scenario()) is False


def test_remote_engine_skips_a_request_that_is_already_cancelled():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"data": []})

    settings = Settings(_env_file=None, engine_backend="remote", remote_base_url="http://upstream")
    engine = RemoteEngine(settings, transport=httpx.MockTransport(handler))
    cancel = threading.Event()
    cancel.set()
    result = engine.generate([{"role": "user", "content": "q"}], GenerationParams(), cancel)
    assert result.finish_reason == "cancelled" and result.text == ""
    assert "/v1/chat/completions" not in calls
