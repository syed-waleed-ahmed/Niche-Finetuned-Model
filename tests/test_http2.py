"""The Hypercorn server: cleartext HTTP/2 (h2c) and HTTP/1.1 on one port.

Cloud Run does not report a client's disconnect to the container over HTTP/1.1. Over
HTTP/2 its front end resets the request's stream, on a connection that stays open for
other requests. These tests do the same: they speak h2c to a real Hypercorn server
with the production configuration and reset one stream while it is generating.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest

from fastapi_assistant.api import create_app
from fastapi_assistant.config import Settings
from fastapi_assistant.server import hypercorn_config, serve_hypercorn
from tests.conftest import TOKEN_INTERVAL_S, SlowEngine, free_port

h2_connection = pytest.importorskip("h2.connection")
h2_config = pytest.importorskip("h2.config")
h2_events = pytest.importorskip("h2.events")
h2_errors = pytest.importorskip("h2.errors")


class H2Client:
    """A minimal HTTP/2 client over a plain socket (prior knowledge, no TLS)."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self.conn = h2_connection.H2Connection(
            h2_config.H2Configuration(client_side=True, header_encoding="utf-8")
        )
        self.conn.initiate_connection()
        self._flush()
        self.status: dict[int, int] = {}
        self.body: dict[int, bytes] = {}
        self.ended: set[int] = set()

    def _flush(self) -> None:
        data = self.conn.data_to_send()
        if data:
            self.sock.sendall(data)

    def post(self, path: str, payload: dict) -> int:
        body = json.dumps(payload).encode()
        stream = self.conn.get_next_available_stream_id()
        self.conn.send_headers(
            stream,
            [
                (":method", "POST"),
                (":path", path),
                (":scheme", "http"),
                (":authority", f"127.0.0.1:{self.port}"),
                ("content-type", "application/json"),
                ("content-length", str(len(body))),
            ],
        )
        self.conn.send_data(stream, body, end_stream=True)
        self._flush()
        return stream

    def reset(self, stream: int) -> None:
        """Abandon one request, as a proxy does when its own client goes away."""
        self.conn.reset_stream(stream, error_code=h2_errors.ErrorCodes.CANCEL)
        self._flush()

    def pump(self, until, timeout: float = 10.0) -> None:
        """Read frames until ``until()`` is true."""
        deadline = time.monotonic() + timeout
        while not until():
            remaining = deadline - time.monotonic()
            assert remaining > 0, "timed out waiting for the HTTP/2 response"
            self.sock.settimeout(remaining)
            data = self.sock.recv(65536)
            assert data, "the server closed the HTTP/2 connection"
            for event in self.conn.receive_data(data):
                if isinstance(event, h2_events.ResponseReceived):
                    self.status[event.stream_id] = int(dict(event.headers)[":status"])
                elif isinstance(event, h2_events.DataReceived):
                    self.body[event.stream_id] = self.body.get(event.stream_id, b"") + event.data
                    self.conn.acknowledge_received_data(
                        event.flow_controlled_length, event.stream_id
                    )
                elif isinstance(event, h2_events.StreamEnded):
                    self.ended.add(event.stream_id)
            self._flush()

    def close(self) -> None:
        self.sock.close()


@pytest.fixture
def h2_server(tmp_path):
    """Hypercorn with the production configuration, one generation slot and no queue."""
    engine = SlowEngine()
    port = free_port()
    settings = Settings(
        _env_file=None,
        warmup_on_startup=False,
        api_key=None,
        database_url=None,
        rag_index_dir=tmp_path / "no_index",
        max_concurrency=1,
        max_queue=0,
        http_server="hypercorn",
        api_host="127.0.0.1",
        api_port=port,
    )
    app = create_app(settings=settings, engine=engine)
    loop = asyncio.new_event_loop()
    stop = asyncio.Event()

    def run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(serve_hypercorn(app, hypercorn_config(settings), stop.wait))

    thread = threading.Thread(target=run, daemon=True)
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
        pytest.fail("the Hypercorn test server did not start")
    yield url, port, engine, app
    loop.call_soon_threadsafe(stop.set)
    thread.join(timeout=10)


def _wait_for_free_slot(app) -> None:
    deadline = time.monotonic() + 5
    while app.state.service.admission.inflight and time.monotonic() < deadline:
        time.sleep(0.02)
    assert app.state.service.admission.inflight == 0


def test_http11_is_still_served(h2_server):
    # Probes, the image health check and most tools speak HTTP/1.1 to the same port.
    url, _, engine, _ = h2_server
    health = httpx.get(f"{url}/health", timeout=5)
    assert health.status_code == 200 and health.http_version == "HTTP/1.1"
    assert "server" not in health.headers
    answer = httpx.post(f"{url}/generate", json={"question": "q", "max_new_tokens": 3}, timeout=10)
    assert answer.status_code == 200 and engine.emitted == 3
    # The request limits behave as they do under Uvicorn (the image smoke test relies on it).
    too_big = httpx.post(
        f"{url}/generate",
        content=b"x" * 70_000,
        headers={"content-type": "application/json"},
        timeout=10,
    )
    assert too_big.status_code == 413
    assert httpx.post(f"{url}/generate", json={}, timeout=10).status_code == 422


def test_h2c_request_is_answered(h2_server):
    _, port, engine, _ = h2_server
    client = H2Client(port)
    try:
        stream = client.post("/generate", {"question": "q", "max_new_tokens": 5})
        client.pump(lambda: stream in client.ended)
        assert client.status[stream] == 200
        assert json.loads(client.body[stream])["finish_reason"] == "length"
        assert engine.finish == "length" and engine.emitted == 5
    finally:
        client.close()


@pytest.mark.parametrize(
    ("path", "long", "short"),
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
def test_resetting_an_h2_stream_cancels_a_non_streaming_generation(h2_server, path, long, short):
    _, port, engine, app = h2_server
    client = H2Client(port)
    try:
        # 1000 tokens at 20 ms would hold the only slot for 20 seconds.
        abandoned = client.post(path, long)
        assert engine.started.wait(5), "the generation never started"
        time.sleep(5 * TOKEN_INTERVAL_S)
        client.reset(abandoned)

        assert engine.finished.wait(5), "the generation kept running after the stream was reset"
        assert engine.finish == "cancelled"
        assert 0 < engine.emitted < 200
        _wait_for_free_slot(app)

        # The connection is still usable, and with max_queue=0 a leaked slot would be 429.
        follow_up = client.post(path, short)
        client.pump(lambda: follow_up in client.ended)
        assert client.status[follow_up] == 200
    finally:
        client.close()


def test_resetting_an_h2_stream_cancels_a_streamed_generation(h2_server):
    _, port, engine, app = h2_server
    client = H2Client(port)
    try:
        stream = client.post("/generate", {"question": "q", "max_new_tokens": 1000, "stream": True})
        client.pump(lambda: b"event: token" in client.body.get(stream, b""))
        assert client.status[stream] == 200
        client.reset(stream)

        # The server stops consuming the stream without closing it, so the service's
        # disconnect watch has to stop the engine and release the slot.
        assert engine.finished.wait(5), "the stream kept generating after it was reset"
        assert engine.finish == "cancelled"
        assert 0 < engine.emitted < 200
        _wait_for_free_slot(app)

        follow_up = client.post("/generate", {"question": "q", "max_new_tokens": 3})
        client.pump(lambda: follow_up in client.ended)
        assert client.status[follow_up] == 200
    finally:
        client.close()
