"""Admission control: bounded concurrency, bounded queue, and 429 back-pressure."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from fastapi_assistant.admission import AdmissionController, QueueFullError, QueueTimeoutError
from fastapi_assistant.api import create_app
from tests.conftest import FakeEngine


def test_slots_limit_concurrency():
    async def scenario():
        ctl = AdmissionController(max_concurrency=1, max_queue=1, queue_timeout_s=5)
        first = await ctl.acquire()
        waiter = asyncio.create_task(ctl.acquire())
        await asyncio.sleep(0.01)
        assert ctl.inflight == 1 and ctl.waiting == 1
        # Queue is full now: the next request is rejected immediately.
        with pytest.raises(QueueFullError) as info:
            await ctl.acquire()
        assert info.value.retry_after_s >= 1
        first.release()
        first.release()  # idempotent
        second = await waiter
        assert ctl.inflight == 1 and ctl.waiting == 0
        second.release()
        assert ctl.inflight == 0

    asyncio.run(scenario())


def test_queue_timeout():
    async def scenario():
        ctl = AdmissionController(max_concurrency=1, max_queue=5, queue_timeout_s=0.05)
        slot = await ctl.acquire()
        with pytest.raises(QueueTimeoutError):
            await ctl.acquire()
        assert ctl.waiting == 0
        slot.release()

    asyncio.run(scenario())


def test_http_429_when_queue_full(make_settings):
    """With one slot and no queue, a concurrent second request gets 429 + Retry-After."""
    engine = FakeEngine(delay_s=0.3)
    app = create_app(settings=make_settings(max_concurrency=1, max_queue=0), engine=engine)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(client.post("/generate", json={"question": "slow"}))
            await asyncio.sleep(0.05)
            second = await client.post("/generate", json={"question": "fast"})
            return await first, second

    first, second = asyncio.run(scenario())
    assert first.status_code == 200
    assert second.status_code == 429
    assert int(second.headers["retry-after"]) >= 1


def test_stream_releases_slot(make_settings):
    from fastapi.testclient import TestClient

    app = create_app(settings=make_settings(max_concurrency=1, max_queue=0), engine=FakeEngine())
    with TestClient(app) as client:
        for _ in range(3):  # would 429 on the 2nd call if the slot leaked
            resp = client.post("/generate", json={"question": "x", "stream": True})
            assert resp.status_code == 200
        assert app.state.service.admission.inflight == 0
