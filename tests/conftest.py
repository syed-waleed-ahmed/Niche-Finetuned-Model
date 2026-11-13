"""Shared pytest fixtures.

These tests never load a real model or hit the network — a lightweight fake engine
stands in for the real backends, so the suite runs fast and deterministically in CI.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fastapi_assistant.api import create_app
from fastapi_assistant.config import Settings
from fastapi_assistant.engines.base import BaseEngine, GenerationParams, Message, StreamEvent


class FakeEngine(BaseEngine):
    """A stand-in engine that echoes the last user message instead of running a model."""

    backend = "fake"

    def __init__(
        self,
        *,
        loaded: bool = True,
        uses_adapter: bool = True,
        fail_load: bool = False,
        delay_s: float = 0.0,
        fail_generate: bool = False,
    ) -> None:
        super().__init__("fake/model")
        self._uses_adapter = uses_adapter
        self._fail_load = fail_load
        self._delay_s = delay_s
        self._fail_generate = fail_generate
        self.calls: list[dict] = []
        if loaded:
            self._state = "ready"

    def _load_impl(self) -> None:
        if self._fail_load:
            raise RuntimeError("boom")

    def stream(
        self,
        messages: Sequence[Message],
        params: GenerationParams,
        cancel: threading.Event | None = None,
    ) -> Iterator[StreamEvent]:
        self.load()
        self.calls.append({"messages": list(messages), "params": params})
        if self._fail_generate:
            raise RuntimeError("generation exploded")
        if self._delay_s:
            time.sleep(self._delay_s)
        answer = f"echo: {messages[-1]['content']}"
        words = answer.split(" ")
        for i, word in enumerate(words):
            yield StreamEvent(delta=word if i == 0 else f" {word}")
        yield StreamEvent(finish_reason="stop", prompt_tokens=7, completion_tokens=len(words))


@pytest.fixture
def make_settings():
    """Factory for isolated Settings that never read a real .env file."""

    def _make(**overrides) -> Settings:
        base = {
            "warmup_on_startup": False,
            "api_key": None,
            "database_url": None,
            # Never pick up a locally built index (outputs/rag_index) in tests.
            "rag_index_dir": Path(__file__).parent / "_no_rag_index",
        }
        base.update(overrides)
        return Settings(_env_file=None, **base)

    return _make


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def client(make_settings, fake_engine) -> TestClient:
    app = create_app(settings=make_settings(), engine=fake_engine)
    with TestClient(app) as test_client:
        yield test_client
