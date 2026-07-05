"""Shared pytest fixtures.

These tests never load a real model or hit the network — a lightweight fake engine
stands in for :class:`~fastapi_assistant.inference.AssistantEngine`, so the suite
runs fast and deterministically in CI.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fastapi_assistant.api import create_app
from fastapi_assistant.config import Settings


class FakeEngine:
    """A stand-in engine that echoes input instead of running a model."""

    def __init__(self, *, loaded: bool = True, uses_adapter: bool = True, fail_load: bool = False):
        self._loaded = loaded
        self._uses_adapter = uses_adapter
        self._fail_load = fail_load
        self.calls: list[dict] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def uses_adapter(self) -> bool:
        return self._uses_adapter

    def load(self) -> None:
        if self._fail_load:
            from fastapi_assistant.inference import ModelNotReadyError

            raise ModelNotReadyError("boom")
        self._loaded = True

    def generate(self, question: str, **kwargs) -> str:
        self.calls.append({"question": question, **kwargs})
        return f"echo: {question}"


@pytest.fixture
def make_settings():
    """Factory for isolated Settings that never read a real .env file."""

    def _make(**overrides) -> Settings:
        base = {"warmup_on_startup": False, "api_key": None}
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
