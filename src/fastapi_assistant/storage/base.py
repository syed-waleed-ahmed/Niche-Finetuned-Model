"""Persistence contract: API keys, usage events, and feedback.

Keys are stored as SHA-256 hashes; the plaintext is shown once at creation and
never persisted. Usage and feedback are append-only events keyed by request id,
which lets feedback be joined back to the exact answer it rates.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

KEY_PREFIX = "fa_"


def generate_api_key() -> str:
    """Return a new random API key (about 256 bits of entropy)."""
    return KEY_PREFIX + secrets.token_urlsafe(32)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def display_prefix(key: str) -> str:
    """The non-secret part of a key shown in listings (e.g. ``fa_AbC123``)."""
    return key[: len(KEY_PREFIX) + 6]


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""

    key_id: int | None
    name: str


ANONYMOUS = Principal(key_id=None, name="anonymous")
STATIC_KEY_PRINCIPAL = Principal(key_id=None, name="static-key")


@dataclass(frozen=True)
class ApiKeyRecord:
    id: int
    name: str
    prefix: str
    active: bool
    created_at: datetime


@dataclass
class UsageEvent:
    request_id: str
    endpoint: str
    status_code: int
    latency_ms: float
    model_version: str
    key_id: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    question: str | None = None
    answer: str | None = None


@dataclass
class Feedback:
    request_id: str
    rating: int  # -1 or 1
    key_id: int | None = None
    comment: str | None = None
    extra: dict = field(default_factory=dict)


class Store(Protocol):
    def ping(self) -> bool: ...

    def lookup_key(self, key_hash: str) -> Principal | None: ...

    def create_key(self, name: str) -> tuple[str, ApiKeyRecord]: ...

    def revoke_key(self, prefix: str) -> int: ...

    def list_keys(self) -> list[ApiKeyRecord]: ...

    def record_usage(self, event: UsageEvent) -> None: ...

    def record_feedback(self, feedback: Feedback) -> None: ...

    def close(self) -> None: ...
