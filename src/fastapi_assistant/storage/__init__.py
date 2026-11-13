"""Persistence for keys, usage, and feedback."""

from __future__ import annotations

from ..config import Settings
from .base import (
    ANONYMOUS,
    STATIC_KEY_PRINCIPAL,
    ApiKeyRecord,
    Feedback,
    Principal,
    Store,
    UsageEvent,
    generate_api_key,
    hash_api_key,
)
from .memory import MemoryStore

__all__ = [
    "ANONYMOUS",
    "STATIC_KEY_PRINCIPAL",
    "ApiKeyRecord",
    "Feedback",
    "MemoryStore",
    "Principal",
    "Store",
    "UsageEvent",
    "create_store",
    "generate_api_key",
    "hash_api_key",
]


def create_store(settings: Settings) -> Store | None:
    """Return a Postgres store when ``DATABASE_URL`` is set, else ``None``."""
    if not settings.database_url:
        return None
    from .postgres import PostgresStore

    return PostgresStore(settings.database_url, max_size=settings.db_pool_max_size)
