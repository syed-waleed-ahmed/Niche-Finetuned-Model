"""In-process store for development and tests (not durable, single replica)."""

from __future__ import annotations

import dataclasses
import threading
from datetime import datetime, timezone

from .base import (
    ApiKeyRecord,
    Feedback,
    Principal,
    UsageEvent,
    display_prefix,
    generate_api_key,
    hash_api_key,
)


class MemoryStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: dict[str, ApiKeyRecord] = {}  # key hash -> record
        self._next_id = 1
        self.usage: list[UsageEvent] = []
        self.feedback: list[Feedback] = []

    def ping(self) -> bool:
        return True

    def lookup_key(self, key_hash: str) -> Principal | None:
        record = self._keys.get(key_hash)
        if record is None or not record.active:
            return None
        return Principal(key_id=record.id, name=record.name)

    def create_key(self, name: str) -> tuple[str, ApiKeyRecord]:
        key = generate_api_key()
        with self._lock:
            record = ApiKeyRecord(
                id=self._next_id,
                name=name,
                prefix=display_prefix(key),
                active=True,
                created_at=datetime.now(timezone.utc),
            )
            self._next_id += 1
            self._keys[hash_api_key(key)] = record
        return key, record

    def revoke_key(self, prefix: str) -> int:
        count = 0
        with self._lock:
            for key_hash, record in self._keys.items():
                if record.prefix == prefix and record.active:
                    self._keys[key_hash] = dataclasses.replace(record, active=False)
                    count += 1
        return count

    def list_keys(self) -> list[ApiKeyRecord]:
        return sorted(self._keys.values(), key=lambda r: r.id)

    def record_usage(self, event: UsageEvent) -> None:
        with self._lock:
            self.usage.append(event)

    def record_feedback(self, feedback: Feedback) -> None:
        with self._lock:
            self.feedback.append(feedback)

    def close(self) -> None:
        pass
