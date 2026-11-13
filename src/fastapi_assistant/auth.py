"""Request authentication.

Two key sources, usable together:

* ``API_KEY`` — one static shared key (simple deployments, tests).
* ``DATABASE_URL`` — per-user keys in Postgres, stored as SHA-256 hashes.

Clients send the key as ``X-API-Key`` or ``Authorization: Bearer <key>`` (the form
OpenAI SDKs use). Database lookups are cached briefly in-process so a hot key does
not cost a query per request; revocation therefore takes effect within
``KEY_CACHE_TTL_S`` seconds.
"""

from __future__ import annotations

import hmac
import threading
import time

from .storage import ANONYMOUS, STATIC_KEY_PRINCIPAL, Principal, Store, hash_api_key


class AuthError(Exception):
    """Missing or invalid credentials (HTTP 401)."""


def extract_api_key(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return None


class _TTLCache:
    """Tiny thread-safe TTL cache for key lookups (positive and negative)."""

    def __init__(self, ttl_s: float, max_entries: int = 10_000) -> None:
        self._ttl = ttl_s
        self._max = max_entries
        self._data: dict[str, tuple[float, Principal | None]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[bool, Principal | None]:
        with self._lock:
            hit = self._data.get(key)
            if hit is None or hit[0] < time.monotonic():
                return False, None
            return True, hit[1]

    def put(self, key: str, value: Principal | None, ttl_s: float | None = None) -> None:
        with self._lock:
            if len(self._data) >= self._max:
                self._data.clear()  # crude but bounded; keys are cheap to re-check
            self._data[key] = (
                time.monotonic() + (ttl_s if ttl_s is not None else self._ttl),
                value,
            )


class Authenticator:
    def __init__(
        self,
        static_key: str | None,
        store: Store | None,
        cache_ttl_s: float = 60.0,
        allow_anonymous: bool = False,
    ) -> None:
        self._static_key = static_key
        self._store = store
        # Public demo: a request without a key is accepted as anonymous (the edge
        # rate-limits it per client IP). A key that IS sent must still be valid.
        self._allow_anonymous = allow_anonymous
        self._cache = _TTLCache(cache_ttl_s)
        self._negative_ttl = min(cache_ttl_s, 10.0)

    @property
    def enabled(self) -> bool:
        return bool(self._static_key or self._store)

    def authenticate(self, key: str | None) -> Principal:
        """Return the caller, or raise :class:`AuthError`. Blocking (may hit the DB)."""
        if not self.enabled:
            return ANONYMOUS
        if not key:
            if self._allow_anonymous:
                return ANONYMOUS
            raise AuthError("Missing API key")
        if self._static_key and hmac.compare_digest(key, self._static_key):
            return STATIC_KEY_PRINCIPAL
        if self._store is not None:
            key_hash = hash_api_key(key)
            found, principal = self._cache.get(key_hash)
            if not found:
                principal = self._store.lookup_key(key_hash)
                self._cache.put(
                    key_hash, principal, None if principal is not None else self._negative_ttl
                )
            if principal is not None:
                return principal
        raise AuthError("Invalid API key")
