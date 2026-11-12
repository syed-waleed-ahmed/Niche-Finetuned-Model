"""Postgres store (Neon free tier, or any Postgres 14+ with pgvector).

Uses a small psycopg 3 connection pool that opens lazily and keeps no idle
connections (``min_size=0``), so a scale-to-zero database is not kept awake by an
idle replica. Schema changes are plain SQL files applied in order by
:func:`migrate`.
"""

from __future__ import annotations

import logging
from importlib import resources

from .base import (
    ApiKeyRecord,
    Feedback,
    Principal,
    UsageEvent,
    display_prefix,
    generate_api_key,
    hash_api_key,
)

log = logging.getLogger(__name__)


def _migration_files() -> list[tuple[str, str]]:
    root = resources.files("fastapi_assistant.storage") / "migrations"
    files = sorted(
        (entry for entry in root.iterdir() if entry.name.endswith(".sql")),
        key=lambda e: e.name,
    )
    return [(entry.name.removesuffix(".sql"), entry.read_text(encoding="utf-8")) for entry in files]


class PostgresStore:
    def __init__(self, database_url: str, max_size: int = 4) -> None:
        from psycopg_pool import ConnectionPool

        self._pool = ConnectionPool(
            conninfo=database_url,
            min_size=0,
            max_size=max_size,
            max_idle=60.0,
            open=False,
            kwargs={"autocommit": True},
        )
        self._opened = False

    def _conn(self):
        if not self._opened:
            self._pool.open(wait=False)
            self._opened = True
        return self._pool.connection(timeout=10.0)

    # -- schema --------------------------------------------------------------
    def migrate(self) -> list[str]:
        """Apply pending migrations; return the versions applied."""
        applied: list[str] = []
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " version TEXT PRIMARY KEY,"
                " applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
            done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
            for version, sql in _migration_files():
                if version in done:
                    continue
                log.info("Applying migration %s", version)
                with conn.transaction():
                    conn.execute(sql)
                    conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
                applied.append(version)
        return applied

    # -- Store protocol --------------------------------------------------------
    def ping(self) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            log.exception("Database ping failed")
            return False

    def lookup_key(self, key_hash: str) -> Principal | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, name FROM api_keys WHERE key_hash = %s AND active",
                (key_hash,),
            ).fetchone()
        return Principal(key_id=row[0], name=row[1]) if row else None

    def create_key(self, name: str) -> tuple[str, ApiKeyRecord]:
        key = generate_api_key()
        with self._conn() as conn:
            row = conn.execute(
                "INSERT INTO api_keys (name, key_prefix, key_hash) VALUES (%s, %s, %s)"
                " RETURNING id, name, key_prefix, active, created_at",
                (name, display_prefix(key), hash_api_key(key)),
            ).fetchone()
        return key, ApiKeyRecord(*row)

    def revoke_key(self, prefix: str) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE api_keys SET active = FALSE, revoked_at = now()"
                " WHERE key_prefix = %s AND active",
                (prefix,),
            )
            return cur.rowcount

    def list_keys(self) -> list[ApiKeyRecord]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, key_prefix, active, created_at FROM api_keys ORDER BY id"
            ).fetchall()
        return [ApiKeyRecord(*row) for row in rows]

    def record_usage(self, event: UsageEvent) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO usage_events (request_id, key_id, endpoint, status_code,"
                " prompt_tokens, completion_tokens, latency_ms, finish_reason, model_version,"
                " question, answer) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    event.request_id,
                    event.key_id,
                    event.endpoint,
                    event.status_code,
                    event.prompt_tokens,
                    event.completion_tokens,
                    event.latency_ms,
                    event.finish_reason,
                    event.model_version,
                    event.question,
                    event.answer,
                ),
            )

    def record_feedback(self, feedback: Feedback) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO feedback (request_id, key_id, rating, comment) VALUES (%s, %s, %s, %s)",
                (feedback.request_id, feedback.key_id, feedback.rating, feedback.comment),
            )

    def close(self) -> None:
        if self._opened:
            self._pool.close()
            self._opened = False
