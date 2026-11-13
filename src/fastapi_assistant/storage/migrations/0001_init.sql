-- Core schema: API keys, usage events, and answer feedback.

CREATE TABLE IF NOT EXISTS api_keys (
    id          BIGSERIAL PRIMARY KEY,
    name        TEXT        NOT NULL,
    key_prefix  TEXT        NOT NULL,
    key_hash    TEXT        NOT NULL UNIQUE,
    active      BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS api_keys_prefix_idx ON api_keys (key_prefix);

CREATE TABLE IF NOT EXISTS usage_events (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ      NOT NULL DEFAULT now(),
    request_id        TEXT             NOT NULL,
    key_id            BIGINT           REFERENCES api_keys (id),
    endpoint          TEXT             NOT NULL,
    status_code       INTEGER          NOT NULL,
    prompt_tokens     INTEGER          NOT NULL DEFAULT 0,
    completion_tokens INTEGER          NOT NULL DEFAULT 0,
    latency_ms        DOUBLE PRECISION NOT NULL,
    finish_reason     TEXT,
    model_version     TEXT             NOT NULL,
    question          TEXT,
    answer            TEXT
);

CREATE INDEX IF NOT EXISTS usage_events_created_idx ON usage_events (created_at);
CREATE INDEX IF NOT EXISTS usage_events_key_created_idx ON usage_events (key_id, created_at);
CREATE INDEX IF NOT EXISTS usage_events_request_idx ON usage_events (request_id);

CREATE TABLE IF NOT EXISTS feedback (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    request_id  TEXT        NOT NULL,
    key_id      BIGINT      REFERENCES api_keys (id),
    rating      SMALLINT    NOT NULL CHECK (rating IN (-1, 1)),
    comment     TEXT
);

CREATE INDEX IF NOT EXISTS feedback_request_idx ON feedback (request_id);
