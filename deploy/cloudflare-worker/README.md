# fastapi-assistant-edge (Cloudflare Worker)

Edge layer in front of the FastAPI inference service. Each origin replica runs a
~1B model on CPU, so an uncached answer costs 5-20 s. The Worker:

- **rate limits** per API key (or per client IP when anonymous) in a fixed one-minute window;
- serves a **shared answer cache** for non-streaming `POST /generate` and `POST /v1/chat/completions`;
- **fails over** across `ORIGINS` in order (Cloud Run first, then backups);
- **locks the origins**: it adds `X-Origin-Secret`, and origins reject requests without it.

It fits the Workers free plan: one Upstash pipeline call before the origin, at most
one after (inside `waitUntil`), and only hashing/small JSON work on the CPU path.

## Request flow

1. `/metrics` and unknown paths: 404 at the edge (no Redis, no origin).
2. `/generate`, `/v1/*` without `X-API-Key` / `Authorization: Bearer`: 401 at the edge
   (when `REQUIRE_API_KEY=true`). CORS preflights (`OPTIONS`) skip this check.
3. One Upstash `/pipeline`: `INCR` + `EXPIRE 70` on `rl:{identity}:{epochMinute}`, plus
   `GET` of the cache entry and key marker for cacheable requests.
4. Over the limit: 429 `{"error": {"message", "type": "rate_limited"}}` with `Retry-After`.
5. Cache hit allowed: answer from Redis (`X-Cache: HIT`).
6. Otherwise forward. Failover on network error, timeout, 429, 502, 503, 504; any other
   status (including 400/401/500) is returned as is. SSE bodies are streamed, never buffered.

Identity is the first 32 hex chars of SHA-256(API key), or `ip:<CF-Connecting-IP>`;
the raw key is never logged or stored. Public paths (`/health`, `/ready`, `/docs`,
`/redoc`, `/openapi.json`) and preflights are always limited per IP with the anon limit.

Response headers added: `X-Cache: HIT|MISS|BYPASS`, `X-Served-By: origin-<i>` (or
`edge-cache`), `X-Request-ID` (client value kept if log-safe, else generated),
`X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` (Unix seconds).

## Configuration

| Name | Kind | Default | Meaning |
|---|---|---|---|
| `ORIGINS` | var | (required) | Comma-separated origin base URLs, primary first |
| `ORIGIN_SHARED_SECRET` | secret | (required) | Sent as `X-Origin-Secret`; must match the origins |
| `UPSTASH_REDIS_REST_URL` | secret | unset | Upstash REST URL; unset disables rate limit + cache |
| `UPSTASH_REDIS_REST_TOKEN` | secret | unset | Upstash REST token |
| `RATE_LIMIT_PER_MINUTE` | var | `20` | Requests/minute per API key |
| `RATE_LIMIT_PER_MINUTE_ANON` | var | `10` | Requests/minute per IP (anonymous and public paths) |
| `CACHE_TTL_SECONDS` | var | `86400` | Lifetime of a cached answer |
| `ORIGIN_TIMEOUT_MS` | var | `60000` | Per-origin wait for response headers before failover |
| `MODEL_VERSION` | var | `v1` | Cache namespace; bump to invalidate every cached answer |
| `REQUIRE_API_KEY` | var | `true` | Reject keyless calls to protected paths at the edge |
| `KEY_VERIFIED_TTL_SECONDS` | var | `300` | Lifetime of the "origin accepted this key" marker |

Redis errors or timeouts (2 s) fail open: the request is proxied without rate limiting
or caching (`X-Cache: BYPASS`) and the error is logged.

## Cache-safety rule

The cache is shared across users, but only the origin can validate a key. So:

- Cache key: `c:{MODEL_VERSION}:{sha256(path + canonical JSON)}`; canonical JSON sorts
  object keys recursively, so key order never matters. `stream` must be absent/false.
- A hit is served only if `kv:{identity}` exists. That marker is written (TTL
  `KEY_VERIFIED_TTL_SECONDS`) only after the origin returned 2xx for that key, so a
  revoked key loses cache access within that TTL. With `REQUIRE_API_KEY=false` hits
  are served to anyone.
- Only `200` JSON responses up to 64 KB are stored. Error responses never are.
- Cache hits are answered at the edge and carry no origin CORS headers.

## Upstash command budget per request

| Request | Before origin | After origin | Commands |
|---|---|---|---|
| 401 (no key), 404, `/metrics` | none | none | 0 |
| Public path, preflight, non-cacheable or streaming call | INCR, EXPIRE | none | 2 |
| Cacheable, HIT | INCR, EXPIRE, GET entry, GET marker | none | 4 |
| Cacheable, MISS, origin 200 | same 4 | SET entry (+ SET marker if absent) | 5-6 |

Each row is at most two HTTP subrequests to Upstash. At ~4 commands per request the
500k/month free tier covers roughly 125k requests/month (~4k/day), well below the
Workers limit of 100k/day, so Upstash is the quota to watch.

## Local development

```bash
npm ci
npm test            # vitest in plain Node (no workerd needed)
npm run typecheck   # tsc --noEmit, strict
```

`fetch` and the Redis client are injected via `createHandler()` in `src/index.ts`, so
the tests use an in-memory Redis and mocked origins. Source layout: `config.ts`,
`routes.ts`, `identity.ts`, `ratelimit.ts`, `cache.ts`, `upstash.ts`, `proxy.ts`.

The npm scripts call `node ./node_modules/...` directly: Windows `.cmd` shims break
when the checkout path contains `&`. If `npx` fails for that reason, use
`node ./node_modules/wrangler/bin/wrangler.js` in place of `npx wrangler`.

## Deploy

1. Set `ORIGIN_SHARED_SECRET` on every origin replica (FastAPI setting
   `origin_shared_secret`), then set `ORIGINS` in `wrangler.toml`.
2. Log in and set the secrets:

   ```bash
   npx wrangler login
   npx wrangler secret put ORIGIN_SHARED_SECRET
   npx wrangler secret put UPSTASH_REDIS_REST_URL
   npx wrangler secret put UPSTASH_REDIS_REST_TOKEN
   ```

3. Deploy and smoke-test:

   ```bash
   npx wrangler deploy
   curl -i https://fastapi-assistant-edge.<your-subdomain>.workers.dev/health
   curl -i -X POST https://fastapi-assistant-edge.<your-subdomain>.workers.dev/generate \
     -H "X-API-Key: $KEY" -H "Content-Type: application/json" -d '{"question":"hi"}'
   ```

   Send the second request twice: the repeat should return `X-Cache: HIT`.
4. Watch logs with `npx wrangler tail` (one JSON line per request) or in the dashboard
   (observability is enabled). To drop every cached answer, bump `MODEL_VERSION`.
