/**
 * Edge layer in front of the FastAPI inference service.
 *
 * Per request, in order:
 *   1. route check (/metrics and unknown paths -> 404 at the edge)
 *   2. API key presence check on protected paths (401 without Redis/origin)
 *   3. body size cap (413), so oversized bodies never reach Redis or the origin
 *   4. ONE Upstash pipeline: rate-limit counter (+ cache entry + key marker);
 *      an in-memory per-isolate counter when Upstash is absent or failing
 *   5. 429 when over the limit, cached answer when allowed
 *   6. forward with failover and a circuit breaker; optional single pipeline
 *      write in waitUntil
 */

import {
  cacheKey,
  cacheWriteCommand,
  isJsonContentType,
  keyMarker,
  markerWriteCommand,
  parseCacheableBody,
  type CacheStatus,
} from "./cache";
import { readBodyCapped } from "./body";
import { DEFAULTS, loadConfig, type Config, type Env } from "./config";
import { jsonError, sanitizeRequestId, sanitizeResponseHeaders, SECURITY_HEADERS } from "./http";
import { clientIp, extractApiKey, ipIdentity, keyIdentity, type Identity } from "./identity";
import {
  buildForwardHeaders,
  CircuitBreaker,
  forwardWithFailover,
  RETRYABLE_STATUSES,
  rewriteLocation,
} from "./proxy";
import {
  evaluateRateLimit,
  LocalRateLimiter,
  rateLimitCommands,
  rateLimitHeaders,
  type RateLimitState,
} from "./ratelimit";
import { classifyPath, isCacheableEndpoint } from "./routes";
import {
  createUpstashClient,
  type FetchLike,
  type RedisClient,
  type RedisCommand,
  type RedisReply,
} from "./upstash";

export type { Env } from "./config";

export interface WaitUntilContext {
  waitUntil(promise: Promise<unknown>): void;
}

export interface Deps {
  /** Used for origin requests and (unless `redis` is given) for Upstash. */
  fetch: FetchLike;
  /** Overrides the Redis client. `null` disables Redis; undefined builds it from config. */
  redis?: RedisClient | null;
  now: () => number;
  log: (entry: Record<string, unknown>) => void;
  randomId: () => string;
  /** Per-isolate fallback rate limiter (created per handler when not given). */
  localLimiter?: LocalRateLimiter;
  /** Per-isolate origin circuit breaker (created per handler when not given). */
  breaker?: CircuitBreaker;
}

/** Hex characters of the key identity (a truncated SHA-256) written to the request log. */
const LOG_KEY_ID_LENGTH = 12;

/** State that lives as long as the isolate: one instance per handler. */
interface IsolateState {
  limiter: LocalRateLimiter;
  breaker: CircuitBreaker;
}

interface RequestLog {
  event: "request";
  method: string;
  path: string;
  status: number;
  cache: CacheStatus | null;
  rl_identity: Identity["type"] | null;
  /** The first characters of the presented key's SHA-256; never the key, never an IP. */
  rl_key_id?: string;
  rl_scope?: "redis" | "local";
  upstream_normalized?: boolean;
  /** The client disconnected before an origin answered; the origin request was aborted. */
  client_gone?: boolean;
  origin_index: number | null;
  skipped_origins?: number[];
  attempts: number;
  duration_ms: number;
  request_id: string;
  redis_error?: string;
  upstream_failures?: string[];
  error?: string;
}

const defaultDeps: Deps = {
  fetch: (input, init) => fetch(input, init),
  now: () => Date.now(),
  log: (entry) => console.log(JSON.stringify(entry)),
  randomId: () => crypto.randomUUID(),
};

const decoder = new TextDecoder();

/** Retry hint for platform-generated 429/5xx that carry none (a new instance warming). */
const PLATFORM_RETRY_AFTER_SECONDS = 10;

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function redisFor(config: Config, deps: Deps): RedisClient | null {
  if (deps.redis !== undefined) return deps.redis;
  if (!config.upstash) return null;
  return createUpstashClient({ ...config.upstash, timeoutMs: config.redisTimeoutMs }, deps.fetch);
}

function toCount(reply: RedisReply | undefined): number | null {
  const value = reply?.result;
  if (typeof value === "number") return value;
  if (typeof value === "string" && /^\d+$/.test(value)) return Number(value);
  return null;
}

function setAll(headers: Headers, values: Record<string, string>): void {
  for (const [name, value] of Object.entries(values)) headers.set(name, value);
}

export function createHandler(overrides: Partial<Deps> = {}) {
  const deps: Deps = { ...defaultDeps, ...overrides };
  const state: IsolateState = {
    limiter: deps.localLimiter ?? new LocalRateLimiter(),
    breaker:
      deps.breaker ?? new CircuitBreaker(DEFAULTS.BREAKER_FAILURES, DEFAULTS.BREAKER_COOLDOWN_MS),
  };
  return async (request: Request, env: Env, ctx: WaitUntilContext): Promise<Response> => {
    const started = deps.now();
    const url = new URL(request.url);
    const log: RequestLog = {
      event: "request",
      method: request.method,
      path: url.pathname,
      status: 0,
      cache: null,
      rl_identity: null,
      origin_index: null,
      attempts: 0,
      duration_ms: 0,
      request_id: sanitizeRequestId(request.headers.get("x-request-id")) ?? deps.randomId(),
    };
    const finish = (response: Response): Response => {
      log.status = response.status;
      log.duration_ms = deps.now() - started;
      deps.log({ ...log });
      return response;
    };

    try {
      return finish(await handle(request, url, env, ctx, deps, state, log));
    } catch (err) {
      log.error = errorMessage(err);
      return finish(
        jsonError(500, "Internal edge error", "edge_error", { "x-request-id": log.request_id }),
      );
    }
  };
}

async function handle(
  request: Request,
  url: URL,
  env: Env,
  ctx: WaitUntilContext,
  deps: Deps,
  state: IsolateState,
  log: RequestLog,
): Promise<Response> {
  const config = loadConfig(env);
  const path = url.pathname;
  const method = request.method.toUpperCase();
  const requestId = log.request_id;
  const idHeader = { "x-request-id": requestId };

  // 1. Routing: internal and unknown paths never reach Redis or the origin.
  const route = classifyPath(path);
  if (route === "blocked" || route === "unknown") {
    return jsonError(404, "Not Found", "not_found", idHeader);
  }

  // 2. Identity. CORS preflights carry no credentials, so they are treated as public.
  const isProtected = route === "protected" && method !== "OPTIONS";
  const apiKey = isProtected ? extractApiKey(request.headers) : null;
  if (isProtected && config.requireApiKey && !apiKey) {
    return jsonError(401, "Missing API key. Send X-API-Key or Authorization: Bearer <key>.", "authentication_error", {
      ...idHeader,
      "www-authenticate": "Bearer",
    });
  }
  const identity = apiKey ? await keyIdentity(apiKey) : ipIdentity(request.headers);
  const limit = identity.type === "key" ? config.rateLimitPerMinute : config.rateLimitPerMinuteAnon;
  log.rl_identity = identity.type;
  // Enough to tell callers apart and to find the key to revoke; an IP identity is
  // left out, so the logs hold no personal data.
  if (identity.type === "key") log.rl_key_id = identity.id.slice(0, LOG_KEY_ID_LENGTH);

  // 3. Buffer the body once (capped) so it can be hashed and replayed across origins.
  const read = await readBodyCapped(request, config.maxRequestBytes);
  if (!read.ok) {
    return jsonError(
      413,
      "Request body is larger than " + config.maxRequestBytes + " bytes",
      "request_too_large",
      idHeader,
    );
  }
  const body = read.body;

  // 4. One Redis pipeline: rate limit counter, and for cacheable requests the
  //    cache entry and the verified-key marker.
  const redis = redisFor(config, deps);
  let cKey: string | null = null;
  if (redis && isProtected && isCacheableEndpoint(method, path)) {
    const parsed = parseCacheableBody(body, config.maxCacheableRequestBytes);
    if (parsed) cKey = await cacheKey(config.modelVersion, path, parsed);
  }
  const checkMarker = cKey !== null && config.requireApiKey && identity.type === "key";

  const now = deps.now();
  let rateLimit: RateLimitState | null = null;
  let cached: string | null = null;
  let markerPresent = false;
  let redisOk = false;
  if (redis) {
    const commands: RedisCommand[] = rateLimitCommands(identity.id, now);
    const cacheIdx = cKey ? commands.push(["GET", cKey]) - 1 : -1;
    const markerIdx = checkMarker ? commands.push(["GET", keyMarker(identity.id)]) - 1 : -1;
    try {
      const replies = await redis.pipeline(commands);
      redisOk = true;
      const commandErrors = replies.filter((r) => r?.error).map((r) => r.error);
      if (commandErrors.length) log.redis_error = commandErrors.join("; ");
      const count = toCount(replies[0]);
      if (count !== null) rateLimit = evaluateRateLimit(count, limit, now);
      if (cacheIdx >= 0) {
        const value = replies[cacheIdx]?.result;
        cached = typeof value === "string" ? value : null;
      }
      if (markerIdx >= 0) {
        const value = replies[markerIdx]?.result;
        markerPresent = value !== null && value !== undefined;
      }
    } catch (err) {
      // No caching for this request; rate limiting falls back to this isolate.
      log.redis_error = errorMessage(err);
    }
  }
  // Redis answered but the INCR reply was unusable (a command-level error): count
  // locally rather than letting the request through unlimited.
  if (redisOk && rateLimit !== null) {
    log.rl_scope = "redis";
  } else {
    rateLimit = evaluateRateLimit(state.limiter.hit(identity.id, now), limit, now);
    log.rl_scope = "local";
  }
  const rlHeaders = rateLimit ? rateLimitHeaders(rateLimit) : {};

  // 5a. Over the limit.
  if (rateLimit?.exceeded) {
    return jsonError(
      429,
      `Rate limit exceeded: ${rateLimit.limit} requests per minute. Retry after ${rateLimit.retryAfterSeconds} s.`,
      "rate_limited",
      { ...idHeader, ...rlHeaders, "retry-after": String(rateLimit.retryAfterSeconds) },
    );
  }

  // 5b. Shared cache: a hit is only served to a key the origin recently accepted.
  const cacheActive = cKey !== null && redisOk;
  const cacheStatus: CacheStatus = cacheActive ? "MISS" : "BYPASS";
  log.cache = cacheStatus;
  if (cacheActive && cached !== null && (markerPresent || !config.requireApiKey)) {
    log.cache = "HIT";
    return new Response(cached, {
      status: 200,
      headers: {
        "content-type": "application/json",
        "x-cache": "HIT",
        "x-served-by": "edge-cache",
        ...SECURITY_HEADERS,
        ...idHeader,
        ...rlHeaders,
      },
    });
  }

  // 6. Forward with failover.
  const result = await forwardWithFailover(
    {
      method: request.method,
      pathAndQuery: path + url.search,
      headers: buildForwardHeaders(request.headers, {
        originSecret: config.originSecret,
        requestId,
        clientIp: clientIp(request.headers),
      }),
      body,
    },
    {
      origins: config.origins,
      timeoutMs: config.originTimeoutMs,
      fetch: deps.fetch,
      breaker: state.breaker,
      now: deps.now,
      clientSignal: request.signal,
    },
  );
  log.attempts = result.attempts;
  log.origin_index = result.originIndex;
  if (result.failures.length) log.upstream_failures = result.failures;
  if (result.skipped.length) log.skipped_origins = result.skipped;

  const upstream = result.response;
  if (!upstream) {
    const errorHeaders = { ...idHeader, ...rlHeaders, "x-cache": cacheStatus };
    if (result.clientGone) {
      // Nobody reads this response; the status is for the request log (499 is the
      // conventional "client closed request").
      log.client_gone = true;
      return jsonError(499, "Client closed the request", "client_closed_request", errorHeaders);
    }
    if (result.timedOut) {
      return jsonError(504, "The origin did not respond in time", "upstream_timeout", errorHeaders);
    }
    const message = config.origins.length ? "No origin available" : "No origin configured";
    return jsonError(502, message, "upstream_unavailable", errorHeaders);
  }

  // The platform in front of the origin (e.g. Cloud Run's front end while it scales
  // out) can answer 429/5xx itself, as plain text without Retry-After. Clients get the
  // same JSON error shape and a retry hint whichever layer refused the request.
  if (
    RETRYABLE_STATUSES.has(upstream.status) &&
    !isJsonContentType(upstream.headers.get("content-type"))
  ) {
    log.upstream_normalized = true;
    await upstream.body?.cancel();
    const retryAfter = upstream.headers.get("retry-after") ?? String(PLATFORM_RETRY_AFTER_SECONDS);
    return jsonError(
      upstream.status,
      upstream.status === 429
        ? "The service is at capacity. Retry after " + retryAfter + " s."
        : "The origin is temporarily unavailable. Retry after " + retryAfter + " s.",
      upstream.status === 429 ? "overloaded" : "upstream_unavailable",
      {
        ...idHeader,
        ...rlHeaders,
        "x-cache": cacheStatus,
        "x-served-by": "origin-" + result.originIndex,
        "retry-after": retryAfter,
      },
    );
  }

  // Cache write-back (MISS only): answer on 200 JSON, key marker on any 2xx.
  let responseBody: ReadableStream | ArrayBuffer | null = upstream.body;
  if (cacheActive && redis && cKey) {
    const writes: RedisCommand[] = [];
    const ok2xx = upstream.status >= 200 && upstream.status < 300;
    if (ok2xx && checkMarker && !markerPresent) {
      writes.push(markerWriteCommand(identity.id, config.keyVerifiedTtlSeconds));
    }
    const declaredLength = Number(upstream.headers.get("content-length") ?? "0");
    if (
      upstream.status === 200 &&
      cached === null &&
      isJsonContentType(upstream.headers.get("content-type")) &&
      declaredLength <= config.maxCachedBodyBytes
    ) {
      const buffered = await upstream.arrayBuffer();
      responseBody = buffered;
      if (buffered.byteLength <= config.maxCachedBodyBytes) {
        writes.push(cacheWriteCommand(cKey, decoder.decode(buffered), config.cacheTtlSeconds));
      }
    }
    if (writes.length) {
      ctx.waitUntil(
        redis.pipeline(writes).then(
          () => undefined,
          (err: unknown) =>
            deps.log({ event: "cache_write_error", request_id: requestId, error: errorMessage(err) }),
        ),
      );
    }
  }

  const headers = new Headers(upstream.headers);
  sanitizeResponseHeaders(headers);
  if (result.originBase) rewriteLocation(headers, result.originBase, url.origin);
  if (!headers.has("x-request-id")) headers.set("x-request-id", requestId);
  headers.set("x-cache", cacheStatus);
  headers.set("x-served-by", `origin-${result.originIndex}`);
  setAll(headers, rlHeaders);
  return new Response(responseBody, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers,
  });
}

export default {
  fetch: createHandler(),
} satisfies ExportedHandler<Env>;
