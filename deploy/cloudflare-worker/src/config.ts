/**
 * Worker configuration: `[vars]` from wrangler.toml plus secrets.
 *
 * Wrangler may deliver a var as a string, number or boolean depending on how it
 * was written in wrangler.toml, so every value is parsed defensively and falls
 * back to its documented default when absent or invalid.
 */

type VarValue = string | number | boolean | undefined;

export interface Env {
  ORIGINS?: VarValue;
  ORIGIN_SHARED_SECRET?: VarValue;
  UPSTASH_REDIS_REST_URL?: VarValue;
  UPSTASH_REDIS_REST_TOKEN?: VarValue;
  RATE_LIMIT_PER_MINUTE?: VarValue;
  RATE_LIMIT_PER_MINUTE_ANON?: VarValue;
  CACHE_TTL_SECONDS?: VarValue;
  ORIGIN_TIMEOUT_MS?: VarValue;
  MODEL_VERSION?: VarValue;
  REQUIRE_API_KEY?: VarValue;
  KEY_VERIFIED_TTL_SECONDS?: VarValue;
  MAX_REQUEST_BYTES?: VarValue;
}

export interface Config {
  /** Origin base URLs without trailing slash, primary first. */
  origins: string[];
  originSecret: string;
  /** Null when either Upstash var is missing: rate limiting and caching are then disabled. */
  upstash: { url: string; token: string } | null;
  rateLimitPerMinute: number;
  rateLimitPerMinuteAnon: number;
  cacheTtlSeconds: number;
  originTimeoutMs: number;
  modelVersion: string;
  requireApiKey: boolean;
  keyVerifiedTtlSeconds: number;
  /** Largest origin response body (bytes) written to the shared cache. */
  maxCachedBodyBytes: number;
  /** Largest request body (bytes) that is parsed and hashed for a cache lookup. */
  maxCacheableRequestBytes: number;
  /** Timeout for the Upstash REST call; Redis must never stall a request for long. */
  redisTimeoutMs: number;
  /** Largest request body accepted at the edge (413 above it). */
  maxRequestBytes: number;
}

export const DEFAULTS = {
  RATE_LIMIT_PER_MINUTE: 20,
  RATE_LIMIT_PER_MINUTE_ANON: 10,
  CACHE_TTL_SECONDS: 86_400,
  ORIGIN_TIMEOUT_MS: 60_000,
  MODEL_VERSION: "v1",
  REQUIRE_API_KEY: true,
  KEY_VERIFIED_TTL_SECONDS: 300,
  MAX_CACHED_BODY_BYTES: 64 * 1024,
  MAX_CACHEABLE_REQUEST_BYTES: 64 * 1024,
  REDIS_TIMEOUT_MS: 2_000,
  MAX_REQUEST_BYTES: 64 * 1024,
  /** Consecutive network failures or timeouts that open an origin's circuit. */
  BREAKER_FAILURES: 3,
  /** How long an open circuit skips its origin (while another origin is available). */
  BREAKER_COOLDOWN_MS: 30_000,
} as const;

function str(value: VarValue): string {
  return value === undefined ? "" : String(value).trim();
}

function positiveInt(value: VarValue, fallback: number): number {
  const parsed = Number.parseInt(str(value), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function bool(value: VarValue, fallback: boolean): boolean {
  const s = str(value).toLowerCase();
  if (s === "true" || s === "1" || s === "yes") return true;
  if (s === "false" || s === "0" || s === "no") return false;
  return fallback;
}

/** Strip trailing "/" characters (a loop, not a regex: linear on any input). */
export function trimTrailingSlashes(value: string): string {
  let end = value.length;
  while (end > 0 && value.charCodeAt(end - 1) === 47) end--;
  return value.slice(0, end);
}

export function parseOrigins(value: VarValue): string[] {
  return str(value)
    .split(",")
    .map((o) => trimTrailingSlashes(o.trim()))
    .filter((o) => o.length > 0);
}

export function loadConfig(env: Env): Config {
  const redisUrl = trimTrailingSlashes(str(env.UPSTASH_REDIS_REST_URL));
  const redisToken = str(env.UPSTASH_REDIS_REST_TOKEN);
  return {
    origins: parseOrigins(env.ORIGINS),
    originSecret: str(env.ORIGIN_SHARED_SECRET),
    upstash: redisUrl && redisToken ? { url: redisUrl, token: redisToken } : null,
    rateLimitPerMinute: positiveInt(env.RATE_LIMIT_PER_MINUTE, DEFAULTS.RATE_LIMIT_PER_MINUTE),
    rateLimitPerMinuteAnon: positiveInt(
      env.RATE_LIMIT_PER_MINUTE_ANON,
      DEFAULTS.RATE_LIMIT_PER_MINUTE_ANON,
    ),
    cacheTtlSeconds: positiveInt(env.CACHE_TTL_SECONDS, DEFAULTS.CACHE_TTL_SECONDS),
    originTimeoutMs: positiveInt(env.ORIGIN_TIMEOUT_MS, DEFAULTS.ORIGIN_TIMEOUT_MS),
    modelVersion: str(env.MODEL_VERSION) || DEFAULTS.MODEL_VERSION,
    requireApiKey: bool(env.REQUIRE_API_KEY, DEFAULTS.REQUIRE_API_KEY),
    keyVerifiedTtlSeconds: positiveInt(
      env.KEY_VERIFIED_TTL_SECONDS,
      DEFAULTS.KEY_VERIFIED_TTL_SECONDS,
    ),
    maxCachedBodyBytes: DEFAULTS.MAX_CACHED_BODY_BYTES,
    maxCacheableRequestBytes: DEFAULTS.MAX_CACHEABLE_REQUEST_BYTES,
    redisTimeoutMs: DEFAULTS.REDIS_TIMEOUT_MS,
    maxRequestBytes: positiveInt(env.MAX_REQUEST_BYTES, DEFAULTS.MAX_REQUEST_BYTES),
  };
}
