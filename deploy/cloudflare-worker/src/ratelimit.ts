/**
 * Fixed one-minute window rate limiting on Redis: `INCR rl:{identity}:{epochMinute}`
 * plus `EXPIRE ... 70`, sent in the same pipeline as the cache lookup.
 */

import type { RedisCommand } from "./upstash";

export const WINDOW_SECONDS = 60;
export const KEY_EXPIRY_SECONDS = 70;

export interface RateLimitState {
  limit: number;
  remaining: number;
  /** Unix time (seconds) at which the current window ends. */
  resetEpochSeconds: number;
  /** Seconds until the window ends (at least 1). */
  retryAfterSeconds: number;
  exceeded: boolean;
}

export function epochMinute(nowMs: number): number {
  return Math.floor(nowMs / 1000 / WINDOW_SECONDS);
}

export function rateLimitKey(identityId: string, nowMs: number): string {
  return `rl:${identityId}:${epochMinute(nowMs)}`;
}

export function rateLimitCommands(identityId: string, nowMs: number): RedisCommand[] {
  const key = rateLimitKey(identityId, nowMs);
  return [
    ["INCR", key],
    ["EXPIRE", key, String(KEY_EXPIRY_SECONDS)],
  ];
}

export function evaluateRateLimit(count: number, limit: number, nowMs: number): RateLimitState {
  const resetEpochSeconds = (epochMinute(nowMs) + 1) * WINDOW_SECONDS;
  const retryAfterSeconds = Math.max(1, resetEpochSeconds - Math.floor(nowMs / 1000));
  return {
    limit,
    remaining: Math.max(0, limit - count),
    resetEpochSeconds,
    retryAfterSeconds,
    exceeded: count > limit,
  };
}

/**
 * Fallback limiter used when Upstash is not configured or does not answer: the same
 * fixed one-minute window, counted in this isolate's memory.
 *
 * It is weaker than the shared Redis counter. Cloudflare runs many isolates, each
 * counting on its own, so the effective limit is a multiple of the configured one,
 * and counts reset when an isolate is recycled. It still bounds a single client
 * hammering one location, and it never fails open.
 */
export class LocalRateLimiter {
  private readonly windows = new Map<string, { minute: number; count: number }>();

  constructor(private readonly maxIdentities = 10_000) {}

  /** Records one request and returns the count for the current window. */
  hit(identityId: string, nowMs: number): number {
    const minute = epochMinute(nowMs);
    const entry = this.windows.get(identityId);
    if (entry && entry.minute === minute) {
      entry.count++;
      return entry.count;
    }
    if (this.windows.size >= this.maxIdentities) this.evict(minute);
    this.windows.set(identityId, { minute, count: 1 });
    return 1;
  }

  get size(): number {
    return this.windows.size;
  }

  private evict(currentMinute: number): void {
    for (const [id, entry] of this.windows) {
      if (entry.minute !== currentMinute) this.windows.delete(id);
    }
    // Still full of identities active this minute: start over rather than grow.
    if (this.windows.size >= this.maxIdentities) this.windows.clear();
  }
}

export function rateLimitHeaders(state: RateLimitState): Record<string, string> {
  return {
    "x-ratelimit-limit": String(state.limit),
    "x-ratelimit-remaining": String(state.remaining),
    "x-ratelimit-reset": String(state.resetEpochSeconds),
  };
}
