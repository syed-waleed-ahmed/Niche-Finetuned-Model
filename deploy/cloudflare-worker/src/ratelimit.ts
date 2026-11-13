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

export function rateLimitHeaders(state: RateLimitState): Record<string, string> {
  return {
    "x-ratelimit-limit": String(state.limit),
    "x-ratelimit-remaining": String(state.remaining),
    "x-ratelimit-reset": String(state.resetEpochSeconds),
  };
}
