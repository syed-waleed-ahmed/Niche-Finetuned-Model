/**
 * Shared answer cache.
 *
 * Key:   c:{MODEL_VERSION}:{sha256(path + canonical JSON of body)}
 * Value: the origin's 200 JSON body, stored verbatim (no re-serialisation, so a
 *        hit costs no JSON work on the Worker).
 *
 * The cache is shared across users, so a hit may only be served to a key that
 * the origin has recently accepted: `kv:{identity}` is written with a short TTL
 * whenever the origin returns 2xx for that key.
 */

import { sha256Hex } from "./identity";
import type { RedisCommand } from "./upstash";

export type CacheStatus = "HIT" | "MISS" | "BYPASS";

/** JSON with object keys sorted recursively, so key order never changes the cache key. */
export function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(",")}]`;
  }
  const obj = value as Record<string, unknown>;
  const parts: string[] = [];
  for (const key of Object.keys(obj).sort()) {
    if (obj[key] === undefined) continue;
    parts.push(`${JSON.stringify(key)}:${canonicalJson(obj[key])}`);
  }
  return `{${parts.join(",")}}`;
}

/** True when the request asks for SSE. Anything other than absent/null/false is treated as streaming. */
export function wantsStream(body: Record<string, unknown>): boolean {
  const s = body["stream"];
  return !(s === undefined || s === null || s === false);
}

const decoder = new TextDecoder();

/**
 * Returns the parsed JSON object when this request body may be served from or
 * stored in the cache, or null when the request must bypass the cache.
 */
export function parseCacheableBody(
  body: ArrayBuffer | null,
  maxBytes: number,
): Record<string, unknown> | null {
  if (!body || body.byteLength === 0 || body.byteLength > maxBytes) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(decoder.decode(body));
  } catch {
    return null;
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const obj = parsed as Record<string, unknown>;
  return wantsStream(obj) ? null : obj;
}

export async function cacheKey(
  modelVersion: string,
  pathname: string,
  body: Record<string, unknown>,
): Promise<string> {
  return `c:${modelVersion}:${await sha256Hex(pathname + canonicalJson(body))}`;
}

export function keyMarker(identityId: string): string {
  return `kv:${identityId}`;
}

export function cacheWriteCommand(key: string, body: string, ttlSeconds: number): RedisCommand {
  return ["SET", key, body, "EX", String(ttlSeconds)];
}

export function markerWriteCommand(identityId: string, ttlSeconds: number): RedisCommand {
  return ["SET", keyMarker(identityId), "1", "EX", String(ttlSeconds)];
}

export function isJsonContentType(contentType: string | null): boolean {
  if (!contentType) return false;
  const mime = contentType.split(";")[0]?.trim().toLowerCase() ?? "";
  return mime === "application/json" || mime.endsWith("+json");
}
