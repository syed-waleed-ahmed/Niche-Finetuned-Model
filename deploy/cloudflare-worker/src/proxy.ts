/**
 * Forwarding to the origin replicas with ordered failover.
 *
 * The request body is buffered once by the caller (an ArrayBuffer can be sent
 * any number of times). Failover happens only before a usable response exists:
 * the first response that is not 429/502/503/504 is returned untouched, so a
 * streaming (SSE) body is passed straight through and never retried.
 */

import type { FetchLike } from "./upstash";

/** Statuses that mean "this replica cannot take the request right now". */
export const RETRYABLE_STATUSES: ReadonlySet<number> = new Set([429, 502, 503, 504]);

/** Request headers never forwarded to the origin (hop-by-hop, spoofable or recomputed). */
const STRIPPED_REQUEST_HEADERS: ReadonlySet<string> = new Set([
  "host",
  "x-origin-secret",
  "x-forwarded-for",
  "x-request-id",
  "content-length",
  "connection",
  "keep-alive",
  "proxy-connection",
  "transfer-encoding",
  "upgrade",
  "te",
  "trailer",
]);

export function buildForwardHeaders(
  incoming: Headers,
  options: { originSecret: string; requestId: string; clientIp: string },
): Headers {
  const headers = new Headers();
  incoming.forEach((value, name) => {
    const lower = name.toLowerCase();
    // cf-* headers are set by Cloudflare itself; forwarding them can break
    // subrequests to other Cloudflare-fronted hosts.
    if (STRIPPED_REQUEST_HEADERS.has(lower) || lower.startsWith("cf-")) return;
    headers.set(name, value);
  });
  if (options.originSecret) headers.set("x-origin-secret", options.originSecret);
  headers.set("x-request-id", options.requestId);
  headers.set("x-forwarded-for", options.clientIp);
  return headers;
}

export interface ForwardRequest {
  method: string;
  /** Pathname plus query string, e.g. `/generate?x=1`. */
  pathAndQuery: string;
  headers: Headers;
  body: ArrayBuffer | null;
}

export interface ForwardResult {
  response: Response | null;
  /** Index into ORIGINS of the replica that produced `response`. */
  originIndex: number | null;
  originBase: string | null;
  attempts: number;
  /** One short reason per failed attempt, for logging. */
  failures: string[];
  /** True when the last attempt ended because the origin did not answer in time. */
  timedOut: boolean;
  /** Origins skipped because their circuit was open. */
  skipped: number[];
}

/**
 * Per-isolate circuit breaker over origin indices.
 *
 * After `threshold` consecutive network failures or timeouts, an origin is skipped
 * for `cooldownMs` while another origin is available, so a dead primary does not
 * cost every request a full ORIGIN_TIMEOUT_MS before failover. Any HTTP response
 * (even a 503) proves the origin is reachable and closes its circuit. When every
 * origin is open they are all tried anyway: with a single origin, a slow cold start
 * must never turn into self-inflicted errors.
 */
export class CircuitBreaker {
  private readonly state = new Map<number, { failures: number; openUntil: number }>();

  constructor(
    private readonly threshold: number,
    private readonly cooldownMs: number,
  ) {}

  isOpen(index: number, nowMs: number): boolean {
    const entry = this.state.get(index);
    return entry !== undefined && entry.openUntil > nowMs;
  }

  success(index: number): void {
    this.state.delete(index);
  }

  failure(index: number, nowMs: number): void {
    const entry = this.state.get(index) ?? { failures: 0, openUntil: 0 };
    entry.failures++;
    if (entry.failures >= this.threshold) {
      entry.openUntil = nowMs + this.cooldownMs;
      entry.failures = 0;
    }
    this.state.set(index, entry);
  }
}

function describeError(err: unknown): string {
  if (err instanceof Error) return `${err.name}: ${err.message}`;
  return String(err);
}

/** Re-creates a small error response with its body in memory so it can be returned later. */
async function bufferResponse(res: Response): Promise<Response> {
  const body = res.status === 204 || res.status === 304 ? null : await res.arrayBuffer();
  return new Response(body, { status: res.status, statusText: res.statusText, headers: res.headers });
}

export async function forwardWithFailover(
  req: ForwardRequest,
  options: {
    origins: readonly string[];
    timeoutMs: number;
    fetch: FetchLike;
    breaker?: CircuitBreaker;
    now?: () => number;
  },
): Promise<ForwardResult> {
  const { origins, timeoutMs, breaker } = options;
  const now = options.now ?? (() => Date.now());
  const failures: string[] = [];
  let attempts = 0;
  let timedOut = false;
  let last: { response: Response; index: number } | null = null;

  const all = origins.map((_, i) => i);
  const startedAt = now();
  const closed = breaker ? all.filter((i) => !breaker.isOpen(i, startedAt)) : all;
  const order = closed.length > 0 ? closed : all;
  const skipped = all.filter((i) => !order.includes(i));

  for (let pos = 0; pos < order.length; pos++) {
    const i = order[pos] as number;
    const base = origins[i] as string;
    const hasNext = pos < order.length - 1;
    attempts++;

    // The timeout covers the wait for response headers (and buffering of a
    // retryable error body), but never the streaming of a successful body.
    const controller = new AbortController();
    let expired = false;
    const timer = setTimeout(() => {
      expired = true;
      controller.abort(new Error(`origin timeout after ${timeoutMs} ms`));
    }, timeoutMs);

    let res: Response;
    try {
      res = await options.fetch(base + req.pathAndQuery, {
        method: req.method,
        headers: req.headers,
        body: req.body,
        redirect: "manual",
        signal: controller.signal,
      });
    } catch (err) {
      clearTimeout(timer);
      timedOut = expired;
      breaker?.failure(i, now());
      failures.push(`origin-${i}: ${expired ? `timeout after ${timeoutMs} ms` : describeError(err)}`);
      continue;
    }
    timedOut = false;
    breaker?.success(i);

    if (RETRYABLE_STATUSES.has(res.status) && hasNext) {
      failures.push(`origin-${i}: HTTP ${res.status}`);
      try {
        last = { response: await bufferResponse(res), index: i };
      } catch {
        last = {
          response: new Response(null, { status: res.status, headers: res.headers }),
          index: i,
        };
      } finally {
        clearTimeout(timer);
      }
      continue;
    }

    clearTimeout(timer);
    return {
      response: res,
      originIndex: i,
      originBase: base,
      attempts,
      failures,
      timedOut: false,
      skipped,
    };
  }

  return {
    response: last?.response ?? null,
    originIndex: last?.index ?? null,
    originBase: last ? (origins[last.index] as string) : null,
    attempts,
    failures,
    timedOut,
    skipped,
  };
}

/**
 * Rewrites an absolute `Location` that points at the origin (e.g. FastAPI's
 * trailing-slash redirects) so clients stay on the edge hostname.
 */
export function rewriteLocation(headers: Headers, originBase: string, publicOrigin: string): void {
  const location = headers.get("location");
  if (!location) return;
  try {
    const base = new URL(originBase);
    const target = new URL(location, originBase);
    if (target.host !== base.host) return;
    const prefix = base.pathname.replace(/\/+$/, "");
    let path = target.pathname;
    if (prefix && path.startsWith(prefix)) path = path.slice(prefix.length) || "/";
    headers.set("location", publicOrigin + path + target.search + target.hash);
  } catch {
    // Leave unparseable Location headers alone.
  }
}
