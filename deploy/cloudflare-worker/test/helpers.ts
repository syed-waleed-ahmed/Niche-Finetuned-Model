import { createHandler, type Deps, type Env } from "../src/index";
import type { RedisClient, RedisCommand, RedisReply } from "../src/upstash";

/** 2023-11-14T22:13:50Z: 30 s into epoch minute 28333333, so the window resets in 10 s. */
export const NOW = 1_700_000_030_000;
export const EPOCH_MINUTE = 28_333_333;
export const RESET_EPOCH_SECONDS = 1_700_000_040;

/** SHA-256("abc") truncated to 32 hex chars: the identity of API key "abc". */
export const ABC_KEY_ID = "ba7816bf8f01cfea414140de5dae2223";

export const ORIGIN_0 = "https://o0.test";
export const ORIGIN_1 = "https://o1.test";
export const EDGE = "https://edge.test";

/** In-memory Redis implementing the handful of commands the Worker uses. */
export class FakeRedis implements RedisClient {
  readonly store = new Map<string, string>();
  readonly ttl = new Map<string, number>();
  readonly calls: string[][][] = [];
  fail: Error | null = null;

  async pipeline(commands: readonly RedisCommand[]): Promise<RedisReply[]> {
    const call = commands.map((c) => c.map(String));
    this.calls.push(call);
    if (this.fail) throw this.fail;
    return call.map((c) => this.exec(c));
  }

  private exec([op = "", key = "", ...args]: string[]): RedisReply {
    switch (op.toUpperCase()) {
      case "INCR": {
        const next = Number(this.store.get(key) ?? "0") + 1;
        this.store.set(key, String(next));
        return { result: next };
      }
      case "EXPIRE":
        this.ttl.set(key, Number(args[0]));
        return { result: 1 };
      case "GET":
        return { result: this.store.get(key) ?? null };
      case "SET":
        this.store.set(key, args[0] ?? "");
        if (args[1]?.toUpperCase() === "EX") this.ttl.set(key, Number(args[2]));
        return { result: "OK" };
      default:
        return { error: `ERR unknown command '${op}'` };
    }
  }

  /** Every command sent, flattened, as "OP key" strings. */
  get commands(): string[] {
    return this.calls.flat().map((c) => `${c[0]} ${c[1]}`);
  }
}

export interface OriginCall {
  url: string;
  method: string;
  headers: Headers;
  body: string | null;
}

export type OriginHandler = (call: OriginCall, signal: AbortSignal | null) => Response | Promise<Response>;

/** A fetch mock that routes by URL prefix and records every call. */
export function mockFetch(routes: Record<string, OriginHandler>) {
  const calls: OriginCall[] = [];
  const fetchFn = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const raw = init?.body;
    const body =
      raw == null
        ? null
        : raw instanceof ArrayBuffer
          ? new TextDecoder().decode(raw)
          : typeof raw === "string"
            ? raw
            : "[stream]";
    const call: OriginCall = {
      url,
      method: init?.method ?? "GET",
      headers: new Headers(init?.headers),
      body,
    };
    calls.push(call);
    const prefix = Object.keys(routes).find((p) => url.startsWith(p));
    if (!prefix) throw new TypeError(`fetch failed: no mock for ${url}`);
    return (routes[prefix] as OriginHandler)(call, init?.signal ?? null);
  };
  return { fetch: fetchFn, calls };
}

export function json(body: unknown, status = 200, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

/** Never resolves until the request is aborted (simulates a hung origin). */
export function hang(_call: OriginCall, signal: AbortSignal | null): Promise<Response> {
  return new Promise((_, reject) => {
    signal?.addEventListener("abort", () => reject(signal.reason ?? new Error("aborted")));
  });
}

export function networkError(): Promise<Response> {
  return Promise.reject(new TypeError("fetch failed"));
}

export class TestContext {
  readonly pending: Promise<unknown>[] = [];
  waitUntil(promise: Promise<unknown>): void {
    this.pending.push(promise);
  }
  async flush(): Promise<void> {
    await Promise.all(this.pending);
  }
}

export function baseEnv(overrides: Partial<Env> = {}): Env {
  return {
    ORIGINS: `${ORIGIN_0},${ORIGIN_1}`,
    ORIGIN_SHARED_SECRET: "the-real-secret",
    MODEL_VERSION: "test-v1",
    ...overrides,
  };
}

export interface Harness {
  run: (request: Request, env?: Env) => Promise<Response>;
  ctx: TestContext;
  logs: Record<string, unknown>[];
}

export function harness(deps: Partial<Deps> & Pick<Deps, "fetch">, env: Env = baseEnv()): Harness {
  const logs: Record<string, unknown>[] = [];
  const ctx = new TestContext();
  const handler = createHandler({
    now: () => NOW,
    log: (entry) => logs.push(entry),
    randomId: () => "generated-id",
    ...deps,
  });
  return { run: (request, e = env) => handler(request, e, ctx), ctx, logs };
}

export function post(path: string, body: unknown, headers: Record<string, string> = {}): Request {
  return new Request(EDGE + path, {
    method: "POST",
    headers: { "content-type": "application/json", "cf-connecting-ip": "203.0.113.7", ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

export function get(path: string, headers: Record<string, string> = {}): Request {
  return new Request(EDGE + path, { headers: { "cf-connecting-ip": "203.0.113.7", ...headers } });
}
