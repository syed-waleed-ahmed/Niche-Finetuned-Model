import { describe, expect, it } from "vitest";
import { cacheKey } from "../src/cache";
import {
  ABC_KEY_ID,
  EPOCH_MINUTE,
  FakeRedis,
  ORIGIN_0,
  ORIGIN_1,
  RESET_EPOCH_SECONDS,
  baseEnv,
  get,
  harness,
  hang,
  json,
  mockFetch,
  networkError,
  post,
} from "./helpers";

const KEY = { "x-api-key": "abc" };
const QUESTION = { question: "What is the refund policy?", max_new_tokens: 64 };
const ANSWER = { answer: "Thirty days.", tokens: 4 };

async function generateKey(body: Record<string, unknown> = QUESTION): Promise<string> {
  return cacheKey("test-v1", "/generate", body);
}

describe("authentication at the edge", () => {
  it("returns 401 for a protected path without a key, touching neither Redis nor the origin", async () => {
    const redis = new FakeRedis();
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch, redis });

    for (const path of ["/generate", "/v1/chat/completions", "/v1/models"]) {
      const res = await h.run(path === "/v1/models" ? get(path) : post(path, QUESTION));
      expect(res.status).toBe(401);
      expect(await res.json()).toMatchObject({ error: { type: "authentication_error" } });
    }
    expect(origin.calls).toHaveLength(0);
    expect(redis.calls).toHaveLength(0);
  });

  it("does not contact Upstash either when it is configured", async () => {
    const f = mockFetch({});
    const h = harness(
      { fetch: f.fetch },
      baseEnv({ UPSTASH_REDIS_REST_URL: "https://r.upstash.test", UPSTASH_REDIS_REST_TOKEN: "t" }),
    );
    expect((await h.run(post("/generate", QUESTION))).status).toBe(401);
    expect(f.calls).toHaveLength(0);
  });

  it("passes CORS preflights through without a key", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => new Response(null, { status: 204 }) });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() });
    const res = await h.run(new Request("https://edge.test/generate", { method: "OPTIONS" }));
    expect(res.status).toBe(204);
    expect(origin.calls[0]?.method).toBe("OPTIONS");
  });

  it("allows anonymous calls when REQUIRE_API_KEY=false", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() }, baseEnv({ REQUIRE_API_KEY: "false" }));
    const res = await h.run(post("/generate", QUESTION));
    expect(res.status).toBe(200);
    expect(h.logs[0]).toMatchObject({ rl_identity: "ip" });
  });
});

describe("rate limiting", () => {
  it("returns 429 with Retry-After and X-RateLimit-* once the per-key limit is exceeded", async () => {
    const redis = new FakeRedis();
    const origin = mockFetch({ [ORIGIN_0]: () => json([{ id: "m" }]) });
    const h = harness({ fetch: origin.fetch, redis }, baseEnv({ RATE_LIMIT_PER_MINUTE: "2" }));

    const first = await h.run(get("/v1/models", KEY));
    expect(first.status).toBe(200);
    expect(first.headers.get("x-ratelimit-limit")).toBe("2");
    expect(first.headers.get("x-ratelimit-remaining")).toBe("1");
    expect(first.headers.get("x-ratelimit-reset")).toBe(String(RESET_EPOCH_SECONDS));
    expect((await h.run(get("/v1/models", KEY))).headers.get("x-ratelimit-remaining")).toBe("0");

    const limited = await h.run(get("/v1/models", KEY));
    expect(limited.status).toBe(429);
    expect(limited.headers.get("retry-after")).toBe("10");
    expect(limited.headers.get("x-ratelimit-limit")).toBe("2");
    expect(limited.headers.get("x-ratelimit-remaining")).toBe("0");
    expect(limited.headers.get("x-ratelimit-reset")).toBe(String(RESET_EPOCH_SECONDS));
    const body = (await limited.json()) as { error: { message: string; type: string } };
    expect(body.error.type).toBe("rate_limited");
    expect(body.error.message).toEqual(expect.any(String));
    expect(origin.calls).toHaveLength(2);

    const key = `rl:${ABC_KEY_ID}:${EPOCH_MINUTE}`;
    expect(redis.store.get(key)).toBe("3");
    expect(redis.ttl.get(key)).toBe(70);
  });

  it("treats X-API-Key and Bearer as the same identity", async () => {
    const redis = new FakeRedis();
    const h = harness({ fetch: mockFetch({ [ORIGIN_0]: () => json([]) }).fetch, redis });
    await h.run(get("/v1/models", { "x-api-key": "abc" }));
    await h.run(get("/v1/models", { authorization: "Bearer abc" }));
    expect(redis.store.get(`rl:${ABC_KEY_ID}:${EPOCH_MINUTE}`)).toBe("2");
  });

  it("applies the anonymous per-IP limit to public paths", async () => {
    const redis = new FakeRedis();
    const origin = mockFetch({ [ORIGIN_0]: () => json({ status: "ok" }) });
    const h = harness({ fetch: origin.fetch, redis }, baseEnv({ RATE_LIMIT_PER_MINUTE_ANON: "1" }));

    expect((await h.run(get("/health"))).status).toBe(200);
    const limited = await h.run(get("/health", KEY)); // a key does not lift the public-path limit
    expect(limited.status).toBe(429);
    expect(limited.headers.get("x-ratelimit-limit")).toBe("1");
    expect((await h.run(get("/ready", { "cf-connecting-ip": "198.51.100.1" }))).status).toBe(200);
    expect(redis.store.get(`rl:ip:203.0.113.7:${EPOCH_MINUTE}`)).toBe("2");
    expect(h.logs.every((l) => l["rl_identity"] === "ip")).toBe(true);
  });
});

describe("shared answer cache", () => {
  it("does not serve a hit to a key without the verified marker; the origin decides", async () => {
    const redis = new FakeRedis();
    redis.store.set(await generateKey(), JSON.stringify(ANSWER));
    const origin = mockFetch({ [ORIGIN_0]: () => json({ detail: "Invalid API key" }, 401) });
    const h = harness({ fetch: origin.fetch, redis });

    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(401);
    expect(res.headers.get("x-cache")).toBe("MISS");
    expect(origin.calls).toHaveLength(1);
    await h.ctx.flush();
    expect(redis.store.has(`kv:${ABC_KEY_ID}`)).toBe(false);
  });

  it("serves a hit once the key is verified, without calling the origin", async () => {
    const redis = new FakeRedis();
    const cKey = await generateKey();
    redis.store.set(cKey, JSON.stringify(ANSWER));
    redis.store.set(`kv:${ABC_KEY_ID}`, "1");
    const origin = mockFetch({ [ORIGIN_0]: () => json({ unexpected: true }) });
    const h = harness({ fetch: origin.fetch, redis });

    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(200);
    expect(res.headers.get("x-cache")).toBe("HIT");
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(res.headers.get("x-ratelimit-remaining")).toBe("19");
    expect(await res.json()).toEqual(ANSWER);
    expect(origin.calls).toHaveLength(0);
    // Rate limit, cache entry and marker were fetched in ONE pipeline call.
    expect(redis.calls).toHaveLength(1);
    expect(redis.commands).toEqual([
      `INCR rl:${ABC_KEY_ID}:${EPOCH_MINUTE}`,
      `EXPIRE rl:${ABC_KEY_ID}:${EPOCH_MINUTE}`,
      `GET ${cKey}`,
      `GET kv:${ABC_KEY_ID}`,
    ]);
  });

  it("stores the answer and the key marker on a 200 miss, then hits regardless of key order", async () => {
    const redis = new FakeRedis();
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch, redis });

    const miss = await h.run(post("/generate", QUESTION, KEY));
    expect(miss.status).toBe(200);
    expect(miss.headers.get("x-cache")).toBe("MISS");
    expect(await miss.json()).toEqual(ANSWER);
    await h.ctx.flush();

    const cKey = await generateKey();
    expect(redis.store.get(cKey)).toBe(JSON.stringify(ANSWER));
    expect(redis.ttl.get(cKey)).toBe(86_400);
    expect(redis.store.get(`kv:${ABC_KEY_ID}`)).toBe("1");
    expect(redis.ttl.get(`kv:${ABC_KEY_ID}`)).toBe(300);
    expect(redis.calls).toHaveLength(2); // one read pipeline + one write pipeline

    const reordered = `{"max_new_tokens":64,"question":"What is the refund policy?"}`;
    const hit = await h.run(post("/generate", reordered, { authorization: "Bearer abc" }));
    expect(hit.headers.get("x-cache")).toBe("HIT");
    expect(await hit.json()).toEqual(ANSWER);
    expect(origin.calls).toHaveLength(1);
  });

  it("only caches 200 responses", async () => {
    for (const status of [201, 400, 500]) {
      const redis = new FakeRedis();
      const origin = mockFetch({ [ORIGIN_0]: () => json({ status }, status) });
      const h = harness({ fetch: origin.fetch, redis });
      const res = await h.run(post("/generate", QUESTION, KEY));
      expect(res.status).toBe(status);
      expect(res.headers.get("x-cache")).toBe("MISS");
      await h.ctx.flush();
      expect(redis.store.has(await generateKey())).toBe(false);
      // A 2xx still proves the key is valid.
      expect(redis.store.has(`kv:${ABC_KEY_ID}`)).toBe(status === 201);
    }
  });

  it("does not cache bodies over the size cap", async () => {
    const redis = new FakeRedis();
    const big = { answer: "x".repeat(70 * 1024) };
    const h = harness({ fetch: mockFetch({ [ORIGIN_0]: () => json(big) }).fetch, redis });
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(((await res.json()) as { answer: string }).answer).toHaveLength(70 * 1024);
    await h.ctx.flush();
    expect(redis.store.has(await generateKey())).toBe(false);
    expect(redis.store.get(`kv:${ABC_KEY_ID}`)).toBe("1");
  });

  it("caches /v1/chat/completions under its own key", async () => {
    const redis = new FakeRedis();
    const chat = { model: "m", messages: [{ role: "user", content: "hi" }] };
    const h = harness({ fetch: mockFetch({ [ORIGIN_0]: () => json({ id: "c1" }) }).fetch, redis });
    await h.run(post("/v1/chat/completions", chat, KEY));
    await h.ctx.flush();
    expect(redis.store.get(await cacheKey("test-v1", "/v1/chat/completions", chat))).toBe('{"id":"c1"}');
  });

  it("marks non-cacheable requests BYPASS", async () => {
    const h = harness({ fetch: mockFetch({ [ORIGIN_0]: () => json({ ok: true }) }).fetch, redis: new FakeRedis() });
    expect((await h.run(post("/v1/feedback", { rating: 1 }, KEY))).headers.get("x-cache")).toBe("BYPASS");
    expect((await h.run(post("/generate", "not json", KEY))).headers.get("x-cache")).toBe("BYPASS");
  });
});

describe("streaming", () => {
  it("bypasses the cache and passes the SSE body through without buffering", async () => {
    const redis = new FakeRedis();
    let push!: (chunk: string) => void;
    let close!: () => void;
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const enc = new TextEncoder();
        push = (chunk) => controller.enqueue(enc.encode(chunk));
        close = () => controller.close();
      },
    });
    const origin = mockFetch({
      [ORIGIN_0]: () => {
        push("data: first\n\n");
        return new Response(stream, { headers: { "content-type": "text/event-stream" } });
      },
    });
    const h = harness({ fetch: origin.fetch, redis });

    // The origin stream is still open: a buffering Worker would never resolve here.
    const res = await Promise.race([
      h.run(post("/generate", { ...QUESTION, stream: true }, KEY)),
      new Promise<never>((_, reject) => setTimeout(() => reject(new Error("response was buffered")), 1000)),
    ]);
    expect(res.headers.get("x-cache")).toBe("BYPASS");
    expect(res.headers.get("content-type")).toBe("text/event-stream");

    const reader = res.body!.getReader();
    const dec = new TextDecoder();
    expect(dec.decode((await reader.read()).value)).toBe("data: first\n\n");
    push("data: [DONE]\n\n");
    close();
    expect(dec.decode((await reader.read()).value)).toBe("data: [DONE]\n\n");
    expect((await reader.read()).done).toBe(true);

    await h.ctx.flush();
    expect(h.ctx.pending).toHaveLength(0);
    expect(redis.commands.some((c) => c.startsWith("GET"))).toBe(false);
  });
});

describe("failover", () => {
  const cases: [string, Parameters<typeof mockFetch>[0][string]][] = [
    ["503", () => json({ detail: "model not ready" }, 503)],
    ["429", () => json({ detail: "queue full" }, 429, { "retry-after": "5" })],
    ["502", () => new Response("bad gateway", { status: 502 })],
    ["504", () => new Response("gateway timeout", { status: 504 })],
    ["network error", networkError],
    ["timeout", hang],
  ];

  for (const [name, firstOrigin] of cases) {
    it(`moves to the next origin on ${name}`, async () => {
      const origin = mockFetch({ [ORIGIN_0]: firstOrigin, [ORIGIN_1]: () => json(ANSWER) });
      const h = harness(
        { fetch: origin.fetch, redis: new FakeRedis() },
        baseEnv({ ORIGIN_TIMEOUT_MS: "30" }),
      );
      const res = await h.run(post("/generate", QUESTION, KEY));
      expect(res.status).toBe(200);
      expect(res.headers.get("x-served-by")).toBe("origin-1");
      expect(origin.calls.map((c) => c.url)).toEqual([`${ORIGIN_0}/generate`, `${ORIGIN_1}/generate`]);
      expect(origin.calls[1]?.body).toBe(JSON.stringify(QUESTION)); // body replayed intact
      expect(h.logs[0]).toMatchObject({ origin_index: 1, attempts: 2, status: 200 });
    });
  }

  for (const status of [400, 401, 404, 422, 500]) {
    it(`returns ${status} from the first origin without failover`, async () => {
      const origin = mockFetch({ [ORIGIN_0]: () => json({ status }, status), [ORIGIN_1]: () => json(ANSWER) });
      const h = harness({ fetch: origin.fetch, redis: new FakeRedis() });
      const res = await h.run(post("/generate", QUESTION, KEY));
      expect(res.status).toBe(status);
      expect(res.headers.get("x-served-by")).toBe("origin-0");
      expect(origin.calls).toHaveLength(1);
    });
  }

  it("returns the last origin response when every origin is unavailable", async () => {
    const origin = mockFetch({
      [ORIGIN_0]: () => json({ detail: "first" }, 503),
      [ORIGIN_1]: () => json({ detail: "second" }, 503, { "retry-after": "7" }),
    });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() });
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(503);
    expect(res.headers.get("retry-after")).toBe("7");
    expect(res.headers.get("x-served-by")).toBe("origin-1");
    expect(await res.json()).toEqual({ detail: "second" });
  });

  it("returns an earlier origin's response when the last one did not respond", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json({ detail: "busy" }, 429), [ORIGIN_1]: networkError });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() });
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(429);
    expect(res.headers.get("x-served-by")).toBe("origin-0");
    expect(await res.json()).toEqual({ detail: "busy" });
  });

  it("returns 502 JSON when no origin responded", async () => {
    const origin = mockFetch({ [ORIGIN_0]: networkError, [ORIGIN_1]: networkError });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() });
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(502);
    expect(await res.json()).toMatchObject({ error: { type: "upstream_unavailable" } });
    expect(h.logs[0]).toMatchObject({ attempts: 2, origin_index: null, status: 502 });
  });
});

describe("forwarding", () => {
  it("overwrites a client X-Origin-Secret and sets forwarding headers", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json([]) });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() });
    const res = await h.run(
      get("/v1/models?limit=5&x=y", {
        ...KEY,
        "x-origin-secret": "attacker-guess",
        "x-forwarded-for": "6.6.6.6",
        "x-request-id": "client-req-1",
      }),
    );
    expect(res.status).toBe(200);
    const call = origin.calls[0]!;
    expect(call.url).toBe(`${ORIGIN_0}/v1/models?limit=5&x=y`);
    expect(call.method).toBe("GET");
    expect(call.headers.get("x-origin-secret")).toBe("the-real-secret");
    expect(call.headers.get("x-forwarded-for")).toBe("203.0.113.7");
    expect(call.headers.get("x-request-id")).toBe("client-req-1");
    expect(call.headers.get("x-api-key")).toBe("abc");
    expect(call.headers.get("host")).toBeNull();
    expect(res.headers.get("x-request-id")).toBe("client-req-1");
  });

  it("generates a request id when the client sends none and preserves origin headers", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json({ ok: true }, 200, { "x-custom": "1" }) });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() });
    const res = await h.run(post("/v1/feedback", { rating: 5 }, KEY));
    expect(origin.calls[0]?.headers.get("x-request-id")).toBe("generated-id");
    expect(origin.calls[0]?.body).toBe('{"rating":5}');
    expect(res.headers.get("x-request-id")).toBe("generated-id");
    expect(res.headers.get("x-custom")).toBe("1");
  });

  it("passes unknown methods on known paths through to the origin", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json({ detail: "Method Not Allowed" }, 405) });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() });
    const res = await h.run(new Request("https://edge.test/generate", { method: "PUT", headers: KEY, body: "{}" }));
    expect(res.status).toBe(405);
    expect(origin.calls[0]?.method).toBe("PUT");
  });

  it("returns 404 for /metrics and unknown paths without contacting anything", async () => {
    const redis = new FakeRedis();
    const origin = mockFetch({ [ORIGIN_0]: () => json({}) });
    const h = harness({ fetch: origin.fetch, redis });
    for (const path of ["/metrics", "/wp-login.php", "/"]) {
      expect((await h.run(get(path, KEY))).status).toBe(404);
    }
    expect(origin.calls).toHaveLength(0);
    expect(redis.calls).toHaveLength(0);
  });
});

describe("Redis failures and absence", () => {
  it("serves without the cache and limits in-isolate when the Redis pipeline throws", async () => {
    const redis = new FakeRedis();
    redis.fail = new Error("connection reset");
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch, redis });
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(200);
    expect(res.headers.get("x-cache")).toBe("BYPASS");
    expect(res.headers.get("x-ratelimit-limit")).toBe("20");
    expect(res.headers.get("x-ratelimit-remaining")).toBe("19");
    expect(h.logs[0]).toMatchObject({
      redis_error: "connection reset",
      rl_scope: "local",
      status: 200,
    });
    await h.ctx.flush();
    expect(redis.calls).toHaveLength(1); // no write attempted after a failed read
  });

  it("fails open when Upstash answers with an HTTP error", async () => {
    const f = mockFetch({
      "https://r.upstash.test": () => new Response("unavailable", { status: 500 }),
      [ORIGIN_0]: () => json(ANSWER),
    });
    const h = harness(
      { fetch: f.fetch },
      baseEnv({ UPSTASH_REDIS_REST_URL: "https://r.upstash.test", UPSTASH_REDIS_REST_TOKEN: "tok" }),
    );
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(200);
    expect(h.logs[0]?.["redis_error"]).toContain("500");
  });

  it("talks to Upstash with one /pipeline call per request", async () => {
    const f = mockFetch({
      "https://r.upstash.test": (call) => {
        const cmds = JSON.parse(call.body ?? "[]") as string[][];
        return json(cmds.map((c) => ({ result: c[0] === "INCR" ? 1 : c[0] === "GET" ? null : "OK" })));
      },
      [ORIGIN_0]: () => json(ANSWER),
    });
    const h = harness(
      { fetch: f.fetch },
      baseEnv({ UPSTASH_REDIS_REST_URL: "https://r.upstash.test/", UPSTASH_REDIS_REST_TOKEN: "tok" }),
    );
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.headers.get("x-cache")).toBe("MISS");
    await h.ctx.flush();
    const redisCalls = f.calls.filter((c) => new URL(c.url).host === "r.upstash.test");
    expect(redisCalls.map((c) => c.url)).toEqual([
      "https://r.upstash.test/pipeline",
      "https://r.upstash.test/pipeline",
    ]);
    expect(redisCalls[0]?.headers.get("authorization")).toBe("Bearer tok");
    expect((JSON.parse(redisCalls[0]?.body ?? "") as string[][]).map((c) => c[0])).toEqual([
      "INCR",
      "EXPIRE",
      "GET",
      "GET",
    ]);
    expect((JSON.parse(redisCalls[1]?.body ?? "") as string[][]).map((c) => c[0])).toEqual(["SET", "SET"]);
  });

  it("works without Upstash configured: in-isolate rate limit, no cache", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch });
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(200);
    expect(res.headers.get("x-cache")).toBe("BYPASS");
    expect(res.headers.get("x-ratelimit-limit")).toBe("20");
    expect(h.logs[0]).toMatchObject({ rl_scope: "local" });
    expect(origin.calls).toHaveLength(1);
  });

  it("enforces the in-isolate limit with 429 and Retry-After", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch }, baseEnv({ RATE_LIMIT_PER_MINUTE: "2" }));
    const statuses: number[] = [];
    for (let i = 0; i < 3; i++) statuses.push((await h.run(post("/generate", QUESTION, KEY))).status);
    expect(statuses).toEqual([200, 200, 429]);
    const last = await h.run(post("/generate", QUESTION, KEY));
    expect(last.headers.get("retry-after")).toBe("10");
    expect(origin.calls).toHaveLength(2); // rejected requests never reach the origin
  });

  it("counts identities separately in-isolate", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch }, baseEnv({ RATE_LIMIT_PER_MINUTE: "1" }));
    expect((await h.run(post("/generate", QUESTION, KEY))).status).toBe(200);
    expect((await h.run(post("/generate", QUESTION, { "x-api-key": "other" }))).status).toBe(200);
    expect((await h.run(post("/generate", QUESTION, KEY))).status).toBe(429);
  });

  it("serves hits without a marker when REQUIRE_API_KEY=false", async () => {
    const redis = new FakeRedis();
    redis.store.set(await generateKey(), JSON.stringify(ANSWER));
    const origin = mockFetch({ [ORIGIN_0]: () => json({}) });
    const h = harness({ fetch: origin.fetch, redis }, baseEnv({ REQUIRE_API_KEY: "false" }));
    const res = await h.run(post("/generate", QUESTION));
    expect(res.headers.get("x-cache")).toBe("HIT");
    expect(origin.calls).toHaveLength(0);
    expect(redis.commands.some((c) => c.startsWith("GET kv:"))).toBe(false);
  });
});

describe("logging", () => {
  it("emits one structured line per request without the raw key", async () => {
    const h = harness({ fetch: mockFetch({ [ORIGIN_0]: () => json(ANSWER) }).fetch, redis: new FakeRedis() });
    await h.run(post("/generate", QUESTION, { "x-api-key": "super-secret-key-value" }));
    expect(h.logs).toHaveLength(1);
    expect(h.logs[0]).toMatchObject({
      event: "request",
      path: "/generate",
      status: 200,
      cache: "MISS",
      rl_identity: "key",
      origin_index: 0,
      attempts: 1,
      duration_ms: 0,
    });
    expect(JSON.stringify(h.logs)).not.toContain("super-secret-key-value");
  });

  it("identifies the key by a prefix of its SHA-256, never by the key or the IP", async () => {
    const h = harness({ fetch: mockFetch({ [ORIGIN_0]: () => json(ANSWER) }).fetch, redis: new FakeRedis() });
    await h.run(post("/generate", QUESTION, { "x-api-key": "abc" }));
    expect(h.logs[0]).toMatchObject({ rl_identity: "key", rl_key_id: ABC_KEY_ID.slice(0, 12) });
    expect(h.logs[0]?.["rl_key_id"]).toHaveLength(12);
    const line = JSON.stringify(h.logs[0]);
    expect(line).not.toContain('"abc"');
    expect(line).not.toContain(ABC_KEY_ID); // only a prefix of the identity is logged
    expect(line).not.toContain("203.0.113.7"); // the caller's IP in the test requests
  });

  it("logs no key id and no IP for an anonymous request", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() }, baseEnv({ REQUIRE_API_KEY: "false" }));
    await h.run(post("/generate", QUESTION));
    expect(h.logs[0]).toMatchObject({ rl_identity: "ip" });
    expect(h.logs[0]).not.toHaveProperty("rl_key_id");
    expect(JSON.stringify(h.logs[0])).not.toContain("203.0.113.7");
  });
});

describe("request body cap", () => {
  it("rejects a declared oversized body with 413 before Redis or the origin", async () => {
    const redis = new FakeRedis();
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch, redis }, baseEnv({ MAX_REQUEST_BYTES: "1024" }));
    const res = await h.run(post("/generate", { question: "hi", pad: "a".repeat(5000) }, KEY));
    expect(res.status).toBe(413);
    expect(await res.json()).toMatchObject({ error: { type: "request_too_large" } });
    expect(origin.calls).toHaveLength(0);
    expect(redis.calls).toHaveLength(0);
  });

  it("rejects a chunked body as soon as it crosses the cap", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch }, baseEnv({ MAX_REQUEST_BYTES: "1024" }));
    const enc = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (let i = 0; i < 8; i++) controller.enqueue(enc.encode("a".repeat(512)));
        controller.close();
      },
    });
    const req = new Request("https://edge.test/generate", {
      method: "POST",
      headers: { ...KEY, "content-type": "application/json" },
      body: stream,
      // @ts-expect-error: Node's fetch needs duplex for a streamed request body
      duplex: "half",
    });
    const res = await h.run(req);
    expect(res.status).toBe(413);
    expect(origin.calls).toHaveLength(0);
  });

  it("forwards bodies under the cap intact", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch }, baseEnv({ MAX_REQUEST_BYTES: "1024" }));
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(200);
    expect(origin.calls[0]?.body).toBe(JSON.stringify(QUESTION));
  });

  it("applies the 401 check before reading the body", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch }, baseEnv({ MAX_REQUEST_BYTES: "1024" }));
    const res = await h.run(post("/generate", { pad: "a".repeat(5000) }));
    expect(res.status).toBe(401);
  });
});

describe("origin timeouts and the circuit breaker", () => {
  it("returns 504 when the only origin times out", async () => {
    const origin = mockFetch({ [ORIGIN_0]: hang });
    const h = harness(
      { fetch: origin.fetch, redis: new FakeRedis() },
      baseEnv({ ORIGINS: ORIGIN_0, ORIGIN_TIMEOUT_MS: "20" }),
    );
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(504);
    expect(await res.json()).toMatchObject({ error: { type: "upstream_timeout" } });
  });

  it("skips a primary after repeated network failures, then retries it after the cooldown", async () => {
    let clock = 1_700_000_030_000;
    const origin = mockFetch({ [ORIGIN_0]: networkError, [ORIGIN_1]: () => json(ANSWER) });
    // No Redis: every request reaches an origin (no shared cache).
    const h = harness({ fetch: origin.fetch, redis: null, now: () => clock });
    for (let i = 0; i < 3; i++) await h.run(post("/generate", QUESTION, KEY));
    expect(origin.calls.filter((c) => c.url.startsWith(ORIGIN_0))).toHaveLength(3);

    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(200);
    expect(origin.calls.filter((c) => c.url.startsWith(ORIGIN_0))).toHaveLength(3); // skipped
    expect(h.logs.at(-1)).toMatchObject({ skipped_origins: [0], attempts: 1, origin_index: 1 });

    clock += 31_000;
    await h.run(post("/generate", QUESTION, KEY));
    expect(origin.calls.filter((c) => c.url.startsWith(ORIGIN_0))).toHaveLength(4); // half-open
  });

  it("still tries every origin when all circuits are open", async () => {
    const origin = mockFetch({ [ORIGIN_0]: networkError });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() }, baseEnv({ ORIGINS: ORIGIN_0 }));
    for (let i = 0; i < 5; i++) expect((await h.run(post("/generate", QUESTION, KEY))).status).toBe(502);
    expect(origin.calls).toHaveLength(5);
  });

  it("an HTTP error response does not count against the circuit", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json({}, 503), [ORIGIN_1]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch, redis: null });
    for (let i = 0; i < 4; i++) await h.run(post("/generate", QUESTION, KEY));
    expect(origin.calls.filter((c) => c.url.startsWith(ORIGIN_0))).toHaveLength(4);
  });
});

describe("response headers", () => {
  it("strips origin infrastructure headers and adds nosniff", async () => {
    const origin = mockFetch({
      [ORIGIN_0]: () =>
        json(ANSWER, 200, { "x-cloud-trace-context": "abc;o=1", server: "Google Frontend" }),
    });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() });
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.headers.get("x-cloud-trace-context")).toBeNull();
    expect(res.headers.get("server")).toBeNull();
    expect(res.headers.get("x-content-type-options")).toBe("nosniff");
  });

  it("adds nosniff to edge-generated errors and cache hits", async () => {
    const redis = new FakeRedis();
    redis.store.set(await generateKey(), JSON.stringify(ANSWER));
    redis.store.set(`kv:${ABC_KEY_ID}`, "1");
    const h = harness({ fetch: mockFetch({ [ORIGIN_0]: () => json({}) }).fetch, redis });
    const unauth = await h.run(post("/generate", QUESTION));
    expect(unauth.headers.get("x-content-type-options")).toBe("nosniff");
    const hit = await h.run(post("/generate", QUESTION, KEY));
    expect(hit.headers.get("x-content-type-options")).toBe("nosniff");
  });
});

describe("platform errors from the origin's host", () => {
  it("turns a plain-text 429 into the JSON error shape with Retry-After", async () => {
    const origin = mockFetch({
      [ORIGIN_0]: () => new Response("Rate exceeded.", { status: 429, headers: { "content-type": "text/plain" } }),
    });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() }, baseEnv({ ORIGINS: ORIGIN_0 }));
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("10");
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(await res.json()).toMatchObject({ error: { type: "overloaded" } });
    expect(h.logs[0]).toMatchObject({ upstream_normalized: true });
  });

  it("keeps the origin's own Retry-After when present", async () => {
    const origin = mockFetch({
      [ORIGIN_0]: () => new Response("unavailable", { status: 503, headers: { "retry-after": "30" } }),
    });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() }, baseEnv({ ORIGINS: ORIGIN_0 }));
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(503);
    expect(res.headers.get("retry-after")).toBe("30");
    expect(await res.json()).toMatchObject({ error: { type: "upstream_unavailable" } });
  });

  it("passes the application's own JSON 429 through untouched", async () => {
    const body = { detail: "Queue full", extra: 1 };
    const origin = mockFetch({ [ORIGIN_0]: () => json(body, 429, { "retry-after": "3" }) });
    const h = harness({ fetch: origin.fetch, redis: new FakeRedis() }, baseEnv({ ORIGINS: ORIGIN_0 }));
    const res = await h.run(post("/generate", QUESTION, KEY));
    expect(res.status).toBe(429);
    expect(res.headers.get("retry-after")).toBe("3");
    expect(await res.json()).toEqual(body);
  });
});

describe("rate limit when Redis replies with a command error", () => {
  it("falls back to the in-isolate limiter instead of letting the request through", async () => {
    const redis = new FakeRedis();
    const original = redis.pipeline.bind(redis);
    redis.pipeline = async (commands) => {
      const replies = await original(commands);
      replies[0] = { error: "ERR wrong type" }; // the INCR reply
      return replies;
    };
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch, redis }, baseEnv({ RATE_LIMIT_PER_MINUTE: "1" }));
    expect((await h.run(post("/generate", { question: "a" }, KEY))).status).toBe(200);
    expect((await h.run(post("/generate", { question: "b" }, KEY))).status).toBe(429);
    expect(h.logs[0]).toMatchObject({ rl_scope: "local", redis_error: "ERR wrong type" });
  });
});

describe("a client that disconnects before the origin answers", () => {
  /** A POST whose `signal` the test can abort, as the runtime does when the client leaves. */
  function abortable(): { request: Request; leave: () => void } {
    const controller = new AbortController();
    const request = new Request("https://edge.test/generate", {
      method: "POST",
      headers: { "content-type": "application/json", "cf-connecting-ip": "203.0.113.7", ...KEY },
      body: JSON.stringify(QUESTION),
      signal: controller.signal,
    });
    return { request, leave: () => controller.abort() };
  }

  it("aborts the origin request, asks no other origin, and leaves the circuit closed", async () => {
    const signals: (AbortSignal | null)[] = [];
    let primaryAnswers = false;
    const origin = mockFetch({
      [ORIGIN_0]: (call, signal) => {
        if (primaryAnswers) return json(ANSWER);
        signals.push(signal);
        return hang(call, signal);
      },
      [ORIGIN_1]: () => json(ANSWER),
    });
    // No Redis, so nothing is answered from the cache.
    const h = harness({ fetch: origin.fetch, redis: null });

    for (let i = 0; i < 3; i++) {
      const { request, leave } = abortable();
      const pending = h.run(request);
      await new Promise((resolve) => setTimeout(resolve, 5)); // the origin request is in flight
      leave();
      const res = await pending;
      expect(res.status).toBe(499);
      expect(signals[i]?.aborted).toBe(true);
    }
    // The backup origin was never asked to start the abandoned work.
    expect(origin.calls.every((c) => c.url.startsWith(ORIGIN_0))).toBe(true);
    expect(h.logs.at(-1)).toMatchObject({ status: 499, client_gone: true, attempts: 1, origin_index: null });
    expect(h.logs.at(-1)?.["upstream_failures"]).toEqual(["origin-0: client disconnected"]);

    // Three disconnects would open the primary's circuit if they counted as failures;
    // the next request would then skip it and be answered by the backup.
    primaryAnswers = true;
    const next = await h.run(post("/generate", QUESTION, KEY));
    expect(next.status).toBe(200);
    expect(h.logs.at(-1)).toMatchObject({ origin_index: 0, attempts: 1 });
    expect(h.logs.at(-1)).not.toHaveProperty("skipped_origins");
  });

  it("does not contact an origin for a request that is already abandoned", async () => {
    const origin = mockFetch({ [ORIGIN_0]: () => json(ANSWER), [ORIGIN_1]: () => json(ANSWER) });
    const h = harness({ fetch: origin.fetch, redis: null });
    const { request, leave } = abortable();
    leave();
    const res = await h.run(request);
    expect(res.status).toBe(499);
    expect(origin.calls).toHaveLength(0);
    expect(h.logs[0]).toMatchObject({ client_gone: true, attempts: 0 });
  });

  it("still reports a timeout as a timeout", async () => {
    const origin = mockFetch({ [ORIGIN_0]: hang });
    const h = harness(
      { fetch: origin.fetch, redis: null },
      baseEnv({ ORIGINS: ORIGIN_0, ORIGIN_TIMEOUT_MS: "20" }),
    );
    const { request } = abortable(); // the client stays connected
    const res = await h.run(request);
    expect(res.status).toBe(504);
    expect(h.logs[0]).not.toHaveProperty("client_gone");
  });
});
