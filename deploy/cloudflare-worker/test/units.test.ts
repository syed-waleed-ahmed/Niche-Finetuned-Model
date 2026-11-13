import { describe, expect, it } from "vitest";
import { cacheKey, canonicalJson, isJsonContentType, parseCacheableBody } from "../src/cache";
import { DEFAULTS, loadConfig } from "../src/config";
import { extractApiKey, keyIdentity, sha256Hex } from "../src/identity";
import { buildForwardHeaders, rewriteLocation } from "../src/proxy";
import { evaluateRateLimit, rateLimitCommands } from "../src/ratelimit";
import { classifyPath, isCacheableEndpoint } from "../src/routes";
import { ABC_KEY_ID, EPOCH_MINUTE, NOW, RESET_EPOCH_SECONDS } from "./helpers";

const enc = new TextEncoder();
const buf = (s: string): ArrayBuffer => enc.encode(s).buffer as ArrayBuffer;

describe("canonicalJson", () => {
  it("makes object key order irrelevant, recursively", () => {
    const a = { question: "q", options: { b: 1, a: [{ y: 2, x: 1 }] }, temperature: 0.2 };
    const b = { temperature: 0.2, options: { a: [{ x: 1, y: 2 }], b: 1 }, question: "q" };
    expect(canonicalJson(a)).toBe(canonicalJson(b));
    expect(canonicalJson(a)).toBe('{"options":{"a":[{"x":1,"y":2}],"b":1},"question":"q","temperature":0.2}');
  });

  it("keeps array order significant", () => {
    expect(canonicalJson([1, 2])).not.toBe(canonicalJson([2, 1]));
  });

  it("produces the same cache key regardless of key order", async () => {
    const k1 = await cacheKey("v1", "/generate", { question: "hi", max_new_tokens: 64 });
    const k2 = await cacheKey("v1", "/generate", { max_new_tokens: 64, question: "hi" });
    expect(k1).toBe(k2);
    expect(k1).toMatch(/^c:v1:[0-9a-f]{64}$/);
    expect(await cacheKey("v2", "/generate", { question: "hi", max_new_tokens: 64 })).not.toBe(k1);
    expect(await cacheKey("v1", "/v1/chat/completions", { question: "hi", max_new_tokens: 64 })).not.toBe(k1);
  });
});

describe("parseCacheableBody", () => {
  it("accepts a JSON object without stream", () => {
    expect(parseCacheableBody(buf('{"question":"q"}'), 1024)).toEqual({ question: "q" });
    expect(parseCacheableBody(buf('{"question":"q","stream":false}'), 1024)).not.toBeNull();
  });

  it("bypasses streaming, invalid, non-object, empty and oversized bodies", () => {
    expect(parseCacheableBody(buf('{"question":"q","stream":true}'), 1024)).toBeNull();
    expect(parseCacheableBody(buf('{"question":"q","stream":"true"}'), 1024)).toBeNull();
    expect(parseCacheableBody(buf("not json"), 1024)).toBeNull();
    expect(parseCacheableBody(buf("[1,2]"), 1024)).toBeNull();
    expect(parseCacheableBody(null, 1024)).toBeNull();
    expect(parseCacheableBody(buf(`{"question":"${"x".repeat(100)}"}`), 50)).toBeNull();
  });

  it("recognises JSON content types", () => {
    expect(isJsonContentType("application/json; charset=utf-8")).toBe(true);
    expect(isJsonContentType("text/event-stream")).toBe(false);
    expect(isJsonContentType(null)).toBe(false);
  });
});

describe("identity", () => {
  it("hashes the key to the first 32 hex chars of SHA-256", async () => {
    expect(await sha256Hex("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
    expect(await keyIdentity("abc")).toEqual({ type: "key", id: ABC_KEY_ID });
  });

  it("reads X-API-Key or Authorization: Bearer", () => {
    expect(extractApiKey(new Headers({ "x-api-key": " k1 " }))).toBe("k1");
    expect(extractApiKey(new Headers({ authorization: "Bearer k2" }))).toBe("k2");
    expect(extractApiKey(new Headers({ authorization: "bearer   k3" }))).toBe("k3");
    expect(extractApiKey(new Headers({ authorization: "Basic abc" }))).toBeNull();
    expect(extractApiKey(new Headers({ "x-api-key": "  " }))).toBeNull();
    expect(extractApiKey(new Headers())).toBeNull();
  });
});

describe("rate limit math", () => {
  it("uses rl:{identity}:{epochMinute} with INCR + EXPIRE 70", () => {
    expect(rateLimitCommands("abc", NOW)).toEqual([
      ["INCR", `rl:abc:${EPOCH_MINUTE}`],
      ["EXPIRE", `rl:abc:${EPOCH_MINUTE}`, "70"],
    ]);
  });

  it("computes remaining, reset and retry-after", () => {
    expect(evaluateRateLimit(3, 20, NOW)).toEqual({
      limit: 20,
      remaining: 17,
      resetEpochSeconds: RESET_EPOCH_SECONDS,
      retryAfterSeconds: 10,
      exceeded: false,
    });
    expect(evaluateRateLimit(21, 20, NOW)).toMatchObject({ remaining: 0, exceeded: true });
    expect(evaluateRateLimit(20, 20, NOW).exceeded).toBe(false);
  });
});

describe("config", () => {
  it("applies defaults and disables Upstash when vars are missing", () => {
    const c = loadConfig({});
    expect(c.origins).toEqual([]);
    expect(c.upstash).toBeNull();
    expect(c.rateLimitPerMinute).toBe(20);
    expect(c.rateLimitPerMinuteAnon).toBe(10);
    expect(c.cacheTtlSeconds).toBe(86_400);
    expect(c.originTimeoutMs).toBe(60_000);
    expect(c.requireApiKey).toBe(true);
    expect(c.keyVerifiedTtlSeconds).toBe(300);
    expect(c.maxCachedBodyBytes).toBe(DEFAULTS.MAX_CACHED_BODY_BYTES);
  });

  it("parses strings, numbers and booleans and trims origins", () => {
    const c = loadConfig({
      ORIGINS: " https://a.test/ , https://b.test ,",
      UPSTASH_REDIS_REST_URL: "https://r.upstash.io/",
      UPSTASH_REDIS_REST_TOKEN: "tok",
      RATE_LIMIT_PER_MINUTE: 5,
      RATE_LIMIT_PER_MINUTE_ANON: "bogus",
      REQUIRE_API_KEY: "false",
    });
    expect(c.origins).toEqual(["https://a.test", "https://b.test"]);
    expect(c.upstash).toEqual({ url: "https://r.upstash.io", token: "tok" });
    expect(c.rateLimitPerMinute).toBe(5);
    expect(c.rateLimitPerMinuteAnon).toBe(10);
    expect(c.requireApiKey).toBe(false);
  });
});

describe("routes", () => {
  it("classifies paths", () => {
    expect(classifyPath("/generate")).toBe("protected");
    expect(classifyPath("/v1/chat/completions")).toBe("protected");
    expect(classifyPath("/v1/models")).toBe("protected");
    expect(classifyPath("/health")).toBe("public");
    expect(classifyPath("/openapi.json")).toBe("public");
    expect(classifyPath("/metrics")).toBe("blocked");
    expect(classifyPath("/wp-login.php")).toBe("unknown");
    expect(isCacheableEndpoint("POST", "/generate")).toBe(true);
    expect(isCacheableEndpoint("GET", "/generate")).toBe(false);
    expect(isCacheableEndpoint("POST", "/v1/feedback")).toBe(false);
  });
});

describe("forward headers", () => {
  it("overwrites the secret and drops host, cf-* and spoofed forwarding headers", () => {
    const h = buildForwardHeaders(
      new Headers({
        "x-origin-secret": "attacker",
        "x-forwarded-for": "6.6.6.6",
        "cf-connecting-ip": "203.0.113.7",
        "x-api-key": "k",
        "content-type": "application/json",
      }),
      { originSecret: "real", requestId: "rid", clientIp: "203.0.113.7" },
    );
    expect(h.get("x-origin-secret")).toBe("real");
    expect(h.get("x-forwarded-for")).toBe("203.0.113.7");
    expect(h.get("x-request-id")).toBe("rid");
    expect(h.get("cf-connecting-ip")).toBeNull();
    expect(h.get("x-api-key")).toBe("k");
    expect(h.get("content-type")).toBe("application/json");
  });

  it("rewrites origin redirects to the edge hostname", () => {
    const h = new Headers({ location: "http://o0.test/v1/models?x=1" });
    rewriteLocation(h, "https://o0.test", "https://edge.test");
    expect(h.get("location")).toBe("https://edge.test/v1/models?x=1");
    const other = new Headers({ location: "https://elsewhere.test/a" });
    rewriteLocation(other, "https://o0.test", "https://edge.test");
    expect(other.get("location")).toBe("https://elsewhere.test/a");
  });
});

describe("trimTrailingSlashes", () => {
  it("strips any number of trailing slashes in linear time", async () => {
    const { trimTrailingSlashes } = await import("../src/config");
    expect(trimTrailingSlashes("https://a.example///")).toBe("https://a.example");
    expect(trimTrailingSlashes("https://a.example/path")).toBe("https://a.example/path");
    expect(trimTrailingSlashes("////")).toBe("");
    expect(trimTrailingSlashes("/".repeat(100_000) + "x")).toBe("/".repeat(100_000) + "x");
  });
});
