import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  askChat,
  askNative,
  chatMessagesFrom,
  getHealth,
  joinUrl,
  retryAfterSeconds,
  sendFeedback,
  type AnswerEvent,
} from "../src/api/client";

const config = { baseUrl: "", apiKey: "fa_test" };

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  const headers = new Headers(init.headers);
  headers.set("content-type", "application/json");
  return new Response(JSON.stringify(body), { ...init, headers });
}

function sseResponse(chunks: string[], headers: Record<string, string> = {}): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "content-type": "text/event-stream", ...headers },
  });
}

async function collect(iter: AsyncIterable<AnswerEvent>): Promise<AnswerEvent[]> {
  const out: AnswerEvent[] = [];
  for await (const item of iter) out.push(item);
  return out;
}

function mockFetch(response: Response | (() => Response)) {
  const fn = vi.fn(async () => (typeof response === "function" ? response() : response));
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("helpers", () => {
  it("joins base URLs without duplicate slashes", () => {
    expect(joinUrl("", "/generate")).toBe("/generate");
    expect(joinUrl("https://edge.example/", "/v1/models")).toBe("https://edge.example/v1/models");
  });

  it("reads Retry-After as seconds or as an HTTP date", () => {
    expect(retryAfterSeconds(new Headers({ "retry-after": "7" }))).toBe(7);
    expect(retryAfterSeconds(new Headers())).toBeNull();
    const now = Date.UTC(2026, 0, 1, 0, 0, 0);
    const later = new Date(now + 30_000).toUTCString();
    expect(retryAfterSeconds(new Headers({ "retry-after": later }), () => now)).toBe(30);
  });

  it("filters empty turns when building /v1 messages", () => {
    expect(
      chatMessagesFrom([
        { role: "user", content: "a" },
        { role: "assistant", content: "  " },
        { role: "user", content: "b" },
      ]),
    ).toEqual([
      { role: "user", content: "a" },
      { role: "user", content: "b" },
    ]);
  });
});

describe("askNative", () => {
  it("streams token events and finishes with the done payload", async () => {
    const fetchMock = mockFetch(
      sseResponse(
        [
          'event: token\ndata: {"delta": "Use "}\n\n',
          'event: token\ndata: {"delta": "status_code"}\n\n',
          'event: done\ndata: {"finish_reason": "stop", "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}, "sources": [{"url": "https://fastapi.tiangolo.com/x/", "heading": "X", "score": 0.8}], "request_id": "abc", "model_version": "v1"}\n\n',
        ],
        { "x-request-id": "abc", "x-cache": "BYPASS" },
      ),
    );

    const events = await collect(askNative(config, { question: "q", stream: true }));

    expect(events[0]).toMatchObject({ kind: "meta", meta: { requestId: "abc", cache: "BYPASS" } });
    expect(events.filter((e) => e.kind === "delta").map((e) => (e.kind === "delta" ? e.text : ""))).toEqual([
      "Use ",
      "status_code",
    ]);
    expect(events.at(-1)).toMatchObject({
      kind: "done",
      finishReason: "stop",
      usage: { total_tokens: 7 },
      sources: [{ heading: "X" }],
      requestId: "abc",
      modelVersion: "v1",
    });

    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/generate");
    expect(new Headers(init.headers).get("x-api-key")).toBe("fa_test");
    expect(JSON.parse(init.body as string)).toEqual({ question: "q", stream: true });
  });

  it("treats a JSON body to a streaming request as a complete (cached) answer", async () => {
    mockFetch(
      jsonResponse(
        {
          question: "q",
          answer: "cached answer",
          model: "m",
          model_version: "v1",
          uses_adapter: true,
          finish_reason: "stop",
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          sources: [],
          request_id: "orig",
        },
        { headers: { "x-cache": "HIT", "x-request-id": "edge" } },
      ),
    );
    const events = await collect(askNative(config, { question: "q", stream: true }));
    expect(events.map((e) => e.kind)).toEqual(["meta", "delta", "done"]);
    expect(events[0]).toMatchObject({ meta: { cache: "HIT" } });
    expect(events[1]).toMatchObject({ text: "cached answer" });
    // Feedback must attach to the answer that was actually generated.
    expect(events[2]).toMatchObject({ requestId: "orig" });
  });

  it("raises an ApiError with Retry-After on 429", async () => {
    mockFetch(
      jsonResponse(
        { detail: "Queue full" },
        { status: 429, headers: { "retry-after": "3", "x-request-id": "r1" } },
      ),
    );
    const error = await collect(askNative(config, { question: "q" })).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    const apiError = error as ApiError;
    expect(apiError.status).toBe(429);
    expect(apiError.retryAfterS).toBe(3);
    expect(apiError.isRetryable).toBe(true);
    expect(apiError.message).toBe("Queue full");
    expect(apiError.requestId).toBe("r1");
  });

  it("maps 401 to authentication_error from either error shape", async () => {
    mockFetch(jsonResponse({ detail: "Invalid API key" }, { status: 401 }));
    const native = (await collect(askNative(config, { question: "q" })).catch((e: unknown) => e)) as ApiError;
    expect(native.errorType).toBe("authentication_error");

    mockFetch(
      jsonResponse(
        { error: { message: "Missing API key", type: "authentication_error", code: 401 } },
        { status: 401 },
      ),
    );
    const openai = (await collect(askChat(config, { messages: [] })).catch((e: unknown) => e)) as ApiError;
    expect(openai.errorType).toBe("authentication_error");
    expect(openai.message).toBe("Missing API key");
  });

  it("surfaces an in-stream error event", async () => {
    mockFetch(sseResponse(['event: error\ndata: {"message": "Model not ready: loading"}\n\n']));
    const error = (await collect(askNative(config, { question: "q", stream: true })).catch(
      (e: unknown) => e,
    )) as ApiError;
    expect(error.errorType).toBe("model_not_ready");
    expect(error.message).toContain("loading");
  });

  it("fails when the stream ends without a done event", async () => {
    mockFetch(sseResponse(['event: token\ndata: {"delta": "partial"}\n\n']));
    const error = (await collect(askNative(config, { question: "q", stream: true })).catch(
      (e: unknown) => e,
    )) as ApiError;
    expect(error.errorType).toBe("incomplete_stream");
  });
});

describe("askChat", () => {
  it("streams OpenAI chunks until [DONE] and derives the request id", async () => {
    const chunk = (delta: Record<string, string>, finish: string | null = null) =>
      `data: ${JSON.stringify({
        id: "chatcmpl-req42",
        object: "chat.completion.chunk",
        created: 1,
        model: "fastapi-assistant",
        choices: [{ index: 0, delta, finish_reason: finish }],
      })}\n\n`;
    mockFetch(
      sseResponse([chunk({ role: "assistant" }), chunk({ content: "Hello" }), chunk({}, "length"), "data: [DONE]\n\n"]),
    );
    const events = await collect(askChat(config, { messages: [{ role: "user", content: "hi" }], stream: true }));
    expect(events.map((e) => e.kind)).toEqual(["meta", "delta", "done"]);
    expect(events.at(-1)).toMatchObject({ finishReason: "length", requestId: "req42", sources: [] });
  });

  it("returns a buffered completion when not streaming", async () => {
    mockFetch(
      jsonResponse(
        {
          id: "chatcmpl-1",
          object: "chat.completion",
          created: 1,
          model: "fastapi-assistant",
          choices: [{ index: 0, message: { role: "assistant", content: "Answer" }, finish_reason: "stop" }],
          usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
        },
        { headers: { "x-request-id": "hdr" } },
      ),
    );
    const events = await collect(askChat(config, { messages: [{ role: "user", content: "hi" }] }));
    expect(events[1]).toMatchObject({ kind: "delta", text: "Answer" });
    expect(events[2]).toMatchObject({ kind: "done", requestId: "hdr", usage: { total_tokens: 4 } });
  });

  it("raises on an error chunk inside the stream", async () => {
    mockFetch(sseResponse(['data: {"error": {"message": "Generation failed", "type": "server_error"}}\n\n']));
    const error = (await collect(askChat(config, { messages: [], stream: true })).catch((e: unknown) => e)) as ApiError;
    expect(error.errorType).toBe("server_error");
  });
});

describe("feedback and health", () => {
  it("posts feedback with the key and returns response metadata", async () => {
    const fetchMock = mockFetch(jsonResponse({ status: "recorded" }, { headers: { "x-request-id": "f1" } }));
    const meta = await sendFeedback(config, { request_id: "abc", rating: 1 });
    expect(meta.requestId).toBe("f1");
    const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/v1/feedback");
    expect(JSON.parse(init.body as string)).toEqual({ request_id: "abc", rating: 1 });
  });

  it("reads health and rate-limit headers", async () => {
    mockFetch(
      jsonResponse(
        { status: "ok", model_loaded: false, uses_adapter: true, backend: "llamacpp", model_version: "v1", load_state: "loading" },
        { headers: { "x-ratelimit-limit": "10", "x-ratelimit-remaining": "9", "x-ratelimit-reset": "42" } },
      ),
    );
    const { health, meta } = await getHealth({ baseUrl: "https://edge.example", apiKey: "" });
    expect(health.load_state).toBe("loading");
    expect(meta.rateLimit).toEqual({ limit: 10, remaining: 9, reset: 42 });
  });
});
