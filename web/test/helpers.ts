import { vi } from "vitest";

import type { HealthResponse } from "../src/api/types";
import { NARROW_QUERY } from "../src/hooks/useMediaQuery";

export type Handler = (url: string, init: RequestInit) => Response | Promise<Response>;

export const HEALTH: HealthResponse = {
  status: "ok",
  model_loaded: true,
  uses_adapter: true,
  backend: "llamacpp",
  model_version: "v2026.11.12",
  load_state: "loaded",
};

export const EXAMPLE = "How do I return a 201 status code from a POST endpoint?";

export function json(body: unknown, init: ResponseInit = {}): Response {
  const headers = new Headers(init.headers);
  headers.set("content-type", "application/json");
  return new Response(JSON.stringify(body), { ...init, headers });
}

export function sse(chunks: string[], headers: Record<string, string> = {}): Response {
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

export function doneEvent(requestId: string): string {
  return `event: done\ndata: ${JSON.stringify({
    finish_reason: "stop",
    usage: { prompt_tokens: 40, completion_tokens: 12, total_tokens: 52 },
    sources: [
      {
        url: "https://fastapi.tiangolo.com/tutorial/response-status-code/",
        heading: "Response Status Code",
        score: 0.81,
      },
    ],
    request_id: requestId,
    model_version: "v2026.11.12",
  })}\n\n`;
}

/** Stub fetch with one handler per URL; returns every call made. */
export function install(handlers: Record<string, Handler>) {
  const calls: { url: string; init: RequestInit }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      calls.push({ url, init });
      const handler = handlers[url];
      if (!handler) return new Response("not found", { status: 404 });
      return handler(url, init);
    }),
  );
  return calls;
}

/** jsdom has no matchMedia; emulate a narrow (drawer) or wide viewport. */
export function setViewport(width: "narrow" | "wide") {
  Object.defineProperty(window, "matchMedia", {
    configurable: true,
    writable: true,
    value: (query: string) => ({
      matches: width === "narrow" && query === NARROW_QUERY,
      media: query,
      onchange: null,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
      addListener: () => undefined,
      removeListener: () => undefined,
      dispatchEvent: () => false,
    }),
  });
}

export function resetViewport() {
  Reflect.deleteProperty(window, "matchMedia");
}
