import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/App";
import { STORAGE_KEY } from "../src/state/settings";

type Handler = (url: string, init: RequestInit) => Response | Promise<Response>;

const HEALTH = {
  status: "ok",
  model_loaded: true,
  uses_adapter: true,
  backend: "llamacpp",
  model_version: "v2026.11.12",
  load_state: "loaded",
};

function json(body: unknown, init: ResponseInit = {}): Response {
  const headers = new Headers(init.headers);
  headers.set("content-type", "application/json");
  return new Response(JSON.stringify(body), { ...init, headers });
}

function sse(chunks: string[], headers: Record<string, string> = {}): Response {
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

function install(handlers: Record<string, Handler>) {
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

function doneEvent(requestId: string) {
  return `event: done\ndata: ${JSON.stringify({
    finish_reason: "stop",
    usage: { prompt_tokens: 40, completion_tokens: 12, total_tokens: 52 },
    sources: [{ url: "https://fastapi.tiangolo.com/tutorial/response-status-code/", heading: "Response Status Code", score: 0.81 }],
    request_id: requestId,
    model_version: "v2026.11.12",
  })}\n\n`;
}

beforeEach(() => {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ apiKey: "fa_test" }));
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("streams an answer with sources and metadata, then records feedback", async () => {
    const calls = install({
      "/health": () => json(HEALTH),
      "/generate": () =>
        sse(
          [
            'event: token\ndata: {"delta": "Set `status_code=201` on the decorator."}\n\n',
            doneEvent("req-1"),
          ],
          { "x-request-id": "req-1", "x-cache": "BYPASS" },
        ),
      "/v1/feedback": () => json({ status: "recorded" }),
    });
    const user = userEvent.setup();
    render(<App />);

    expect(await screen.findByText("v2026.11.12", { selector: ".badge" })).toBeInTheDocument();

    await user.type(screen.getByLabelText("Your FastAPI question"), "How do I return a 201?");
    await user.keyboard("{Enter}");

    expect(await screen.findByText("How do I return a 201?")).toBeInTheDocument();
    const answer = await screen.findByLabelText("Assistant answer");
    await waitFor(() => expect(within(answer).getByText(/on the decorator/)).toBeInTheDocument());
    expect(within(answer).getByText("Response Status Code")).toBeInTheDocument();
    expect(within(answer).getByText("40 in / 12 out")).toBeInTheDocument();
    expect(within(answer).getByText("cache BYPASS")).toBeInTheDocument();

    await user.click(within(answer).getByRole("button", { name: "Good answer" }));
    expect(await within(answer).findByText("Thanks, recorded.")).toBeInTheDocument();

    const feedback = calls.find((c) => c.url === "/v1/feedback");
    expect(feedback).toBeDefined();
    expect(JSON.parse(feedback?.init.body as string)).toEqual({ request_id: "req-1", rating: 1 });
    expect(new Headers(feedback?.init.headers).get("x-api-key")).toBe("fa_test");

    const generate = calls.find((c) => c.url === "/generate");
    expect(JSON.parse(generate?.init.body as string)).toMatchObject({
      question: "How do I return a 201?",
      stream: true,
      max_new_tokens: 256,
    });
  });

  it("shows the API-key hint on 401 and sends no feedback controls", async () => {
    install({
      "/health": () => json(HEALTH),
      "/generate": () => json({ detail: "Invalid API key" }, { status: 401 }),
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: EXAMPLE }));

    expect(await screen.findByText(/API key required: Invalid API key/)).toBeInTheDocument();
    const answer = screen.getByLabelText("Assistant answer");
    expect(within(answer).getByRole("alert")).toHaveTextContent("Invalid API key");
    expect(within(answer).queryByRole("button", { name: "Good answer" })).toBeNull();
  });

  it("waits for Retry-After on 429 and then succeeds", async () => {
    let attempts = 0;
    install({
      "/health": () => json(HEALTH),
      "/generate": () => {
        attempts += 1;
        if (attempts === 1) {
          return json({ detail: "Queue full" }, { status: 429, headers: { "retry-after": "1" } });
        }
        return sse(['event: token\ndata: {"delta": "After the wait."}\n\n', doneEvent("req-2")], {
          "x-request-id": "req-2",
        });
      },
    });
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: EXAMPLE }));

    expect(await screen.findByRole("status")).toHaveTextContent(/at capacity/);
    expect(await screen.findByText("After the wait.", {}, { timeout: 4000 })).toBeInTheDocument();
    expect(attempts).toBe(2);
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("uses the OpenAI-compatible endpoint with history in conversation mode", async () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ apiKey: "fa_test", mode: "chat", stream: false }));
    const calls = install({
      "/health": () => json(HEALTH),
      "/v1/chat/completions": () =>
        json(
          {
            id: "chatcmpl-r9",
            object: "chat.completion",
            created: 1,
            model: "fastapi-assistant",
            choices: [{ index: 0, message: { role: "assistant", content: "Reply" }, finish_reason: "stop" }],
            usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
          },
          { headers: { "x-request-id": "r9" } },
        ),
    });
    const user = userEvent.setup();
    render(<App />);

    const box = screen.getByLabelText("Your FastAPI question");
    await user.type(box, "first{Enter}");
    await screen.findByText("Reply");
    await user.type(box, "second{Enter}");
    await waitFor(() => expect(calls.filter((c) => c.url === "/v1/chat/completions")).toHaveLength(2));

    const second = calls.filter((c) => c.url === "/v1/chat/completions")[1];
    expect(JSON.parse(second?.init.body as string)).toMatchObject({
      model: "fastapi-assistant",
      messages: [
        { role: "user", content: "first" },
        { role: "assistant", content: "Reply" },
        { role: "user", content: "second" },
      ],
      max_tokens: 256,
    });
  });

  it("renders copyable snippets that follow the draft and never contain the key", async () => {
    install({ "/health": () => json(HEALTH) });
    const user = userEvent.setup();
    render(<App />);

    await user.type(screen.getByLabelText("Your FastAPI question"), "What does Depends do?");
    await user.click(screen.getByRole("tab", { name: "Code" }));

    const panel = screen.getByRole("tabpanel");
    expect(within(panel).getAllByText(/What does Depends do\?/).length).toBeGreaterThan(0);
    expect(panel.textContent).not.toContain("fa_test");
    // user-event installs a working clipboard stub on setup().
    await user.click(within(panel).getAllByRole("button", { name: "Copy" })[0] as HTMLElement);
    const copied = await navigator.clipboard.readText();
    expect(copied).toContain("curl");
    expect(copied).toContain("What does Depends do?");
    expect(copied).not.toContain("fa_test");
  });
});

const EXAMPLE = "How do I return a 201 status code from a POST endpoint?";
