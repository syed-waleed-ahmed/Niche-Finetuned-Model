/**
 * HTTP client for the assistant service.
 *
 * Both APIs (`/generate` and the OpenAI-compatible `/v1/chat/completions`) are
 * exposed through one async-iterator contract (`AnswerEvent`) so the UI renders
 * streamed answers, buffered answers, edge cache hits and errors the same way.
 */

import { readSse } from "./sse";
import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatMessage,
  FeedbackRequest,
  GenerateDoneEvent,
  GenerateRequest,
  GenerateResponse,
  HealthResponse,
  Source,
  Usage,
} from "./types";

export interface ClientConfig {
  /** "" means same origin (production, and the Vite dev proxy). */
  baseUrl: string;
  apiKey: string;
}

/** Headers worth showing: set by the origin or by the edge Worker. */
export interface ResponseMeta {
  status: number;
  requestId: string | null;
  cache: string | null;
  servedBy: string | null;
  processTimeMs: number | null;
  rateLimit: { limit: number | null; remaining: number | null; reset: number | null };
  retryAfterS: number | null;
}

export type AnswerEvent =
  | { kind: "meta"; meta: ResponseMeta }
  | { kind: "delta"; text: string }
  | {
      kind: "done";
      finishReason: string;
      usage: Usage | null;
      sources: Source[];
      requestId: string | null;
      modelVersion: string | null;
    };

export class ApiError extends Error {
  readonly status: number;
  readonly errorType: string;
  readonly retryAfterS: number | null;
  readonly requestId: string | null;

  constructor(
    message: string,
    options: {
      status: number;
      errorType?: string;
      retryAfterS?: number | null;
      requestId?: string | null;
    },
  ) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.errorType = options.errorType ?? statusType(options.status);
    this.retryAfterS = options.retryAfterS ?? null;
    this.requestId = options.requestId ?? null;
  }

  /** 429 and 503 carry Retry-After: the same request can simply be repeated. */
  get isRetryable(): boolean {
    return this.status === 429 || this.status === 503;
  }
}

function statusType(status: number): string {
  if (status === 401) return "authentication_error";
  if (status === 403) return "permission_error";
  if (status === 404) return "not_found";
  if (status === 429) return "rate_limited";
  if (status === 503) return "overloaded";
  if (status >= 500) return "server_error";
  return "invalid_request_error";
}

function numberHeader(headers: Headers, name: string): number | null {
  const raw = headers.get(name);
  if (raw === null || raw.trim() === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) ? value : null;
}

export function retryAfterSeconds(headers: Headers, now: () => number = Date.now): number | null {
  const raw = headers.get("retry-after");
  if (raw === null) return null;
  const seconds = Number(raw);
  if (Number.isFinite(seconds)) return Math.max(0, seconds);
  const date = Date.parse(raw);
  return Number.isNaN(date) ? null : Math.max(0, Math.ceil((date - now()) / 1000));
}

export function metaFromResponse(response: Response): ResponseMeta {
  const h = response.headers;
  return {
    status: response.status,
    requestId: h.get("x-request-id"),
    cache: h.get("x-cache"),
    servedBy: h.get("x-served-by"),
    processTimeMs: numberHeader(h, "x-process-time-ms"),
    rateLimit: {
      limit: numberHeader(h, "x-ratelimit-limit"),
      remaining: numberHeader(h, "x-ratelimit-remaining"),
      reset: numberHeader(h, "x-ratelimit-reset"),
    },
    retryAfterS: retryAfterSeconds(h),
  };
}

export function joinUrl(baseUrl: string, path: string): string {
  return `${baseUrl.replace(/\/+$/, "")}${path}`;
}

function headersFor(config: ClientConfig, json: boolean): Headers {
  const headers = new Headers();
  if (json) headers.set("content-type", "application/json");
  if (config.apiKey) headers.set("x-api-key", config.apiKey);
  return headers;
}

/** Build an ApiError from either error body shape; fall back to the status text. */
export async function errorFromResponse(response: Response): Promise<ApiError> {
  let message = response.statusText || `HTTP ${response.status}`;
  let errorType: string | undefined;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object") {
      const record = body as Record<string, unknown>;
      const error = record["error"];
      if (error && typeof error === "object") {
        const e = error as Record<string, unknown>;
        if (typeof e["message"] === "string") message = e["message"];
        if (typeof e["type"] === "string") errorType = e["type"];
      } else if (typeof record["detail"] === "string") {
        message = record["detail"];
      } else if (Array.isArray(record["detail"])) {
        const first = record["detail"][0] as Record<string, unknown> | undefined;
        if (first && typeof first["msg"] === "string") message = first["msg"];
      }
    }
  } catch {
    // Non-JSON body (empty, or HTML from an intermediary): keep the status text.
  }
  const meta = metaFromResponse(response);
  return new ApiError(message, {
    status: response.status,
    ...(errorType ? { errorType } : {}),
    retryAfterS: meta.retryAfterS,
    requestId: meta.requestId,
  });
}

async function postJson(
  config: ClientConfig,
  path: string,
  payload: unknown,
  signal?: AbortSignal,
): Promise<Response> {
  const response = await fetch(joinUrl(config.baseUrl, path), {
    method: "POST",
    headers: headersFor(config, true),
    body: JSON.stringify(payload),
    ...(signal ? { signal } : {}),
  });
  if (!response.ok) throw await errorFromResponse(response);
  return response;
}

function isEventStream(response: Response): boolean {
  return (response.headers.get("content-type") ?? "").includes("text/event-stream");
}

function incompleteStream(requestId: string | null): ApiError {
  return new ApiError("The stream ended before the answer was complete", {
    status: 200,
    errorType: "incomplete_stream",
    requestId,
  });
}

// -- native API -------------------------------------------------------------------

export async function* askNative(
  config: ClientConfig,
  request: GenerateRequest,
  signal?: AbortSignal,
): AsyncGenerator<AnswerEvent> {
  const response = await postJson(config, "/generate", request, signal);
  const meta = metaFromResponse(response);
  yield { kind: "meta", meta };

  // A JSON body to a streaming request is a shared-cache hit answered at the edge.
  if (!request.stream || !isEventStream(response) || !response.body) {
    const body = (await response.json()) as GenerateResponse;
    yield { kind: "delta", text: body.answer };
    yield {
      kind: "done",
      finishReason: body.finish_reason,
      usage: body.usage,
      sources: body.sources ?? [],
      requestId: body.request_id ?? meta.requestId,
      modelVersion: body.model_version,
    };
    return;
  }

  for await (const event of readSse(response.body, signal)) {
    if (event.event === "token") {
      const { delta } = JSON.parse(event.data) as { delta: string };
      if (delta) yield { kind: "delta", text: delta };
    } else if (event.event === "done") {
      const done = JSON.parse(event.data) as GenerateDoneEvent;
      yield {
        kind: "done",
        finishReason: done.finish_reason,
        usage: done.usage,
        sources: done.sources ?? [],
        requestId: done.request_id ?? meta.requestId,
        modelVersion: done.model_version,
      };
      return;
    } else if (event.event === "error") {
      const { message } = JSON.parse(event.data) as { message: string };
      throw new ApiError(message, {
        status: 200,
        errorType: /not ready/i.test(message) ? "model_not_ready" : "server_error",
        requestId: meta.requestId,
      });
    }
  }
  throw incompleteStream(meta.requestId);
}

// -- OpenAI-compatible API --------------------------------------------------------

export async function* askChat(
  config: ClientConfig,
  request: ChatCompletionRequest,
  signal?: AbortSignal,
): AsyncGenerator<AnswerEvent> {
  const response = await postJson(config, "/v1/chat/completions", request, signal);
  const meta = metaFromResponse(response);
  yield { kind: "meta", meta };

  if (!request.stream || !isEventStream(response) || !response.body) {
    const body = (await response.json()) as ChatCompletionResponse;
    const choice = body.choices[0];
    yield { kind: "delta", text: choice?.message.content ?? "" };
    yield {
      kind: "done",
      finishReason: choice?.finish_reason ?? "stop",
      usage: body.usage ?? null,
      sources: [],
      requestId: meta.requestId ?? requestIdFromCompletion(body.id),
      modelVersion: null,
    };
    return;
  }

  let finishReason = "stop";
  let completionId: string | null = null;
  for await (const event of readSse(response.body, signal)) {
    if (event.data === "[DONE]") {
      yield {
        kind: "done",
        finishReason,
        usage: null,
        sources: [],
        requestId: meta.requestId ?? requestIdFromCompletion(completionId),
        modelVersion: null,
      };
      return;
    }
    const chunk = JSON.parse(event.data) as
      | ChatCompletionChunk
      | { error: { message: string; type: string } };
    if ("error" in chunk) {
      throw new ApiError(chunk.error.message, {
        status: 200,
        errorType: chunk.error.type,
        requestId: meta.requestId,
      });
    }
    completionId = chunk.id;
    const choice = chunk.choices[0];
    if (choice?.delta.content) yield { kind: "delta", text: choice.delta.content };
    if (choice?.finish_reason) finishReason = choice.finish_reason;
  }
  throw incompleteStream(meta.requestId);
}

function requestIdFromCompletion(id: string | null): string | null {
  if (!id) return null;
  return id.startsWith("chatcmpl-") ? id.slice("chatcmpl-".length) : id;
}

/** Build the `/v1` message list from the visible conversation. */
export function chatMessagesFrom(
  turns: { role: "user" | "assistant"; content: string }[],
): ChatMessage[] {
  return turns
    .filter((t) => t.content.trim().length > 0)
    .map((t) => ({ role: t.role, content: t.content }));
}

// -- feedback and health ----------------------------------------------------------

export async function sendFeedback(
  config: ClientConfig,
  request: FeedbackRequest,
  signal?: AbortSignal,
): Promise<ResponseMeta> {
  const response = await postJson(config, "/v1/feedback", request, signal);
  return metaFromResponse(response);
}

export async function getHealth(
  config: ClientConfig,
  signal?: AbortSignal,
): Promise<{ health: HealthResponse; meta: ResponseMeta }> {
  const response = await fetch(joinUrl(config.baseUrl, "/health"), {
    method: "GET",
    headers: headersFor(config, false),
    ...(signal ? { signal } : {}),
  });
  if (!response.ok) throw await errorFromResponse(response);
  return {
    health: (await response.json()) as HealthResponse,
    meta: metaFromResponse(response),
  };
}
