/**
 * Conversation state: a list of turns and the reducer that applies streaming
 * events to the assistant turn in progress. Kept free of React so it is easy
 * to test and reason about.
 */

import type { ResponseMeta } from "../api/client";
import type { EndpointMode } from "../api/snippets";
import type { Source, Usage } from "../api/types";

export type TurnStatus = "pending" | "streaming" | "done" | "error" | "stopped";

export interface UserTurn {
  id: string;
  role: "user";
  content: string;
}

export interface AssistantTurn {
  id: string;
  role: "assistant";
  content: string;
  status: TurnStatus;
  error: string | null;
  errorType: string | null;
  meta: ResponseMeta | null;
  finishReason: string | null;
  usage: Usage | null;
  sources: Source[];
  requestId: string | null;
  modelVersion: string | null;
  startedAt: number;
  firstTokenAt: number | null;
  finishedAt: number | null;
  feedback: 1 | -1 | null;
  mode: EndpointMode;
  streamed: boolean;
}

export type Turn = UserTurn | AssistantTurn;

export type ConversationAction =
  | { type: "start"; user: UserTurn; assistant: AssistantTurn }
  | { type: "meta"; id: string; meta: ResponseMeta }
  | { type: "delta"; id: string; text: string; at: number }
  | {
      type: "done";
      id: string;
      at: number;
      finishReason: string;
      usage: Usage | null;
      sources: Source[];
      requestId: string | null;
      modelVersion: string | null;
    }
  | { type: "error"; id: string; at: number; message: string; errorType: string }
  | { type: "stopped"; id: string; at: number }
  | { type: "pending"; id: string }
  | { type: "feedback"; id: string; rating: 1 | -1 }
  | { type: "clear" };

export function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function newAssistantTurn(mode: EndpointMode, streamed: boolean, now: number): AssistantTurn {
  return {
    id: newId(),
    role: "assistant",
    content: "",
    status: "pending",
    error: null,
    errorType: null,
    meta: null,
    finishReason: null,
    usage: null,
    sources: [],
    requestId: null,
    modelVersion: null,
    startedAt: now,
    firstTokenAt: null,
    finishedAt: null,
    feedback: null,
    mode,
    streamed,
  };
}

function patch(
  turns: Turn[],
  id: string,
  update: (turn: AssistantTurn) => AssistantTurn,
): Turn[] {
  return turns.map((t) => (t.role === "assistant" && t.id === id ? update(t) : t));
}

export function conversationReducer(turns: Turn[], action: ConversationAction): Turn[] {
  switch (action.type) {
    case "start":
      return [...turns, action.user, action.assistant];
    case "meta":
      return patch(turns, action.id, (t) => ({ ...t, meta: action.meta, status: "streaming" }));
    case "delta":
      return patch(turns, action.id, (t) => ({
        ...t,
        content: t.content + action.text,
        status: "streaming",
        firstTokenAt: t.firstTokenAt ?? action.at,
      }));
    case "done":
      return patch(turns, action.id, (t) => ({
        ...t,
        status: "done",
        finishedAt: action.at,
        finishReason: action.finishReason,
        usage: action.usage,
        sources: action.sources,
        requestId: action.requestId ?? t.meta?.requestId ?? null,
        modelVersion: action.modelVersion,
      }));
    case "error":
      return patch(turns, action.id, (t) => ({
        ...t,
        status: "error",
        finishedAt: action.at,
        error: action.message,
        errorType: action.errorType,
      }));
    case "stopped":
      return patch(turns, action.id, (t) => ({
        ...t,
        status: "stopped",
        finishedAt: action.at,
        finishReason: t.content ? "cancelled" : null,
        requestId: t.requestId ?? t.meta?.requestId ?? null,
      }));
    case "pending":
      return patch(turns, action.id, (t) => ({ ...t, status: "pending", meta: null }));
    case "feedback":
      return patch(turns, action.id, (t) => ({ ...t, feedback: action.rating }));
    case "clear":
      return [];
  }
}

/** Prior turns usable as `/v1` history: completed answers and their questions. */
export function historyFor(turns: Turn[]): { role: "user" | "assistant"; content: string }[] {
  const out: { role: "user" | "assistant"; content: string }[] = [];
  for (let i = 0; i < turns.length; i++) {
    const turn = turns[i];
    if (!turn) continue;
    if (turn.role === "user") {
      const reply = turns[i + 1];
      const answered =
        reply?.role === "assistant" &&
        (reply.status === "done" || reply.status === "stopped") &&
        reply.content.trim().length > 0;
      if (answered && reply.role === "assistant") {
        out.push({ role: "user", content: turn.content });
        out.push({ role: "assistant", content: reply.content });
      }
    }
  }
  return out;
}
