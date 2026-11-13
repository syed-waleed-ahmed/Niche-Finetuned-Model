import { describe, expect, it } from "vitest";

import {
  conversationReducer,
  historyFor,
  newAssistantTurn,
  type Turn,
} from "../src/state/conversation";

const meta = {
  status: 200,
  requestId: "hdr-id",
  cache: "MISS",
  servedBy: "origin-0",
  processTimeMs: 12,
  rateLimit: { limit: null, remaining: null, reset: null },
  retryAfterS: null,
};

function started(): { turns: Turn[]; id: string } {
  const assistant = newAssistantTurn("generate", true, 1000);
  const turns = conversationReducer([], {
    type: "start",
    user: { id: "u1", role: "user", content: "q" },
    assistant,
  });
  return { turns, id: assistant.id };
}

describe("conversationReducer", () => {
  it("accumulates deltas and records the first-token time", () => {
    const { turns, id } = started();
    let next = conversationReducer(turns, { type: "meta", id, meta });
    next = conversationReducer(next, { type: "delta", id, text: "Hel", at: 1200 });
    next = conversationReducer(next, { type: "delta", id, text: "lo", at: 1300 });
    const turn = next[1];
    expect(turn?.role).toBe("assistant");
    if (turn?.role !== "assistant") return;
    expect(turn.content).toBe("Hello");
    expect(turn.status).toBe("streaming");
    expect(turn.firstTokenAt).toBe(1200);
    expect(turn.meta?.requestId).toBe("hdr-id");
  });

  it("falls back to the header request id when done carries none", () => {
    const { turns, id } = started();
    let next = conversationReducer(turns, { type: "meta", id, meta });
    next = conversationReducer(next, {
      type: "done",
      id,
      at: 2000,
      finishReason: "stop",
      usage: null,
      sources: [],
      requestId: null,
      modelVersion: null,
    });
    const turn = next[1];
    if (turn?.role !== "assistant") throw new Error("expected assistant turn");
    expect(turn.status).toBe("done");
    expect(turn.requestId).toBe("hdr-id");
  });

  it("marks a stopped turn with partial content as cancelled", () => {
    const { turns, id } = started();
    let next = conversationReducer(turns, { type: "delta", id, text: "partial", at: 1100 });
    next = conversationReducer(next, { type: "stopped", id, at: 1500 });
    const turn = next[1];
    if (turn?.role !== "assistant") throw new Error("expected assistant turn");
    expect(turn.status).toBe("stopped");
    expect(turn.finishReason).toBe("cancelled");
  });

  it("resets to pending (dropping stale meta) before a retry", () => {
    const { turns, id } = started();
    let next = conversationReducer(turns, { type: "meta", id, meta });
    next = conversationReducer(next, { type: "pending", id });
    const turn = next[1];
    if (turn?.role !== "assistant") throw new Error("expected assistant turn");
    expect(turn.status).toBe("pending");
    expect(turn.meta).toBeNull();
  });

  it("clears everything", () => {
    const { turns } = started();
    expect(conversationReducer(turns, { type: "clear" })).toEqual([]);
  });
});

describe("historyFor", () => {
  it("keeps only completed question/answer pairs, in order", () => {
    const a1 = { ...newAssistantTurn("chat", true, 0), id: "a1", content: "A1", status: "done" as const };
    const a2 = { ...newAssistantTurn("chat", true, 0), id: "a2", content: "", status: "error" as const };
    const a3 = { ...newAssistantTurn("chat", true, 0), id: "a3", content: "A3 part", status: "stopped" as const };
    const a4 = { ...newAssistantTurn("chat", true, 0), id: "a4", content: "", status: "pending" as const };
    const turns: Turn[] = [
      { id: "u1", role: "user", content: "Q1" },
      a1,
      { id: "u2", role: "user", content: "Q2" },
      a2,
      { id: "u3", role: "user", content: "Q3" },
      a3,
      { id: "u4", role: "user", content: "Q4" },
      a4,
    ];
    expect(historyFor(turns)).toEqual([
      { role: "user", content: "Q1" },
      { role: "assistant", content: "A1" },
      { role: "user", content: "Q3" },
      { role: "assistant", content: "A3 part" },
    ]);
  });
});
