import { describe, expect, it } from "vitest";

import {
  CONVERSATION_STORAGE_KEY,
  MAX_SAVED_PAIRS,
  conversationReducer,
  historyFor,
  loadTurns,
  newAssistantTurn,
  newId,
  saveTurns,
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

describe("conversation persistence (#41)", () => {
  function answered(question: string, answer: string, status: "done" | "stopped" | "error" = "done"): Turn[] {
    const assistant = { ...newAssistantTurn("chat", true, 1000), content: answer, status };
    return [{ id: newId(), role: "user", content: question }, assistant];
  }

  function memoryStorage(): Storage {
    const data = new Map<string, string>();
    return {
      get length() {
        return data.size;
      },
      clear: () => data.clear(),
      getItem: (k) => data.get(k) ?? null,
      key: (i) => [...data.keys()][i] ?? null,
      removeItem: (k) => void data.delete(k),
      setItem: (k, v) => void data.set(k, String(v)),
    };
  }

  it("saves finished pairs and loads them back", () => {
    const storage = memoryStorage();
    const turns = [...answered("q1", "a1"), ...answered("q2", "partial", "stopped")];
    saveTurns(turns, storage);
    expect(loadTurns(storage)).toEqual(turns);
  });

  it("never saves an answer that is still being written", () => {
    const storage = memoryStorage();
    const { turns, id } = started();
    const streaming = conversationReducer(turns, { type: "delta", id, text: "half", at: 1100 });
    saveTurns([...answered("q1", "a1"), ...streaming], storage);
    const loaded = loadTurns(storage);
    expect(loaded.map((t) => t.content)).toEqual(["q1", "a1"]);
  });

  it("keeps only the most recent pairs", () => {
    const storage = memoryStorage();
    const turns = Array.from({ length: MAX_SAVED_PAIRS + 5 }, (_, i) => answered(`q${i}`, `a${i}`)).flat();
    saveTurns(turns, storage);
    const loaded = loadTurns(storage);
    expect(loaded).toHaveLength(2 * MAX_SAVED_PAIRS);
    expect(loaded[0]?.content).toBe("q5");
  });

  it("removes the saved conversation when it is cleared", () => {
    const storage = memoryStorage();
    saveTurns(answered("q1", "a1"), storage);
    saveTurns([], storage);
    expect(storage.getItem(CONVERSATION_STORAGE_KEY)).toBeNull();
  });

  it("ignores a corrupt or unexpected stored value", () => {
    const storage = memoryStorage();
    storage.setItem(CONVERSATION_STORAGE_KEY, "{not json");
    expect(loadTurns(storage)).toEqual([]);
    storage.setItem(CONVERSATION_STORAGE_KEY, JSON.stringify([{ role: "assistant", content: 1 }]));
    expect(loadTurns(storage)).toEqual([]);
    storage.setItem(CONVERSATION_STORAGE_KEY, JSON.stringify({ turns: [] }));
    expect(loadTurns(storage)).toEqual([]);
  });

  it("works without storage, and survives storage that throws", () => {
    expect(loadTurns(null)).toEqual([]);
    expect(() => saveTurns(answered("q", "a"), null)).not.toThrow();
    const broken = memoryStorage();
    broken.setItem = () => {
      throw new Error("QuotaExceededError");
    };
    broken.getItem = () => {
      throw new Error("SecurityError");
    };
    expect(() => saveTurns(answered("q", "a"), broken)).not.toThrow();
    expect(loadTurns(broken)).toEqual([]);
  });
});
