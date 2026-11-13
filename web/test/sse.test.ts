import { describe, expect, it } from "vitest";

import { SseParser, readSse } from "../src/api/sse";

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect<T>(iter: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const item of iter) out.push(item);
  return out;
}

describe("SseParser", () => {
  it("parses the service's event/data pairs", () => {
    const parser = new SseParser();
    const events = parser.push('event: token\ndata: {"delta": "Hi"}\n\nevent: done\ndata: {}\n\n');
    expect(events).toEqual([
      { event: "token", data: '{"delta": "Hi"}' },
      { event: "done", data: "{}" },
    ]);
  });

  it("handles events split across arbitrary chunk boundaries", () => {
    const parser = new SseParser();
    const text = 'event: token\ndata: {"delta": "a\\nb"}\n\n';
    const events = [];
    for (const char of text) events.push(...parser.push(char));
    expect(events).toEqual([{ event: "token", data: '{"delta": "a\\nb"}' }]);
  });

  it("joins multi-line data, ignores comments and tolerates CRLF", () => {
    const parser = new SseParser();
    const events = parser.push(": keep-alive\r\ndata: first\r\ndata: second\r\n\r\n");
    expect(events).toEqual([{ event: null, data: "first\nsecond" }]);
  });

  it("emits data-only events (OpenAI style) and flushes a trailing event", () => {
    const parser = new SseParser();
    expect(parser.push("data: {}\n\ndata: [DONE]")).toEqual([{ event: null, data: "{}" }]);
    expect(parser.flush()).toEqual([{ event: null, data: "[DONE]" }]);
    expect(parser.flush()).toEqual([]);
  });

  it("does not carry an event name across a blank line without data", () => {
    const parser = new SseParser();
    expect(parser.push("event: token\n\ndata: x\n\n")).toEqual([{ event: null, data: "x" }]);
  });
});

describe("readSse", () => {
  it("decodes multi-byte characters split across chunks", async () => {
    const encoded = new TextEncoder().encode('data: {"delta": "é"}\n\n');
    const split = 15; // inside the two-byte "é"
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoded.slice(0, split));
        controller.enqueue(encoded.slice(split));
        controller.close();
      },
    });
    const events = await collect(readSse(stream));
    expect(events).toEqual([{ event: null, data: '{"delta": "é"}' }]);
  });

  it("yields events in order from several chunks", async () => {
    const events = await collect(readSse(streamOf(["event: a\nda", "ta: 1\n\nevent: b\ndata: 2\n\n"])));
    expect(events.map((e) => e.event)).toEqual(["a", "b"]);
  });
});
