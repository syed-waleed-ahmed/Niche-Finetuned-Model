/**
 * Minimal Server-Sent Events parser.
 *
 * The service emits only `event:` and `data:` lines terminated by a blank line
 * (see `_sse()` in `src/fastapi_assistant/api.py`). The parser still follows the
 * spec where it matters: multi-line `data` joined with "\n", comment lines
 * ignored, CRLF tolerated, and events split correctly across chunk boundaries.
 */

export interface SseEvent {
  event: string | null;
  data: string;
}

export class SseParser {
  private buffer = "";
  private eventName: string | null = null;
  private dataLines: string[] = [];

  /** Feed a decoded text chunk; returns every event completed by it. */
  push(chunk: string): SseEvent[] {
    this.buffer += chunk;
    const events: SseEvent[] = [];
    let newline: number;
    while ((newline = this.buffer.indexOf("\n")) !== -1) {
      let line = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      if (line.endsWith("\r")) line = line.slice(0, -1);
      const event = this.handleLine(line);
      if (event) events.push(event);
    }
    return events;
  }

  /** Dispatch a trailing event that was not followed by a blank line. */
  flush(): SseEvent[] {
    const events: SseEvent[] = [];
    if (this.buffer.length) {
      const event = this.handleLine(this.buffer.replace(/\r$/, ""));
      this.buffer = "";
      if (event) events.push(event);
    }
    const last = this.dispatch();
    if (last) events.push(last);
    return events;
  }

  private handleLine(line: string): SseEvent | null {
    if (line === "") return this.dispatch();
    if (line.startsWith(":")) return null;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.eventName = value;
    else if (field === "data") this.dataLines.push(value);
    // `id` and `retry` are not used by this service.
    return null;
  }

  private dispatch(): SseEvent | null {
    if (this.dataLines.length === 0) {
      this.eventName = null;
      return null;
    }
    const event = { event: this.eventName, data: this.dataLines.join("\n") };
    this.eventName = null;
    this.dataLines = [];
    return event;
  }
}

/** Iterate the SSE events of a fetch response body. */
export async function* readSse(
  body: ReadableStream<Uint8Array>,
  signal?: AbortSignal,
): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();
  const abort = () => {
    reader.cancel().catch(() => undefined);
  };
  signal?.addEventListener("abort", abort, { once: true });
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const event of parser.push(decoder.decode(value, { stream: true }))) yield event;
    }
    for (const event of parser.push(decoder.decode())) yield event;
    for (const event of parser.flush()) yield event;
  } finally {
    signal?.removeEventListener("abort", abort);
    reader.releaseLock();
  }
}
