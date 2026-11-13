import { useEffect, useRef } from "react";

import type { Turn } from "../state/conversation";
import { Message } from "./Message";

export const EXAMPLE_QUESTIONS = [
  "How do I return a 201 status code from a POST endpoint?",
  "What does Depends do and when should I use it?",
  "How do I validate a request body with a Pydantic model?",
  "How do I add a background task to a route?",
  "How do I serve static files?",
  "How do I write a test for an async endpoint with TestClient?",
];

interface Props {
  turns: Turn[];
  onRate: (turnId: string, rating: 1 | -1, comment?: string) => Promise<string | null>;
  onPick: (question: string) => void;
}

export function MessageList({ turns, onRate, onPick }: Props) {
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);

  // Follow the stream unless the reader scrolled up to read something. Content also
  // grows without a state change (lazy Markdown, code blocks, the retry banner
  // resizing the viewport), so the pinned scroll is re-applied on every resize.
  useEffect(() => {
    const el = scroller.current;
    if (el && pinned.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  useEffect(() => {
    const el = scroller.current;
    const inner = el?.firstElementChild;
    if (!el || !inner || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      if (pinned.current) el.scrollTop = el.scrollHeight;
    });
    observer.observe(inner);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const onScroll = () => {
    const el = scroller.current;
    if (!el) return;
    pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 48;
  };

  return (
    <div className="messages" ref={scroller} onScroll={onScroll} role="log" aria-live="polite">
      <div className="messages__inner">
        {turns.length === 0 ? (
          <div className="empty">
            <h2>Ask anything about FastAPI</h2>
            <p>
              Answers come from a small (1.1B-parameter) language model, streamed token by token,
              with the documentation passages they were grounded on. The badge above shows
              whether the fine-tuned or the base model is serving. Check answers against the
              cited sources.
            </p>
            <div className="chips" aria-label="Example questions">
              {EXAMPLE_QUESTIONS.map((q) => (
                <button key={q} type="button" className="chip" onClick={() => onPick(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          turns.map((turn) => <Message key={turn.id} turn={turn} onRate={onRate} />)
        )}
      </div>
    </div>
  );
}
