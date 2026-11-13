import { useEffect, useRef, type ReactNode } from "react";

import type { Turn } from "../state/conversation";
import { Message } from "./Message";

interface Props {
  turns: Turn[];
  onRate: (turnId: string, rating: 1 | -1, comment?: string) => Promise<string | null>;
  /** Rendered while the conversation is empty. */
  welcome: ReactNode;
}

export function MessageList({ turns, onRate, welcome }: Props) {
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

  const empty = turns.length === 0;
  // No aria-live here: announcing every streamed token is noise. App owns a polite
  // live region that reports when an answer starts and finishes instead.
  return (
    <div className="messages" ref={scroller} onScroll={onScroll}>
      <div className={`messages__inner ${empty ? "messages__inner--empty" : ""}`}>
        {empty ? welcome : turns.map((turn) => <Message key={turn.id} turn={turn} onRate={onRate} />)}
      </div>
    </div>
  );
}
