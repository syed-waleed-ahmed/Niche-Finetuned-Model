import { useEffect, useRef, type KeyboardEvent } from "react";

import { LIMITS } from "../state/settings";
import { ArrowUpIcon, StopIcon } from "./Icons";

interface Props {
  draft: string;
  onDraft: (value: string) => void;
  onSend: (question: string) => void;
  onStop: () => void;
  busy: boolean;
  disabled: boolean;
  /** Replaces the keyboard hint, for example when a key must be added first. */
  hint?: string | null;
}

const KEYBOARD_HINT = "Enter to send, Shift+Enter for a new line.";

export function Composer({ draft, onDraft, onSend, onStop, busy, disabled, hint = null }: Props) {
  const textarea = useRef<HTMLTextAreaElement>(null);
  const over = draft.length > LIMITS.questionChars;
  const canSend = !busy && !disabled && draft.trim().length > 0 && !over;

  useEffect(() => {
    const el = textarea.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(200, el.scrollHeight)}px`;
  }, [draft]);

  const submit = () => {
    if (canSend) onSend(draft);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        submit();
      }}
    >
      <div className="composer__inner">
        <div className="composer__box">
          <label htmlFor="question" className="visually-hidden">
            Your FastAPI question
          </label>
          <textarea
            id="question"
            ref={textarea}
            rows={1}
            placeholder="Ask a FastAPI question"
            value={draft}
            onChange={(e) => onDraft(e.target.value)}
            onKeyDown={onKeyDown}
            disabled={disabled}
            aria-describedby="composer-hint"
            autoFocus
          />
          {busy ? (
            <button type="button" className="btn btn--stop composer__button" onClick={onStop}>
              <StopIcon size={14} />
              Stop
            </button>
          ) : (
            <button type="submit" className="btn btn--primary composer__button" disabled={!canSend}>
              <ArrowUpIcon size={16} />
              Send
            </button>
          )}
        </div>
        <div className="composer__hint">
          <span id="composer-hint" className={hint ? "composer__notice" : "composer__keys"}>
            {hint ?? KEYBOARD_HINT}
          </span>
          <span className={`composer__count ${over ? "over" : ""}`}>
            {draft.length} / {LIMITS.questionChars}
          </span>
        </div>
      </div>
    </form>
  );
}
