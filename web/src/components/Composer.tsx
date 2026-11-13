import { useEffect, useRef, type KeyboardEvent } from "react";

import { LIMITS } from "../state/settings";

interface Props {
  draft: string;
  onDraft: (value: string) => void;
  onSend: (question: string) => void;
  onStop: () => void;
  busy: boolean;
  disabled: boolean;
}

export function Composer({ draft, onDraft, onSend, onStop, busy, disabled }: Props) {
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
            autoFocus
          />
          {busy ? (
            <button type="button" className="btn btn--danger" onClick={onStop}>
              Stop
            </button>
          ) : (
            <button type="submit" className="btn btn--primary" disabled={!canSend}>
              Send
            </button>
          )}
        </div>
        <div className="composer__hint">
          <span>Enter to send, Shift+Enter for a new line.</span>
          <span className={over ? "over" : ""}>
            {draft.length} / {LIMITS.questionChars}
          </span>
        </div>
      </div>
    </form>
  );
}
