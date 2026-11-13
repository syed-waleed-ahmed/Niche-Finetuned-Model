import { Suspense, lazy, useState } from "react";

import type { AssistantTurn, Turn } from "../state/conversation";

// The Markdown renderer (react-markdown + highlight.js) is the largest part of the
// bundle and is only needed once an answer arrives, so it loads on demand. Until
// then the raw text is shown, which keeps streaming visible from the first token.
const Markdown = lazy(() => import("./Markdown").then((m) => ({ default: m.Markdown })));

function Answer({ text }: { text: string }) {
  return (
    <Suspense fallback={<div className="md" style={{ whiteSpace: "pre-wrap" }}>{text}</div>}>
      <Markdown text={text} />
    </Suspense>
  );
}

interface Props {
  turn: Turn;
  onRate: (turnId: string, rating: 1 | -1, comment?: string) => Promise<string | null>;
}

export function Message({ turn, onRate }: Props) {
  if (turn.role === "user") {
    return (
      <article className="turn turn--user" aria-label="Your question">
        <div className="turn__label">You</div>
        <div className="bubble">{turn.content}</div>
      </article>
    );
  }
  return <AssistantMessage turn={turn} onRate={onRate} />;
}

function AssistantMessage({ turn, onRate }: { turn: AssistantTurn; onRate: Props["onRate"] }) {
  const streaming = turn.status === "streaming" || turn.status === "pending";
  return (
    <article className="turn turn--assistant" aria-label="Assistant answer" aria-busy={streaming}>
      <div className="turn__label">Assistant</div>
      <div className={`bubble ${turn.status === "error" && !turn.content ? "bubble--error" : ""}`}>
        {turn.content ? <Answer text={turn.content} /> : null}
        {turn.status === "pending" ? (
          <span className="thinking">Waiting for the model</span>
        ) : null}
        {turn.status === "streaming" ? <span className="caret" aria-hidden="true" /> : null}
        {turn.status === "error" ? (
          <div className="banner banner--danger" role="alert" style={{ marginTop: turn.content ? 8 : 0 }}>
            <span className="banner__text">{turn.error}</span>
          </div>
        ) : null}
        {turn.status === "stopped" ? (
          <div className="muted" style={{ marginTop: 6 }}>
            Stopped. The server cancels generation at the next token and frees the slot.
          </div>
        ) : null}
      </div>
      {turn.status === "done" || turn.status === "stopped" ? <MetaRow turn={turn} /> : null}
      {turn.sources.length ? <Sources sources={turn.sources} /> : null}
      {turn.status === "done" && turn.requestId ? <Feedback turn={turn} onRate={onRate} /> : null}
    </article>
  );
}

export function formatDuration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

function MetaRow({ turn }: { turn: AssistantTurn }) {
  const total = turn.finishedAt ? turn.finishedAt - turn.startedAt : null;
  const firstToken = turn.firstTokenAt ? turn.firstTokenAt - turn.startedAt : null;
  const completion = turn.usage?.completion_tokens ?? null;
  const decodeMs =
    turn.finishedAt && turn.firstTokenAt ? turn.finishedAt - turn.firstTokenAt : null;
  const tps =
    completion && decodeMs && decodeMs > 0 && turn.streamed
      ? (completion / (decodeMs / 1000)).toFixed(1)
      : null;
  const cache = turn.meta?.cache;
  return (
    <div className="meta" aria-label="Answer details">
      {turn.modelVersion ? (
        <span className="meta__item">
          model <b className="mono">{turn.modelVersion}</b>
        </span>
      ) : null}
      {turn.finishReason ? (
        <span className="meta__item">
          finish <b>{turn.finishReason}</b>
        </span>
      ) : null}
      {turn.usage ? (
        <span className="meta__item">
          tokens{" "}
          <b>
            {turn.usage.prompt_tokens} in / {turn.usage.completion_tokens} out
          </b>
        </span>
      ) : null}
      {total !== null ? (
        <span className="meta__item">
          time <b>{formatDuration(total)}</b>
        </span>
      ) : null}
      {firstToken !== null && turn.streamed ? (
        <span className="meta__item">
          first token <b>{formatDuration(firstToken)}</b>
        </span>
      ) : null}
      {tps ? (
        <span className="meta__item">
          <b>{tps}</b> tok/s
        </span>
      ) : null}
      {cache ? (
        <span className={`badge ${cache === "HIT" ? "badge--info" : ""}`} title="Edge answer cache">
          cache {cache}
        </span>
      ) : null}
      <span className="meta__item">
        {turn.mode === "generate" ? "/generate" : "/v1/chat/completions"}
      </span>
      {turn.requestId ? (
        <span className="meta__item" title="X-Request-ID">
          id <span className="mono">{turn.requestId}</span>
        </span>
      ) : null}
    </div>
  );
}

function Sources({ sources }: { sources: AssistantTurn["sources"] }) {
  return (
    <div className="sources">
      <div className="sources__title">Sources from the FastAPI docs</div>
      <ol>
        {sources.map((s) => (
          <li key={`${s.url}#${s.heading}`}>
            <a href={s.url} target="_blank" rel="noreferrer noopener">
              {s.heading || s.url}
            </a>
            <span className="sources__score" title="Retrieval similarity">
              {s.score.toFixed(2)}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Feedback({ turn, onRate }: { turn: AssistantTurn; onRate: Props["onRate"] }) {
  const [comment, setComment] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [sending, setSending] = useState(false);

  const rate = async (rating: 1 | -1) => {
    setSending(true);
    const error = await onRate(turn.id, rating, comment);
    setSending(false);
    setStatus(error ? `Not recorded: ${error}` : "Thanks, recorded.");
  };

  return (
    <div className="feedback" aria-label="Rate this answer">
      <button
        type="button"
        className={`btn btn--sm ${turn.feedback === 1 ? "btn--active" : ""}`}
        onClick={() => void rate(1)}
        disabled={sending}
        aria-pressed={turn.feedback === 1}
      >
        Good answer
      </button>
      <button
        type="button"
        className={`btn btn--sm ${turn.feedback === -1 ? "btn--active" : ""}`}
        onClick={() => void rate(-1)}
        disabled={sending}
        aria-pressed={turn.feedback === -1}
      >
        Bad answer
      </button>
      <input
        className="feedback__comment"
        type="text"
        placeholder="Optional comment"
        value={comment}
        maxLength={2000}
        onChange={(e) => setComment(e.target.value)}
        aria-label="Feedback comment"
      />
      {status ? <span className="feedback__status">{status}</span> : null}
    </div>
  );
}
