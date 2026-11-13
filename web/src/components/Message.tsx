import { Suspense, lazy, useId, useState } from "react";

import type { AssistantTurn, Turn } from "../state/conversation";
import { CopyButton } from "./CopyButton";
import { CommentIcon, Logo, ThumbDownIcon, ThumbUpIcon } from "./Icons";

// The Markdown renderer (react-markdown + highlight.js) is the largest part of the
// bundle and is only needed once an answer arrives, so it loads on demand. Until
// then the raw text is shown, which keeps streaming visible from the first token.
const Markdown = lazy(() => import("./Markdown").then((m) => ({ default: m.Markdown })));

function Answer({ text }: { text: string }) {
  return (
    <Suspense fallback={<div className="md md--plain">{text}</div>}>
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
        <div className="bubble">{turn.content}</div>
      </article>
    );
  }
  return <AssistantMessage turn={turn} onRate={onRate} />;
}

function AssistantMessage({ turn, onRate }: { turn: AssistantTurn; onRate: Props["onRate"] }) {
  const streaming = turn.status === "streaming" || turn.status === "pending";
  const settled = turn.status === "done" || turn.status === "stopped";
  const errorOnly = turn.status === "error" && !turn.content;
  return (
    <article className="turn turn--assistant" aria-label="Assistant answer" aria-busy={streaming}>
      <div className="turn__avatar">
        <Logo size={26} />
      </div>
      <div className="turn__body">
        <div className="turn__name" aria-hidden="true">
          Assistant
        </div>
        <div className={`answer ${errorOnly ? "answer--error" : ""}`}>
          {turn.content ? <Answer text={turn.content} /> : null}
          {turn.status === "pending" ? (
            <span className="thinking">
              <span className="thinking__dots" aria-hidden="true">
                <span />
                <span />
                <span />
              </span>
              Waiting for the model
            </span>
          ) : null}
          {turn.status === "streaming" ? <span className="caret" aria-hidden="true" /> : null}
          {turn.status === "error" ? (
            <div className={`banner banner--danger ${turn.content ? "banner--after" : ""}`} role="alert">
              <span className="banner__text">{turn.error}</span>
            </div>
          ) : null}
          {turn.status === "stopped" ? (
            <p className="answer__note">
              Stopped. The server cancels generation at the next token and frees the slot.
            </p>
          ) : null}
          {turn.sources.length ? <Sources sources={turn.sources} /> : null}
        </div>
        {settled ? (
          <div className="answer__footer">
            <div className="answer__actions">
              {turn.content ? (
                <CopyButton text={turn.content} ariaLabel="Copy answer" className="action" />
              ) : null}
              {turn.status === "done" && turn.requestId ? <Feedback turn={turn} onRate={onRate} /> : null}
            </div>
            <MetaRow turn={turn} />
          </div>
        ) : null}
      </div>
    </article>
  );
}

export function formatDuration(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`;
}

/**
 * Per-answer numbers in one compact line (model, time, tokens, speed, cache); the
 * rest (finish reason, first-token latency, endpoint, request id) behind "Details".
 */
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
  const [open, setOpen] = useState(false);
  const detailsId = useId();
  return (
    <>
      <div className="meta">
        <ul className="meta__list" aria-label="Answer details">
          {turn.modelVersion ? (
            <li className="meta__item" title="Model version">
              <span className="mono">{turn.modelVersion}</span>
            </li>
          ) : null}
          {total !== null ? (
            <li className="meta__item" title="Total time">
              <span className="meta__label">Time</span>
              <span className="meta__value">{formatDuration(total)}</span>
            </li>
          ) : null}
          {turn.usage ? (
            <li className="meta__item" title="Prompt and completion tokens">
              <span className="meta__label">Tokens</span>
              <span className="meta__value">
                {turn.usage.prompt_tokens} in / {turn.usage.completion_tokens} out
              </span>
            </li>
          ) : null}
          {tps ? (
            <li className="meta__item" title="Decoding speed">
              <span className="meta__value">{tps}</span>
              <span className="meta__label">tok/s</span>
            </li>
          ) : null}
          {cache ? (
            <li className="meta__item">
              <span className={`pill ${cache === "HIT" ? "pill--info" : ""}`} title="Edge answer cache">
                cache {cache}
              </span>
            </li>
          ) : null}
        </ul>
        <button
          type="button"
          className="meta__toggle"
          aria-expanded={open}
          aria-controls={detailsId}
          onClick={() => setOpen((v) => !v)}
        >
          Details
        </button>
      </div>
      <dl id={detailsId} className="kv kv--compact meta__details" hidden={!open}>
        <dt>Finish reason</dt>
        <dd>{turn.finishReason ?? "-"}</dd>
        {firstToken !== null && turn.streamed ? (
          <>
            <dt>First token</dt>
            <dd>{formatDuration(firstToken)}</dd>
          </>
        ) : null}
        <dt>Endpoint</dt>
        <dd>{turn.mode === "generate" ? "/generate" : "/v1/chat/completions"}</dd>
        <dt>Streamed</dt>
        <dd>{turn.streamed ? "yes" : "no"}</dd>
        {turn.requestId ? (
          <>
            <dt>Request id</dt>
            <dd title="X-Request-ID">{turn.requestId}</dd>
          </>
        ) : null}
      </dl>
    </>
  );
}

function hostAndPath(url: string): string {
  try {
    const u = new URL(url);
    return `${u.host}${u.pathname.replace(/\/$/, "")}`;
  } catch {
    return url;
  }
}

function Sources({ sources }: { sources: AssistantTurn["sources"] }) {
  const titleId = useId();
  return (
    <section className="sources" aria-labelledby={titleId}>
      <h3 className="sources__title" id={titleId}>
        Sources from the FastAPI docs
      </h3>
      <ol className="sources__list">
        {sources.map((s, i) => (
          <li key={`${s.url}#${s.heading}`} className="source">
            <span className="source__index" aria-hidden="true">
              {i + 1}
            </span>
            <a className="source__link" href={s.url} target="_blank" rel="noreferrer noopener">
              <span className="source__heading">{s.heading || s.url}</span>
              <span className="source__url">{hostAndPath(s.url)}</span>
              <span className="visually-hidden"> (opens in a new tab)</span>
            </a>
            <span className="source__score" title="Retrieval similarity">
              <span className="visually-hidden">similarity </span>
              {s.score.toFixed(2)}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Feedback({ turn, onRate }: { turn: AssistantTurn; onRate: Props["onRate"] }) {
  const [comment, setComment] = useState("");
  const [showComment, setShowComment] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [sending, setSending] = useState(false);
  const commentId = useId();

  const rate = async (rating: 1 | -1) => {
    setSending(true);
    const error = await onRate(turn.id, rating, comment);
    setSending(false);
    setStatus(error ? `Not recorded: ${error}` : "Thanks, recorded.");
  };

  return (
    <div className="feedback" role="group" aria-label="Rate this answer">
      <div className="feedback__row">
        <button
          type="button"
          className={`btn btn--icon btn--quiet action ${turn.feedback === 1 ? "btn--active" : ""}`}
          onClick={() => void rate(1)}
          disabled={sending}
          aria-pressed={turn.feedback === 1}
          aria-label="Good answer"
          title="Good answer"
        >
          <ThumbUpIcon size={15} />
        </button>
        <button
          type="button"
          className={`btn btn--icon btn--quiet action ${turn.feedback === -1 ? "btn--active" : ""}`}
          onClick={() => void rate(-1)}
          disabled={sending}
          aria-pressed={turn.feedback === -1}
          aria-label="Bad answer"
          title="Bad answer"
        >
          <ThumbDownIcon size={15} />
        </button>
        <button
          type="button"
          className={`btn btn--sm btn--quiet action ${showComment ? "btn--active" : ""}`}
          onClick={() => setShowComment((v) => !v)}
          aria-expanded={showComment}
          aria-controls={commentId}
        >
          <CommentIcon size={14} />
          Comment
        </button>
        <span className="feedback__status" aria-live="polite">
          {status ?? ""}
        </span>
      </div>
      <div id={commentId} className="feedback__comment-row" hidden={!showComment}>
        <input
          className="input feedback__comment"
          type="text"
          placeholder="Optional comment, sent with your rating"
          value={comment}
          maxLength={2000}
          onChange={(e) => setComment(e.target.value)}
          aria-label="Feedback comment"
        />
      </div>
    </div>
  );
}
