import type { HealthState } from "../hooks/useHealth";
import { DOCS_URL, GITHUB_URL } from "../links";
import type { AccessMode } from "../api/types";
import { KeyRequiredCard, PublicNotice } from "./AccessNotice";
import { ExternalLink } from "./ExternalLink";
import { ArrowRightIcon, Logo } from "./Icons";

export const EXAMPLE_QUESTIONS = [
  "How do I return a 201 status code from a POST endpoint?",
  "What does Depends do and when should I use it?",
  "How do I validate a request body with a Pydantic model?",
  "How do I add a background task to a route?",
  "How do I serve static files?",
  "How do I write a test for an async endpoint with TestClient?",
];

interface Props {
  health: HealthState;
  access: AccessMode | null;
  needsKey: boolean;
  apiDocsUrl: string;
  onPick: (question: string) => void;
  onAddKey: () => void;
}

/** First-run screen: what this is, whether it is ready, how to get access, what to ask. */
export function Welcome({ health, access, needsKey, apiDocsUrl, onPick, onAddKey }: Props) {
  return (
    <section className="welcome" aria-labelledby="welcome-title">
      <Logo className="welcome__logo" size={44} />
      <h2 className="welcome__title" id="welcome-title">
        Ask anything about FastAPI
      </h2>
      <p className="welcome__lead">
        A small language model specialised in FastAPI. It streams its answers and, with
        documentation retrieval on, cites the passages they are grounded on.
      </p>
      <ServiceStatus state={health} />

      {access === "public" ? <PublicNotice /> : null}
      {needsKey ? <KeyRequiredCard onAddKey={onAddKey} /> : null}

      <div className="welcome__examples">
        <h3 className="eyebrow" id="examples-title">
          Try an example
        </h3>
        <ul className="chips" aria-labelledby="examples-title">
          {EXAMPLE_QUESTIONS.map((q) => (
            <li key={q}>
              <button type="button" className="chip" onClick={() => onPick(q)}>
                <span className="chip__text">{q}</span>
                <ArrowRightIcon size={14} />
              </button>
            </li>
          ))}
        </ul>
      </div>

      <nav className="welcome__links" aria-label="Learn more">
        <ExternalLink href={DOCS_URL} icon>
          Docs
        </ExternalLink>
        <ExternalLink href={GITHUB_URL} icon>
          GitHub
        </ExternalLink>
        <ExternalLink href={apiDocsUrl} icon>
          API docs
        </ExternalLink>
      </nav>
      <p className="welcome__caveat">
        Small models make mistakes. Check answers against the FastAPI documentation.
      </p>
    </section>
  );
}

type Tone = "neutral" | "ok" | "warn" | "danger";

export function describeHealth(state: HealthState): { tone: Tone; parts: string[] } {
  const h = state.health;
  if (!h) {
    return state.error
      ? { tone: "danger", parts: ["Service unreachable", "answers cannot be sent until it responds"] }
      : { tone: "neutral", parts: ["Checking the service"] };
  }
  const parts: string[] = [];
  if (h.model_version) parts.push(`Model ${h.model_version}`);
  parts.push(h.uses_adapter ? "Fine-tuned" : "Base model");
  if (!h.model_loaded) return { tone: "warn", parts: [...parts, "Loading"] };
  if (state.error) return { tone: "warn", parts: [...parts, "Ready", "last check failed"] };
  return { tone: "ok", parts: [...parts, "Ready"] };
}

/** One-line service summary from `/health`. */
export function ServiceStatus({ state }: { state: HealthState }) {
  const { tone, parts } = describeHealth(state);
  return (
    <p className={`service service--${tone}`}>
      <span className="service__dot" aria-hidden="true" />
      <span className="visually-hidden">Service status: </span>
      {parts.map((part, i) => (
        <span key={part} className="service__part">
          {i > 0 ? (
            <>
              <span className="service__sep" aria-hidden="true" />
              <span className="visually-hidden">, </span>
            </>
          ) : null}
          {part}
        </span>
      ))}
    </p>
  );
}
