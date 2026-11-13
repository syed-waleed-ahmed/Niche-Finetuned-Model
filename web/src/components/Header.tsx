import type { HealthState } from "../hooks/useHealth";

interface Props {
  health: HealthState;
  onOpenSettings: () => void;
  docsUrl: string;
}

export function Header({ health, onOpenSettings, docsUrl }: Props) {
  return (
    <header className="header">
      <div className="brand">
        <svg className="brand__logo" viewBox="0 0 32 32" aria-hidden="true">
          <rect width="32" height="32" rx="7" fill="var(--accent)" />
          <path d="M17.5 5 9 18h6l-1.5 9L23 14h-6z" fill="var(--accent-fg)" />
        </svg>
        <div>
          <div className="brand__title">FastAPI Assistant</div>
          <div className="brand__subtitle">A small language model that answers FastAPI questions</div>
        </div>
      </div>
      <div className="header__spacer" />
      <HealthBadge state={health} />
      <nav className="header__links" aria-label="Links">
        <a className="btn btn--ghost btn--sm" href={docsUrl} target="_blank" rel="noreferrer noopener">
          API docs
        </a>
        <a
          className="btn btn--ghost btn--sm"
          href="https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model"
          target="_blank"
          rel="noreferrer noopener"
        >
          GitHub
        </a>
        <button type="button" className="btn btn--sm" onClick={onOpenSettings}>
          Settings
        </button>
      </nav>
    </header>
  );
}

export function HealthBadge({ state }: { state: HealthState }) {
  if (!state.health && !state.error) {
    return (
      <span className="badge" title="Checking /health">
        <span className="badge__dot" /> checking
      </span>
    );
  }
  if (state.error && !state.health) {
    return (
      <span className="badge badge--danger" title={state.error}>
        <span className="badge__dot" /> unreachable
      </span>
    );
  }
  const h = state.health;
  if (!h) return null;
  const detail = `${h.backend || "backend"} - ${h.model_version || "model"}${h.load_state ? ` - ${h.load_state}` : ""}`;
  if (!h.model_loaded) {
    return (
      <span className="badge badge--warn" title={detail}>
        <span className="badge__dot" /> loading weights
      </span>
    );
  }
  return (
    <span className={`badge ${state.error ? "badge--warn" : "badge--ok"}`} title={state.error ?? detail}>
      <span className="badge__dot" /> {h.model_version || "ready"}
      {h.uses_adapter ? "" : " (base model)"}
    </span>
  );
}
