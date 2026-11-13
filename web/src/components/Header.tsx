import type { Ref } from "react";

import type { HealthState } from "../hooks/useHealth";
import { DOCS_URL, GITHUB_URL } from "../links";
import type { ThemePreference } from "../state/theme";
import { ExternalLink } from "./ExternalLink";
import { Logo, SlidersIcon } from "./Icons";
import { ThemeToggle } from "./ThemeToggle";

interface Props {
  health: HealthState;
  apiDocsUrl: string;
  theme: ThemePreference;
  onTheme: (theme: ThemePreference) => void;
  onOpenSettings: () => void;
  /** Set while the side panel is an off-canvas drawer: the button then reports its state. */
  drawer: { open: boolean; controls: string } | null;
  settingsRef: Ref<HTMLButtonElement>;
  /** True while a modal drawer is open: the header is then out of reach. */
  inert: boolean;
}

export function Header({
  health,
  apiDocsUrl,
  theme,
  onTheme,
  onOpenSettings,
  drawer,
  settingsRef,
  inert,
}: Props) {
  return (
    <header className="header" inert={inert}>
      <div className="brand">
        <Logo className="brand__logo" />
        <div className="brand__text">
          <h1 className="brand__title">FastAPI Assistant</h1>
          <p className="brand__subtitle">A small language model for FastAPI questions</p>
        </div>
      </div>
      <HealthBadge state={health} />
      <div className="header__spacer" />
      <nav className="header__links" aria-label="Project links">
        <ExternalLink className="header__link" href={DOCS_URL}>
          Docs
        </ExternalLink>
        <ExternalLink className="header__link" href={GITHUB_URL}>
          GitHub
        </ExternalLink>
        <ExternalLink className="header__link" href={apiDocsUrl}>
          API docs
        </ExternalLink>
      </nav>
      <div className="header__actions">
        <ThemeToggle theme={theme} onTheme={onTheme} />
        <button
          ref={settingsRef}
          type="button"
          className="btn header__settings"
          onClick={onOpenSettings}
          {...(drawer
            ? { "aria-expanded": drawer.open, "aria-controls": drawer.controls, "aria-haspopup": "dialog" as const }
            : {})}
        >
          <SlidersIcon size={16} />
          <span>Settings</span>
        </button>
      </div>
    </header>
  );
}

export function HealthBadge({ state }: { state: HealthState }) {
  if (!state.health && !state.error) {
    return (
      <span className="badge header__badge" title="Checking /health">
        <span className="badge__dot" /> checking
      </span>
    );
  }
  if (state.error && !state.health) {
    return (
      <span className="badge badge--danger header__badge" title={state.error}>
        <span className="badge__dot" /> unreachable
      </span>
    );
  }
  const h = state.health;
  if (!h) return null;
  const detail = `${h.backend || "backend"} - ${h.model_version || "model"}${h.load_state ? ` - ${h.load_state}` : ""}`;
  if (!h.model_loaded) {
    return (
      <span className="badge badge--warn header__badge" title={detail}>
        <span className="badge__dot" /> loading weights
      </span>
    );
  }
  return (
    <span
      className={`badge ${state.error ? "badge--warn" : "badge--ok"} header__badge`}
      title={state.error ?? detail}
    >
      <span className="badge__dot" /> {h.model_version || "ready"}
      {h.uses_adapter ? "" : " (base model)"}
    </span>
  );
}
