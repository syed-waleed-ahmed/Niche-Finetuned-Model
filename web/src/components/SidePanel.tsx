import { useId, useState, type KeyboardEvent, type Ref } from "react";

import type { ResponseMeta } from "../api/client";
import {
  curlSnippet,
  javascriptSnippet,
  pythonOpenAISnippet,
  pythonRequestsSnippet,
  type SnippetInput,
} from "../api/snippets";
import type { AccessMode } from "../api/types";
import type { HealthHook } from "../hooks/useHealth";
import { DOCS_URL, GITHUB_URL } from "../links";
import { DEFAULT_SETTINGS, LIMITS, maskKey, type Settings } from "../state/settings";
import { THEMES, THEME_LABELS, type ThemePreference } from "../state/theme";
import { CopyButton } from "./CopyButton";
import { ExternalLink } from "./ExternalLink";
import { CloseIcon } from "./Icons";

export type Tab = "settings" | "status" | "code";

export const SIDE_PANEL_ID = "side-panel";

interface Props {
  tab: Tab;
  onTab: (tab: Tab) => void;
  settings: Settings;
  onSettings: (patch: Partial<Settings>) => void;
  health: HealthHook;
  access: AccessMode | null;
  lastMeta: ResponseMeta | null;
  draft: string;
  turnCount: number;
  onClear: () => void;
  authError: string | null;
  onDismissAuthError: () => void;
  displayBaseUrl: string;
  apiDocsUrl: string;
  theme: ThemePreference;
  onTheme: (theme: ThemePreference) => void;
  /** "drawer" below the narrow breakpoint: off-canvas, modal while open. */
  variant: "inline" | "drawer";
  open: boolean;
  onClose: () => void;
  closeRef: Ref<HTMLButtonElement>;
}

const TABS: { id: Tab; label: string }[] = [
  { id: "settings", label: "Settings" },
  { id: "status", label: "Status" },
  { id: "code", label: "Code" },
];

export function SidePanel(props: Props) {
  const { tab, onTab, variant, open, onClose, closeRef } = props;
  const titleId = useId();
  const drawer = variant === "drawer";
  const hiddenDrawer = drawer && !open;

  // Arrow keys move between tabs (WAI-ARIA tabs pattern, automatic activation).
  const onTabKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const index = TABS.findIndex((t) => t.id === tab);
    const last = TABS.length - 1;
    const target =
      e.key === "ArrowRight"
        ? (index + 1) % TABS.length
        : e.key === "ArrowLeft"
          ? (index - 1 + TABS.length) % TABS.length
          : e.key === "Home"
            ? 0
            : e.key === "End"
              ? last
              : -1;
    const next = TABS[target];
    if (!next) return;
    e.preventDefault();
    onTab(next.id);
    document.getElementById(`tab-${next.id}`)?.focus();
  };

  return (
    <div
      id={SIDE_PANEL_ID}
      className={`side ${drawer ? "side--drawer" : ""} ${drawer && open ? "side--open" : ""}`}
      role={drawer ? (open ? "dialog" : undefined) : "complementary"}
      aria-modal={drawer && open ? true : undefined}
      aria-labelledby={hiddenDrawer ? undefined : titleId}
      aria-hidden={hiddenDrawer ? true : undefined}
      inert={hiddenDrawer}
    >
      {/* Inline, the title is for assistive technology only: the tabs say the rest. */}
      {drawer ? null : (
        <h2 className="visually-hidden" id={titleId}>
          Settings and tools
        </h2>
      )}
      {drawer ? (
        <div className="side__header">
          <h2 className="side__title" id={titleId}>
            Settings and tools
          </h2>
          <button
            ref={closeRef}
            type="button"
            className="btn btn--icon btn--ghost"
            onClick={onClose}
            aria-label="Close panel"
          >
            <CloseIcon size={18} />
          </button>
        </div>
      ) : null}
      <div className="tabs" role="tablist" aria-label="Panel sections" onKeyDown={onTabKeyDown}>
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
            tabIndex={tab === t.id ? 0 : -1}
            className="tab"
            onClick={() => onTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="panel" role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
        {tab === "settings" ? <SettingsPanel {...props} /> : null}
        {tab === "status" ? <StatusPanel {...props} /> : null}
        {tab === "code" ? <SnippetsPanel {...props} /> : null}
      </div>
      {drawer ? (
        <nav className="side__links" aria-label="Project links">
          <ExternalLink href={DOCS_URL}>Docs</ExternalLink>
          <ExternalLink href={GITHUB_URL}>GitHub</ExternalLink>
          <ExternalLink href={props.apiDocsUrl}>API docs</ExternalLink>
        </nav>
      ) : null}
    </div>
  );
}

function keyHelp(access: AccessMode | null, apiKey: string): string {
  const stored = `kept in this browser only${apiKey ? ` (${maskKey(apiKey)})` : ""}`;
  if (access === "public") {
    return `Optional on this server: it answers without a key, rate-limited per IP. A key is sent as X-API-Key and ${stored}.`;
  }
  if (access === "key") return `Required by this server. Sent as X-API-Key and ${stored}.`;
  return `Sent as X-API-Key and ${stored}.`;
}

function SettingsPanel({
  settings,
  onSettings,
  authError,
  onDismissAuthError,
  access,
  theme,
  onTheme,
}: Props) {
  const [showKey, setShowKey] = useState(false);
  return (
    <>
      {authError ? (
        <div className="banner banner--danger" role="alert">
          <span className="banner__text">API key required: {authError}</span>
          <button type="button" className="btn btn--sm btn--quiet" onClick={onDismissAuthError}>
            Dismiss
          </button>
        </div>
      ) : null}

      <section className="section">
        <h3 className="eyebrow">Access</h3>
        <div className="field">
          <label htmlFor="api-key">
            API key{" "}
            {access === "key" ? <span className="field__tag">Required</span> : null}
            {access === "public" ? <span className="field__tag field__tag--muted">Optional</span> : null}
          </label>
          <div className="field__row">
            <input
              id="api-key"
              className="input"
              type={showKey ? "text" : "password"}
              autoComplete="off"
              spellCheck={false}
              placeholder="fa_..."
              value={settings.apiKey}
              onChange={(e) => onSettings({ apiKey: e.target.value.trim() })}
              aria-describedby="api-key-help"
            />
            <button
              type="button"
              className="btn btn--sm"
              onClick={() => setShowKey((v) => !v)}
              aria-pressed={showKey}
            >
              {showKey ? "Hide" : "Show"}
            </button>
          </div>
          <div className="field__help" id="api-key-help">
            {keyHelp(access, settings.apiKey)}
          </div>
        </div>
        <div className="field">
          <label htmlFor="base-url">API base URL</label>
          <input
            id="base-url"
            className="input"
            type="url"
            placeholder="Same origin"
            value={settings.baseUrl}
            onChange={(e) => onSettings({ baseUrl: e.target.value })}
            onBlur={(e) => onSettings({ baseUrl: normalizeUrl(e.target.value) })}
            aria-describedby="base-url-help"
          />
          <div className="field__help" id="base-url-help">
            Leave empty to call the API this page is served from. Another host must allow CORS.
          </div>
        </div>
      </section>

      <section className="section">
        <h3 className="eyebrow">Endpoint</h3>
        <div className="field">
          <span className="field__label" id="mode-label">
            Mode
          </span>
          <div className="segmented" role="group" aria-labelledby="mode-label">
            <button
              type="button"
              aria-pressed={settings.mode === "generate"}
              onClick={() => onSettings({ mode: "generate" })}
            >
              Ask (/generate)
            </button>
            <button
              type="button"
              aria-pressed={settings.mode === "chat"}
              onClick={() => onSettings({ mode: "chat" })}
            >
              Conversation (/v1)
            </button>
          </div>
          <div className="field__help">
            {settings.mode === "generate"
              ? "One question per request through the native API, with cited sources and token usage."
              : "Multi-turn through the OpenAI-compatible API: previous answers are sent as history. No sources in this mode."}
          </div>
        </div>
        <label className="toggle">
          <span className="toggle__text">
            <span className="toggle__label">Stream tokens</span>
            <span className="field__help">Off: one JSON response, eligible for the edge cache.</span>
          </span>
          <input
            type="checkbox"
            className="switch"
            checked={settings.stream}
            onChange={(e) => onSettings({ stream: e.target.checked })}
          />
        </label>
        <div className="field">
          <label htmlFor="retrieval">Documentation retrieval</label>
          <select
            id="retrieval"
            className="input"
            value={settings.useRetrieval === null ? "server" : settings.useRetrieval ? "on" : "off"}
            onChange={(e) =>
              onSettings({
                useRetrieval: e.target.value === "server" ? null : e.target.value === "on",
              })
            }
          >
            <option value="server">Server default</option>
            <option value="on">On (ground in the FastAPI docs)</option>
            <option value="off">Off (model only)</option>
          </select>
        </div>
      </section>

      <section className="section">
        <h3 className="eyebrow">Generation</h3>
        <RangeField
          id="temperature"
          label="Temperature"
          value={settings.temperature}
          limits={LIMITS.temperature}
          onChange={(temperature) => onSettings({ temperature })}
          help="0 is greedy decoding, the setting used by the evaluation harness."
        />
        <RangeField
          id="top-p"
          label="Top-p"
          value={settings.topP}
          limits={LIMITS.topP}
          onChange={(topP) => onSettings({ topP })}
        />
        <RangeField
          id="max-tokens"
          label="Max new tokens"
          value={settings.maxNewTokens}
          limits={LIMITS.maxNewTokens}
          onChange={(maxNewTokens) => onSettings({ maxNewTokens })}
        />
        <div>
          <button
            type="button"
            className="btn btn--sm"
            onClick={() => onSettings({ ...DEFAULT_SETTINGS, apiKey: settings.apiKey, baseUrl: settings.baseUrl })}
          >
            Reset generation settings
          </button>
        </div>
      </section>

      <section className="section">
        <h3 className="eyebrow">Appearance</h3>
        <fieldset className="field field--plain">
          <legend className="field__label">Theme</legend>
          <div className="segmented segmented--radio">
            {THEMES.map((t) => (
              <label key={t} className="segmented__option">
                <input
                  type="radio"
                  name="theme"
                  value={t}
                  checked={theme === t}
                  onChange={() => onTheme(t)}
                />
                <span>{THEME_LABELS[t]}</span>
              </label>
            ))}
          </div>
          <div className="field__help">System follows your device setting.</div>
        </fieldset>
      </section>
    </>
  );
}

function normalizeUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, "");
  return /^https?:\/\//i.test(trimmed) ? trimmed : "";
}

function RangeField({
  id,
  label,
  value,
  limits,
  onChange,
  help,
}: {
  id: string;
  label: string;
  value: number;
  limits: { min: number; max: number; step: number };
  onChange: (value: number) => void;
  help?: string;
}) {
  return (
    <div className="field">
      <div className="field__head">
        <label htmlFor={id}>{label}</label>
        {/* Not <output>: its implicit status role would announce every drag step. */}
        <span className="field__value" aria-hidden="true">
          {value}
        </span>
      </div>
      <input
        id={id}
        className="range"
        type="range"
        min={limits.min}
        max={limits.max}
        step={limits.step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        {...(help ? { "aria-describedby": `${id}-help` } : {})}
      />
      {help ? (
        <div className="field__help" id={`${id}-help`}>
          {help}
        </div>
      ) : null}
    </div>
  );
}

function accessLabel(access: AccessMode | null): string {
  if (access === "public") return "public (no key needed)";
  if (access === "key") return "API key required";
  return "-";
}

function StatusPanel({ health, access, lastMeta, turnCount, onClear, displayBaseUrl }: Props) {
  const h = health.health;
  return (
    <>
      <section className="section">
        <h3 className="eyebrow">Service</h3>
        <dl className="kv">
          <dt>Endpoint</dt>
          <dd>{displayBaseUrl}</dd>
          <dt>Status</dt>
          <dd>{health.error ?? h?.status ?? "checking"}</dd>
          {h ? (
            <>
              <dt>Access</dt>
              <dd>{accessLabel(access)}</dd>
              <dt>Backend</dt>
              <dd>{h.backend || "-"}</dd>
              <dt>Model version</dt>
              <dd>{h.model_version || "-"}</dd>
              <dt>Fine-tuned weights</dt>
              <dd>{h.uses_adapter ? "yes" : "no (base model)"}</dd>
              <dt>Load state</dt>
              <dd>{h.load_state || (h.model_loaded ? "loaded" : "-")}</dd>
            </>
          ) : null}
          <dt>Last check</dt>
          <dd>{health.checkedAt ? new Date(health.checkedAt).toLocaleTimeString() : "-"}</dd>
        </dl>
        <div>
          <button type="button" className="btn btn--sm" onClick={health.refresh} disabled={health.checking}>
            {health.checking ? "Checking" : "Check now"}
          </button>
        </div>
      </section>

      <section className="section">
        <h3 className="eyebrow">Last response</h3>
        {lastMeta ? <MetaTable meta={lastMeta} /> : <p className="muted">No request sent yet.</p>}
      </section>

      <section className="section">
        <h3 className="eyebrow">Conversation</h3>
        <p className="muted">
          {turnCount === 0
            ? "Empty."
            : `${turnCount} turns. Nothing is stored on the server unless transcripts are enabled.`}
        </p>
        <div>
          <button
            type="button"
            className="btn btn--sm btn--danger"
            onClick={onClear}
            disabled={turnCount === 0}
          >
            Clear conversation
          </button>
        </div>
      </section>
    </>
  );
}

function MetaTable({ meta }: { meta: ResponseMeta }) {
  const rl = meta.rateLimit;
  return (
    <dl className="kv">
      <dt>HTTP status</dt>
      <dd>{meta.status}</dd>
      <dt>X-Request-ID</dt>
      <dd>{meta.requestId ?? "-"}</dd>
      <dt>X-Cache</dt>
      <dd>{meta.cache ?? "- (no edge)"}</dd>
      <dt>X-Served-By</dt>
      <dd>{meta.servedBy ?? "-"}</dd>
      <dt>X-Process-Time-Ms</dt>
      <dd>{meta.processTimeMs ?? "-"}</dd>
      <dt>Rate limit</dt>
      <dd>
        {rl.limit === null
          ? "-"
          : `${rl.remaining ?? "?"} of ${rl.limit} left this minute${rl.reset !== null ? `, resets in ${rl.reset}s` : ""}`}
      </dd>
      {meta.retryAfterS !== null ? (
        <>
          <dt>Retry-After</dt>
          <dd>{meta.retryAfterS}s</dd>
        </>
      ) : null}
    </dl>
  );
}

function SnippetsPanel({ settings, draft, displayBaseUrl, access }: Props) {
  const input: SnippetInput = {
    baseUrl: displayBaseUrl,
    question: draft,
    mode: settings.mode,
    stream: settings.stream,
    temperature: settings.temperature,
    topP: settings.topP,
    maxNewTokens: settings.maxNewTokens,
    useRetrieval: settings.useRetrieval,
  };
  const snippets = [
    { title: "curl", text: curlSnippet(input) },
    { title: "Python (requests)", text: pythonRequestsSnippet(input) },
    { title: "Python (OpenAI SDK)", text: pythonOpenAISnippet(input) },
    { title: "JavaScript (fetch)", text: javascriptSnippet(input) },
  ];
  return (
    <>
      <p className="muted">
        The same request as the composer, for your terminal or code. Set{" "}
        <span className="mono">API_KEY</span> in your environment; the key in this browser is never
        included.
        {access === "public" ? " This server also answers without the header." : ""}
      </p>
      {snippets.map((s) => (
        <section key={s.title} className="snippet">
          <div className="snippet__bar">
            <h3 className="snippet__title">{s.title}</h3>
            <CopyButton text={s.text} className="snippet__copy" />
          </div>
          <pre>
            <code>{s.text}</code>
          </pre>
        </section>
      ))}
    </>
  );
}
