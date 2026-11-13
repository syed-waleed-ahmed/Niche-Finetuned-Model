import { useState } from "react";

import type { ResponseMeta } from "../api/client";
import {
  curlSnippet,
  javascriptSnippet,
  pythonOpenAISnippet,
  pythonRequestsSnippet,
  type SnippetInput,
} from "../api/snippets";
import type { HealthHook } from "../hooks/useHealth";
import { DEFAULT_SETTINGS, LIMITS, maskKey, type Settings } from "../state/settings";
import { CopyButton } from "./CopyButton";

export type Tab = "settings" | "status" | "code";

interface Props {
  tab: Tab;
  onTab: (tab: Tab) => void;
  settings: Settings;
  onSettings: (patch: Partial<Settings>) => void;
  health: HealthHook;
  lastMeta: ResponseMeta | null;
  draft: string;
  turnCount: number;
  onClear: () => void;
  authError: string | null;
  onDismissAuthError: () => void;
  displayBaseUrl: string;
}

const TABS: { id: Tab; label: string }[] = [
  { id: "settings", label: "Settings" },
  { id: "status", label: "Status" },
  { id: "code", label: "Code" },
];

export function SidePanel(props: Props) {
  const { tab, onTab } = props;
  return (
    <aside className="side" aria-label="Side panel">
      <div className="tabs" role="tablist">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            id={`tab-${t.id}`}
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
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
    </aside>
  );
}

function SettingsPanel({ settings, onSettings, authError, onDismissAuthError }: Props) {
  const [showKey, setShowKey] = useState(false);
  return (
    <>
      {authError ? (
        <div className="banner banner--danger" role="alert">
          <span className="banner__text">API key required: {authError}</span>
          <button type="button" className="btn btn--sm" onClick={onDismissAuthError}>
            Dismiss
          </button>
        </div>
      ) : null}

      <h3>Access</h3>
      <div className="field">
        <label htmlFor="api-key">API key</label>
        <div className="field__row">
          <input
            id="api-key"
            type={showKey ? "text" : "password"}
            autoComplete="off"
            spellCheck={false}
            placeholder="fa_..."
            value={settings.apiKey}
            onChange={(e) => onSettings({ apiKey: e.target.value.trim() })}
          />
          <button type="button" className="btn btn--sm" onClick={() => setShowKey((v) => !v)}>
            {showKey ? "Hide" : "Show"}
          </button>
        </div>
        <div className="field__help">
          Sent as <span className="mono">X-API-Key</span> and kept in this browser only
          {settings.apiKey ? ` (${maskKey(settings.apiKey)})` : ""}.
        </div>
      </div>
      <div className="field">
        <label htmlFor="base-url">API base URL</label>
        <input
          id="base-url"
          type="url"
          placeholder="Same origin"
          value={settings.baseUrl}
          onChange={(e) => onSettings({ baseUrl: e.target.value })}
          onBlur={(e) => onSettings({ baseUrl: normalizeUrl(e.target.value) })}
        />
        <div className="field__help">
          Leave empty to call the API this page is served from. Another host must allow CORS.
        </div>
      </div>

      <h3>Endpoint</h3>
      <div className="field">
        <span className="field__label">Mode</span>
        <div className="segmented" role="group" aria-label="Endpoint mode">
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
        <span>
          Stream tokens
          <div className="field__help">Off: one JSON response, eligible for the edge cache.</div>
        </span>
        <input
          type="checkbox"
          checked={settings.stream}
          onChange={(e) => onSettings({ stream: e.target.checked })}
        />
      </label>
      <div className="field">
        <label htmlFor="retrieval">Documentation retrieval</label>
        <select
          id="retrieval"
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

      <h3>Generation</h3>
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
      <label htmlFor={id}>{label}</label>
      <div className="field__row">
        <input
          id={id}
          type="range"
          min={limits.min}
          max={limits.max}
          step={limits.step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
        />
        <span className="field__value">{value}</span>
      </div>
      {help ? <div className="field__help">{help}</div> : null}
    </div>
  );
}

function StatusPanel({ health, lastMeta, turnCount, onClear, displayBaseUrl }: Props) {
  const h = health.health;
  return (
    <>
      <h3>Service</h3>
      <dl className="kv">
        <dt>Endpoint</dt>
        <dd>{displayBaseUrl}</dd>
        <dt>Status</dt>
        <dd>{health.error ?? h?.status ?? "checking"}</dd>
        {h ? (
          <>
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

      <h3>Last response</h3>
      {lastMeta ? <MetaTable meta={lastMeta} /> : <p className="muted">No request sent yet.</p>}

      <h3>Conversation</h3>
      <p className="muted">
        {turnCount === 0 ? "Empty." : `${turnCount} turns. Nothing is stored on the server unless transcripts are enabled.`}
      </p>
      <div>
        <button type="button" className="btn btn--sm btn--danger" onClick={onClear} disabled={turnCount === 0}>
          Clear conversation
        </button>
      </div>
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

function SnippetsPanel({ settings, draft, displayBaseUrl }: Props) {
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
      </p>
      {snippets.map((s) => (
        <div key={s.title}>
          <h3>{s.title}</h3>
          <div className="snippet">
            <CopyButton text={s.text} className="snippet__copy" />
            <pre>
              <code>{s.text}</code>
            </pre>
          </div>
        </div>
      ))}
    </>
  );
}
