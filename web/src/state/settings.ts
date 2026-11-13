/**
 * User settings persisted in localStorage. Bounds mirror the server's schema
 * limits (`schemas.py`), so the UI never sends a value the API would reject.
 */

import type { EndpointMode } from "../api/snippets";
import { safeStorage } from "./storage";

export interface Settings {
  apiKey: string;
  /** "" means same origin. Anything else must be an absolute http(s) URL. */
  baseUrl: string;
  mode: EndpointMode;
  stream: boolean;
  temperature: number;
  topP: number;
  maxNewTokens: number;
  /** null: let the server decide (its RAG_ENABLED default). */
  useRetrieval: boolean | null;
}

export const STORAGE_KEY = "fastapi-assistant.settings.v1";

export const LIMITS = {
  temperature: { min: 0, max: 2, step: 0.1 },
  topP: { min: 0.05, max: 1, step: 0.05 },
  maxNewTokens: { min: 1, max: 1024, step: 1 },
  questionChars: 4000,
} as const;

export const DEFAULT_SETTINGS: Settings = {
  apiKey: "",
  baseUrl: "",
  mode: "generate",
  stream: true,
  temperature: 0.2,
  topP: 0.9,
  maxNewTokens: 256,
  useRetrieval: null,
};

function clamp(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === "number" ? value : Number(value);
  if (value === "" || value === null || value === undefined || !Number.isFinite(n)) {
    return fallback;
  }
  return Math.min(max, Math.max(min, n));
}

function normalizeBaseUrl(value: unknown): string {
  if (typeof value !== "string") return "";
  const trimmed = value.trim().replace(/\/+$/, "");
  return /^https?:\/\//i.test(trimmed) ? trimmed : "";
}

/** Coerce any stored object into a valid Settings value. */
export function sanitizeSettings(raw: unknown): Settings {
  const r = (raw && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const d = DEFAULT_SETTINGS;
  return {
    apiKey: typeof r["apiKey"] === "string" ? r["apiKey"] : d.apiKey,
    baseUrl: normalizeBaseUrl(r["baseUrl"]),
    mode: r["mode"] === "chat" ? "chat" : "generate",
    stream: typeof r["stream"] === "boolean" ? r["stream"] : d.stream,
    temperature: clamp(r["temperature"], LIMITS.temperature.min, LIMITS.temperature.max, d.temperature),
    topP: clamp(r["topP"], LIMITS.topP.min, LIMITS.topP.max, d.topP),
    maxNewTokens: Math.round(
      clamp(r["maxNewTokens"], LIMITS.maxNewTokens.min, LIMITS.maxNewTokens.max, d.maxNewTokens),
    ),
    useRetrieval: typeof r["useRetrieval"] === "boolean" ? r["useRetrieval"] : null,
  };
}

export function loadSettings(storage: Storage | null = safeStorage()): Settings {
  if (!storage) return { ...DEFAULT_SETTINGS };
  try {
    const raw = storage.getItem(STORAGE_KEY);
    return raw ? sanitizeSettings(JSON.parse(raw)) : { ...DEFAULT_SETTINGS };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(settings: Settings, storage: Storage | null = safeStorage()): void {
  if (!storage) return;
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Private mode or quota exceeded: settings simply do not persist.
  }
}

/** Mask a key for display: keep a short prefix and the last four characters. */
export function maskKey(key: string): string {
  if (!key) return "";
  if (key.length <= 8) return "*".repeat(key.length);
  return `${key.slice(0, 3)}${"*".repeat(Math.min(12, key.length - 7))}${key.slice(-4)}`;
}
