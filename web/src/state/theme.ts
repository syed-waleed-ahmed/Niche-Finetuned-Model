/**
 * Colour theme preference. "system" follows `prefers-color-scheme` through CSS; "light"
 * and "dark" set `data-theme` on the root element, which overrides the media query.
 */

import { safeStorage } from "./storage";

export type ThemePreference = "system" | "light" | "dark";

export const THEME_STORAGE_KEY = "fastapi-assistant.theme.v1";

export const THEMES: readonly ThemePreference[] = ["system", "light", "dark"];

export const THEME_LABELS: Record<ThemePreference, string> = {
  system: "System",
  light: "Light",
  dark: "Dark",
};

export function parseTheme(value: unknown): ThemePreference {
  return value === "light" || value === "dark" ? value : "system";
}

/** The next preference for the header toggle: system, light, dark, then system again. */
export function nextTheme(theme: ThemePreference): ThemePreference {
  const index = THEMES.indexOf(theme);
  return THEMES[(index + 1) % THEMES.length] ?? "system";
}

export function loadTheme(storage: Storage | null = safeStorage()): ThemePreference {
  if (!storage) return "system";
  try {
    return parseTheme(storage.getItem(THEME_STORAGE_KEY));
  } catch {
    return "system";
  }
}

export function saveTheme(theme: ThemePreference, storage: Storage | null = safeStorage()): void {
  if (!storage) return;
  try {
    storage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Private mode or quota exceeded: the choice lasts for this page only.
  }
}

export function applyTheme(
  theme: ThemePreference,
  root: HTMLElement | null = typeof document !== "undefined" ? document.documentElement : null,
): void {
  if (!root) return;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
}
