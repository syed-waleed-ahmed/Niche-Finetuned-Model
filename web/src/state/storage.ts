/**
 * localStorage, or null where it is unavailable. Merely reading `window.localStorage`
 * throws in some browsers when site data is blocked, so callers also wrap every read
 * and write in try/catch.
 */
export function safeStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.localStorage : null;
  } catch {
    return null;
  }
}
