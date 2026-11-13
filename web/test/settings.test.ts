import { describe, expect, it } from "vitest";

import {
  DEFAULT_SETTINGS,
  STORAGE_KEY,
  loadSettings,
  maskKey,
  sanitizeSettings,
  saveSettings,
} from "../src/state/settings";

describe("settings", () => {
  it("returns defaults for missing or corrupt storage", () => {
    expect(loadSettings()).toEqual(DEFAULT_SETTINGS);
    window.localStorage.setItem(STORAGE_KEY, "{not json");
    expect(loadSettings()).toEqual(DEFAULT_SETTINGS);
  });

  it("round-trips through localStorage", () => {
    saveSettings({ ...DEFAULT_SETTINGS, apiKey: "fa_abc", mode: "chat", temperature: 0.7 });
    expect(loadSettings()).toMatchObject({ apiKey: "fa_abc", mode: "chat", temperature: 0.7 });
  });

  it("clamps numbers to the API limits and drops invalid values", () => {
    const s = sanitizeSettings({
      temperature: 9,
      topP: -1,
      maxNewTokens: "4096",
      mode: "bogus",
      stream: "yes",
      useRetrieval: "true",
      baseUrl: "https://edge.example///",
    });
    expect(s.temperature).toBe(2);
    expect(s.topP).toBe(0.05);
    expect(s.maxNewTokens).toBe(1024);
    expect(s.mode).toBe("generate");
    expect(s.stream).toBe(true);
    expect(s.useRetrieval).toBeNull();
    expect(s.baseUrl).toBe("https://edge.example");
  });

  it("rejects a base URL that is not absolute http(s)", () => {
    expect(sanitizeSettings({ baseUrl: "javascript:alert(1)" }).baseUrl).toBe("");
    expect(sanitizeSettings({ baseUrl: "edge.example" }).baseUrl).toBe("");
    expect(sanitizeSettings({ baseUrl: " http://localhost:8000/ " }).baseUrl).toBe("http://localhost:8000");
  });

  it("masks keys without revealing the middle", () => {
    expect(maskKey("")).toBe("");
    expect(maskKey("short")).toBe("*****");
    const masked = maskKey("fa_AbCdEfGhIjKlMnOp");
    expect(masked.startsWith("fa_")).toBe(true);
    expect(masked.endsWith("MnOp")).toBe(true);
    expect(masked).not.toContain("AbCdEfGh");
  });
});
