import { describe, expect, it } from "vitest";

import {
  THEME_STORAGE_KEY,
  applyTheme,
  loadTheme,
  nextTheme,
  parseTheme,
  saveTheme,
} from "../src/state/theme";

function memoryStorage(): Storage {
  const data = new Map<string, string>();
  return {
    getItem: (k) => data.get(k) ?? null,
    setItem: (k, v) => void data.set(k, String(v)),
    removeItem: (k) => void data.delete(k),
    clear: () => data.clear(),
    key: (i) => [...data.keys()][i] ?? null,
    get length() {
      return data.size;
    },
  };
}

function throwingStorage(): Storage {
  const fail = () => {
    throw new DOMException("blocked", "SecurityError");
  };
  return { ...memoryStorage(), getItem: fail, setItem: fail };
}

describe("theme", () => {
  it("accepts only light and dark; anything else means system", () => {
    expect(parseTheme("light")).toBe("light");
    expect(parseTheme("dark")).toBe("dark");
    expect(parseTheme("system")).toBe("system");
    expect(parseTheme("sepia")).toBe("system");
    expect(parseTheme(null)).toBe("system");
  });

  it("cycles system, light, dark", () => {
    expect(nextTheme("system")).toBe("light");
    expect(nextTheme("light")).toBe("dark");
    expect(nextTheme("dark")).toBe("system");
  });

  it("round-trips through storage and defaults to system", () => {
    const storage = memoryStorage();
    expect(loadTheme(storage)).toBe("system");
    saveTheme("dark", storage);
    expect(storage.getItem(THEME_STORAGE_KEY)).toBe("dark");
    expect(loadTheme(storage)).toBe("dark");
    storage.setItem(THEME_STORAGE_KEY, "{garbage");
    expect(loadTheme(storage)).toBe("system");
  });

  it("survives storage that throws or does not exist", () => {
    expect(loadTheme(throwingStorage())).toBe("system");
    expect(() => saveTheme("light", throwingStorage())).not.toThrow();
    expect(loadTheme(null)).toBe("system");
    expect(() => saveTheme("light", null)).not.toThrow();
  });

  it("sets data-theme for light and dark and removes it for system", () => {
    const root = document.createElement("html");
    applyTheme("dark", root);
    expect(root.getAttribute("data-theme")).toBe("dark");
    applyTheme("light", root);
    expect(root.getAttribute("data-theme")).toBe("light");
    applyTheme("system", root);
    expect(root.hasAttribute("data-theme")).toBe(false);
  });
});
