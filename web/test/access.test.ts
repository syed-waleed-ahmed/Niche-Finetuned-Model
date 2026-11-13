import { describe, expect, it } from "vitest";

import type { HealthResponse } from "../src/api/types";
import { accessMode } from "../src/state/access";
import { HEALTH } from "./helpers";

describe("accessMode", () => {
  it("is unknown until /health has answered", () => {
    expect(accessMode(null)).toBeNull();
  });

  it("treats an absent field (older servers) as key", () => {
    expect(accessMode(HEALTH)).toBe("key");
  });

  it("reads public and key", () => {
    expect(accessMode({ ...HEALTH, access: "public" })).toBe("public");
    expect(accessMode({ ...HEALTH, access: "key" })).toBe("key");
  });

  it("treats an unexpected value as key, the safe default", () => {
    expect(accessMode({ ...HEALTH, access: "open" } as unknown as HealthResponse)).toBe("key");
  });
});
