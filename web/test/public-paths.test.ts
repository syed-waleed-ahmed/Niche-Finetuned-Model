import { describe, expect, it } from "vitest";

/**
 * In production these files are served by Cloudflare Workers Static Assets in
 * front of the edge Worker: a request whose path matches a file never reaches the
 * Worker. A file named after an API path would therefore shadow the API.
 */
const WORKER_PATHS = new Set([
  "generate",
  "v1",
  "health",
  "ready",
  "docs",
  "redoc",
  "openapi.json",
  "metrics",
]);

describe("public/", () => {
  it("contains no file or folder named after a Worker-handled path", () => {
    // Vite resolves the glob at transform time, so no Node filesystem access is needed.
    const files = Object.keys(import.meta.glob("../public/**", { query: "?url" }));
    expect(files.length).toBeGreaterThan(0);
    const topLevel = files.map((f) => f.replace(/^\.\.\/public\//, "").split("/")[0]?.toLowerCase() ?? "");
    const shadowing = topLevel.filter((n) => WORKER_PATHS.has(n));
    expect(shadowing).toEqual([]);
  });
});
