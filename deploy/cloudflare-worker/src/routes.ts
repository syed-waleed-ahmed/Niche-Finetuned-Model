/**
 * Path classification for the fixed origin contract.
 *
 * - protected: needs an API key (checked at the edge, validated by the origin).
 * - public:    no key; rate limited per client IP with the anonymous limit.
 * - blocked:   internal endpoints that must never be reachable through the edge.
 * - unknown:   anything else; answered with 404 at the edge so scanners never
 *              wake a scaled-to-zero origin.
 */

export type RouteKind = "protected" | "public" | "blocked" | "unknown";

const PUBLIC_PATHS: ReadonlySet<string> = new Set([
  "/health",
  "/ready",
  "/docs",
  "/docs/oauth2-redirect",
  "/redoc",
  "/openapi.json",
]);

const BLOCKED_PATHS: ReadonlySet<string> = new Set(["/metrics"]);

/** Paths whose non-streaming POST answers go into the shared cache. */
const CACHEABLE_PATHS: ReadonlySet<string> = new Set(["/generate", "/v1/chat/completions"]);

export function classifyPath(pathname: string): RouteKind {
  if (BLOCKED_PATHS.has(pathname)) return "blocked";
  if (pathname === "/generate" || pathname.startsWith("/v1/")) return "protected";
  if (PUBLIC_PATHS.has(pathname)) return "public";
  return "unknown";
}

export function isCacheableEndpoint(method: string, pathname: string): boolean {
  return method === "POST" && CACHEABLE_PATHS.has(pathname);
}
