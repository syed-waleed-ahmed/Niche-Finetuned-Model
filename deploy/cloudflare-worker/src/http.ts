/** Small helpers for responses generated at the edge. */

export function jsonError(
  status: number,
  message: string,
  type: string,
  extraHeaders: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify({ error: { message, type } }), {
    status,
    headers: { "content-type": "application/json", ...SECURITY_HEADERS, ...extraHeaders },
  });
}

/** Set on every response the Worker produces or relays (JSON and SSE only). */
export const SECURITY_HEADERS: Readonly<Record<string, string>> = {
  "x-content-type-options": "nosniff",
};

/** Origin response headers that describe the origin's infrastructure, not the API. */
const STRIPPED_RESPONSE_HEADERS = ["x-cloud-trace-context", "server", "via", "alt-svc"];

export function sanitizeResponseHeaders(headers: Headers): void {
  for (const name of STRIPPED_RESPONSE_HEADERS) headers.delete(name);
  for (const [name, value] of Object.entries(SECURITY_HEADERS)) headers.set(name, value);
}

const SAFE_REQUEST_ID = /^[A-Za-z0-9._:-]{1,128}$/;

/** Keeps a client-supplied X-Request-ID only if it is short and log-safe. */
export function sanitizeRequestId(value: string | null): string | null {
  const trimmed = value?.trim();
  return trimmed && SAFE_REQUEST_ID.test(trimmed) ? trimmed : null;
}
