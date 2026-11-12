/** Small helpers for responses generated at the edge. */

export function jsonError(
  status: number,
  message: string,
  type: string,
  extraHeaders: Record<string, string> = {},
): Response {
  return new Response(JSON.stringify({ error: { message, type } }), {
    status,
    headers: { "content-type": "application/json", ...extraHeaders },
  });
}

const SAFE_REQUEST_ID = /^[A-Za-z0-9._:-]{1,128}$/;

/** Keeps a client-supplied X-Request-ID only if it is short and log-safe. */
export function sanitizeRequestId(value: string | null): string | null {
  const trimmed = value?.trim();
  return trimmed && SAFE_REQUEST_ID.test(trimmed) ? trimmed : null;
}
