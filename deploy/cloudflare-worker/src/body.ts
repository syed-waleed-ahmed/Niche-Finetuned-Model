/**
 * Request body buffering with a hard size cap.
 *
 * The body is buffered once so it can be hashed for the cache key and replayed to a
 * second origin on failover. Without a cap, a caller could make every isolate (and
 * then the origin) hold a body up to Cloudflare's own 100 MB ceiling. A declared
 * Content-Length over the cap is refused before reading; a chunked body is counted
 * as it streams and abandoned as soon as it crosses the cap.
 */

export type BodyResult = { ok: true; body: ArrayBuffer | null } | { ok: false };

export async function readBodyCapped(request: Request, maxBytes: number): Promise<BodyResult> {
  const method = request.method.toUpperCase();
  if (method === "GET" || method === "HEAD" || request.body === null) {
    return { ok: true, body: null };
  }
  const declared = Number(request.headers.get("content-length") ?? "");
  if (Number.isFinite(declared) && declared > maxBytes) return { ok: false };

  const reader = request.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    total += value.byteLength;
    if (total > maxBytes) {
      await reader.cancel();
      return { ok: false };
    }
    chunks.push(value);
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return { ok: true, body: out.buffer };
}
