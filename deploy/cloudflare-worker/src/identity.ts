/**
 * Client identity for rate limiting and the verified-key marker.
 *
 * The raw API key never leaves this module: callers only see a truncated
 * SHA-256 digest (key identity) or the client IP (anonymous identity).
 */

export type Identity =
  | { type: "key"; id: string }
  | { type: "ip"; id: string };

const HEX: readonly string[] = Array.from({ length: 256 }, (_, i) =>
  i.toString(16).padStart(2, "0"),
);

export function toHex(bytes: ArrayBuffer): string {
  const view = new Uint8Array(bytes);
  let out = "";
  for (let i = 0; i < view.length; i++) out += HEX[view[i] as number];
  return out;
}

const encoder = new TextEncoder();

export async function sha256Hex(input: string): Promise<string> {
  return toHex(await crypto.subtle.digest("SHA-256", encoder.encode(input)));
}

/** API key from `X-API-Key` or `Authorization: Bearer <key>`; null when absent or blank. */
export function extractApiKey(headers: Headers): string | null {
  const direct = headers.get("x-api-key")?.trim();
  if (direct) return direct;
  const auth = headers.get("authorization");
  if (auth) {
    const match = /^bearer\s+(.+)$/i.exec(auth.trim());
    const token = match?.[1]?.trim();
    if (token) return token;
  }
  return null;
}

export function clientIp(headers: Headers): string {
  return headers.get("cf-connecting-ip")?.trim() || "unknown";
}

export async function keyIdentity(apiKey: string): Promise<Identity> {
  return { type: "key", id: (await sha256Hex(apiKey)).slice(0, 32) };
}

export function ipIdentity(headers: Headers): Identity {
  return { type: "ip", id: `ip:${clientIp(headers)}` };
}
