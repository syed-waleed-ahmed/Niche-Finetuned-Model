/**
 * Minimal Upstash Redis REST client: a single `/pipeline` call per use.
 *
 * Every command in a pipeline counts against the Upstash command quota, but a
 * pipeline is one HTTP subrequest, which keeps Worker subrequests and latency low.
 */

export type RedisCommand = readonly (string | number)[];

export interface RedisReply {
  result?: unknown;
  error?: string;
}

export interface RedisClient {
  /** Runs the commands in order; rejects on transport or HTTP errors. */
  pipeline(commands: readonly RedisCommand[]): Promise<RedisReply[]>;
}

export type FetchLike = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

export function createUpstashClient(
  options: { url: string; token: string; timeoutMs: number },
  fetchFn: FetchLike,
): RedisClient {
  const endpoint = `${options.url.replace(/\/+$/, "")}/pipeline`;
  return {
    async pipeline(commands) {
      if (commands.length === 0) return [];
      const res = await fetchFn(endpoint, {
        method: "POST",
        headers: {
          authorization: `Bearer ${options.token}`,
          "content-type": "application/json",
        },
        body: JSON.stringify(commands.map((c) => c.map(String))),
        signal: AbortSignal.timeout(options.timeoutMs),
      });
      if (!res.ok) {
        await res.body?.cancel();
        throw new Error(`upstash pipeline HTTP ${res.status}`);
      }
      const data: unknown = await res.json();
      if (!Array.isArray(data) || data.length !== commands.length) {
        throw new Error("upstash pipeline: unexpected response shape");
      }
      return data as RedisReply[];
    },
  };
}
