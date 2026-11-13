import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, getHealth, type ClientConfig, type ResponseMeta } from "../api/client";
import type { HealthResponse } from "../api/types";

export interface HealthState {
  health: HealthResponse | null;
  meta: ResponseMeta | null;
  error: string | null;
  checkedAt: number | null;
  checking: boolean;
}

/** `/health` is rate limited per IP at the edge (10/min): never poll faster than this. */
export const HEALTH_INTERVAL_MS = 30_000;
const BACKOFF_MS = 60_000;

export type HealthHook = HealthState & { refresh: () => void };

export function useHealth(config: ClientConfig): HealthHook {
  const [state, setState] = useState<HealthState>({
    health: null,
    meta: null,
    error: null,
    checkedAt: null,
    checking: false,
  });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const inflight = useRef<AbortController | null>(null);
  const configRef = useRef(config);
  configRef.current = config;
  const checkRef = useRef<() => void>(() => undefined);

  const schedule = useCallback((delay: number) => {
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => checkRef.current(), delay);
  }, []);

  const check = useCallback(async () => {
    if (typeof document !== "undefined" && document.hidden) {
      schedule(HEALTH_INTERVAL_MS);
      return;
    }
    inflight.current?.abort();
    const controller = new AbortController();
    inflight.current = controller;
    setState((s) => ({ ...s, checking: true }));
    let nextDelay = HEALTH_INTERVAL_MS;
    try {
      const { health, meta } = await getHealth(configRef.current, controller.signal);
      setState({ health, meta, error: null, checkedAt: Date.now(), checking: false });
    } catch (err) {
      if (controller.signal.aborted) return;
      const message = err instanceof Error ? err.message : String(err);
      if (err instanceof ApiError && err.status === 429) {
        nextDelay = Math.max(BACKOFF_MS, (err.retryAfterS ?? 0) * 1000);
      }
      setState((s) => ({ ...s, error: message, checkedAt: Date.now(), checking: false }));
    }
    if (inflight.current === controller) schedule(nextDelay);
  }, [schedule]);
  checkRef.current = () => void check();

  useEffect(() => {
    void check();
    const onVisible = () => {
      if (!document.hidden) void check();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      if (timer.current) clearTimeout(timer.current);
      inflight.current?.abort();
    };
  }, [check, config.baseUrl]);

  return { ...state, refresh: () => void check() };
}
