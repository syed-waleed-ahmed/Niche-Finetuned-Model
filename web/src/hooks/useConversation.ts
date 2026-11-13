import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import {
  ApiError,
  askChat,
  askNative,
  chatMessagesFrom,
  sendFeedback,
  type AnswerEvent,
  type ClientConfig,
  type ResponseMeta,
} from "../api/client";
import {
  conversationReducer,
  historyFor,
  newAssistantTurn,
  newId,
  type Turn,
} from "../state/conversation";
import type { Settings } from "../state/settings";

export const MAX_ATTEMPTS = 3;
const DEFAULT_RETRY_S = 5;
const MAX_RETRY_S = 60;
/** Client-side cap on history sent to /v1; the server trims further (MAX_CHAT_TURNS). */
const MAX_HISTORY_TURNS = 12;

export interface RetryState {
  untilMs: number;
  seconds: number;
  attempt: number;
  maxAttempts: number;
  status: number;
  message: string;
}

export interface Conversation {
  turns: Turn[];
  busy: boolean;
  retry: RetryState | null;
  lastMeta: ResponseMeta | null;
  authError: string | null;
  send: (question: string) => Promise<void>;
  stop: () => void;
  clear: () => void;
  rate: (turnId: string, rating: 1 | -1, comment?: string) => Promise<string | null>;
  dismissAuthError: () => void;
}

function wait(ms: number, signal: AbortSignal): Promise<boolean> {
  return new Promise((resolve) => {
    if (signal.aborted) return resolve(false);
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve(true);
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      resolve(false);
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export function useConversation(settings: Settings): Conversation {
  const [turns, dispatch] = useReducer(conversationReducer, []);
  const [busy, setBusy] = useState(false);
  const [retry, setRetry] = useState<RetryState | null>(null);
  const [lastMeta, setLastMeta] = useState<ResponseMeta | null>(null);
  const [authError, setAuthError] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const turnsRef = useRef<Turn[]>(turns);
  turnsRef.current = turns;
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  useEffect(() => () => controller.current?.abort(), []);

  const send = useCallback(async (question: string) => {
    const text = question.trim();
    if (!text || controller.current) return;
    const s = settingsRef.current;
    const config: ClientConfig = { baseUrl: s.baseUrl, apiKey: s.apiKey };
    const history = historyFor(turnsRef.current).slice(-MAX_HISTORY_TURNS);
    const assistant = newAssistantTurn(s.mode, s.stream, Date.now());
    const id = assistant.id;
    dispatch({ type: "start", user: { id: newId(), role: "user", content: text }, assistant });

    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setAuthError(null);

    const request = (): AsyncGenerator<AnswerEvent> =>
      s.mode === "generate"
        ? askNative(
            config,
            {
              question: text,
              stream: s.stream,
              temperature: s.temperature,
              top_p: s.topP,
              max_new_tokens: s.maxNewTokens,
              ...(s.useRetrieval === null ? {} : { use_retrieval: s.useRetrieval }),
            },
            abort.signal,
          )
        : askChat(
            config,
            {
              model: "fastapi-assistant",
              messages: chatMessagesFrom([...history, { role: "user", content: text }]),
              stream: s.stream,
              temperature: s.temperature,
              top_p: s.topP,
              max_tokens: s.maxNewTokens,
            },
            abort.signal,
          );

    try {
      for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
        try {
          for await (const event of request()) {
            if (event.kind === "meta") {
              setLastMeta(event.meta);
              dispatch({ type: "meta", id, meta: event.meta });
            } else if (event.kind === "delta") {
              dispatch({ type: "delta", id, text: event.text, at: Date.now() });
            } else {
              dispatch({ type: "done", id, at: Date.now(), ...event });
            }
          }
          return;
        } catch (err) {
          if (abort.signal.aborted) {
            dispatch({ type: "stopped", id, at: Date.now() });
            return;
          }
          if (err instanceof ApiError && err.isRetryable && attempt < MAX_ATTEMPTS) {
            const seconds = Math.min(MAX_RETRY_S, Math.max(1, err.retryAfterS ?? DEFAULT_RETRY_S));
            setRetry({
              untilMs: Date.now() + seconds * 1000,
              seconds,
              attempt,
              maxAttempts: MAX_ATTEMPTS,
              status: err.status,
              message: err.message,
            });
            dispatch({ type: "pending", id });
            const proceed = await wait(seconds * 1000, abort.signal);
            setRetry(null);
            if (!proceed) {
              dispatch({ type: "stopped", id, at: Date.now() });
              return;
            }
            continue;
          }
          const message = err instanceof Error ? err.message : String(err);
          const errorType = err instanceof ApiError ? err.errorType : "network_error";
          if (err instanceof ApiError && err.status === 401) setAuthError(message);
          dispatch({ type: "error", id, at: Date.now(), message, errorType });
          return;
        }
      }
    } finally {
      controller.current = null;
      setBusy(false);
      setRetry(null);
    }
  }, []);

  const stop = useCallback(() => controller.current?.abort(), []);

  const clear = useCallback(() => {
    controller.current?.abort();
    dispatch({ type: "clear" });
  }, []);

  const rate = useCallback(async (turnId: string, rating: 1 | -1, comment?: string) => {
    const turn = turnsRef.current.find((t) => t.id === turnId);
    if (!turn || turn.role !== "assistant" || !turn.requestId) {
      return "This answer has no request id to attach feedback to.";
    }
    const s = settingsRef.current;
    try {
      await sendFeedback(
        { baseUrl: s.baseUrl, apiKey: s.apiKey },
        { request_id: turn.requestId, rating, ...(comment?.trim() ? { comment: comment.trim() } : {}) },
      );
      dispatch({ type: "feedback", id: turnId, rating });
      return null;
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) setAuthError(err.message);
      return err instanceof Error ? err.message : String(err);
    }
  }, []);

  return {
    turns,
    busy,
    retry,
    lastMeta,
    authError,
    send,
    stop,
    clear,
    rate,
    dismissAuthError: () => setAuthError(null),
  };
}
