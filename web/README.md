# Web UI

A browser playground for the FastAPI Niche Assistant: React 19, TypeScript and Vite,
no UI framework and no server of its own. It exists to make the service's behaviour
visible: streamed tokens, cited documentation sources, token usage and latency, the
edge cache and rate-limit headers, explicit back-pressure (429/503 with
`Retry-After`), feedback, and the two API styles.

## Run it

```bash
npm install
npm run dev          # http://localhost:5173
```

The dev server proxies `/generate`, `/v1`, `/health`, `/ready`, `/docs` and
`/openapi.json` to `http://localhost:8000` (override with `VITE_DEV_API_TARGET`), so
start the API first, for example:

```bash
API_KEY=dev fastapi-assistant serve
```

Then paste `dev` into Settings. Without a key the API answers 401 and the UI opens
the settings panel for you.

```bash
npm run typecheck    # tsc --noEmit
npm test             # vitest (jsdom), including an end-to-end App test with mocked fetch
npm run build        # writes dist/
```

## How it is deployed

`deploy/cloudflare-worker/wrangler.toml` declares `dist/` as Workers Static Assets.
Requests that match a built file (`/`, `/assets/*`, `/favicon.svg`) are served by
Cloudflare without running the Worker; every other path falls through to the Worker,
so the API and its 404 rule are unchanged. UI and API share one hostname, which is
why the UI needs no CORS configuration and can read the edge's `X-Cache` and
`X-RateLimit-*` headers. The release workflow runs `npm ci && npm run build` here
before `wrangler deploy`; `wrangler dev` also needs `dist/` to exist.

Do not add a file to `public/` named after an API path (`generate`, `v1`, `health`,
`ready`, `docs`, `redoc`, `openapi.json`, `metrics`): it would shadow the API. A unit
test checks this.

Settings allow an explicit API base URL for pointing a local UI at another host.
In that case only responses that come from the origin carry CORS headers; responses
generated at the edge (401, 429, cache hits) do not, and the browser reports them as
network errors.

## Layout

```text
src/
  api/types.ts       request/response types mirroring schemas.py
  api/sse.ts         dependency-free SSE parser (fetch + ReadableStream; EventSource cannot POST)
  api/client.ts      askNative (/generate), askChat (/v1/chat/completions), feedback, health,
                     ApiError with status/Retry-After/request id; one AnswerEvent stream for both APIs
  api/snippets.ts    curl / Python / JavaScript snippet builders
  state/settings.ts  persisted settings, bounded to the API's limits
  state/conversation.ts  turn model and reducer (streaming, done, error, stopped, feedback)
  hooks/             useSettings, useHealth (30 s poll, paused when hidden), useConversation (send,
                     stop, retry on 429/503, rate)
  components/        Header + health badge, MessageList, Message (Markdown, meta row, sources,
                     feedback), Composer, RetryBanner, SidePanel (Settings, Status, Code)
test/                unit tests for the API layer and reducer, plus App.test.tsx
```

Pinned dependencies: react, react-dom, react-markdown, remark-gfm, rehype-highlight,
highlight.js (only the languages the model emits are registered). Dev: vite,
@vitejs/plugin-react, typescript, vitest, jsdom, Testing Library.
