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

Then paste `dev` into Settings. A server that requires a key and has none saved
shows a card explaining how to add one, and a question asked anyway opens Settings at
the key field instead of being sent; a 401 (a wrong key) still opens Settings too.

```bash
npm run typecheck    # tsc --noEmit
npm test             # vitest (jsdom), including an end-to-end App test with mocked fetch
npm run build        # writes dist/
```

## What it does

- **First run.** The empty conversation is a welcome screen: one sentence on what the
  assistant is, a status line from `/health` (model version, fine-tuned or base model,
  ready or loading), example questions, and links to the docs site, GitHub and the
  API reference (`/docs`).
- **Access.** `/health` may report `access`: `"public"` (no key needed, rate-limited
  per IP at the edge) or `"key"`. Public servers get a "Public demo" note and the key
  becomes optional. Key servers, and older servers that omit the field, get the key
  card described above until a key is saved.
- **Theme.** System, light or dark, from the header toggle (cycles) or Settings,
  Appearance. The choice is stored in `localStorage` (`fastapi-assistant.theme.v1`)
  and applied as `data-theme` on `<html>`; "system" follows `prefers-color-scheme`.
  Colours are CSS custom properties in `src/styles.css`; text pairs meet WCAG AA in
  both themes.
- **Layout.** Below 900px the side panel is an off-canvas drawer opened from the
  header Settings button: a modal dialog with a close button and backdrop, closed by
  Escape, with focus moved into it and back to the trigger. The composer stays pinned
  to the bottom of the viewport (`100dvh`).
- **Answers.** Streamed Markdown with highlighted, copyable code blocks; cited sources
  as a numbered list; a compact line of model, time, tokens, speed and cache status
  with the rest behind Details; copy, rating and an optional comment.
- **Accessibility.** Visible focus rings, labelled controls, arrow-key tabs, and one
  polite live region that announces when an answer starts and finishes rather than
  every token. Animations stop under `prefers-reduced-motion`.

The Content-Security-Policy in `public/_headers` allows no external or inline scripts
and styles, so the UI uses the system font stack and inline SVG icons, and the theme is
applied from `main.tsx` before the first render rather than from an inline script.

## How it is deployed

`deploy/cloudflare-worker/wrangler.toml` declares `dist/` as Workers Static Assets.
Requests that match a built file (`/`, `/assets/*`, `/favicon.svg`) are served by
Cloudflare without running the Worker; every other path falls through to the Worker,
so the API and its 404 rule are unchanged. UI and API share one hostname, which is
why the UI needs no CORS configuration and can read the edge's `X-Cache` and
`X-RateLimit-*` headers. The release workflow runs `npm ci && npm run build` here
before `wrangler deploy` on every release, so a merge to `main` that changes the UI
goes live with the next Release run. A manual `wrangler deploy`
(`docs/operations/RUNBOOK.md`, section 7.7) is only for local testing or an
emergency. `wrangler dev` also needs `dist/` to exist.

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
  state/theme.ts     theme preference: load, save, apply (data-theme on <html>)
  state/access.ts    access mode from /health (absent means "key")
  state/conversation.ts  turn model and reducer (streaming, done, error, stopped, feedback)
  hooks/             useSettings, useTheme, useMediaQuery, useHealth (30 s poll, paused when
                     hidden), useConversation (send, stop, retry on 429/503, rate)
  components/        Header (health badge, theme toggle), Welcome (first-run screen),
                     AccessNotice (public note, key card), MessageList, Message (Markdown,
                     sources, meta row, feedback), Composer, RetryBanner, SidePanel (inline
                     panel or drawer: Settings, Status, Code), Icons
test/                unit tests for the API layer, reducer, theme and access; App, onboarding
                     and shell (theme, drawer) tests with mocked fetch and matchMedia
```

Pinned dependencies: react, react-dom, react-markdown, remark-gfm, rehype-highlight,
highlight.js (only the languages the model emits are registered). Dev: vite,
@vitejs/plugin-react, typescript, vitest, jsdom, Testing Library.
