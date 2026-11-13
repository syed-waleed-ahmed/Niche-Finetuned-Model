# Examples

Each example reads the endpoint and key from the environment, so no key ever lands in
code or shell history:

```bash
export ASSISTANT_URL=https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev
export ASSISTANT_API_KEY=fa_...   # or http://localhost:8000 and your local API_KEY
```

A server in public demo mode, like the live demo above, needs no key.
`fetch_stream.mjs` then runs with `ASSISTANT_API_KEY` unset; the other examples always
ask for one.

| File | What it shows | Needs |
|---|---|---|
| [openai_sdk.py](openai_sdk.py) | The official OpenAI SDK against `/v1`: models, a blocking chat completion, and a streamed one | `pip install "openai>=1.40"` |
| [native_stream.py](native_stream.py) | The native `/generate` API with retrieval, cited sources, and SSE parsing | `pip install httpx` |
| [curl.sh](curl.sh) | Every endpoint with curl, including streaming (`-N`) and an error | bash, curl |
| [fetch_stream.mjs](fetch_stream.mjs) | JavaScript with the built-in `fetch`: a streamed `/v1/chat/completions` answer, parsing server-sent events by hand, and a plain message for `401`, `413`, `429` and `503` | Node.js 22 (no packages); the same code runs in a browser |

All four are exercised in CI by `tests/test_examples.py`. The test runs each example
against a real local server (backed by a fake engine), so a change to the API contract
that breaks an example fails the build.

## Limits worth knowing

| Condition | Response |
|---|---|
| Missing or invalid key | `401` |
| Body over 64 KiB | `413` |
| Prompt longer than the model's context window | `400` with `code: context_length_exceeded` |
| Over the per-minute rate limit at the edge | `429` with `Retry-After` |
| Generation queue full | `429` with `Retry-After` |
| Queue wait timed out, or model still loading | `503` with `Retry-After` |
| Origin did not answer in time | `504` |

The served model is small (1.1B parameters). Treat its answers as drafts and check them
against the cited documentation sources.
