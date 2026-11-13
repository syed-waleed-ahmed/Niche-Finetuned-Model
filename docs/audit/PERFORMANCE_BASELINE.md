# Performance baseline

Measured on 2026-11-13 against the live deployment:

- **Edge:** Cloudflare Worker.
- **Origin:** Cloud Run, 2 vCPU and 2 GiB per instance, `maxScale: 2`, `containerConcurrency: 4`, `MAX_CONCURRENCY=1`, `MAX_QUEUE=3`.
- **Model:** TinyLlama-1.1B-Chat Q4_K_M with llama.cpp, `n_ctx` 2048.

Every figure is one observed run from a single client, not a statistical benchmark. No
sustained load test has been run, and none should be run against the public deployment
without a plan, a time window and a cost check.

## Single requests (warm instance)

| Request | Result |
|---|---|
| `GET /health` through the edge | 683 ms (includes edge and TLS setup from the client) |
| `POST /generate`, 64 new tokens, blocking | 8.6 s first call, 6.3 s repeat |
| OpenAI SDK chat completion, 64 tokens | 7.3 s, `finish_reason: length` |
| Native SSE stream, 64-token budget | time to first token 298 ms, 6.8 s total, 65 events |
| OpenAI SDK stream, 48 tokens | time to first token 1.06 s, 5.5 s total |
| `POST /generate` with retrieval, 128 tokens | 521 prompt tokens (retrieved context included) |

Implied decode rate for one request: roughly 8 to 10 tokens per second.

## Concurrency probes

Two controlled, one-off bursts of short requests (32 new tokens each), one minute apart:

| Burst | Outcome | Latencies |
|---|---|---|
| 6 simultaneous, one warm instance | 2 x 200, 4 x 429 | 200s at 3.8 s and 7.2 s; the 429s all at about 10.5 s |
| 4 simultaneous, a minute later | 4 x 200 | 3.5, 5.8, 8.8, 11.9 s (served one after another) |

### What the probes show

1. **One generation at a time per instance.** The 4-request burst finished in a staircase,
   about 3 s per 32-token answer. That is `MAX_CONCURRENCY=1`: the queue holds the
   others, so latency grows linearly with queue position.
2. **Scale-out is not instant.** In the 6-request burst, Cloud Run routed part of the
   burst towards a second instance that was still starting.
   - **What the client saw:** after about 10 seconds, Cloud Run's front end answered
     those requests itself with `429 Rate exceeded.` (plain text, no `Retry-After`).
   - **Why:** a cold instance has to download and load the model before it is ready.
     The application's own queue never saw those requests.
3. **The app's admission control behaves as designed once traffic reaches it.** It
   returned no 429 or 503 in either burst.

### Changes made because of this

- **Edge:** the Worker now rewrites a non-JSON 429, 502, 503 or 504 from the origin's
  platform into the API's JSON error shape (`overloaded` or `upstream_unavailable`).
  It adds `Retry-After: 10` when the platform sent none and logs
  `upstream_normalized: true`. Clients see one error contract, whichever layer refused
  the request.
- **New metrics:** `assistant_queue_wait_seconds` and
  `assistant_time_to_first_token_seconds` make the staircase visible without probing.

### Options not taken (they cost money or need owner approval)

| Option | Effect | Cost |
|---|---|---|
| `minScale: 1` | No cold start for the first instance | An always-allocated instance, billed while idle |
| Higher `maxScale` | More parallel generations during bursts | Raises the worst-case bill |
| Bake weights into the image | Shorter cold start (no 0.7 GB download) | Image exceeds Artifact Registry's 0.5 GB free storage |

## Capacity statement

From this evidence:

- **Concurrency:** the deployment serves about one generation at a time per instance,
  with at most 2 instances.
- **Short bursts:** a burst of a few short requests completes, serially.
- **Larger bursts:** a burst that forces a cold scale-out can see some requests refused
  with 429 and a retry hint while the second instance starts.

No number of supported users is claimed. Establishing one needs a controlled load test
(see `tests/load/k6.js`), with an explicit request budget, against a non-production
copy or an agreed window.
