# Initial audit

Date: 2026-11-13. Commit audited: `155e580` (main) plus the uncommitted tweaks carried
on `feat/production-hardening`. Live deployment audited: Cloud Run revision built from
`684a10d`, edge Worker version `8ec9a2fc`.

Every finding below cites the command or code path that produced it. Nothing here is
inferred from documentation alone. Severity:

| Level | Meaning |
|---|---|
| P0 | Exploitable now, or data or secret exposure. Fix before anything else. |
| P1 | Production correctness, abuse, or cost risk. Fix in this hardening pass. |
| P2 | Operability, maintainability, or supply-chain gap. Fix or schedule. |
| P3 | Hygiene or documentation. Fix when convenient, or accept with a reason. |

## 1. Repository inventory

| Area | Path | Size |
|---|---|---|
| Python package (service, engines, training, eval, RAG, storage) | `src/fastapi_assistant/` | 32 modules, 4,895 lines |
| Python tests | `tests/` | 15 files, 1,625 lines |
| Edge Worker (TypeScript) | `deploy/cloudflare-worker/` | 9 source files, 55 tests |
| Web UI (React + Vite) | `web/` | 46 tests |
| TypeScript total (Worker + web source) | | 3,251 lines |
| Deploy targets | `deploy/cloudrun`, `deploy/gcp`, `deploy/hf-space`, `deploy/vm`, `deploy/grafana` | |
| Training on Kaggle | `kaggle/` | |
| Data | `data/fastapi_qa_train.jsonl` (351), `data/fastapi_qa_eval.jsonl` (108) | |
| Evaluation baseline | `eval/baseline.json`, composite 0.410 | |
| Workflows | `ci`, `release`, `eval`, `train`, `uptime`, `codeql`, plus Dependabot | 6 |
| Tracked generated output | `graphify-out/` (658 KB, tracked deliberately in `788c9e0`) | |

151 tracked files in total.

## 2. Baseline checks

| Check | Command | Result |
|---|---|---|
| Python tests + coverage | `python -m pytest --cov` | 93 passed, 2 skipped (Postgres-only, run in CI), line coverage 79% |
| Lint | `ruff check src tests` | clean |
| Format | `ruff format --check src tests` | 47 files clean |
| Type check | `mypy src/fastapi_assistant --ignore-missing-imports` | 29 errors in 8 files (see F-08); not run in CI |
| Worker | `npm run typecheck && npm test` | typecheck clean, 55 passed |
| Web UI | `npm run typecheck && npm test` | typecheck clean, 46 passed |
| Python dependency audit | `pip-audit -r requirements/serve.txt` | 1 advisory: `diskcache` 5.6.3, PYSEC-2026-2447, no fixed version (see F-09) |
| npm audit (Worker, web; prod and dev) | `npm audit` | 0 vulnerabilities |
| Secret scan, full git history | gitleaks in CI with `fetch-depth: 0` | passing on `155e580` |
| Static analysis | CodeQL | passing; two open alerts are intended behaviour (F-15) |
| Container scan | Trivy in the Release workflow | gate passing on `684a10d` (CRITICAL only); 13 HIGH, MEDIUM and LOW image alerts open (F-20), plus 1 false-positive secret alert (F-21) |
| CI on main after the Dependabot floor bumps | `gh run list` | CI success on `155e580` |

Coverage gaps: `storage/postgres.py` 0% locally (covered by the CI Postgres job),
`cli.py` 0%, `__main__.py` 40%, `retrieval.py` 61%, `hub.py` 58%. The torch and
llama.cpp engines are excluded from coverage because they need model weights.

## 3. Live deployment baseline

A scripted end-to-end check against the public edge passed 15 of 15. It reads the API
key from a private file and never prints it.

| Check | Result |
|---|---|
| `GET /` (web UI) | 200, `text/html` |
| `GET /health` via edge | 200, `model_loaded: true`, backend `llamacpp`, 683 ms |
| `GET /metrics` via edge | 404 (not exposed at the edge) |
| `POST /generate` without a key | 401 at the edge |
| `POST /generate` with an invalid key | 401 from the origin |
| `POST /generate` direct to Cloud Run, no edge secret | 403 |
| `GET /metrics` direct to Cloud Run, no token | 401 |
| `POST /generate`, valid key, 64 tokens | 200, 8.6 s (first call), 6.3 s (repeat) |
| Validation error | 422 |
| CORS preflight | 200, `access-control-allow-origin: *` |
| Native SSE stream | 200, 65 events, done event received, TTFT 298 ms, 6.8 s total |
| OpenAI SDK `models.list` | `["fastapi-assistant"]` |
| OpenAI SDK chat completion, 64 tokens | 7.3 s, `finish_reason: length` |
| OpenAI SDK streaming, 48 tokens | 48 chunks, TTFT 1.06 s, 5.5 s total |

Observed decode throughput on Cloud Run (2 vCPU, Q4_K_M) is roughly 8 to 10 tokens per
second for one request. These are single-request figures, not capacity figures.
Section 7 covers what may and may not be claimed about capacity.

## 4. Findings

### P0

None found. Specifically verified:

- the origin rejects direct traffic without the edge secret (`hmac.compare_digest`, 403);
- invalid keys are rejected at the origin;
- `/metrics` needs a token;
- no secret appears in tracked files or git history (gitleaks, full history);
- client errors return fixed messages, not stack traces.

### P1

**F-01. Token-dense input under the character limit returns HTTP 500.**
- Evidence: a 5,800-character prompt of hex tokens sent to `/v1/chat/completions`
  returned `500 Generation failed`, and a 3,990-character question to `/generate`
  did the same. Streaming returned `200`, then an in-band `server_error` event.
- Cause: the only guard is `max_input_chars = 6000` (`api.py`, characters). The
  llama.cpp context is `n_ctx = 2048` tokens, and `Llama.create_chat_completion`
  raises `ValueError` when the prompt exceeds it. Plain English passes (a 15,900-character
  conversation is rejected with 400 by the character check). Code, identifiers and
  non-Latin scripts tokenize at one to two characters per token and slip through.
- Impact: client errors are reported as server errors. They inflate the 5xx rate the
  alerts watch and waste a model slot.
- Fix: count prompt tokens with the loaded tokenizer before generating. Return
  `400 context_length_exceeded` when the prompt does not fit, and clamp `max_tokens` to
  the remaining context.

**F-02. Request bodies are unbounded at the edge and the origin.**
- Evidence: a 2 MB JSON body (`{"question": "hi", "pad": "aaa..."}`) was accepted end to
  end and answered with 200.
- Cause: the Worker reads `await request.arrayBuffer()` with no cap
  (`deploy/cloudflare-worker/src/index.ts:174`); Cloudflare's own ceiling is 100 MB.
  FastAPI reads and parses the body before it resolves the auth dependency, and the
  edge checks only that some key is present. So a caller with any made-up key can make
  the origin buffer and parse very large bodies.
- Fix: reject bodies over 64 KiB with 413 at the edge (declared length and a streaming
  count) and at the origin (ASGI middleware, before parsing).

**F-03. No rate limiting is active in the live deployment.**
- Evidence: there are no `X-RateLimit-*` headers on responses, and `X-Cache: BYPASS`.
  Rate limiting and caching both depend on Upstash, which is not configured.
- Impact: request volume is bounded only by Cloud Run `maxScale: 2` and the origin's
  admission queue. Billing is enabled on the GCP project, so sustained abuse could
  exceed the free request and CPU allowance. The budget alert notifies; it does not cap.
- Fix:
  - add an in-memory, per-isolate fallback limiter that applies when Upstash is absent. It
    is weaker than a shared limiter, because each Cloudflare isolate counts on its own,
    but it is never off.
  - document Upstash as the recommended configuration. It needs a credential only the
    owner can create, so it is BLOCKED here.

**F-04. The live model is the base TinyLlama, not the fine-tuned adapter.**
- Evidence: Cloud Run env `FINETUNED_WEIGHTS=false`, `MODEL_VERSION=base-tinyllama-q4km`.
  The answer to "How do I declare a path parameter in FastAPI?" claimed there is a
  "`path` argument in the `Path` class", which is incorrect.
- Cause: the Kaggle training run has not happened. It needs Kaggle and Hugging Face
  tokens that only the owner can create.
- Impact: the project's central claim (a FastAPI-specialised model) is not yet true in
  production. Retrieval (RAG) can still ground answers per request.
- Fix: BLOCKED on credentials. The training, eval-gate and release pipeline is ready
  (`train.yml`, `eval.yml`, `model-release.json`).

**F-20. The production image ships HIGH-severity CVEs that have fixes.**
- Evidence: open code-scanning alerts #7 to #19, uploaded by the Release Trivy scan of
  the deployed image:
  - **HIGH:** `zlib1g` (Debian, #7 and #8), `wheel` 0.45.1 vendored in setuptools (#19),
    `jaraco.context` 5.3.0 vendored in setuptools (#10).
  - **MEDIUM and LOW:** `pip` 24.0 (#11 to #16), `setuptools` 79.0.1 (#17 and #18).
- Cause:
  - the Trivy gate fails the release only on CRITICAL (`release.yml`, second Trivy
    step), so HIGH findings are reported but shipped;
  - `pip`, `setuptools` and `wheel` come from the `python:3.11-slim` base and are not
    needed at runtime;
  - Debian security updates are not applied in the runtime stage.
- Fix:
  - apply Debian security updates in the runtime stage;
  - uninstall pip, setuptools and wheel from the final image;
  - raise the release gate to fail on HIGH (still ignoring unfixed issues).

**F-21. A false-positive secret alert on the baked RAG index.**
- Evidence: alert #20 (`jwt-token`) at `/models/rag_index/chunks.jsonl:1383`.
- Cause: the FastAPI security tutorial contains an example JWT, and the baked
  documentation index contains that tutorial.
- Fix: skip that directory in the image secret scan, with the reason recorded next to
  the setting. It holds documentation text only, no packages.

### P2

**F-05. 5xx errors on `/v1/*` are labelled `invalid_request_error`.**
- Evidence: the F-01 response body.
- Cause: in the `http_error` handler, any status not in {401, 403, 404} becomes
  `invalid_request_error`.
- Fix: map 5xx to `server_error`, 429 and 503 to `overloaded`, and other 4xx to
  `invalid_request_error`.

**F-06. Cold starts download the 0.7 GB GGUF from Hugging Face.**
- Evidence: `BAKE_MODEL=false` (Dockerfile default; the Release workflow does not set it).
- Impact: Hugging Face availability is a runtime dependency. A cold start pays the download.
- Integrity: downloads are pinned to a Hugging Face commit (`GGUF_REVISION`), but the file
  hash is not checked after download.
- Fix:
  - record the GGUF SHA-256 in `model-release.json` and verify it on load (no cost).
  - baking the weights into the image is the other lever. It would push the image past
    Artifact Registry's 0.5 GB free storage, a small recurring charge, so it is left off
    pending owner approval.

**F-07. No browser security headers; an origin header leaks through the edge.**
- Evidence: the web UI `_headers` sets only `Cache-Control`. API responses carry no
  `X-Content-Type-Options`. `x-cloud-trace-context` from Google's front end is passed to
  clients.
- Fix: add a CSP, `X-Content-Type-Options`, `Referrer-Policy` and `frame-ancestors` for
  the UI; add `nosniff` and `no-store` on API responses; strip the trace header at the edge.

**F-08. Type checking is not enforced.**
- Evidence: mypy reports 29 errors, but none is a confirmed runtime bug:
  - Optional attributes that are filled on load (`self._llm = None`);
  - `Literal` narrowing;
  - a module-level `__getattr__` that mypy cannot follow;
  - a retriever that is `None` only when RAG is disabled (the caller forces it on).
- Fix: type the lazily loaded attributes and the `Literal`s, then add mypy to CI.

**F-09. Dependency audit is not in CI.**
- Current coverage: Trivy scans the image; Dependabot raises PRs.
- Gap: nothing fails a pull request on a known-vulnerable Python or npm dependency.
- Current advisory: `diskcache` 5.6.3 (PYSEC-2026-2447, unsafe deserialisation of cached
  values, no fixed release). It is pulled in by llama-cpp-python for `LlamaDiskCache`. The
  service never constructs a disk cache, so the vulnerable path is unreachable. Accept it
  with a documented ignore and revisit when a fix ships.
- Fix: add a security workflow with pip-audit, npm audit and dependency review on PRs.

**F-10. Edge deploys are manual.**
- Evidence: the Release `deploy-edge` job is skipped, because there is no
  `CLOUDFLARE_ACCOUNT_ID` variable and no `CLOUDFLARE_API_TOKEN` secret.
- Impact: the live Worker was deployed from a workstation, so the Worker version can
  drift from `main`.
- Fix: BLOCKED. The owner creates a scoped API token in the Cloudflare dashboard. The
  workflow and the `EDGE_ORIGINS` variable are already in place.

**F-11. Only one origin is live.**
- Evidence: `EDGE_ORIGINS` lists Cloud Run only. The Hugging Face Space and VM backup
  origins exist as configuration only.
- Impact: failover code is unit-tested but has never run in production.
- Fix: deploying a backup origin is optional. The Hugging Face Space free CPU tier costs
  nothing, but it needs an HF token (BLOCKED). Document it as a known limitation.

**F-12. A dead primary costs every request the full timeout.**
- Evidence: `forwardWithFailover` tries origins in order, and each try may wait
  `ORIGIN_TIMEOUT_MS` (60 s). There is no memory of recent failures.
- Also: when every origin fails at the network level, the edge returns 502, even when the
  failure was a timeout (where 504 is the correct status).
- Fix: add a per-isolate circuit breaker that skips an origin for a cool-down period after
  consecutive network failures, and return 504 when the last failure was a timeout.

**F-13. Queue wait and time-to-first-token are not measured.**
- Evidence: `metrics.py` has request, generation, tokens-per-second, queue depth and
  in-flight metrics. It has no histogram for admission wait or TTFT, which are the two
  numbers that explain user-visible latency under load.
- Fix: add `queue_wait_seconds` and `time_to_first_token_seconds`.

**F-14. Open-source governance files are missing.**
- Missing: CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG, CITATION, MODEL_CARD,
  issue and PR templates, `.editorconfig`, pre-commit.

### P3

**F-15. Two CodeQL alerts are open, both intended behaviour.**
- **#1** `py/clear-text-logging-sensitive-data` at `__main__.py:317`. The
  `keys create` command prints a newly issued key once to the operator's terminal,
  the same pattern as `gh auth token`; the server never logs it.
- **#2** `py/weak-sensitive-data-hashing` at `storage/base.py:25`. API keys are 256-bit
  random values, not passwords, so a fast hash is the right choice: brute force is
  infeasible, and a deliberately slow hash would add latency to every request.
- Dismissal is a repository-admin action reserved for the owner.

**F-16. The probes are public.**
- `/health` and `/ready` are public through the edge and report the backend and model
  version.
- Accepted: the information is low-sensitivity and useful to uptime monitoring.

**F-17. Legacy entry points.**
- Files: `main.py`, `fastapi_assistant/inference.py`, `fastapi_assistant/cli.py`.
- Kept for backward compatibility (`cli.py` backs `python -m fastapi_assistant chat`). Accepted.

**F-18. CORS allows any origin.**
- Accepted: authentication is a bearer or `X-API-Key` header, never a cookie, so a
  wildcard origin does not enable cross-site request forgery.

**F-19. `graphify-out/` (658 KB of generated files) is tracked.**
- This was a deliberate choice in `788c9e0`. Accepted; it is excluded from the Docker
  build context.

## 5. Strengths confirmed

These need no change, and the hardening pass must not regress them:

**Supply chain**
- Hash-pinned serving lock (`pip wheel --require-hashes`).
- Third-party GitHub Actions pinned to commit SHAs.

**Container**
- Non-root user (uid 10001).
- Multi-arch images scanned by Trivy.

**Deployment and secrets**
- Keyless deploys through Workload Identity Federation.
- Runtime secrets live in Secret Manager.
- Cloud Run `maxScale` is capped at 2.
- A GCP budget alert is configured.

**Request handling and auth**
- Admission control with bounded concurrency and queue (429 with `Retry-After`, 503 on
  queue timeout).
- API keys stored as SHA-256 hashes.
- Fixed, non-revealing error messages.

**Model and evaluation**
- Model revisions pinned in `model-release.json`.
- An evaluation gate against a committed baseline.

## 6. Plan for the hardening pass

| Finding | Action in this pass |
|---|---|
| F-01, F-05 | Token-accurate prompt check, `context_length_exceeded`, max-tokens clamp, error type mapping |
| F-02 | 64 KiB body cap at edge and origin |
| F-03 | Fallback in-memory limiter; Upstash documented as recommended (BLOCKED) |
| F-04 | BLOCKED (Kaggle and HF tokens); documented in the model card |
| F-06 | GGUF SHA-256 recorded and verified on load |
| F-07 | Security headers; strip `x-cloud-trace-context` |
| F-08, F-09 | mypy clean and in CI; security workflow |
| F-10, F-11 | BLOCKED (owner credentials); documented |
| F-12 | Circuit breaker; 504 on timeout |
| F-20, F-21 | Patched runtime image, no packaging tools, release gate on HIGH, documented scan exclusion |
| F-13 | Queue-wait and TTFT histograms |
| F-14 | Open-source documentation set |
| F-15 to F-19 | Accepted, reasons above |

## 7. Capacity statement

No load test has been run against the public deployment, and none will be run
uncontrolled. The only measurements so far are single-request figures:

- about 8 to 10 tokens per second;
- one generation at a time per instance (`MAX_CONCURRENCY=1`);
- at most 2 instances.

Until a controlled test says otherwise, the documented capacity is therefore "about two
concurrent generations, with further requests queued up to `MAX_QUEUE` and then
rejected with 429". No figure for supported users is claimed.
