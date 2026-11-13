# Architecture decisions

Short records of the decisions that shape the system. Each states the context, the
decision, and its consequences, with the files that implement it. Earlier design
notes are in [docs/ARCHITECTURE.md](../ARCHITECTURE.md), section 13; the findings
referenced as F-nn are in [docs/audit/INITIAL_AUDIT.md](../audit/INITIAL_AUDIT.md).

| ADR | Decision | Status |
| --- | --- | --- |
| [ADR-001](#adr-001-keep-the-current-repository-layout) | Keep the current repository layout | Accepted |
| [ADR-002](#adr-002-serve-a-q4_k_m-gguf-with-llamacpp-on-cpu) | Serve a Q4_K_M GGUF with llama.cpp on CPU | Accepted |
| [ADR-003](#adr-003-cloud-run-with-request-based-billing-and-a-maxscale-cap-as-the-primary-origin) | Cloud Run with request-based billing and a `maxScale` cap | Accepted |
| [ADR-004](#adr-004-a-cloudflare-worker-as-the-edge) | A Cloudflare Worker as the edge | Accepted |
| [ADR-005](#adr-005-upstash-and-neon-are-optional) | Upstash and Neon are optional | Accepted |
| [ADR-006](#adr-006-token-accurate-prompt-budgeting-before-admission) | Token-accurate prompt budgeting before admission | Accepted |
| [ADR-007](#adr-007-weights-are-not-baked-into-the-image) | Weights are not baked into the image | Accepted |
| [ADR-008](#adr-008-fall-back-to-in-isolate-rate-limiting-instead-of-failing-open) | Fall back to in-isolate rate limiting instead of failing open | Accepted |

All records: recorded 2026-11-13.

---

## ADR-001: Keep the current repository layout

**Context.** A restructure into a monorepo layout (`apps/`, `packages/`, `ml/`) was
proposed as a target shape, not as a requirement. The current layout already
separates each concern:

| Concern | Current location | Proposed equivalent |
| --- | --- | --- |
| Inference service | `src/fastapi_assistant/` | `apps/api` |
| Edge Worker | `deploy/cloudflare-worker/` | `apps/edge` |
| Web UI | `web/` | `apps/web` |
| Training, data, eval | `kaggle/`, `scripts/`, `data/`, `eval/`, plus training modules in the package | `ml/` |
| Deployment and CI | `deploy/`, `.github/` | `infra/` |

Many paths are load-bearing:

- `deploy/cloudflare-worker/wrangler.toml` serves `../../web/dist`;
- `ci.yml`, `release.yml` and `security.yml` use `deploy/cloudflare-worker` and `web`
  as working directories and lockfile paths;
- `eval.yml` triggers on `src/fastapi_assistant/prompts.py` and
  `src/fastapi_assistant/engines/**`;
- `Dockerfile` copies `src` and `requirements/serve.txt`, and `.dockerignore`
  excludes `deploy`, `web`, `kaggle`, `data` and `eval` from the build context;
- `config.py` locates `data/` and `outputs/` relative to the repository root;
- `kaggle/pipeline.py` calls `scripts/export_gguf.sh`;
- `deploy/vm/cloud-init.yaml` downloads `deploy/vm/*` by URL from `main`;
- `.github/dependabot.yml` lists directories.

**Decision.** Keep the current layout.

Training, evaluation and serving stay in one Python package because they must share
`prompts.py`, `evaluation.py` and `engines/` (invariant 1 in
[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)). Splitting them into `ml/` and
`apps/api` would need a third shared package and would gain no function.

**Consequences.**
- No churn in CI paths, the Docker context, imports, the VM bootstrap URLs or the
  documentation.
- Training modules (`training.py`, `export.py`, `datagen.py`) ship as source in the
  serving package. Their heavy dependencies are optional extras (`[train]`,
  `[datagen]` in `pyproject.toml`), and the production image installs only the
  `serve` lock.
- Revisit if a second deployable service or a second model family is added.

---

## ADR-002: Serve a Q4_K_M GGUF with llama.cpp on CPU

**Context.** The origin runs on small CPU instances (Cloud Run, 2 vCPU, 2 GiB). The
torch + LoRA path needs torch in the image and is slow on CPU. A GPU server costs
money continuously.

**Decision.**
- Merge the adapter into the base model and convert it to GGUF, quantised to
  `Q4_K_M` (about 0.7 GB).
- Serve it with llama-cpp-python (`ENGINE_BACKEND=llamacpp`).
- Keep the torch backend for training-time evaluation, and the `remote` backend for
  a future GPU server.

Implemented in `engines/llamacpp_engine.py`, `scripts/export_gguf.sh` and the
`Dockerfile`.

**Consequences.**
- The image has no torch and no compiler.
- Measured decode is about 8 to 10 tokens/s per request on Cloud Run (audit
  section 3).
- One generation runs at a time per process (an engine lock), so
  `MAX_CONCURRENCY=1`. Capacity grows only with instances or origins.
- The context is 2,048 tokens (`LLAMACPP_N_CTX`), which makes prompt budgeting
  necessary (ADR-006).
- Quantisation changes outputs, so the release gate scores the GGUF on llama.cpp,
  not the fp16 adapter (`eval.yml`).
- llama.cpp is built with `GGML_NATIVE=OFF`, so one image runs on any CPU of its
  architecture. It gives up host-specific CPU tuning.

---

## ADR-003: Cloud Run with request-based billing and a maxScale cap as the primary origin

**Context.** The primary origin needs:
- HTTPS without managing servers;
- scale to zero when idle;
- a hard bound on spend. A GCP budget alert notifies but does not stop spending.

**Decision.**
- Run the service on Cloud Run with `run.googleapis.com/cpu-throttling: "true"`
  (request-based billing) and `minScale: 0`.
- Set `maxScale` from `CLOUD_RUN_MAX_INSTANCES` (default 2), with 2 vCPU, 2 GiB and
  `containerConcurrency: 4`.
- Deploy from Release through Workload Identity Federation
  (`deploy/cloudrun/service.yaml`, `.github/workflows/release.yml`,
  `deploy/gcp/bootstrap.sh`).

**Consequences.**
- Worst-case spend is bounded by `maxScale`. Lowering it to 1 is the emergency
  brake.
- Documented capacity is about two concurrent generations; further requests queue up
  to `MAX_QUEUE` and are then rejected with 429 (audit section 7).
- Cold starts include the GGUF download, hash check and load. The startup probe
  allows up to 240 s. `uptime.yml` keeps one instance warm once `EDGE_URL` is set.
- Scale-out is not instant. While a second instance starts, Cloud Run's front end
  can refuse requests itself with a plain-text 429
  ([docs/audit/PERFORMANCE_BASELINE.md](../audit/PERFORMANCE_BASELINE.md)); the edge
  rewrites it into the API's JSON error with a `Retry-After`.
- Ingress is `all` with `allUsers` as invoker, so the origin secret, not IAM, is what
  blocks direct use (ADR-004).
- The GCP project must have billing enabled, even within the free tier.

---

## ADR-004: A Cloudflare Worker as the edge

**Context.**
- A scaled-to-zero origin should not wake for scanners, unauthenticated calls or
  oversized bodies.
- Clients need one stable hostname for the API and the UI.
- Several origins should be usable behind it.

**Decision.** Put a Worker in front of the origins (`deploy/cloudflare-worker/src/`).
It does, in order:
1. route allowlisting;
2. a key presence check;
3. a 64 KiB body cap;
4. rate limiting;
5. a shared answer cache with a verified-key rule;
6. ordered failover with a per-isolate circuit breaker;
7. origin-secret injection.

The web UI is served from the same Worker as static assets.

**Consequences.**
- Unknown paths and missing keys are answered without waking the origin.
- The edge has no key store, so the origin remains the authority on key validity.
  A cache hit is served only to a key the origin accepted within 300 s
  (`src/cache.ts`).
- One hostname serves the UI and the API, so the UI makes same-origin calls.
- Failover and the circuit breaker are unit-tested but have not run in production,
  because only one origin is configured (F-11).
- Until Release can deploy the Worker (F-10), the live Worker is deployed by hand and
  can drift from `main`.

---

## ADR-005: Upstash and Neon are optional

**Context.** Each extra managed service adds an account, a credential and a failure
mode. The owner has not created the Upstash credentials (F-03), and the service must
work without them.

**Decision.** The system runs with only the edge and one origin.

| Missing piece | Behaviour | Code |
| --- | --- | --- |
| Upstash not configured | Per-isolate rate limiting; cache `BYPASS` | `src/config.ts` (`upstash: null`), `src/index.ts` |
| Upstash failing at runtime | The same fallback, for that request | `src/index.ts` |
| Neon not configured | Static `API_KEY` only; usage events dropped; feedback logged | `storage/__init__.py` `create_store`, `service.py` `record_usage`, `api.py` `feedback` |
| Optional secrets absent | Release stores `disabled`, which `config.py` treats as unset | `release.yml`, `config.py` `_unset_placeholders` |
| No pgvector | The local documentation index is baked into the image (`BAKE_RAG_INDEX=true`) | `Dockerfile`, `retrieval.py` |

**Consequences.**
- A working deployment needs no database or Redis.
- Until they are configured, there is no shared rate limit, no shared cache, no
  per-user keys and no usage or feedback history.
- When Postgres is configured but unreachable, the static key keeps working and
  per-user keys fail closed: the lookup error propagates as a 500 (`auth.py`,
  `storage/postgres.py`).

---

## ADR-006: Token-accurate prompt budgeting before admission

**Context.** The only guard on prompt size was a 6,000-character cap, but the model
context is 2,048 tokens. Code, identifiers and non-Latin text tokenize densely and
slipped through. llama.cpp then raised, which surfaced as a 500, or as a 200
followed by an in-band error on streams, and wasted an admission slot (F-01).

**Decision.**
- `service.fit` runs before `service.admit` and calls `Engine.fit`
  (`engines/base.py`).
- The llama.cpp engine counts tokens with the loaded tokenizer and over-estimates
  template overhead: 8 tokens per message plus 8 (`engines/llamacpp_engine.py`).
- If fewer than 16 tokens remain, the request is rejected with 400 and
  `code: context_length_exceeded`. Otherwise `max_tokens` is clamped to the room
  left.
- A backstop maps llama.cpp's own overflow error to the same exception.
- Rejections are counted in `assistant_prompt_too_long_total`.

**Consequences.**
- Client mistakes are 4xx, not 5xx.
- Streaming clients get the 400 before any 200.
- Oversized prompts never take a slot.
- Tokenising costs some CPU per request, outside the generation lock.
- The deliberate over-estimate can reject a prompt that would have fit by a few
  tokens.
- A request that arrives while the model loads waits in `fit`, because the
  tokenizer needs the loaded model.
- Engines that cannot measure their context (`remote`) skip the check.
- The character cap stays as a cheap first filter.
- CI asserts the 400 on a token-dense prompt (`ci.yml`, `image` job).

---

## ADR-007: Weights are not baked into the image

**Context.** The GGUF is about 0.7 GB, and Artifact Registry's free storage is
0.5 GB (F-06). Baking the weights in would make every stored image exceed it, a
small recurring charge. Downloading at start makes the Hugging Face Hub a runtime
dependency and lengthens cold starts.

**Decision.**
- Keep `BAKE_MODEL=false` (the `Dockerfile` default; Release does not override it).
- Download the GGUF at the revision pinned in `model-release.json` on first load.
- Verify its SHA-256 (`GGUF_SHA256`) before use, failing closed on a mismatch
  (`hub.py` `verify_sha256`).
- `BAKE_MODEL=true` remains available for deployments where cold-start time matters
  more.

**Consequences.**
- The image stays small, and the Artifact Registry cleanup policy keeps two images.
- Every cold start pays the download and the hash check (covered by the startup
  probe).
- A Hub outage blocks new instances, not running ones.
- Integrity comes from the recorded hash rather than the image digest.
- A rollback needs the previous file to still exist on the Hub, which the immutable
  release tags provide.

---

## ADR-008: Fall back to in-isolate rate limiting instead of failing open

**Context.** Rate limiting depended entirely on Upstash. Without it the Worker
applied no limit at all, so load was bounded only by `maxScale` and the origin's
admission queue, on a project with billing enabled (F-03).

**Decision.**
- When Upstash is not configured, or its pipeline call fails, count the same fixed
  one-minute window per identity in the isolate's memory: `LocalRateLimiter` in
  `deploy/cloudflare-worker/src/ratelimit.ts`.
- The limiter holds at most 10,000 identities, evicting old windows first.
- Each request logs `rl_scope` (`redis` or `local`).

**Consequences.**
- Rate limiting is never off, and clients see the same `429` and `X-RateLimit-*`
  headers either way.
- It is weaker than a shared limiter:
  - each Cloudflare isolate counts separately, so the effective limit is a multiple
    of the configured one;
  - counts reset when an isolate is recycled.
- Upstash remains the recommended configuration.
- This supersedes the "Redis unavailable: edge fails open" row in
  [docs/ARCHITECTURE.md](../ARCHITECTURE.md), section 12.
