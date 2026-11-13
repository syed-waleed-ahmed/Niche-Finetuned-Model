# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Version numbers are the `version` field in `pyproject.toml`.

The deployed model is versioned separately, by `model_version` in
`model-release.json`. The deployed model is the first fine-tuned release,
`v2026.11.13-2b04940` (`finetuned: true`), which replaced the base TinyLlama
(`base-tinyllama-q4km`) in #33. See [MODEL_CARD.md](MODEL_CARD.md).

## [Unreleased]

### Added

- Edge Worker: the request log records `rl_key_id`, the first 12 hex characters of
  the SHA-256 of the presented API key, so requests can be attributed to a key.
  The key itself and the caller's IP are still never logged (#27).
- Grafana dashboard: panels for queue wait (p50 and p95), time to first token by
  endpoint, and prompts rejected for length. A new alert rule, `AssistantLongQueueWait`,
  fires when the p95 wait for a generation slot stays above 20 s. `tests/test_grafana.py`
  checks every dashboard and alert query against the metrics the service exports (#28).

### Fixed

- A non-streaming request (`POST /generate`, or `POST /v1/chat/completions` without
  `stream`) stops generating when its caller disconnects, as streamed requests
  already did. Until now it kept the only generation slot busy for up to
  `GENERATION_TIMEOUT_S`. `Engine.generate` takes an optional `cancel` event; the
  remote engine honours it only before the request is sent (#26).

## [2.2.0] - 2026-11-13

The first fine-tuned model in production, a public demo that needs no key, a
documentation website, and a redesigned web UI. Merged in #23 to #36.

### Added

- A dev container (`.devcontainer/`) for GitHub Codespaces and VS Code Dev Containers:
  Python 3.11 with CPU-only torch, Node.js 22, the test dependencies and the
  pre-commit hooks. Dependabot keeps its features up to date.
- `ROADMAP.md`, GitHub Discussions for questions and ideas, starter issues labelled
  `good first issue`, and a social-preview image (`docs/assets/social-preview.png`).
- Public demo mode. With the repository variable `PUBLIC_DEMO=true`, Release deploys
  the Worker without a key requirement and the origin with `PUBLIC_DEMO=true`: requests
  without a key are served, rate limited per IP at the edge, while a wrong key is
  still rejected. `GET /health` reports `access` (`public` or `key`) (#24).
- A documentation site built with MkDocs Material and published to GitHub Pages
  from `main`, with a strict build that fails on broken links and anchors (#30).
- A redesigned web UI: a welcome screen with the model's status and example
  questions, onboarding driven by `/health` `access`, light and dark themes, a drawer
  on narrow screens, code blocks with copy buttons, and WCAG AA contrast (#32).
- `fastapi-assistant export publish-outputs`, and a `publish_only` input on the Train
  workflow: CI can publish a finished Kaggle run to the Hugging Face Hub when the
  kernel could not (#23).

### Changed

- The served model is the first fine-tuned release, `v2026.11.13-2b04940`: a LoRA
  fine-tune of TinyLlama-1.1B-Chat, merged and quantized to Q4_K_M, published on the
  Hugging Face Hub. On the held-out set it scores 0.546 against the base model's
  0.402 (code validity 0.846 against 0.490), scored with the production backend in
  the same run. `eval/baseline.json` now holds its scores (#33).
- The live deployment runs as a public demo, with `CLOUD_RUN_MAX_INSTANCES=1` to
  bound the worst-case cost of keyless traffic.
- When the Train workflow cannot open the release pull request, it now says which
  repository setting to enable and links to open the pull request from the pushed
  branch.
- The model cards published to the Hugging Face Hub link to the source, the demo and
  the documentation, show how to load the exact release with llama.cpp or `peft`,
  give the system prompt and the stop sequences, compare the scores with the base
  model's, and list the limitations. `export push` gains `--base-report`, and the
  Kaggle pipeline passes the base model's report (#36).
- `scripts/setup_secrets.sh --storage-only` adds Upstash or Neon to a working
  deployment without rotating the origin secret or the API key, which a full run
  does (#36).

### Fixed

- Answers no longer run on into an invented next turn. The chat template's role
  markers (`<|user|>`, `<|system|>`, `<|assistant|>`) are plain text to TinyLlama, so
  when the model skipped end-of-sequence it wrote a made-up follow-up question. The
  llama.cpp and remote engines now stop at them (`prompts.STOP_SEQUENCES`) (#35).
- The retrieval index no longer contains the mkdocstrings directives of the FastAPI
  reference pages (`::: fastapi.BackgroundTasks` and their options blocks), which the
  model copied into answers. The index has 1,524 chunks instead of 1,534 (#35).

## [2.1.0] - 2026-11-13

Production hardening, driven by a baseline audit (`docs/audit/INITIAL_AUDIT.md`):
request limits and token-accurate prompt budgeting, a more resilient edge, model
integrity checks, stricter supply-chain gates, a working training pipeline, and the
documentation and community files of an open-source project. Merged in #13 to #16.

### Added

- Prometheus histograms for admission queue wait and time to first token, and a
  counter for rejected prompts.
- `gguf.sha256` in `model-release.json`. The llama.cpp engine verifies the downloaded
  GGUF against it on load and fails closed on a mismatch. The Kaggle pipeline records
  the hash for new releases.
- Edge Worker: an in-isolate fallback rate limiter that applies when Upstash is not
  configured or is failing (previously rate limiting was off in that case).
- Edge Worker: a per-isolate circuit breaker over origins.
- Security workflow (`.github/workflows/security.yml`): `pip-audit` on the serving
  lock, `npm audit` for the Worker and the web UI, and dependency review on pull
  requests. It also runs weekly.
- mypy configuration in `pyproject.toml`; mypy runs in CI and in `make lint`.
- `examples/`: the OpenAI SDK (blocking and streaming), the native `/generate` API
  with Server-Sent Events and sources, and curl for every endpoint.
  `tests/test_examples.py` runs each one against a real uvicorn server; `openai` is
  added to the `dev` extra for that test.
- Documentation: `docs/audit/INITIAL_AUDIT.md` (the baseline audit, with evidence for
  every finding), `docs/audit/PERFORMANCE_BASELINE.md` (measured single-request
  latency and two controlled concurrency probes), `docs/architecture/` (system,
  request lifecycle, deployment, security and model pipeline documents, and
  architecture decision records ADR-001 to ADR-008), and
  `docs/IMPLEMENTATION_STATUS.md`.
- Community and release files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`
  (Contributor Covenant 2.1 by reference), `SECURITY.md`, this changelog,
  `CITATION.cff`, `MODEL_CARD.md`, `docs/legal/LICENSE_AUDIT.md`, issue and pull
  request templates, `.editorconfig` and `.pre-commit-config.yaml`.
- Operations and security documentation: `docs/operations/` (runbook, incident
  response, rollback, cost management, backup and recovery) and
  `docs/security/SECURITY_AUDIT.md` (STRIDE threat model).
- License notices in shipped artifacts: `THIRD_PARTY_NOTICES.md` (llama.cpp, the
  FastAPI documentation index, the embedding model) is copied into the image with
  `LICENSE`; the web build writes `THIRD_PARTY_NOTICES.txt` for its npm dependencies;
  published model folders get an Apache-2.0 attribution section and a `NOTICE` file.
- The uptime workflow also calls the authenticated `/v1/models` through the edge, so
  an origin-secret or key mismatch is detected, not only a dead origin.
- Continuous delivery of the edge: with `CLOUDFLARE_API_TOKEN` and
  `CLOUDFLARE_ACCOUNT_ID` configured, the Release workflow deploys the Worker and runs
  the k6 smoke test after every Cloud Run deploy.
- Train workflow: a `push_only` input creates the Kaggle kernel without running it, so
  the `HF_TOKEN` secret can be attached before the first training run. The Kaggle
  username is read from a repository variable, and release pull requests carry the
  `model-release` label.
- `SUPPORT.md`, `.github/CODEOWNERS`, and `.github/release.yml` (categories for
  generated release notes). The issue chooser links to private vulnerability
  reporting and to SUPPORT.md.
- `.env.example` documents every setting, including the 22 that were missing, among
  them `GGUF_SHA256`, `MAX_REQUEST_BYTES` and `VALIDATION_FRACTION`. A test fails when
  a new setting is added without an entry, and another loads the file to check that
  its values are valid.
- README: CI, release, security and CodeQL badges, quick links, a screenshot of the
  live UI, a table showing which services the live deployment uses, and sections on
  contributing, security reporting, citation and acknowledgements.
- Tests for the hardening changes: Python 93 to 134 passing (plus 2 Postgres-only
  tests and 1 POSIX-only test), Worker 55 to 75.

### Changed

- `max_tokens` is clamped to the room left in the model's context window.
- Edge Worker: returns 504 instead of 502 when the last origin attempt timed out.
- Edge Worker: `ORIGINS` defaults to empty, so an unconfigured Worker fails closed
  instead of forwarding to placeholder hosts.
- Edge Worker: a non-JSON 429, 502, 503 or 504 from the origin's platform is turned
  into the JSON error shape with a `Retry-After` hint, so clients see one error
  contract.
- `ruff check` and `ruff format --check` cover `examples/` in CI and in the Makefile.
- Timeout budget: on Cloud Run, queue wait (60 s) plus generation (60 s) now stays
  under the edge's `ORIGIN_TIMEOUT_MS` (raised from 60 to 150 s), so the edge no longer
  returns 504 while the origin keeps generating an answer nobody will receive.
- Kaggle gate 1 requires the fine-tune to beat the base model by at least 0.005
  (`min_improvement`); a tie no longer passes.
- JSON logs carry a `severity` field, which Cloud Logging maps to the entry severity.
- README: the deployed base-model status, measured throughput, a full error table, the
  new settings (`GGUF_SHA256`, `LLAMACPP_N_CTX`, `MAX_REQUEST_BYTES`), and a
  documentation index. The unsupported claim of "hundreds to a few thousand answers
  per day" is replaced by the measured capacity statement, in the README and in
  `docs/DEPLOYMENT.md`.
- Corrected comments in `api.py` (middleware order) and `deploy/gcp/bootstrap.sh`, and
  stale lines in `docs/ARCHITECTURE.md`.
- CI and Release Trivy gates fail on fixable HIGH as well as CRITICAL CVEs.
- Container image: the runtime stage applies Debian security updates, and pip,
  setuptools and wheel are removed from the final image.
- `.dockerignore` keeps `eval/`, tooling and local entry points out of the build
  context; `.gitignore` ignores per-run eval reports; the README lists `main.py`, the
  serving lock, the Docker files and `graphify-out/` in the project structure.
- Dependency floors raised by Dependabot: `transformers>=5.20.2`, `peft>=0.21.2`,
  `sentencepiece>=0.2.3`, `datasets>=5.2.0`, `pytest>=9.2.0`, `pytest-cov>=7.1.0`, and
  the build requirement `setuptools>=84.2.1`. The `torch` floor was raised to 2.14.1
  and then restored to 2.2.0 (see Fixed). Dependabot now raises minimum versions only
  when they stop allowing the latest release (`versioning-strategy:
  increase-if-necessary`).
- Dependabot updates: `highlight.js` 11.12.0 (web UI); `wrangler` 4.159.0,
  `vitest` 5.1.0 and `@cloudflare/workers-types` 5.20261110.1 (Worker tooling);
  `grafana/alloy` v1.21.3 (`deploy/vm`).

### Fixed

- The web UI no longer describes the model as fine-tuned while the base model is
  serving; the health badge states which one is live.
- A prompt under the character limit but over the llama.cpp context window (2,048
  tokens by default) returned HTTP 500 (or, when streaming, an in-band error after a
  200). The prompt is now counted
  with the loaded tokenizer before admission and rejected with 400
  `context_length_exceeded`.
- 5xx errors under `/v1` are typed `server_error` instead of `invalid_request_error`.
- 29 mypy errors resolved.
- `kaggle/pipeline.py` cloned with `git clone --branch <ref>`, which rejects the commit
  SHA that the Train workflow passes, so the first training run would have failed at
  the clone. It now runs `git init`, a shallow `git fetch` of the ref, and a checkout
  of `FETCH_HEAD`.
- Training no longer reads the held-out eval file. Checkpoint selection used it as the
  trainer's evaluation split (`load_best_model_at_end` on its loss), which leaked the
  benchmark into model selection. A seeded validation slice of the training file
  (`VALIDATION_FRACTION`, default 0.1) is used instead. No fine-tune had been trained
  with the old behaviour.
- The `eval.yml` error for a changed eval set now says how to regenerate the baseline.
- The package reported `__version__` 1.0.0 (and so did the OpenAPI document) while
  `pyproject.toml` said 2.0.0; both now read 2.1.0, and a test keeps them equal.
- The first Kaggle training runs failed before training: a raised `torch` floor made
  the training install replace the image's CUDA torch (breaking its torchvision), and
  the image's old `torchao` made peft refuse to apply LoRA. The pipeline now keeps the
  preinstalled torch, torchvision and torchaudio as pip constraints and removes
  `torchao`; the Kaggle CLI is installed with its undeclared `six` dependency.
- The Release workflow's edge deploy failed when the optional Upstash secrets were
  absent; only configured secrets are uploaded to the Worker now.
- Streams report a mid-generation context overflow with fixed text, so no exception
  text reaches a client (CodeQL `py/stack-trace-exposure`).

### Security

- Request bodies are capped at 64 KiB (413), at the origin before parsing and
  authentication, and at the edge with a streaming byte counter.
- API responses carry `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY` and
  `Referrer-Policy: no-referrer`. The edge strips origin infrastructure headers and
  adds `nosniff` to every response.
- The web UI is served with a strict Content-Security-Policy and related headers.
- Without a database, feedback logging records the rating and the comment length,
  not the user's comment text.
- Edge: when Upstash answers but the `INCR` reply is an error, the request is counted
  by the in-isolate limiter instead of passing unlimited.
- `API_KEY` set to a placeholder (`disabled`, `none`, `null`, `change-me`) is rejected
  at startup instead of being read as "unset", which silently disabled authentication
  when no database was configured. An origin with no authentication at all now logs
  a warning at startup.

## [2.0.0] - 2026-11-13

The production scale-out, merged to `main` in pull request #1 (`684a10d`). This
version was set in `pyproject.toml` but not tagged in git.

### Added

- Dataset grown from 32 train / 8 eval to 351 train / 108 eval examples. Eval rows
  carry an `id`, a `category`, and `must_include` identifiers; CI checks that the eval
  set does not leak into the training set.
- An `Engine` contract with three backends: llama.cpp GGUF (production CPU serving),
  torch with a LoRA adapter, and remote OpenAI-compatible servers (vLLM, TGI).
  Models load once in the background, with deadlines and cancellation.
- Admission control: bounded concurrency and a bounded queue; overload returns 429
  with `Retry-After`.
- OpenAI-compatible `/v1/chat/completions` and `/v1/models`, Server-Sent Events
  streaming on both APIs, `/v1/feedback`, an origin lock (`X-Origin-Secret`), and
  `PORT` support.
- Per-user hashed API keys, usage and feedback in Postgres (connection pool, SQL
  migrations), with an in-memory store for development and tests.
- Retrieval over the FastAPI documentation (local NumPy index or pgvector), opt-in per
  request.
- Prometheus metrics with a per-app registry; optional Sentry.
- Evaluation harness and release gate, with the recorded baseline in
  `eval/baseline.json` and `eval/baseline-rag.json`.
- `fastapi-assistant data generate`: drafts training pairs from the FastAPI docs with
  Claude, for human review.
- `model-release.json`, which pins the deployed weights; merge and publish tooling.
- A production image that builds llama.cpp portably, ships without torch, and installs
  from a hash-pinned lock (`requirements/serve.txt`).
- A Cloudflare Worker edge: key checks, per-key and per-IP rate limits, a shared answer
  cache in Upstash, ordered origin failover, and the origin lock.
- Deployment targets: Cloud Run (scale to zero, `maxScale` cost cap), a GCP bootstrap
  with Workload Identity Federation, an Oracle VM origin behind a Cloudflare Tunnel
  with Grafana Alloy, and an optional Hugging Face Space origin.
- A Grafana dashboard, alert rules, and a k6 load test.
- A Kaggle GPU pipeline: train, gate against the base model, merge, convert to GGUF,
  publish to the Hugging Face Hub, and write the release manifest.
- Workflows: CI (Python 3.10 to 3.12, Postgres with pgvector, Worker, gitleaks, image
  boot and smoke test, Trivy), multi-architecture release with gated deploys, the eval
  gate on the production backend, Kaggle training, uptime checks, CodeQL, and
  Dependabot. Third-party actions are pinned to commit SHAs.
- A React and TypeScript web playground in `web/`, served by the Worker as static
  assets.
- `scripts/setup_secrets.sh`, with a non-interactive mode, which generates and stores
  deployment secrets without printing them.
- `docs/DEPLOYMENT.md`, the deployment runbook.

### Changed

- README and architecture document rewritten for the multi-origin system;
  `ARCHITECTURE.md` moved to `docs/`, and the k6 test to `tests/load/`.
- The Release workflow copies GitHub secrets into Google Secret Manager, adding a
  version only when a value changed. `deploy/gcp/bootstrap.sh` no longer generates or
  prints secrets.
- Every secret pasted into the setup script is optional; a `disabled` placeholder is
  treated as unset, and a non-URL `SENTRY_DSN` leaves Sentry off.

### Fixed

- Client errors return fixed messages instead of exception text in `/ready`,
  `/generate` (JSON and SSE) and `/v1` errors (CodeQL `py/stack-trace-exposure`).
- Worker: trailing slashes are trimmed with a linear loop instead of a regular
  expression (CodeQL `js/polynomial-redos`).
- The `anthropic` SDK import is optional, so CI passes without the `datagen` extra.
- Training fixes for transformers 5.x.

## [1.0.0] - 2026-07-05

Tag `v1.0.0` (commit `0c6094c`). The GitHub release "v1.0.0: tested, containerized
FastAPI model service" was published on 2026-11-13.

### Added

- An installable `src`-layout package, `fastapi_assistant`, with settings from
  environment variables (pydantic-settings), structured JSON logging, a thread-safe
  engine, a FastAPI app factory, schemas, and a CLI.
- HTTP API: `/health`, `/ready`, and `/generate`, with request-id and timing
  middleware, optional `X-API-Key` authentication, CORS, and bounded generation
  parameters.
- A pytest suite that uses a fake engine (no model or network), ruff, GitHub Actions
  CI on Python 3.10 and 3.11, a non-root Dockerfile with a health check, a Makefile,
  the MIT license, and `ARCHITECTURE.md`.
- Dataset grown from 5 train / 2 eval to 32 train / 8 eval question-answer pairs.
- A compatibility shim for the `torch_dtype` to `dtype` rename in transformers 5.x.
- `.gitattributes` that normalises text files to LF; the `graphify-out/` knowledge
  graph outputs are tracked.

### Fixed

- Inference loaded the adapter-only directory with `AutoModelForCausalLM`. It now
  loads the base model and applies the LoRA adapter with `peft`.
- Training and inference share the tokenizer's native chat template (`prompts.py`).
- Training masks the loss to answer tokens (completion-only loss) and pads
  dynamically.

### Removed

- Training outputs (the LoRA adapter and checkpoint in `outputs/`) are no longer
  tracked in git; the adapter is regenerated with `make train`.

## Prototype (untagged) - 2025-11-17

The original project, before any version number: scripts that fine-tuned
TinyLlama-1.1B-Chat with LoRA (`transformers` and `peft`) on 5 training and 2 eval
question-answer pairs, an interactive CLI (`main.py`), and the resulting adapter
committed under `outputs/` (a 2-step training run). A Gradio `app.py` for Hugging Face
hosting was added and removed the same day.

[Unreleased]: https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/compare/v2.2.0...HEAD
[2.2.0]: https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/compare/v2.0.0...v2.1.0
[2.0.0]: https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/syed-waleed-ahmed/Niche-Finetuned-Model/releases/tag/v1.0.0
