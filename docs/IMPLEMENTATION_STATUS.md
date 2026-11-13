# Implementation status

Last updated: 2026-11-13, after pull requests #13 to #16 were merged into `main`. The
hardening work (#13, merge `b146406`) and its follow-ups are live: the Release workflow
deploys both Cloud Run and the edge Worker from `main`. Version 2.1.0 is being
prepared (tag `v2.1.0` to follow).

| PR | Change |
|---|---|
| #13 | Production hardening (milestones 3 to 9 below) |
| #14 | Post-deploy verification, accurate UI copy for the base model, CodeQL alert #3 fix |
| #15 | Train workflow: Kaggle username from a repository variable, the Kaggle CLI's `six` dependency, a `push_only` input. Release uploads only the Worker secrets that are configured. |
| #16 | Kaggle pipeline keeps the GPU image's CUDA `torch`, `torchvision` and `torchaudio` as pip constraints and uninstalls `torchao`. The `torch` floor is back to `>=2.2.0`; Dependabot uses `versioning-strategy: increase-if-necessary`. |

## Milestones

| # | Milestone | Status | Evidence |
|---|---|---|---|
| 1 | Audit and baseline | Done | [audit/INITIAL_AUDIT.md](audit/INITIAL_AUDIT.md): 21 findings with evidence; baseline test, lint, type, audit and live checks |
| 2 | Architecture decisions | Done | [architecture/DECISIONS.md](architecture/DECISIONS.md): ADRs, including keeping the current layout |
| 3 | Restructure and developer experience | Done, with a documented deviation | The proposed `apps/ packages/ ml/` layout was not adopted (ADR-001). Added: `examples/`, `.editorconfig`, pre-commit, mypy and `make lint` |
| 4 | Backend and inference hardening | Done (#13) | Token-accurate context budget (400 instead of 500), 64 KiB body cap, typed errors, security headers, GGUF SHA-256 verification, queue-wait and TTFT metrics |
| 5 | Deploy and end-to-end verification | Done | Cloud Run deployed by Release from `b146406`; the original 15 live checks and 12 hardening checks pass (see the final verification, item 18). Since #15, Release also deploys the Worker and runs the k6 edge smoke test (Release run 43460008003). |
| 6 | Upstash, Neon, backups | Optional, not configured | The service degrades gracefully without them (ADR-005). Needs owner credentials; see "Optional, owner's choice" |
| 7 | Security, observability, performance | Done (#13) | [security/SECURITY_AUDIT.md](security/SECURITY_AUDIT.md), [audit/PERFORMANCE_BASELINE.md](audit/PERFORMANCE_BASELINE.md); fallback rate limiter, circuit breaker, platform-error normalisation |
| 8 | CI/CD gates | Done (#13) | mypy in CI; `security.yml` (pip-audit, npm audit, dependency review); Trivy gate on HIGH and CRITICAL; image smoke test asserts 400 and 413 against real llama.cpp |
| 9 | Open-source preparation | Done (#13) | CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG, CITATION, MODEL_CARD, license audit, issue and PR templates; private vulnerability reporting enabled |
| 10 | Final verification | See [audit/FINAL_VERIFICATION.md](audit/FINAL_VERIFICATION.md) | |

## What is live

| Component | State |
|---|---|
| Edge Worker and web UI | Live at https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev, deployed by the Release workflow from `main` (`deploy-edge`), followed by a k6 smoke test through the edge (`smoke-edge`) |
| Origin | Cloud Run `fastapi-assistant` in us-central1, deployed by the Release workflow from `main`, `maxScale: 2`, scale to zero |
| Model | Base TinyLlama-1.1B-Chat Q4_K_M GGUF, pinned revision (`finetuned: false`). A fine-tune is being trained; see "In progress" |
| Auth | Static owner API key (Secret Manager), origin locked to the edge with a shared secret |
| Uptime probe | GitHub Actions every 10 minutes through the edge (`EDGE_URL` is set): `/health`, plus the authenticated `/v1/models` |
| Branch protection | "Protect main" ruleset: pull requests and the core CI and security checks required, no force-push or deletion |
| Rate limiting | Live, per isolate (20 per minute per key, 10 per IP for anonymous calls). Upstash would make it shared across isolates |
| Answer cache | Off (needs Upstash) |
| Per-user keys, usage, feedback storage | Off (needs Neon Postgres) |
| Error tracking, dashboards | Off (Sentry DSN and Grafana Cloud not connected) |
| Backup origins | Not deployed (Hugging Face Space and VM configurations exist) |
| Code scanning and reporting | CodeQL alerts #1 and #2 dismissed as "won't fix" (intended behaviour, F-15); #3 fixed in #14. The 14 Trivy image alerts (#7 to #20) are closed as fixed by the scan of the deployed image. Dependabot alert #1 (`diskcache`) dismissed. Private vulnerability reporting is enabled. |

## In progress

| Item | State | Next step |
|---|---|---|
| Fine-tuned model (audit F-04) | Credentials are configured: the `KAGGLE_API_TOKEN` secret, the `KAGGLE_USERNAME` and `HF_NAMESPACE` variables, and `HF_TOKEN` attached as a Kaggle secret to the `fastapi-assistant-train` kernel. The first runs exposed environment problems in the Kaggle image, fixed in #15 and #16. A training run is in progress. | The run must pass both eval gates and open a release PR. Until that PR is merged, the live model stays the base GGUF (`finetuned: false`). |

## Blocked on the owner

Each item needs an account setting or a decision that only the repository owner can
make. None of them is required for the service to work.

| Item | Why blocked | What to do |
|---|---|---|
| Release `v2.1.0` | A public release is the owner's decision | Publish `v2.1.0` on the merge commit of the 2.1.0 pull request, with the CHANGELOG.md 2.1.0 section as notes (`v2.0.0` is published) |

### Optional, owner's choice

| Item | Why blocked | What to do |
|---|---|---|
| Shared rate limits and answer cache (F-03) | Needs an Upstash database | Run `bash scripts/setup_secrets.sh` and paste the REST URL and token |
| Per-user keys, usage, feedback | Needs a Neon database | Same script, paste `DATABASE_URL` |
| Error tracking and dashboards | Needs a Sentry project and a Grafana Cloud stack | Set the `SENTRY_DSN` secret; add a Grafana scrape job ([DEPLOYMENT.md](DEPLOYMENT.md) section 7) |
| Backup origin (F-11) | Needs a Hugging Face Space or a VM | See `deploy/hf-space/` or `deploy/vm/` |
