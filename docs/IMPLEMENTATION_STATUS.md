# Implementation status

Last updated: 2026-11-13, after pull requests #13 to #34 were merged into `main`.
Version 2.1.0 is released (tag `v2.1.0`). Since then the service runs as a public
demo, the documentation is published as a website, the web UI was redesigned, and
the first fine-tuned model replaced the base model (#33).

| PR | Change |
|---|---|
| #34 | Documentation of the public demo, the docs site and the fine-tuned release; clearer Train workflow failure when Actions may not open pull requests |
| #33 | Release the first fine-tuned model, `v2026.11.13-2b04940` |
| #32 | Web UI redesign: welcome screen, access onboarding, themes, mobile drawer, WCAG AA contrast |
| #30 | Documentation website (MkDocs Material on GitHub Pages) |
| #25 | Dev container, ROADMAP.md, Discussions links, social preview |
| #24 | Opt-in public demo mode (`PUBLIC_DEMO`), `access` in `/health` |
| #23 | Publish a finished Kaggle run to the Hub from CI (`publish_only`) |
| #20 to #22 | Release 2.1.0; Dependabot holds `huggingface-hub` at 1.x for fastembed; the Kaggle pipeline imports the package in its own process |
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
| Edge Worker and web UI | Live at https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev (the redesigned UI from #32), deployed by the Release workflow from `main` (`deploy-edge`), followed by a k6 smoke test through the edge (`smoke-edge`) |
| Documentation website | Live at https://syed-waleed-ahmed.github.io/Niche-Finetuned-Model/, deployed by the Docs workflow from `main` |
| Origin | Cloud Run `fastapi-assistant` in us-central1, deployed by the Release workflow from `main`, `maxScale: 1` (`CLOUD_RUN_MAX_INSTANCES=1`, to bound the cost of keyless traffic), scale to zero |
| Model | The fine-tuned release `v2026.11.13-2b04940` (`finetuned: true`): a LoRA fine-tune of TinyLlama-1.1B-Chat as a Q4_K_M GGUF, pinned to its Hub tag and SHA-256. Deployed by Release run 43514221696 (the merge of #33); `/health` reports `"uses_adapter": true` and the web UI "Fine-tuned". On the held-out set it scores 0.546 against the base model's 0.402 ([MODEL_CARD.md](../MODEL_CARD.md)) |
| Access | Public demo (`PUBLIC_DEMO=true`): requests without a key are served, 10 per minute per IP at the edge; `/health` reports `"access": "public"`. A wrong key is still rejected. The static owner API key (Secret Manager) still works, and the origin is locked to the edge with a shared secret |
| Uptime probe | GitHub Actions every 10 minutes through the edge (`EDGE_URL` is set): `/health`, plus the authenticated `/v1/models` |
| Branch protection | "Protect main" ruleset: pull requests and the core CI and security checks required, no force-push or deletion |
| Rate limiting | Live, per isolate (20 per minute per key, 10 per IP for anonymous calls). Upstash would make it shared across isolates |
| Answer cache | Off (needs Upstash) |
| Per-user keys, usage, feedback storage | Off (needs Neon Postgres) |
| Error tracking, dashboards | Off (Sentry DSN and Grafana Cloud not connected) |
| Backup origins | Not deployed (Hugging Face Space and VM configurations exist) |
| Code scanning and reporting | CodeQL alerts #1 and #2 dismissed as "won't fix" (intended behaviour, F-15); #3 fixed in #14. The 14 Trivy image alerts (#7 to #20) are closed as fixed by the scan of the deployed image. Dependabot alert #1 (`diskcache`) dismissed. Private vulnerability reporting is enabled. |

## Done since 2.1.0

| Item | Evidence |
|---|---|
| Fine-tuned model (audit F-04) | Trained on Kaggle and published to the Hub by CI (Train run 43489041415, `publish_only`): [GGUF](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-gguf) and [LoRA adapter](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-lora), both tagged `v2026.11.13-2b04940`. It passed both gates: on the Kaggle GPU, 0.538 against the base model's 0.383; in the production eval (GGUF on llama.cpp), 0.546 against 0.402. Merged in #33 and deployed by Release run 43514221696; a keyless question through the live web UI was answered by `v2026.11.13-2b04940` in 5.4 s. |
| Answers that ran on into an invented turn | Seen live after the fine-tune was deployed: with retrieval on, an answer went on past its end with the template's user role marker and a made-up follow-up question, and it copied a `::: fastapi.BackgroundTasks` directive from a retrieved reference page. The engines now stop at the template's role markers, and the index drops mkdocstrings directives (CHANGELOG, Unreleased, Fixed). |

## Blocked on the owner

Each item needs an account setting or a decision that only the repository owner can
make. None of them is required for the service to work.

| Item | Why blocked | What to do |
|---|---|---|
| Release pull requests from the Train workflow | The repository does not allow GitHub Actions to create pull requests, so the Train workflow pushes the release branch but cannot open the pull request (#33 was opened by hand) | Settings > Actions > General > "Allow GitHub Actions to create and approve pull requests" |

### Optional, owner's choice

| Item | Why blocked | What to do |
|---|---|---|
| Shared rate limits and answer cache (F-03) | Needs an Upstash database | Run `bash scripts/setup_secrets.sh` and paste the REST URL and token |
| Per-user keys, usage, feedback | Needs a Neon database | Same script, paste `DATABASE_URL` |
| Error tracking and dashboards | Needs a Sentry project and a Grafana Cloud stack | Set the `SENTRY_DSN` secret; add a Grafana scrape job ([DEPLOYMENT.md](DEPLOYMENT.md) section 7) |
| Backup origin (F-11) | Needs a Hugging Face Space or a VM | See `deploy/hf-space/` or `deploy/vm/` |
