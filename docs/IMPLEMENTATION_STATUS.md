# Implementation status

Last updated: 2026-11-13. Branch: `feat/production-hardening` (open as a pull request
against `main`; nothing on it is live until it is merged and released).

## Milestones

| # | Milestone | Status | Evidence |
|---|---|---|---|
| 1 | Audit and baseline | Done | [audit/INITIAL_AUDIT.md](audit/INITIAL_AUDIT.md): 21 findings with evidence; baseline test, lint, type, audit and live checks |
| 2 | Architecture decisions | Done | [architecture/DECISIONS.md](architecture/DECISIONS.md): ADRs, including keeping the current layout |
| 3 | Restructure and developer experience | Done, with a documented deviation | The proposed `apps/ packages/ ml/` layout was not adopted (ADR-001). Added: `examples/`, `.editorconfig`, pre-commit, mypy and `make lint` |
| 4 | Backend and inference hardening | Done on branch | Token-accurate context budget (400 instead of 500), 64 KiB body cap, typed errors, security headers, GGUF SHA-256 verification, queue-wait and TTFT metrics |
| 5 | Deploy and end-to-end verification | Done for the current `main` | Cloud Run deployed by Release; Worker connected to the origin; 15 of 15 live checks pass. The branch deploys after merge. |
| 6 | Upstash, Neon, backups | Optional, not configured | The service degrades gracefully without them (ADR-005). Needs owner credentials; see Blocked |
| 7 | Security, observability, performance | Done on branch | [security/SECURITY_AUDIT.md](security/SECURITY_AUDIT.md), [audit/PERFORMANCE_BASELINE.md](audit/PERFORMANCE_BASELINE.md); fallback rate limiter, circuit breaker, platform-error normalisation |
| 8 | CI/CD gates | Done on branch | mypy in CI; `security.yml` (pip-audit, npm audit, dependency review); Trivy gate on HIGH and CRITICAL; image smoke test asserts 400 and 413 against real llama.cpp |
| 9 | Open-source preparation | Done on branch | CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, CHANGELOG, CITATION, MODEL_CARD, license audit, issue and PR templates |
| 10 | Final verification | See [audit/FINAL_VERIFICATION.md](audit/FINAL_VERIFICATION.md) | |

## What is live

| Component | State |
|---|---|
| Edge Worker and web UI | Live at https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev, deployed manually with wrangler |
| Origin | Cloud Run `fastapi-assistant` in us-central1, deployed by the Release workflow from `main`, `maxScale: 2`, scale to zero |
| Model | Base TinyLlama-1.1B-Chat Q4_K_M GGUF, pinned revision (`finetuned: false`) |
| Auth | Static owner API key (Secret Manager), origin locked to the edge with a shared secret |
| Uptime probe | GitHub Actions every 10 minutes through the edge (`EDGE_URL` is set) |
| Rate limiting | None live yet. Upstash is not configured, and the in-isolate fallback ships with this branch |
| Answer cache | Off (needs Upstash) |
| Per-user keys, usage, feedback storage | Off (needs Neon Postgres) |
| Error tracking, dashboards | Off (Sentry DSN and Grafana Cloud not connected) |
| Backup origins | Not deployed (Hugging Face Space and VM configurations exist) |

## Blocked on the owner

Each item needs a credential or an account action that only the repository owner can
take. None of them is required for the service to work.

| Item | Why blocked | What to do |
|---|---|---|
| Fine-tuned model (audit F-04) | Needs Kaggle and Hugging Face tokens | Add the `KAGGLE_USERNAME` and `KAGGLE_KEY` repository secrets (or `KAGGLE_API_TOKEN`) and the `HF_NAMESPACE` variable. Attach an `HF_TOKEN` secret to the kernel on Kaggle. Then run the "Train on Kaggle" workflow |
| CI deploys of the Worker (F-10) | Needs a Cloudflare API token | Create a token with the "Edit Cloudflare Workers" template. Add it as the `CLOUDFLARE_API_TOKEN` secret and the account id as the `CLOUDFLARE_ACCOUNT_ID` variable |
| Shared rate limits and answer cache (F-03) | Needs an Upstash database | Run `bash scripts/setup_secrets.sh` and paste the REST URL and token |
| Per-user keys, usage, feedback | Needs a Neon database | Same script, paste `DATABASE_URL` |
| Backup origin (F-11) | Needs a Hugging Face token or a VM | See `deploy/hf-space/` or `deploy/vm/` |
| CodeQL alerts #1 and #2 (F-15) | Dismissal is an admin action | Dismiss both as "won't fix / intended", with the reasons in the audit |
| Dependabot alert #1 (`diskcache`, F-09) | Dismissal is an admin action; no fixed version exists | Dismiss as "vulnerable code is not actually used" (`LlamaDiskCache` is never constructed) |
| Private vulnerability reporting | A repository setting; it is currently off, so SECURITY.md's reporting button does not appear | Settings, Code security, enable "Private vulnerability reporting" |
| Tag `v2.0.0` | A public release is the owner's decision | Optionally tag `684a10d` (the 2.0.0 merge); CHANGELOG.md already has the section |
| Deploy the hardened Worker | Production change; done after the pull request merges | See the pull request description |
