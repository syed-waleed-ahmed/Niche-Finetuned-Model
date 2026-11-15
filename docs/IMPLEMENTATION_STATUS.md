# Implementation status

Last updated: 2026-11-15, after the Qwen model release (#53) was deployed.
Version 2.2.0 (tag `v2.2.0`) covers #13 to #36: since 2.1.0 the service runs as a
public demo, the documentation is published as a website, the web UI was redesigned,
and the first fine-tuned model replaced the base model (#33). The pull requests from
#37 to #53 are unreleased; among them, #44 and #53 replaced the TinyLlama fine-tune
with a fine-tune of Qwen2.5-Coder-1.5B-Instruct.

| PR | Change |
|---|---|
| #53 | Release the Qwen fine-tune, `v2026.11.14-72d6b37`: `model-release.json` and `eval/baseline.json`. Gate 2 scored it 0.6663 against the previous release's 0.5463 |
| #52 | Issues #40 to #43: GitHub's own actions pinned to commit SHAs; the eval report compares each category with the baseline; the web UI keeps the conversation across a page reload and downloads it as Markdown |
| #44 | Train on Qwen2.5-Coder-1.5B-Instruct (LoRA also on the MLP projections, Kaggle batch 4 x 4) with 151 new training rows (351 to 502); stop sequences for both chat templates; the image falls back to the official Qwen GGUF |
| #38 | Abandoned generations free the slot on the live service: the origin is served over HTTP/2 (Hypercorn) so Cloud Run reports disconnects; streams get a disconnect watch; the Worker aborts its origin request when the client leaves |
| #37 | Issues #26 to #29: non-streaming generations stop when the caller disconnects; the edge log identifies the key; Grafana panels and an alert for the 2.1.0 metrics; a JavaScript streaming example. Docs brought up to date with 2.2.0 |
| #36 | Release 2.2.0; Hub model cards with usage, prompt format, a base-model comparison and limitations |
| #35 | Stop answers at the chat template's role markers; drop mkdocstrings directives from the retrieval index; docs for the fine-tuned release |
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
| 6 | Upstash, Neon, backups | Upstash and Neon live (2026-11-14); backups optional | Release run 43625595256 deployed both: a repeated deterministic request is an edge cache hit (`x-cache: HIT`, 0.24 s against 3.9 s for the first one), and the edge smoke test passed with a database-issued key. Backups: see "Optional, owner's choice" |
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
| Model | The fine-tuned release `v2026.11.14-72d6b37` (`finetuned: true`): a LoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct as a Q4_K_M GGUF (986,048,160 bytes), pinned to its Hub tag and SHA-256. Live since 2026-11-15, deployed by Release run 43696171917 after #53; `/health` reports `"model_version":"v2026.11.14-72d6b37"` and `"uses_adapter":true`. In the production eval it scores 0.6663 against 0.5463 for the previous release, `v2026.11.13-2b04940` (a TinyLlama-1.1B-Chat fine-tune, deployed from #33). On Cloud Run it generates about 7.6 tokens/s. Its answers are better but still often wrong ([MODEL_CARD.md](../MODEL_CARD.md)) |
| Access | Public demo (`PUBLIC_DEMO=true`): requests without a key are served, 10 per minute per IP at the edge; `/health` reports `"access": "public"`. A wrong key is still rejected. The static owner API key (Secret Manager) still works, and the origin is locked to the edge with a shared secret |
| Uptime probe | GitHub Actions every 10 minutes through the edge (`EDGE_URL` is set): `/health`, plus the authenticated `/v1/models` |
| Branch protection | "Protect main" ruleset: pull requests and the core CI and security checks required, no force-push or deletion |
| Rate limiting | Live and shared across isolates through Upstash Redis (20 per minute per key, 10 per IP for anonymous calls). A per-isolate counter takes over if Upstash fails (`rl_scope: "local"`) |
| Answer cache | Live (Upstash): an identical deterministic request is answered from the edge, namespaced by `MODEL_VERSION` |
| Per-user keys, usage, feedback storage | Live (Neon Postgres, database `fastapi_assistant`): migrations applied; the CI smoke-test key is a database key |
| Dashboards and alert rules | Live: A Grafana Cloud Metrics Endpoint job scrapes the Cloud Run `/metrics` every minute with `METRICS_TOKEN`; the dashboard (`deploy/grafana/dashboard.json`) and the 7 rules in `deploy/grafana/alerts.yaml` are loaded in the stack as Grafana-managed rules, routed to an email contact point (a test notification was delivered) |
| Error tracking | Off (no Sentry DSN) |
| Backup origins | Not deployed (Hugging Face Space and VM configurations exist) |
| Code scanning and reporting | CodeQL alerts #1 and #2 dismissed as "won't fix" (intended behaviour, F-15); #3 fixed in #14. The 14 Trivy image alerts (#7 to #20) are closed as fixed by the scan of the deployed image. Dependabot alert #1 (`diskcache`) dismissed. Private vulnerability reporting is enabled. |

## Done since 2.1.0

| Item | Evidence |
|---|---|
| Fine-tuned model (audit F-04) | Trained on Kaggle and published to the Hub by CI (Train run 43489041415, `publish_only`): [GGUF](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-gguf) and [LoRA adapter](https://huggingface.co/syed-waleed-ahmed/fastapi-assistant-lora), both tagged `v2026.11.13-2b04940`. It passed both gates: on the Kaggle GPU, 0.538 against the base model's 0.383; in the production eval (GGUF on llama.cpp), 0.546 against 0.402. Merged in #33 and deployed by Release run 43514221696; a keyless question through the live web UI was answered by `v2026.11.13-2b04940` in 5.4 s. |
| Qwen base model (#44, #53) | Train run 43622001406 fine-tuned Qwen2.5-Coder-1.5B-Instruct from commit `72d6b37` on a Kaggle T4 GPU (fp16) with 502 training rows; the GGUF and adapter are on the Hub, tagged `v2026.11.14-72d6b37`. It passed both gates: on Kaggle, 0.6716 against the base Qwen model's 0.5576; in the production eval, 0.6663 against the deployed TinyLlama fine-tune's 0.5463 (keyword recall 0.6852 against 0.4259). Merged in #53 and deployed by Release run 43696171917. Live, two answers took 14.2 s (104 tokens) and 26.3 s (192 tokens); one described a header as a query parameter and another over-complicated a 201 response. |
| Answers that ran on into an invented turn | Seen live after the fine-tune was deployed: with retrieval on, an answer went on past its end with the template's user role marker and a made-up follow-up question, and it copied a `::: fastapi.BackgroundTasks` directive from a retrieved reference page. The engines now stop at the template's role markers, and the index drops mkdocstrings directives (#35). |
| Release pull requests from the Train workflow | "Allow GitHub Actions to create and approve pull requests" is now on (Settings > Actions > General; the default token permission stays read-only). Until then the Train workflow pushed the release branch but could not open the pull request, so #33 was opened by hand. |
| Abandoned requests kept the generation slot (O-2) | Measured live after #37 with `scripts/check_abandoned_request.sh`: a request sent right behind an abandoned 60 s generation waited 59.8 s, and 58.4 s when the abandoned one was streamed, because Cloud Run does not report a client's disconnect to the container over HTTP/1.1. #38 serves the origin over HTTP/2 (ADR-009), adds a disconnect watch for streams and makes the Worker abort its origin request. Measured live after #38 (2026-11-14), the request behind an abandoned generation was answered in 1.3 s (non-streaming) and 2.2 s (streamed, twice), against 59.8 s and 58.4 s after #37; in the web UI, the first token after Stop came in 0.8 s. |

## Optional, owner's choice

Each item needs an account or a decision that only the repository owner can make.
None of them is required for the service to work.

| Item | Why blocked | What to do |
|---|---|---|
| Error tracking | Needs a Sentry project | Set the `SENTRY_DSN` secret |
| Backup origin (F-11) | Needs a Hugging Face Space or a VM | See `deploy/hf-space/` or `deploy/vm/` |
