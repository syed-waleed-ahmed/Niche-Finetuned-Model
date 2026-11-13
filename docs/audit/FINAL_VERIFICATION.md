# Final verification

Date: 2026-11-13.

| Item | Identifier |
|---|---|
| Branch | `feat/production-hardening`, merged into `main` in #13 (the branch has since been deleted) |
| Pull request | #13; follow-ups #14, #15 and #16, also merged (section 7) |
| Commit verified | `05129c5` (branch head at verification); merged as `b146406`. `main` is now at `eced3bb`, the merge of #16 |
| Live deployment verified | Before merge: Cloud Run from `155e580` and Worker `8ec9a2fc`. After merge: Cloud Run from `b146406` and the Worker deployed from `main` (item 18) |

PR #13 was merged and deployed the same day; item 18 records the post-deploy checks.
Section 7 records what changed after the follow-up merges. The evidence in sections 1
to 4 is left as it was recorded.

Statuses:

| Status | Meaning |
|---|---|
| PASS | Verified with the evidence shown |
| PASS WITH LIMITATIONS | Verified, with a stated, accepted gap |
| IN PROGRESS | No longer blocked and under way, but not yet verifiable |
| BLOCKED | Needs something only the repository owner can provide (credentials, account settings, a merge) |
| FAIL | Verified broken |

**No item is FAIL.**

## 1. Code quality and tests

| # | Check | Status | Evidence |
|---|---|---|---|
| 1 | Python tests | PASS | **Local:** 134 passed, 3 skipped (2 need Postgres, 1 needs POSIX bash). **CI:** Python 3.10, 3.11 and 3.12 pass, and the Postgres + pgvector integration job passes. |
| 2 | Lint, format, types | PASS | `ruff check` and `ruff format --check` are clean on `src`, `tests` and `examples`. `mypy` reports no issues in 33 files (the baseline had 29 errors). Both run in CI. |
| 3 | Edge Worker | PASS | `tsc --noEmit` is clean and 75 vitest tests pass (55 at baseline). |
| 4 | Web UI | PASS | Typecheck is clean and 46 tests pass. The build writes `THIRD_PARTY_NOTICES.txt` for 110 production packages. |
| 5 | Examples | PASS | `tests/test_examples.py` runs each example against a real uvicorn server, and the curl example runs on Linux CI. Run manually against the live edge, all three exited 0. |
| 6 | Documentation integrity | PASS | All 12 Mermaid diagrams in 8 files parse with the Mermaid 11 parser. 136 relative Markdown links checked; every one resolves once this file exists. A scan of new files found no secrets, no Cloud Run hostname, no project id and no email addresses. |

## 2. Security and supply chain

| # | Check | Status | Evidence |
|---|---|---|---|
| 7 | Container build and real-model smoke test | PASS | CI run 43417061327 boots the image, downloads the GGUF and logs "Verified sha256 of tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf in 0.4s". It answers with retrieval and sources, and returns 401 without a key. A token-dense prompt returns 400 `{"detail":"The prompt is 3546 tokens; the model's context window is 2048 tokens, including at least 16 for the answer"}`, and a 70 KB body returns 413. |
| 8 | Image vulnerability gate | PASS | The CI Trivy gate now fails on fixable HIGH and CRITICAL CVEs, and it passed on the hardened image. Debian updates are applied, and pip, setuptools and wheel are removed. The Release gate uses the same rule and runs after merge. |
| 9 | Dependency audits | PASS WITH LIMITATIONS | pip-audit on the serving lock passes with one documented exception, PYSEC-2026-2447 (`diskcache`, no fix exists, code path unreachable). npm audit finds 0 vulnerabilities for the Worker and the web UI. Dependency review passes. |
| 10 | Secret scanning | PASS | gitleaks over the full history passes in CI. No secret values were printed or committed during this work, and the API key was read from a private file without being echoed. |
| 11 | Static analysis | PASS WITH LIMITATIONS | CodeQL passes for Python, JavaScript/TypeScript and Actions. Two alerts are open, and both are intended behaviour: #1 prints a new key once in the CLI, and #2 hashes 256-bit random API keys with SHA-256. Dismissing them is an owner action. **Update:** both were dismissed as "won't fix" after the merge (section 7). |
| 12 | Threat model | PASS | `docs/security/SECURITY_AUDIT.md` has a STRIDE analysis with a status for every finding. Five observations raised during documentation were fixed on this branch (O-3, O-6, O-7, O-8, O-9). |
| 13 | Licenses | PASS WITH LIMITATIONS | `docs/legal/LICENSE_AUDIT.md`. Notices now ship in the image, the web bundle and future model folders. Still unverified: the Debian layer inventory, the native libraries in `psycopg-binary`, and the model-provider terms for drafted dataset rows. The dataset is not cleared for standalone publication. |

## 3. Model quality

| # | Check | Status | Evidence |
|---|---|---|---|
| 14 | Eval gate | PASS | The CI eval job ran the 108-item held-out set with llama.cpp in 24 minutes. Composite 0.4023 (keyword recall 0.4321, code validity 0.4904, ROUGE-L 0.2891, latency p50 6.86 s). The allowed regression from the committed 0.410 baseline is 0.02. No eval data or baseline was modified. |
| 15 | Held-out discipline | PASS | Training no longer reads the eval file; checkpoint selection uses a validation slice of the training file. Covered by tests. No model had been trained with the old behaviour. |
| 16 | Fine-tuned model in production | PASS | Was BLOCKED: needed Kaggle and Hugging Face credentials (audit F-04). The live model was the base TinyLlama GGUF, labelled as such everywhere. The clone defect that would have failed the first training run is fixed. **Update:** the first fine-tuned release, `v2026.11.13-2b04940`, passed both gates and is live since #33 (section 7). |

## 4. Deployment

| # | Check | Status | Evidence |
|---|---|---|---|
| 17 | Live stack end to end (current `main`) | PASS | 15 of 15 scripted checks against the public edge. Covered: health, web UI, `/metrics` hidden, 401 without a key, 401 with a bad key, 403 direct to origin, a metrics token required, generate, 422, CORS, native SSE, and the OpenAI SDK list, chat and stream. |
| 18 | Hardened code live | PASS | Merged as `b146406`. Release 43435695506 succeeded: build, a scan gated on HIGH, and the Cloud Run deploy. The Worker was deployed from `main`, with `ORIGIN_TIMEOUT_MS` 150000 and `MAX_REQUEST_BYTES` 65536 confirmed in its bindings. Live afterwards: the original 15 end-to-end checks pass again. The new checks pass too: token-dense prompt gives 400 (3,560 tokens counted by the live model); the streaming variant gives 400 before the stream; a 70 KB body gives 413 at the edge; the fallback rate limit shows `x-ratelimit-limit: 20`; responses carry `nosniff`, `DENY` and `no-referrer`; `x-cloud-trace-context` and the origin's `Server` header are removed (Cloudflare adds its own); the UI is served with a CSP and its notices file; a made-up key gets 401. In Chrome, the UI loads under the CSP with no violations and the health badge reads "base model". |
| 19 | Capacity | PASS WITH LIMITATIONS | Two controlled probes only (`PERFORMANCE_BASELINE.md`). One generation runs at a time per instance, with a maximum of 2 instances. A burst during scale-out drew plain-text 429s from the platform; the edge normalises them on this branch. No user capacity is claimed. |
| 20 | CI-driven edge deploys | PASS | Was BLOCKED on `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` (F-10). **Update:** both are set. In Release run 43460008003 (the merge of #15), the `deploy-edge` and `smoke-edge` jobs succeeded (section 7). |
| 21 | Shared rate limits, cache, per-user keys | BLOCKED | Optional. Needs Upstash and Neon credentials. The service degrades gracefully without them (ADR-005). |
| 22 | Backup origin | BLOCKED | Optional. Needs a Hugging Face token or a VM (F-11). |
| 23 | Observability backends | BLOCKED | Optional. Sentry DSN and Grafana Cloud are not connected. Metrics, JSON logs (now with `severity`) and the uptime probe work without them. |

## 5. Known residual risks

- **Abandoned non-streaming requests:** a caller that disconnects early did not cancel
  its generation, which ran for up to 60 s. **Update:** fixed in #26; the generation
  now stops and frees its slot.
- **Edge key check:** the edge checks only that a key is present. Each made-up key gets
  its own rate-limit bucket, and the origin rejects it only after reading the body.
  That body is now capped at 64 KiB.
- **Fallback rate limiting:** it is per isolate, so the effective limit is a multiple of
  the configured one.
- **Deploy identity:** the deploy service account has project-wide Secret Manager and
  Cloud Run admin roles (O-4). The "Protect main" ruleset now requires pull requests
  and passing checks for `main` and blocks force-pushes, which narrows this; the
  roles themselves are still project-wide.

## 6. Owner actions

Listed with instructions in [../IMPLEMENTATION_STATUS.md](../IMPLEMENTATION_STATUS.md):

- merge #13: done (and the follow-ups #14 to #16);
- create the Cloudflare API token: done (Release deploys the Worker since #15);
- add the training credentials: done (the first fine-tuned release is live since #33);
- dismiss CodeQL #1 and #2 and Dependabot #1: done;
- private vulnerability reporting: done (enabled after the merge);
- apply the branch-protection ruleset to `main` (O-4): done ("Protect main");
- optionally add Upstash, Neon, Sentry, Grafana Cloud and a backup origin: open;
- publish `v2.0.0` at `684a10d`: done; `v2.1.0` and `v2.2.0`: done (both published on
  2026-11-13).

## 7. Update after the merges (2026-11-13)

Recorded after #14, #15 and #16 were merged into `main`. The evidence above is
unchanged; only the statuses of items 16 and 20 moved.

- **Edge deploys (item 20, F-10): resolved.** The `CLOUDFLARE_API_TOKEN` secret and
  the `CLOUDFLARE_ACCOUNT_ID` variable are set. Release run 43460008003 (the merge of
  #15) deployed the Worker in `deploy-edge` and passed the k6 `smoke-edge` job. Every
  Release now deploys the Worker from `main`; a manual `wrangler deploy` is an
  emergency fallback only.
- **Training (item 16, F-04): done.** The `KAGGLE_API_TOKEN` secret, the
  `KAGGLE_USERNAME` and `HF_NAMESPACE` variables, and the `HF_TOKEN` Kaggle secret on
  the `fastapi-assistant-train` kernel are set. The first runs exposed environment
  problems in the Kaggle image: the Kaggle CLI's undeclared `six` dependency (#15),
  and the image's CUDA `torch` being replaced and an incompatible `torchao` blocking
  LoRA (#16). A later run trained `v2026.11.13-2b04940`; CI published it to the Hub
  (Train run 43489041415, `publish_only`, #23). It passed gate 1 on the Kaggle GPU
  (0.538 against the base model's 0.383) and gate 2 in CI with llama.cpp (0.5463
  against 0.4023). Release pull request #33 was merged, and Release run 43514221696
  deployed it: every job succeeded, including the k6 smoke test through the edge.
  Afterwards `/health` through the edge reported `model_version`
  `v2026.11.13-2b04940` and `"uses_adapter": true`, the web UI showed "Fine-tuned",
  and a keyless question was answered in 5.4 s.
- **Code scanning.** CodeQL alerts #1 and #2 were dismissed by the owner as "won't
  fix" (intended behaviour, F-15). Alert #3 (`py/stack-trace-exposure`) was introduced
  and fixed within #14. All 14 Trivy image alerts (#7 to #20) are closed as fixed by
  the scan of the deployed image. Dependabot alert #1 (`diskcache`, PYSEC-2026-2447)
  is dismissed.
- **Private vulnerability reporting** is enabled.
- **Branch protection:** the "Protect main" ruleset is active (pull requests and the
  core CI and security checks required; no force-push or deletion).
- **Releases:** `v2.0.0` is published at `684a10d`, `v2.1.0` at `6518441` and `v2.2.0`
  at `c4d1041`.
- **Still open:** project-wide deployer roles (O-4) and the
  optional services in items 21 to 23.
