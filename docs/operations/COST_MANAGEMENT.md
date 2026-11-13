# Cost management

The deployment is built to run inside free tiers. This document lists:

- the free-tier limits that matter;
- the guardrails in this repository that keep usage inside them;
- how to watch spend;
- which changes cost money.

Commands assume the session setup in [RUNBOOK.md](RUNBOOK.md) section 2.

**How to read the numbers.**

- **[vendor, re-check]** marks a vendor's free-tier figure as understood in November
  2026. Vendors change these. Confirm each one on the vendor's pricing page before you
  rely on it.
- **Derived** numbers are arithmetic on those figures and on this deployment's
  configuration or measurements.
- No prices are given here. Price any overage on the vendor's page.

**No capacity claim.** Nothing in this document is a claim about how many users the
service supports. The only capacity evidence is the set of one-off probes in
[../audit/PERFORMANCE_BASELINE.md](../audit/PERFORMANCE_BASELINE.md).

## 1. The one hard limit

**`maxScale` is the only setting that caps spend.**

- Cloud Run never runs more than `CLOUD_RUN_MAX_INSTANCES` instances (default 2,
  rendered into `deploy/cloudrun/service.yaml`).
- The GCP budget created by `deploy/gcp/bootstrap.sh` only notifies:
  - default `BUDGET_USD=5`;
  - emails at 50% and 100%;
  - it never stops spending.
- Nothing caps spend on the other vendors except their own free-plan behaviour.

## 2. Free-tier limits that matter

| Service | Free allowance | What uses it here | When exceeded |
|---|---|---|---|
| Cloud Run, request-based billing | 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests per month per billing account [vendor, re-check]; a small monthly outbound data allowance [vendor, re-check] | Every request that reaches the origin: API calls, `/health` from the uptime probe and from open UI tabs, and instance start-up (model download and load) | Billed per unit beyond the allowance. Section 3 has the arithmetic. |
| Cloudflare Workers, Free plan | 100,000 requests per day, 10 ms CPU per invocation [vendor, re-check]. Requests that Workers Static Assets serves (the web UI files) are free and do not count [vendor, re-check]. Workers Logs has a daily event allowance and a few days of retention [vendor, re-check]. | Every API call, `/health` and `/ready` call, and edge-generated 401, 404, 413 or 429 counts as one Worker request | Not billed on the Free plan. Further requests fail until the daily reset [vendor, re-check], and the whole edge goes down with them: API and probe. |
| Artifact Registry | 0.5 GB of storage per month [vendor, re-check] | One amd64 image per Release. The cleanup policy keeps the two most recent versions and deletes older ones after a day. | Billed per GB-month above 0.5 GB |
| Secret Manager | 6 active secret versions and 10,000 access operations per month [vendor, re-check] | 5 secrets (`origin-shared-secret`, `api-key`, `metrics-token`, `database-url`, `sentry-dsn`). Every rotation adds versions. Each new Cloud Run instance reads all 5 at start; each Release reads them to compare. | Billed per version and per 10,000 operations above the allowance |
| Cloud Logging and Monitoring | A monthly log-ingestion allowance per project; Google Cloud's own metrics are free [vendor, re-check] | One JSON line per request plus Cloud Run's request log | Billed per GiB ingested above the allowance |
| Cloud Billing budgets | Free | One budget | |
| GitHub Actions | Standard GitHub-hosted runners, including the `ubuntu-24.04-arm` runner used by Release, are free for public repositories [vendor, re-check] | CI, Security, CodeQL, Release (amd64 and arm64 builds), and Uptime (about 4,320 runs per 30 days) | Free while the repository is public; see section 5 |
| GitHub Container Registry | Free for public packages [vendor, re-check] | Multi-arch images, tagged `sha-<12>` and `latest` | Private packages count against account storage [vendor, re-check] |
| Hugging Face Hub | Free public model repositories [vendor, re-check] | GGUF download at every cold start | Not billed. Availability and download limits are outside our control (see [BACKUP_AND_RECOVERY.md](BACKUP_AND_RECOVERY.md)). |
| Kaggle | A weekly GPU quota, not officially published (about 30 hours per week is commonly cited) [vendor, re-check] | Training, about 1 hour per run. Not running today (BLOCKED on credentials). | The kernel stops |

Optional services. **None is configured today**; check them before enabling.

| Service | Free allowance | Usage pattern in this repository |
|---|---|---|
| Upstash Redis | A monthly command allowance (`docs/DEPLOYMENT.md` cites about 500,000 commands per month) [vendor, re-check] | 0 to 6 commands per Worker request (table in `deploy/cloudflare-worker/README.md`). An HTTP error from Upstash, such as a quota error, makes the Worker fall back to the in-memory limiter, with no cache. |
| Neon Postgres | A storage limit per project (about 0.5 GB) and a monthly compute allowance; compute scales to zero when idle [vendor, re-check] | Key lookups (cached 60 s per instance), usage and feedback inserts. The connection pool keeps no idle connections (`min_size=0`), so an idle origin does not keep the database awake. |
| Sentry | Developer plan event quota [vendor, re-check] | Unhandled errors only; tracing is off (`SENTRY_TRACES_SAMPLE_RATE` defaults to 0) |
| Grafana Cloud | Active-series, log and retention limits (`docs/DEPLOYMENT.md` cites 10k series, 50 GB logs, 14-day retention) [vendor, re-check] | 17 metric families in `metrics.py`, each with a few bounded labels |

## 3. Cloud Run: what consumes the allowance

The configuration (`deploy/cloudrun/service.yaml`) is:

| Setting | Value |
|---|---|
| CPU and memory per instance | 2 vCPU, 2 GiB |
| Billing | Request-based (`run.googleapis.com/cpu-throttling: "true"`) |
| `minScale` | 0 |
| `maxScale` | 2 |
| `containerConcurrency` | 4 |
| `MAX_CONCURRENCY` | 1 |
| `MAX_QUEUE` | 3 |

**What request-based billing charges for** [vendor, re-check]:

- time during which an instance is processing at least one request, rounded up to
  100 ms;
- instance start-up and shutdown time.

An idle instance is not billed. Requests that overlap on one instance are billed once,
for wall-clock time. An instance working through its queue costs the same as one long
request.

Derived from the allowance:

| Quantity | Arithmetic | Result |
|---|---|---|
| Billable instance time covered by the vCPU allowance | 180,000 vCPU-s / 2 vCPU | 90,000 s, about **25 hours** per month |
| Billable instance time covered by the memory allowance | 360,000 GiB-s / 2 GiB | 180,000 s, about 50 hours per month |
| Limiting allowance | | vCPU-seconds |
| Worst case at `maxScale 2`, both instances busy for a 30-day month | 2 x 2 vCPU x 2,592,000 s | 10,368,000 vCPU-s, about **58 times** the free vCPU allowance |
| Worst case at `maxScale 1` | 1 x 2 vCPU x 2,592,000 s | 5,184,000 vCPU-s, about 29 times the free allowance |

Per-request examples come from the single-request measurements in the audit. They are
not averages.

| Request | Measured duration | vCPU-seconds if alone on the instance |
|---|---|---|
| `POST /generate`, 64 tokens, warm | 6.3 to 8.6 s | about 13 to 17 |
| 32-token answers served back to back | about 3 s each | about 6 each |

Other inputs:

- **Cold starts.** Every cold start is billable. It includes downloading and hashing
  the 0.7 GB GGUF and loading it. The log lines `Verified sha256 of ... in N s` and
  `Model ready ... in N s` give the time for each start.
- **Probe traffic.** It is cheap per request but continuous:
  - the uptime probe sends about 4,320 `/health` calls per 30 days (one every
    10 minutes, plus up to 3 retries when a run fails);
  - each UI tab left open sends one `/health` every 30 seconds, about 86,400 calls
    per 30 days.

  Both are far below 2 million requests. Each call can also keep an instance warm or
  start one.

**Long generations.** A non-streaming request with a large `max_new_tokens` (up to
1,024) generates for at most `GENERATION_TIMEOUT_S` (60 s on Cloud Run). The edge waits
150 s, longer than queue plus generation, so that time always produces an answer
rather than a 504. A client that disconnects early from a non-streaming request still
leaves its generation running until it finishes or hits the budget.

## 4. Guardrails in this repository

| Guardrail | Where | What it protects |
|---|---|---|
| `maxScale` from `CLOUD_RUN_MAX_INSTANCES` (default 2) | `service.yaml`, `release.yml` | Hard cap on parallel billable instances |
| `minScale 0` and request-based billing | `service.yaml` | No charge while idle |
| One generation per instance, a queue of 3, and `containerConcurrency 4` | `service.yaml` | Excess load is refused quickly (429 or 503) instead of piling up billable work |
| `QUEUE_TIMEOUT_S=60`, `GENERATION_TIMEOUT_S=60` (sum under the edge's 150 s) | `service.yaml`, `config.py`, `wrangler.toml` | Bound how long one request can hold an instance |
| Budget alert at 50% and 100% of `BUDGET_USD` (default 5 USD) | `deploy/gcp/bootstrap.sh` | Early warning only |
| Edge returns 404 for unknown paths and `/metrics` | `deploy/cloudflare-worker/src/routes.ts` | Scanners never wake a scaled-to-zero origin |
| Edge returns 401 when no key is sent | `deploy/cloudflare-worker/src/index.ts` | Keyless traffic never reaches Cloud Run |
| 64 KiB body cap at the edge and the origin | `body.ts`, `limits.py` | Large bodies are refused before parsing |
| Per-key and per-IP rate limits | `ratelimit.ts` | Bounds a single client. Today this is per isolate (Upstash not configured), so weaker than a shared limit. |
| Prompt token check before admission | `engines/base.py` (`fit`) | Over-long prompts get a 400 without taking a queue slot |
| Weights not baked into the image (`BAKE_MODEL=false`) | `Dockerfile`, Release does not set it | The 0.7 GB GGUF alone exceeds Artifact Registry's 0.5 GB free storage |
| Artifact Registry cleanup policy: keep 2 versions, delete older after a day | `deploy/gcp/bootstrap.sh` | Keeps image storage near the free allowance |
| Secret Manager version added only when a value changed | `release.yml`, "Sync secrets" | Rotations, not deploys, create versions |
| Uptime probe every 10 minutes, not more often | `uptime.yml` | Keeps probe volume to about 4,300 requests per month |
| Build artifacts kept for 1 day | `release.yml` (`retention-days: 1`) | Minimal artifact storage |

## 5. Watching spend

**GCP billing.**

1. Find the billing account and list its budgets:

   ```bash
   BA="$(gcloud billing projects describe "$CLOUDSDK_CORE_PROJECT" --format='value(billingAccountName)')"
   gcloud billing budgets list --billing-account "${BA#billingAccounts/}"
   unset BA
   ```

   `bootstrap.sh` creates the budget with `|| true`. Re-running it can add a second
   budget with the same name, or fail silently. Confirm that exactly one exists.
2. In the Cloud console, open Billing, then Reports. Filter by the project and group by
   service.
3. In Cloud Monitoring, chart the Cloud Run metric
   `run.googleapis.com/container/billable_instance_time` for the service. Billable
   instance time is the quantity in section 3.

**Artifact Registry.** `gcloud artifacts repositories describe fastapi-assistant --location us-central1`
reports the repository size. Compare it with 0.5 GB.

**Secret Manager.** Count enabled versions per secret (RUNBOOK 7.3 explains how to
clean up):

```bash
for s in origin-shared-secret api-key metrics-token database-url sentry-dsn; do
  printf '%s: ' "$s"; gcloud secrets versions list "$s" --filter='state=ENABLED' --format='value(name)' | wc -l
done
```

**Cloudflare.** In the dashboard, Workers & Pages, `fastapi-assistant-edge`, Metrics
shows daily requests, errors and CPU time. Compare daily requests with 100,000.

**GitHub Actions.** Free while the repository is public. If it ever becomes private:

- Actions minutes are metered, with each job rounded up to a minute [vendor, re-check];
- the Uptime workflow alone would use about 4,320 minutes per 30 days, more than the
  minutes included with GitHub Free for private repositories [vendor, re-check].

In that case, lower the probe frequency or move the probe elsewhere first.

## 6. When a budget alert fires

1. **50% alert.** Find the cause:
   - which service (billing report);
   - when (billable instance time);
   - who (Worker logs: one path or identity type dominating, `upstream_normalized`
     bursts).

   Expected causes include a deliberate test, a cold-start pattern, or storage over
   0.5 GB.
2. **100% alert, or a 50% alert with no explanation and usage still rising.** Contain
   first:
   - cap at one instance, or remove public invocation
     ([INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) section 4);
   - then fix the cause: rotate a leaked key, lower edge rate limits, or delete images
     and secret versions.
3. Record what happened. If the budget amount no longer fits, raise `BUDGET_USD`
   deliberately. That is an owner decision.

## 7. Changes that cost money

Each of these needs owner approval, because it moves usage beyond the free tiers or
raises the worst case.

| Change | Benefit | Cost effect |
|---|---|---|
| `minScale 1` | No cold start for the first instance | One instance kept running, and billed, while idle |
| `maxScale` above 2 | More parallel generations during bursts. The measured burst 429s came from Cloud Run scaling out (see PERFORMANCE_BASELINE.md). | Raises the worst case linearly (section 3) |
| Bake weights into the image (`BAKE_MODEL=true`, needs a Release change) | Faster cold starts; no runtime dependency on Hugging Face | Image storage above 0.5 GB |
| More CPU or memory per instance | Faster generation per request | Fewer free billable hours (the 25-hour figure shrinks) |
| Workers Paid plan | Removes the 100,000-requests-per-day ceiling | Monthly plan fee |
| Paid Upstash, Neon, Sentry or Grafana tiers | Higher quotas | Monthly fees |
| GPU inference (`ENGINE_BACKEND=remote` against a GPU server) | Much higher throughput | A GPU server; not a free tier |

Backup origins that are free on paper, the Oracle VM and the Hugging Face Space (audit
F-11), add no Cloud Run cost. They have their own limits and are not deployed.
