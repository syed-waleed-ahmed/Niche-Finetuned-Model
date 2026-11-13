# Runbook

Day-to-day operation of the FastAPI assistant: how to check that it works, where the
logs and metrics are, and how to do the routine tasks. For outages use
[INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md); to undo a change use
[ROLLBACK.md](ROLLBACK.md).

## 1. What is running

| Layer | What | Where it is defined |
|---|---|---|
| Edge | Cloudflare Worker `fastapi-assistant-edge`, which also serves the web UI from `web/dist` | `deploy/cloudflare-worker/`, `web/` |
| Origin | Cloud Run service `fastapi-assistant` in `us-central1`: llama.cpp on CPU, 2 vCPU and 2 GiB per instance, `minScale 0`, `maxScale` from the `CLOUD_RUN_MAX_INSTANCES` variable (1 on the live public demo) | `deploy/cloudrun/service.yaml`, `.github/workflows/release.yml` |
| Model | The fine-tuned release `v2026.11.13-2b04940` (TinyLlama 1.1B Chat with a merged LoRA fine-tune), Q4_K_M GGUF, downloaded from the Hugging Face Hub at its release tag and checked against a SHA-256. To roll back to the base model, revert release pull request #33 | `model-release.json` |
| Uptime probe | GitHub Actions every 10 minutes through the edge: `GET /health`, then the authenticated `GET /v1/models` when `SMOKE_API_KEY` is set | `.github/workflows/uptime.yml` |
| Secrets | GitHub Actions secrets, copied into Google Secret Manager by Release; one Worker secret | `scripts/setup_secrets.sh` |

**Not configured today.** Each of these changes what you can observe:

- **Upstash Redis:** no edge answer cache (`X-Cache: BYPASS`). Rate limits use a
  per-isolate in-memory counter (`rl_scope: "local"`).
- **Neon Postgres:** only the static `API_KEY` works. There are no per-user keys, and
  usage and feedback are not stored.
- **Sentry:** no error tracking.
- **Grafana Cloud scraping:** nothing collects `/metrics`, so the rules in
  `deploy/grafana/alerts.yaml` are not active.

**Edge deploys run in CI.** The Release workflow deploys the Worker and the web UI
from `main` (job `deploy-edge`, with `ORIGINS` from the `EDGE_ORIGINS` variable and
`MODEL_VERSION` from `model-release.json`), then runs a k6 smoke test through the edge
(`smoke-edge`). A merge to `main` therefore redeploys Cloud Run and the Worker
together (audit finding F-10, resolved). A manual `wrangler deploy` (7.7) is an
emergency fallback only.

**Paging today.** Only two signals reach a person:

- GitHub's notification for a failed Uptime run;
- the GCP budget alert emails (at 50% and 100% of the budget).

## 2. Session setup

Run from the repository root in bash (Git Bash on Windows). Every command in the
operations docs assumes these variables. None of them prints a secret, the GCP
project id or the Cloud Run hostname.

```bash
export REGION=us-central1 SERVICE=fastapi-assistant
export EDGE_URL=https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev

# gcloud project for this shell only, read from the repository variable without echoing it.
export CLOUDSDK_CORE_PROJECT="$(gh variable get GCP_PROJECT_ID)"

# Origin URL, kept in a variable. Do not paste it into tickets or docs.
export CLOUD_RUN_URL="$(gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)')"

# API key and metrics token, read from Secret Manager into the environment (not echoed).
export FA_API_KEY="$(gcloud secrets versions access latest --secret api-key)"
export METRICS_TOKEN="$(gcloud secrets versions access latest --secret metrics-token)"

# wrangler through node: the .cmd shim breaks on paths that contain "&".
wr() { (cd deploy/cloudflare-worker && node ./node_modules/wrangler/bin/wrangler.js "$@"); }

# The project virtualenv (the repository does not assume a working global python or jq).
PY=.venv/Scripts/python.exe; [ -x "$PY" ] || PY=.venv/bin/python
```

Notes:

- You can also type the API key at a hidden prompt instead:
  `read -rs FA_API_KEY; export FA_API_KEY`.
- `scripts/setup_secrets.sh` keeps a copy of the metrics token in
  `~/.config/fastapi-assistant/metrics_token`.
- In non-interactive mode, the script also writes the API key to
  `~/.config/fastapi-assistant/api_key`.
- Secret Manager matches what Cloud Run uses only after a Release has synced it.
- `curl -H "X-API-Key: $FA_API_KEY"` puts the key on the curl command line. Run these
  checks on a single-user machine.

## 3. Health and readiness endpoints

| Endpoint | Through the edge | Direct to Cloud Run | Meaning |
|---|---|---|---|
| `GET /health` | Public. Rate limited per client IP with the anonymous limit (10 per minute). | Open; no origin secret needed. | **Liveness.** 200 while the process runs, *even when the model failed to load*. The body has `status: "ok"`, `model_loaded`, `uses_adapter`, `backend`, `model_version` and `load_state` (`idle`, `loading`, `ready` or `failed`). Cloud Run's liveness probe calls it every 30 s. |
| `GET /ready` | Public, same limit | Open | **Readiness.** 200 with `status: "ready"` once the GGUF has been downloaded, hash-checked and loaded. Otherwise 503 with `{"detail": "Model not ready: <load_state>"}`. A 503 call also starts a load if none is running, so a failed load is retried. Cloud Run's startup probe calls it every 5 s and allows 48 failures (240 s) before it restarts the instance. |
| `GET /metrics` | 404 (blocked at the edge) | Needs `Authorization: Bearer $METRICS_TOKEN`; otherwise 401 | Prometheus text for **one** instance (see section 6). |

**What the uptime probe covers.** `uptime.yml` runs
`curl -fsS --retry 3 ... "$EDGE_URL/health"`, which proves that the edge, Cloud Run
and one container answer. When the `SMOKE_API_KEY` secret is set, it then calls
`GET /v1/models` through the edge with that key. That call is authenticated end to
end and passes the origin secret check, so a secret mismatch (every API call 403) or
a rejected key (401) fails the probe. It does not exercise:

- `/ready`;
- generation.

Without `SMOKE_API_KEY`, the second step is skipped and a secret mismatch leaves the
probe green.

**Scheduled runs are best effort.** GitHub can delay scheduled runs. It also disables
them in a public repository after 60 days without repository activity. Check that the
probe is still running with `gh run list --workflow uptime.yml --limit 5`.

**Other `/health` traffic.** The web UI calls `/health` every 30 seconds per open tab
(`web/src/hooks/useHealth.ts`), so `/health` traffic in the logs is normal.

## 4. Checking the edge and the origin

Run these in order. The expected results are what a healthy deployment returns.

```bash
# 1. Web UI is served by the edge (static assets).
curl -sS -o /dev/null -w '%{http_code} %{content_type}\n' "$EDGE_URL/"

# 2. Edge reaches an origin instance; the body shows the deployed model.
curl -sS "$EDGE_URL/health"; echo
curl -sS -o /dev/null -w 'ready: %{http_code}\n' "$EDGE_URL/ready"

# 3. Keyless API calls stop at the edge (401), unless PUBLIC_DEMO is on (200; /health
#    then reports "access": "public").
curl -sS -o /dev/null -w 'no key: %{http_code}\n' -X POST "$EDGE_URL/generate" \
  -H 'Content-Type: application/json' -d '{"question":"hi","max_new_tokens":8}'

# 4. End to end with a valid key (short answer to keep cost down); print headers only.
curl -sS -D - -o /dev/null -X POST "$EDGE_URL/generate" \
  -H "X-API-Key: $FA_API_KEY" -H 'Content-Type: application/json' \
  -d '{"question":"How do I declare a path parameter?","max_new_tokens":32}'

# 5. The origin refuses traffic that did not come through the edge.
curl -sS -w '\ndirect: %{http_code}\n' -X POST "$CLOUD_RUN_URL/generate" \
  -H 'Content-Type: application/json' -d '{"question":"hi"}'

# 6. Metrics are locked at the origin and hidden at the edge.
curl -sS -o /dev/null -w 'metrics, no token: %{http_code}\n' "$CLOUD_RUN_URL/metrics"
curl -sS -o /dev/null -w 'metrics at edge: %{http_code}\n' "$EDGE_URL/metrics"
```

| Check | Healthy result |
|---|---|
| 1 | `200 text/html` |
| 2 | 200 JSON with `model_loaded: true` and `load_state: "ready"` on a warm instance; `ready: 200` |
| 3 | `no key: 401` (edge body type `authentication_error`) |
| 4 | `HTTP 200` with `x-request-id`, `x-served-by: origin-0`, `x-cache: BYPASS` (no Upstash), and `x-ratelimit-limit`, `x-ratelimit-remaining` and `x-ratelimit-reset`. |
| 5 | `{"detail":"Direct origin access is not allowed"}`, `direct: 403` |
| 6 | `metrics, no token: 401`, `metrics at edge: 404` |

**Cold starts.** The first request after an idle period starts a new instance. The
instance downloads the 0.7 GB GGUF, verifies its SHA-256 and loads it before Cloud Run
routes traffic to it. The edge waits at most 150 s (`ORIGIN_TIMEOUT_MS`) for response
headers, so a cold start slower than that shows up as a 504. Check 2 is the cheapest way to
warm an instance before testing.

**Burst testing.** Do not send bursts to test capacity. The only capacity evidence is
in [../audit/PERFORMANCE_BASELINE.md](../audit/PERFORMANCE_BASELINE.md) (one-off
probes).

## 5. Logs

### 5.1 Correlating a request

Every request has one id across both layers:

1. The Worker keeps a client `X-Request-ID` if it is log-safe (`[A-Za-z0-9._:-]`,
   1 to 128 characters). Otherwise it generates a UUID.
2. It forwards that id to the origin.
3. The origin logs it as `request_id` and returns it in `x-request-id`.

Start from the `x-request-id` the client saw.

### 5.2 Worker logs

Workers observability is enabled (`[observability]` in `wrangler.toml`). Recent history
is in the Cloudflare dashboard (Workers, `fastapi-assistant-edge`, Logs). For live
logs:

```bash
wr tail --format pretty                         # every invocation
wr tail --format json --search upstream_failures  # origin attempts that failed
wr tail --format json --search upstream_normalized
wr tail --format json --search <request-id>
wr tail --status error                          # invocations that threw
```

The Worker writes one JSON line per request (`event: "request"`):

| Field | Meaning |
|---|---|
| `request_id` | The correlation id above. |
| `method`, `path`, `status`, `duration_ms` | The request and the status the client received. |
| `rl_identity` | `key` (counted per SHA-256 of the API key) or `ip` (anonymous and public paths). Neither the key nor the IP is logged. |
| `rl_key_id` | Only with `rl_identity: key`: the first 12 hex characters of the SHA-256 of the key that was presented, valid or not. Use it to tell callers apart and to find the key to revoke. For a key you hold, `printf '%s' "$KEY" \| sha256sum \| cut -c1-12` prints the same value. |
| `rl_scope` | `redis`: the shared Upstash counter. `local`: the per-isolate fallback counter. `local` is expected while Upstash is not configured. It also appears when Upstash fails; see `redis_error`. |
| `cache` | `HIT`, `MISS` or `BYPASS`. `null` when the request ended before the cache step (404, 401, 413, edge 429). |
| `origin_index` | Index in `ORIGINS` of the origin that answered (0 is Cloud Run). `null` when none did. |
| `attempts` | How many origin fetches were made. |
| `upstream_failures` | One entry per failed attempt, for example `origin-0: timeout after 150000 ms`, `origin-0: TypeError: ...`, or `origin-0: HTTP 503`. An HTTP status is recorded only when another origin was tried after it. |
| `skipped_origins` | Origins skipped because their circuit breaker was open: 3 consecutive network failures or timeouts open it for 30 s. With a single origin nothing is skipped, because the last origin is always tried. |
| `client_gone` | `true`, with `status: 499`, when the client disconnected before an origin answered. The Worker aborted its origin request; the origin is not counted as failed and no backup origin is tried. |
| `upstream_normalized` | `true` when the origin's platform (Cloud Run's front end) answered 429, 502, 503 or 504 with a non-JSON body. The edge rewrote it into the JSON error shape and added `Retry-After: 10` when it was missing. |
| `redis_error` | Upstash transport or command errors (only when Upstash is configured). |
| `error` | An exception inside the Worker. The client got `500 Internal edge error` (`edge_error`). |

A separate `event: "cache_write_error"` line means an answer could not be written to
the cache (only with Upstash).

### 5.3 Cloud Run logs

The image sets `LOG_JSON=true`, so each application log line is a JSON object on
stdout. Its fields are:

- always: `ts`, `level`, `severity`, `logger`, `message`;
- per request: `request_id`, `method`, `path`, `status_code`, `duration_ms`;
- on errors: `exc_info`.

`severity` carries the same value as `level`, and Cloud Logging maps it to the entry's
severity, so both `severity>=ERROR` and `jsonPayload.level="ERROR"` work as filters.
The queries below use `jsonPayload.level`.

```bash
BASE='resource.type="cloud_run_revision" AND resource.labels.service_name="fastapi-assistant"'

# One request, end to end on the origin.
gcloud logging read "$BASE AND jsonPayload.request_id=\"<request-id>\"" --freshness 1d --format json

# Application errors in the last hour (load failures, generation failures, unhandled errors).
gcloud logging read "$BASE AND jsonPayload.level=\"ERROR\"" --freshness 1h --limit 50 --format json

# Model loading: download, checksum, load time.
gcloud logging read "$BASE AND (jsonPayload.message:\"Fetching\" OR jsonPayload.message:\"Verified sha256\" OR jsonPayload.message:\"Model ready\" OR jsonPayload.message:\"Failed to load\")" --freshness 1d --limit 20

# Cloud Run's own request log (the platform view, including requests it refused itself).
gcloud logging read "$BASE AND log_id(\"run.googleapis.com/requests\") AND httpRequest.status>=429" --freshness 1h --limit 50
```

Useful application messages:

| Message | Meaning |
|---|---|
| `request` | One line per HTTP request, with status and duration. |
| `Fetching <file> from <repo>@<revision>` | Cold-start download started. |
| `Verified sha256 of <file> in N s` | Integrity check passed. |
| `Model ready (backend=llamacpp, ...) in N s` | Instance can serve. |
| `Failed to load model (llamacpp backend)` | Load failed. `exc_info` has the cause, for example `Checksum mismatch for ...`. |
| `Generation failed` / `Streaming generation failed` | An engine error. The client got 500, or an SSE `error` event after a 200. |
| `Retrieval failed; answering without context` | Retrieval degraded; the answer was still produced. |
| `feedback rating=... comment_chars=...` | Feedback, logged because there is no database. Only the rating and the comment length are logged, never the comment text. |

## 6. Metrics

Nothing scrapes `/metrics` today. Read it directly from Cloud Run:

```bash
curl -sS "$CLOUD_RUN_URL/metrics" -H "Authorization: Bearer $METRICS_TOKEN" | grep -E '^assistant_'
```

Caveats:

- Each call reaches whichever instance Cloud Run picks. Values are per instance.
- Counters reset when an instance stops, and instances scale to zero when idle. Treat
  a reading as a snapshot of one instance since it started.
- For a histogram, `_sum / _count` gives the mean.

The names come from `src/fastapi_assistant/metrics.py`.

| Metric | Type, labels | Normal | Investigate when |
|---|---|---|---|
| `assistant_model_loaded` | gauge | 1 on any instance that is serving | 0 on an instance that receives traffic |
| `assistant_build_info` | info: `backend`, `model`, `model_version` | Matches `model-release.json` | Version differs from what was released |
| `assistant_inflight_generations` | gauge | 0 or 1 (`MAX_CONCURRENCY=1`) | Stuck at 1 with no traffic in the logs. A non-streaming generation keeps its slot until it finishes, even after the client or edge gave up. |
| `assistant_queue_depth` | gauge | 0 | At 3 (`MAX_QUEUE`), the next request on that instance is rejected with 429 |
| `assistant_queue_wait_seconds` | histogram | Near 0 when idle; otherwise about queue position times generation time | A p95 above 20 s for 15 minutes (the `AssistantLongQueueWait` rule). `QUEUE_TIMEOUT_S=60` turns a 60 s wait into a 503; the edge waits 150 s (`ORIGIN_TIMEOUT_MS`), so it does not give up first. |
| `assistant_time_to_first_token_seconds` | histogram, `endpoint` | Streaming only. Measured once at 0.3 to 1.1 s on a warm instance. Excludes queue wait. | Several seconds or more: CPU contention or long prompts (retrieval adds prompt tokens) |
| `assistant_generation_duration_seconds` | histogram, `endpoint`, `stream` | Scales with tokens: about 8 to 10 tokens per second for one request | p95 above 60 s (the `AssistantSlowGenerations` rule) |
| `assistant_generation_tokens_per_second` | histogram | About 8 to 10 measured on Cloud Run | Mean below 4 (the `AssistantThroughputDrop` rule) |
| `assistant_generation_finish_total` | counter, `reason` | Mostly `stop` and `length` | Any `timeout` (hit `GENERATION_TIMEOUT_S`, 60 s on Cloud Run); many `cancelled` (clients disconnecting before their answer is complete) |
| `assistant_generation_errors_total` | counter | 0 | Any increase. Find the matching `Generation failed` log line. |
| `assistant_admission_rejections_total` | counter, `reason` = `queue_full` or `queue_timeout` | 0 | Any sustained rate (the `AssistantSaturated` rule) |
| `assistant_prompt_too_long_total` | counter | Rare | A rising count: clients send prompts that do not fit the 2,048-token context. Each gets `400 context_length_exceeded`. A client bug or probing, not a server fault. |
| `assistant_http_requests_total` | counter, `route`, `method`, `status` | 5xx near 0 | 5xx above 5% of requests (the `AssistantHighErrorRate` rule) |
| `assistant_http_request_duration_seconds` | histogram, `route` | Time to response headers (for streams, until the stream starts); about queue wait plus generation time for blocking calls | Much larger than queue wait plus generation time: time is lost elsewhere (tokenizing, retrieval, the database) |
| `assistant_prompt_tokens_total`, `assistant_completion_tokens_total` | counters | | Sudden growth with no matching traffic change |
| `assistant_retrieval_duration_seconds` | histogram | Only when a request sets `use_retrieval` | p95 of several seconds |

**Not visible here.** Platform 429s from Cloud Run's front end never reach the
application. They show up only in the Worker logs (`upstream_normalized`) and in
Cloud Run's request log.

**Cloud Run's own metrics** (request count, latencies, instance count, billable
instance time) are in Cloud Monitoring and need no setup.

## 7. Routine tasks

### 7.1 Rotate all generated secrets

`scripts/setup_secrets.sh` generates **new** values every time it runs:

| Secret | Where the new value goes |
|---|---|
| `ORIGIN_SHARED_SECRET` | Worker secret and GitHub secret |
| `METRICS_TOKEN` | GitHub secret and the local file |
| `API_KEY` | GitHub secret |
| `SMOKE_API_KEY` | GitHub secret (set to the new API key unless a database is configured) |

Upstash and Neon prompts can be skipped with Enter. Skipping leaves the existing
values in place.

1. Tell API key holders that the key changes. Every client using the static key gets
   401 once the new key is deployed.
2. Log in: `gh auth status` and `wr whoami`.
3. Run `bash scripts/setup_secrets.sh`. The new API key is printed once.
4. **Immediately** deploy Cloud Run with the new values:

   ```bash
   gh workflow run release.yml --ref main
   gh run watch "$(gh run list --workflow release.yml --limit 1 --json databaseId -q '.[0].databaseId')"
   ```

   Between step 3 and the end of this Release, the Worker already sends the new origin
   secret but Cloud Run still expects the old one. Every API call gets 403 during that
   window, while `/health` stays green. Do this in a quiet period.
5. Verify with section 4. Check 4 must return 200 with the new key.
6. Clean up old secret versions (7.3).

If the Release fails before `Deploy Cloud Run` (build, or the Trivy gate), the 403s
continue. Fix the Release, or restore the previous origin secret as described in
[ROLLBACK.md](ROLLBACK.md) section 6.

### 7.2 Rotate one secret

The script cannot rotate a single value. To rotate one, do by hand what it does.
Example for the origin secret:

```bash
secret="$("$PY" -c 'import secrets, sys; sys.stdout.write(secrets.token_hex(32))')"
printf '%s' "$secret" | gh secret set ORIGIN_SHARED_SECRET
printf '%s' "$secret" | wr secret put ORIGIN_SHARED_SECRET
unset secret
gh workflow run release.yml --ref main
```

For the API key, use `fa_` followed by 64 hex characters and set only the GitHub
secret `API_KEY`. Update `SMOKE_API_KEY` too if it held the old key. For the metrics
token, set `METRICS_TOKEN`.

Release adds a Secret Manager version only when a value changed.

### 7.3 Clean up old Secret Manager versions

Rotation leaves earlier versions enabled. The Secret Manager free tier counts active
versions (see [COST_MANAGEMENT.md](COST_MANAGEMENT.md)), and old values remain
readable while they exist.

1. List the versions: `gcloud secrets versions list origin-shared-secret`.
2. Disable the older ones: `gcloud secrets versions disable <N> --secret origin-shared-secret`.
3. After a day without problems, destroy them:
   `gcloud secrets versions destroy <N> --secret origin-shared-secret`.

Destroying is permanent and removes the option to restore that value
([ROLLBACK.md](ROLLBACK.md) section 6). The secrets are `origin-shared-secret`,
`api-key`, `metrics-token`, `database-url` and `sentry-dsn`.

### 7.4 Issue and revoke database keys (only once Neon is configured)

Today `DATABASE_URL` is a `disabled` placeholder, and the command exits with
`DATABASE_URL is not set`. Once Neon exists, use the commands below.
`"$PY" -m fastapi_assistant` is the same CLI as `fastapi-assistant`; use either.

```bash
read -rs DATABASE_URL; export DATABASE_URL        # paste the Neon URL at the hidden prompt
"$PY" -m fastapi_assistant keys create <name>     # key printed once
"$PY" -m fastapi_assistant keys list              # id, display prefix, status, name
"$PY" -m fastapi_assistant keys revoke fa_AbC123  # revoke by display prefix
unset DATABASE_URL
```

How long a revoked key keeps working:

- each origin instance caches a positive lookup for `KEY_CACHE_TTL_S` (60 s);
- with Upstash, the edge may serve cached answers to that key for up to
  `KEY_VERIFIED_TTL_SECONDS` (300 s).

The static `API_KEY` cannot be revoked. Rotate it instead (7.2).

### 7.5 Deploy a code change

1. Merge the PR to `main`. The Release workflow then:
   - builds and scans the image (it fails on fixable HIGH or CRITICAL CVEs);
   - copies the amd64 image to Artifact Registry;
   - syncs the secrets;
   - deploys Cloud Run;
   - polls `/health` and `/ready`;
   - builds `web/` and deploys the Worker with its static assets (`deploy-edge`);
   - runs the k6 smoke test through the edge (`smoke-edge`).

   Merges that change only `**.md`, `docs/**`, `eval/**`, `graphify-out/**` or
   `tests/load/**` do not trigger it.
2. Watch it: `gh run list --workflow release.yml --limit 3`, then `gh run watch <id>`.
3. Verify with section 4.

### 7.6 Deploy a new model release

1. Normal path (credentials configured; the first release, #33, was made this way):
   1. Run the "Train on Kaggle" workflow. It trains, publishes the GGUF and runs the
      eval gate. On a fresh setup, run it once with `push_only: true`, attach the
      `HF_TOKEN` secret to the new `fastapi-assistant-train` kernel on Kaggle, then
      run it normally.
   2. It opens a PR that changes `model-release.json` and `eval/baseline.json`.
   3. Review the eval table and merge. Release deploys it.
2. Manual pin, for example a different published GGUF:
   1. Edit `model-release.json`: `repo_id`, `filename`, a commit `revision` and its
      `sha256`.
   2. Compute the hash from the exact file at that revision:

      ```bash
      "$PY" -c "import json; from pathlib import Path; from huggingface_hub import hf_hub_download; from fastapi_assistant.hub import sha256_file; g = json.load(open('model-release.json'))['gguf']; print(sha256_file(Path(hf_hub_download(g['repo_id'], g['filename'], revision=g['revision']))))"
      ```

   3. Open a PR. The "Model eval gate" workflow runs on PRs that touch
      `model-release.json`.
   4. Merge. Release deploys it.
3. A wrong `sha256` makes the new revision fail its startup probe. The deploy fails
   and the previous revision keeps serving (see INCIDENT_RESPONSE.md, "Checksum
   mismatch").
4. The answer cache is namespaced by `MODEL_VERSION`. Release's `deploy-edge` job sets
   it on the Worker from `model-release.json`, so no manual step is needed (it matters
   only once Upstash is configured).
5. Verify. `curl -sS "$EDGE_URL/health"` must show the new `model_version`.

### 7.7 Deploy the Worker by hand (emergency fallback)

Release deploys the Worker on every run. Deploy by hand only when Release cannot, for
example during a Cloudflare token problem or a CI outage. The next Release run
replaces whatever you deploy here with `main`.

`wrangler deploy` **replaces all vars** with those in `wrangler.toml`, where
`ORIGINS = ""`, unless you pass them. A deploy without
`--var ORIGINS:...` makes every API call return `502 No origin configured`. Always
pass both `--var ORIGINS:...` and `--var MODEL_VERSION:...`, as below.

```bash
git switch main && git pull --ff-only
(cd deploy/cloudflare-worker && npm ci)
(cd web && npm ci && npm run build)                  # the Worker serves web/dist
ORIGINS="$(gh variable get EDGE_ORIGINS)"            # Cloud Run URL; not echoed
MODEL_VERSION="$("$PY" -c "import json; print(json.load(open('model-release.json'))['model_version'])")"
wr deploy --dry-run --var "ORIGINS:$ORIGINS" --var "MODEL_VERSION:$MODEL_VERSION"
wr deploy --var "ORIGINS:$ORIGINS" --var "MODEL_VERSION:$MODEL_VERSION" \
  --message "main@$(git rev-parse --short HEAD)"
unset ORIGINS
```

Then:

1. Run `wr deployments list` and confirm the new version.
2. Rerun section 4.

The `--message` records which commit is live, so a manual deploy can be told apart
from a Release deploy in `wr deployments list`. Secrets carry over between deploys.

### 7.8 Change the instance cap

The cap is the repository variable `CLOUD_RUN_MAX_INSTANCES` (default 2). It is the
only hard spending limit. Raising it costs money beyond the free tier and needs owner
approval.

```bash
gh variable set CLOUD_RUN_MAX_INSTANCES --body 1      # persists across Releases
gcloud run services update "$SERVICE" --region "$REGION" --max-instances 1   # applies now
```

A manual `gcloud` change alone is overwritten by the next Release. Set the variable as
well.

## 8. Known gaps

- **No active metric alerts.** `deploy/grafana/alerts.yaml` is not loaded anywhere.
- **The uptime probe does not exercise generation** (section 3).
- **Metrics are per instance and short-lived** (section 6).
- **A manual Worker deploy drifts from `main` until the next Release.** Use
  `wr deployments list` and the deploy message to see what is live.
- **Abuse cannot be attributed.** Worker logs do not include the key identity or the
  client IP, and the origin sees only the edge.

## Related documents

- [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md)
- [ROLLBACK.md](ROLLBACK.md)
- [COST_MANAGEMENT.md](COST_MANAGEMENT.md)
- [BACKUP_AND_RECOVERY.md](BACKUP_AND_RECOVERY.md)
- [../security/SECURITY_AUDIT.md](../security/SECURITY_AUDIT.md)
- [../DEPLOYMENT.md](../DEPLOYMENT.md)
- [../audit/INITIAL_AUDIT.md](../audit/INITIAL_AUDIT.md)
- [../audit/PERFORMANCE_BASELINE.md](../audit/PERFORMANCE_BASELINE.md)
