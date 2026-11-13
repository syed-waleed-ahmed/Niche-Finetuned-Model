# Incident response

What to do when the service misbehaves. The commands assume the session setup in
[RUNBOOK.md](RUNBOOK.md) section 2 (`EDGE_URL`, `CLOUD_RUN_URL`, `FA_API_KEY`,
`METRICS_TOKEN`, `wr`, `PY`, and `CLOUDSDK_CORE_PROJECT`).

This is a single-maintainer project with no on-call rotation:

- **Incident lead.** The repository owner, or whoever holds the GCP, Cloudflare and
  GitHub access needed to act.
- **Paging.** The only automatic alerts are GitHub's notification for a failed Uptime
  run and the GCP budget emails.
- **Grafana alert rules** exist in `deploy/grafana/alerts.yaml` but are not loaded
  anywhere.

## 1. Severity levels

| Level | Definition | Examples | Response |
|---|---|---|---|
| SEV1 | The API is unavailable or wrong for every client, a secret is exposed, or spend is climbing with no cap in sight | Every authenticated call fails (401, 403, 5xx or 502); the edge or UI is down; a key or the origin secret is leaked; a budget alert at 100% | Start now. Mitigate first (roll back, kill switch, rotate), investigate after. |
| SEV2 | Degraded: some requests fail or are slow, or one feature is broken | Bursts of 429, 503 or 504; slow cold starts; streaming broken while blocking calls work; the uptime probe flapping | Start within hours. Mitigate if the cause is a recent change. |
| SEV3 | Little or no user impact | A new dependency advisory; one failed scheduled workflow; isolated 500s; a budget alert at 50% with a known cause | Track in an issue; fix in normal work. |

Raise the severity if the impact grows or the cause is unknown after 15 minutes.
Treat any suspected secret exposure as SEV1 until it is ruled out.

## 2. First 15 minutes

1. **Start a log.** Note the start time in UTC and keep a running list of what you saw
   and did. Use a private place for anything security-related, not a public issue.
2. **Measure the blast radius.** Run RUNBOOK section 4, checks 2 to 5. They tell apart:
   - edge down (check 2 fails before reaching an origin);
   - origin down (`/health` through the edge fails, or returns 502 or 504);
   - authentication broken (401 with a valid key);
   - origin lock-out (403 with a valid key);
   - generation broken (200 health, 5xx on generate).
3. **Check what changed recently.** Most incidents follow a change.

   ```bash
   gh run list --workflow release.yml --limit 5            # Cloud Run deploys and secret syncs
   git log origin/main --oneline -5
   wr deployments list                                      # Worker deploys, rollbacks, secret changes
   gcloud run revisions list --service fastapi-assistant --region us-central1 --limit 5
   gcloud secrets versions list origin-shared-secret --limit 3
   gcloud secrets versions list api-key --limit 3
   ```

   Also ask whether someone ran `scripts/setup_secrets.sh`. It changes the Worker
   secret immediately and the origin only at the next Release.
4. **Read the logs** for one failing request. Take `x-request-id` from a failing
   response, then:
   - Worker: `wr tail --format json --search <id>`;
   - origin: RUNBOOK section 5.3.

   Look at `status`, `upstream_failures`, `upstream_normalized`, `origin_index` and
   the origin's `exc_info`.
5. **Match the symptom** in section 3.
6. **Mitigate:**
   - if a recent change lines up with the start time, roll it back first
     ([ROLLBACK.md](ROLLBACK.md)) and debug later;
   - if spend or abuse is the problem, use a kill switch (section 4);
   - if a secret is exposed, rotate it (RUNBOOK 7.1 or 7.2).
7. **At 15 minutes, decide** and write the decision in the log:
   - mitigated, and monitoring;
   - rollback in progress;
   - blocked on an owner-only action (credentials, billing).

## 3. Symptoms, causes and actions

There are three sources of error responses:

- **The edge Worker.** JSON `{"error": {"message", "type"}}`; the Worker log has
  `origin_index: null`.
- **The application.** `{"detail": ...}` on `/generate` and the probes, and the OpenAI
  error shape on `/v1/*`.
- **Cloud Run's front end.** Plain text from the platform. The Worker rewrites it to
  JSON and logs `upstream_normalized: true`.

`x-served-by: origin-0` means an origin response was relayed.

| Symptom | How to recognise it | Likely cause | Action |
|---|---|---|---|
| **401 everywhere** | **(a)** Edge 401, type `authentication_error`, "Missing API key...", `origin_index: null`. **(b)** Origin 401: `{"detail":"Invalid API key"}`, or type `authentication_error` with `code` 401 on `/v1`. | **(a)** The client sends no key. Typical cases: UI Settings empty; header not `X-API-Key` or `Authorization: Bearer`. **(b)** The key is not the origin's `API_KEY`. Typical cases: the key was rotated by `setup_secrets.sh` and the Release has completed, but clients still use the old key; a database key is used while Neon is not configured (always true today). | **(a)** Fix the client. **(b)** Give clients the current key: read it with `gcloud secrets versions access latest --secret api-key` into a variable; do not paste it into chat. If the rotation itself was the mistake, restore the previous `api-key` version ([ROLLBACK.md](ROLLBACK.md) section 6). |
| **403 from the origin** | `{"detail":"Direct origin access is not allowed"}` with `x-served-by: origin-0`. `/health` stays 200; the uptime probe fails only at its authenticated `/v1/models` step (when `SMOKE_API_KEY` is set). | The origin secret differs between the edge and the origin. Causes: (1) `setup_secrets.sh` or a manual `wrangler secret put` changed the Worker, but no Release has synced Cloud Run (or the Release failed before deploying); (2) a Worker rollback to a version from before a rotation; (3) the GitHub secret changed and a Release deployed Cloud Run, but its `deploy-edge` job failed, so the Worker kept the old value. | **(1)** Run Release now: `gh workflow run release.yml --ref main`. **(2)** Put the current origin secret on the Worker: `gcloud secrets versions access latest --secret origin-shared-secret \| wr secret put ORIGIN_SHARED_SECRET`. If wrangler refuses because the latest version is not deployed, redeploy the Worker (RUNBOOK 7.7). **(3)** Fix the edge job and re-run Release; `deploy-edge` uploads the GitHub value to the Worker. Verify with RUNBOOK check 4. |
| **413** | Edge: type `request_too_large`. Origin: `{"detail":"Request body is larger than 65536 bytes"}`. | The body is over 64 KiB (`MAX_REQUEST_BYTES`, `max_request_bytes`). This is expected for oversized or abusive bodies. | No action unless legitimate clients hit it. Prompts are already capped at 4,000 characters for a `/generate` question and 6,000 for a whole conversation (and 2,048 tokens), so a legitimate request is far below 64 KiB. |
| **429, edge limit** | Type `rate_limited`, "Rate limit exceeded: N requests per minute...", `X-RateLimit-*` headers, `Retry-After` up to 60 s; Worker log `status: 429` with `attempts: 0`. | More than 20 requests per minute for one key (`RATE_LIMIT_PER_MINUTE`), or more than 10 per minute for one IP on public or anonymous paths. Today the count is per isolate (`rl_scope: "local"`), so the effective limit is higher and uneven. | A client loop or abuse: ask the client to back off, or rotate the key if it leaked. A legitimate need: raise the var in `wrangler.toml` and merge it; Release redeploys the Worker. |
| **429, application queue full** | `/generate`: `{"detail":"Server is at capacity"}`. `/v1`: type `overloaded`. Both are JSON with a computed `Retry-After`. `assistant_admission_rejections_total{reason="queue_full"}` rises. | An instance already has 1 generation running and 3 queued (`MAX_CONCURRENCY=1`, `MAX_QUEUE=3`). | Expected under bursts; clients should honour `Retry-After`. Sustained: see "Capacity" below. |
| **Bursts of 429 `overloaded` with `upstream_normalized: true`** (before the edge fix in #13, clients saw plain text `Rate exceeded.` with no `Retry-After`) | Worker log `upstream_normalized: true`, `status: 429`. The application metrics show no rejection, because Cloud Run's front end refused the request before the app saw it. It may also appear in Cloud Run's request log (RUNBOOK 5.3). | Cloud Run is scaling out to a second, cold instance, or both instances (`maxScale 2`) are at `containerConcurrency 4`. **Measured 2026-11-13:** 6 simultaneous short requests with one warm instance gave 2 x 200 and 4 x 429 after about 10.5 s. Four simultaneous requests a minute later all returned 200, served one after another (3.5 to 11.9 s). See [../audit/PERFORMANCE_BASELINE.md](../audit/PERFORMANCE_BASELINE.md). | Usually none: the second instance becomes ready, and clients retry after `Retry-After: 10`. Options that cost money and need owner approval: `minScale 1` (no cold start for the first instance; billed while idle) or a higher `maxScale`. These probes are the only capacity evidence; do not extrapolate a user count from them. |
| **502 `No origin configured`** | Type `upstream_unavailable`, `attempts: 0`. | The Worker's `ORIGINS` var is empty. Deploys replace vars with `wrangler.toml`, where `ORIGINS = ""`. Causes: a manual `wrangler deploy` without `--var ORIGINS:...`, or a Release run while the `EDGE_ORIGINS` variable was empty. | Roll back the Worker to the previous version ([ROLLBACK.md](ROLLBACK.md) section 5). Then fix the cause: set `EDGE_ORIGINS` and re-run Release, or redeploy by hand with both vars (RUNBOOK 7.7). |
| **502 `No origin available`** | Type `upstream_unavailable`, `upstream_failures` such as `origin-0: TypeError: ...` (not a timeout). | Network-level failure to reach Cloud Run. Possible causes: the service was deleted or its URL changed (new project or region); ingress was changed; a Google front-end incident. | Run `gcloud run services describe fastapi-assistant --region us-central1`. If the URL changed, update `EDGE_ORIGINS` (`gh variable set EDGE_ORIGINS`) and re-run Release, which redeploys the Worker with it. |
| **502 or 503 `upstream_unavailable` with `upstream_normalized: true`** | "The origin is temporarily unavailable..." | Cloud Run's front end answered with a plain-text error. Usually no instance could start: a broken revision, a failing startup probe, or a quota problem. | Check `gcloud run revisions list` and the application errors (RUNBOOK 5.3). Roll back the revision if a deploy caused it. |
| **504** | Type `upstream_timeout`, "The origin did not respond in time"; `upstream_failures` has `timeout after 150000 ms`. | No response headers within `ORIGIN_TIMEOUT_MS` (150 s). On Cloud Run, queue wait (60 s) plus generation (60 s) is designed to fit inside that, so a 504 means something outside the budget: (1) a cold start (GGUF download, hash and load) that has not finished; (2) the origin hung or the network to it stalled; (3) someone changed one of the three timeouts and broke `QUEUE_TIMEOUT_S + GENERATION_TIMEOUT_S < ORIGIN_TIMEOUT_MS`. | **(1)** Check the load log lines (RUNBOOK 5.3) and wait for readiness. **(2)** Check origin logs for the request id; restart the revision if it is wedged. **(3)** Restore the invariant (`service.yaml`, `wrangler.toml`). |
| **503 `Model not ready`** | `/generate`: `{"detail":"Model not ready"}`. `/v1`: type `model_not_ready`. Both carry `Retry-After: 30`. `/ready` returns 503 `Model not ready: failed`. Log `Failed to load model (llamacpp backend)` with `exc_info`. On Cloud Run, an instance whose load failed never passes its startup probe and gets no traffic, so clients more often see a normalized `upstream_unavailable` or a 504. | The model load failed: checksum mismatch (next row), Hugging Face download error or unavailability, out of memory, or a bad `GGUF_REVISION` or filename. Note: `Model not ready: loading` on `/ready` is normal during a cold start. | Read `exc_info`. A download error usually clears on retry, because every `/ready` call retries a failed load. A deploy that introduced it: roll back the revision. A Hugging Face outage: see [BACKUP_AND_RECOVERY.md](BACKUP_AND_RECOVERY.md) section 4.3. |
| **503 queue timeout** | `/generate`: `{"detail":"Timed out waiting for capacity"}`. `/v1`: type `overloaded`. `assistant_admission_rejections_total{reason="queue_timeout"}`. | A request waited `QUEUE_TIMEOUT_S` (60 s) for the single generation slot. Non-streaming callers usually see the edge's 504 first. | As for the queue-full 429. Check `assistant_queue_wait_seconds` and look for long generations holding the slot. |
| **Cold-start latency** | The first request after idle takes tens of seconds, or 504s. Logs show `Fetching ...`, `Verified sha256 ... in N s` and `Model ready ... in N s` just before. | `minScale 0`. A new instance downloads the 0.7 GB GGUF from Hugging Face (weights are not baked into the image), hashes it and loads it before Cloud Run routes to it. The 10-minute uptime probe reduces cold starts but cannot prevent them, because Cloud Run decides when idle instances stop. | Expected behaviour. Options that cost money or storage and need owner approval: `minScale 1`, or baking the weights into the image (pushes Artifact Registry over 0.5 GB). Use the log timings to see which phase dominates. |
| **Checksum mismatch on load** | `exc_info` contains `Checksum mismatch for <file>: expected sha256 <a>, got <b>`. The new revision never becomes ready, and the Release fails at `Deploy Cloud Run` or the origin smoke test. Traffic stays on the previous revision, because Cloud Run does not route to a revision that fails its startup probe. | **(1)** `model-release.json` has a wrong `sha256`, filename or revision. **(2)** The file at the pinned revision is not what was reviewed (upstream substitution). **(3)** A corrupted download (rare; retried on the next load). | **Never remove the `sha256` to get a deploy through.** Recompute the hash from the pinned revision (RUNBOOK 7.6). If it matches `model-release.json`, it was a transient download problem; re-run Release. If it does not match a hash you reviewed, treat it as a supply-chain security incident (SEV1): keep the old revision serving, do not deploy, and compare against the SHA-256 that Hugging Face lists for that file at that revision. |
| **500 `Internal edge error`** | Type `edge_error`; Worker log has `error`. | An exception in the Worker. | If it started after a Worker deploy, roll back the Worker. |
| **500 `Generation failed`, or a 200 stream that ends in an `error` event** | Origin log `Generation failed` or `Streaming generation failed` with `exc_info`; `assistant_generation_errors_total` rises. | An engine error. | Read `exc_info`. If it started after a deploy, roll back the revision. |
| **400 `context_length_exceeded`** | `/v1` code `context_length_exceeded`, or a `/generate` detail about the prompt length; `assistant_prompt_too_long_total` rises. | The prompt does not fit the 2,048-token context with at least 16 tokens left for the answer. A client error by design. | Only a sudden surge matters: identify the client. |
| **Cost spike** | Budget alert email (50% or 100% of `BUDGET_USD`, default 5 USD); unusual billing in the Cloud console. | Sustained traffic keeping instances busy (abuse, or a client loop with a valid key); `maxScale` raised; Artifact Registry over 0.5 GB; secret versions piling up. See [COST_MANAGEMENT.md](COST_MANAGEMENT.md). | 1. Check Cloud Run request and instance metrics in Cloud Monitoring, and the Worker logs for a single identity type or path dominating. 2. If unexplained and still rising, use a kill switch (section 4). 3. Then fix the cause: rotate a leaked key, lower the limits, or clean up storage. |

**Capacity.** The deployment runs one generation at a time per instance, on at most 2
instances. Sustained 429s or 503s mean demand exceeds that. Raising capacity costs
money and is an owner decision; see [COST_MANAGEMENT.md](COST_MANAGEMENT.md). No
number of supported users has been measured.

## 4. Containment switches

Use these when the cause is unknown and the impact (cost or abuse) is growing. Each
switch is reversible.

**Stop all origin traffic and spend.**

```bash
gcloud run services remove-iam-policy-binding fastapi-assistant --region us-central1 \
  --member allUsers --role roles/run.invoker
gh workflow disable release.yml     # so a merge does not re-add the binding
```

- Effect: Google's front end refuses every request before an instance starts, so no
  instance time is billed. The edge relays the refusal. The web UI stays up, because
  it is served by Cloudflare.
- Undo:

  ```bash
  gcloud run services add-iam-policy-binding fastapi-assistant --region us-central1 \
    --member allUsers --role roles/run.invoker
  gh workflow enable release.yml
  ```

  The Release workflow also re-adds the binding on every deploy.

**Cap at one instance.**

```bash
gh variable set CLOUD_RUN_MAX_INSTANCES --body 1
gcloud run services update fastapi-assistant --region us-central1 --max-instances 1
```

**Throttle at the edge.** Lower `RATE_LIMIT_PER_MINUTE` or `RATE_LIMIT_PER_MINUTE_ANON`
in `wrangler.toml` and merge the change; Release redeploys the Worker. If Release
cannot run, deploy by hand (RUNBOOK 7.7). Without Upstash, the limits are per
isolate.

**Cut off a leaked key.** Rotate `API_KEY` (RUNBOOK 7.2). Database keys are revoked
with `keys revoke` once Neon exists.

## 5. Communication

- **Who to tell.** People who hold an API key, plus anyone relying on the public demo.
  There is no status page. For outages longer than an hour, use a pinned GitHub issue
  labelled `incident`.
- **Security incidents.** Keep the details private until the fix is deployed (for
  example, a leaked key, an exposed origin secret, or a supply-chain issue). Follow
  the repository's security policy (`SECURITY.md`) when it is present. Do not put
  secret values, the Cloud Run hostname or the GCP project id into issues, chat or
  commit messages.
- **Cadence.** Post an update when the incident starts, when the cause is known, when
  it is mitigated, and when it is resolved. For a SEV1, also post at least every hour
  in between.

Update template:

```text
[SEV<n>] <short title> - <investigating | identified | mitigated | resolved>
Started: <YYYY-MM-DD HH:MM UTC>   Updated: <HH:MM UTC>
Impact: <who or what is affected, e.g. "all /generate calls return 403">
Cause (if known): <one line>
Action taken / next step: <one line>
Next update by: <HH:MM UTC>
```

## 6. Post-incident review

Write one for every SEV1 and SEV2 within a week. The review is blameless: record what
the system and the process allowed, not who made a mistake. Store it next to the
incident issue, or privately for security incidents.

```markdown
# Incident review: <title>

- Severity: SEV<n>
- Start / detection / mitigation / resolution (UTC): <times>
- Detected by: <uptime probe | budget alert | user report | operator>
- Duration of user impact: <minutes>

## Summary
<Two or three sentences: what happened and what users saw.>

## Timeline
| Time (UTC) | Event |
|---|---|

## Root cause
<The technical cause, and the condition that let it reach production.>

## What went well / what did not

## Detection gap
<Why it was not caught earlier. Example: the uptime probe only checks /health.>

## Actions
| Action | Owner | Type (prevent / detect / mitigate) | Due | Issue |
|---|---|---|---|---|

## Evidence
<Request ids, log queries, workflow run ids. No secret values.>
```

Feed confirmed lessons back into [RUNBOOK.md](RUNBOOK.md), this document, and the
alert rules once Grafana scraping exists.
