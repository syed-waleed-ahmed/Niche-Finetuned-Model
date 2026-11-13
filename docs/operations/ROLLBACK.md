# Rollback

How to undo a bad change at each layer, fastest first. The commands assume the session
setup in [RUNBOOK.md](RUNBOOK.md) section 2.

## 1. Choose the mechanism

| What went wrong | Fastest safe rollback | Section |
|---|---|---|
| A Cloud Run deploy (code, image or config) | Send traffic to the previous revision. No build is needed. | 2 |
| The model (a bad `model-release.json`) | The previous revision also carries the previous model, so send traffic to it. Then revert the file. | 2, then 4 |
| Anything on `main` that must not be redeployed | Revert the commit on `main`; Release redeploys Cloud Run. | 3 |
| The Worker or web UI (a manual `wrangler deploy`) | `wrangler rollback` to an explicit earlier version | 5 |
| A secret rotation that broke traffic | Restore the previous Secret Manager version to GitHub and the Worker, then run Release | 6 |

**Fix the source as well.** Rolling back a deployment does not change `main`. The next
push to `main` redeploys whatever `main` contains. After an emergency rollback, revert
or fix on `main` (section 3) before merging anything else. If necessary, pause
deploys with `gh workflow disable release.yml` and re-enable them afterwards.

## 2. Cloud Run: send traffic to a previous revision

Each Release creates a new revision. A revision fixes three things:

- the image;
- the model pins (`GGUF_REPO_ID`, `GGUF_FILENAME`, `GGUF_REVISION`, `GGUF_SHA256`,
  `MODEL_VERSION`);
- the settings in `service.yaml`.

So moving traffic restores code and model together, in seconds, with no build.

1. **List revisions.** The newest are first; `ACTIVE` marks the ones receiving
   traffic.

   ```bash
   gcloud run revisions list --service fastapi-assistant --region us-central1 --limit 10
   ```

2. **Inspect a candidate.** Its image tag is `sha-<first 12 characters of the commit>`,
   and its `model-version` label comes from `model-release.json`.

   ```bash
   gcloud run revisions describe <revision> --region us-central1 \
     --format='yaml(metadata.labels, metadata.creationTimestamp, spec.containers[0].image)'
   ```

3. **Move all traffic to it.**

   ```bash
   gcloud run services update-traffic fastapi-assistant --region us-central1 \
     --to-revisions <revision>=100
   ```

4. **Confirm the split.**

   ```bash
   gcloud run services describe fastapi-assistant --region us-central1 --format='yaml(status.traffic)'
   ```

5. **Verify** (section 7).

Caveats:

- **Secrets are not rolled back.** The service references every secret as `latest`,
  and instances read them when they start. An old revision uses the current origin
  secret, API key and metrics token. That is usually what you want; if a rotation
  was the problem, use section 6.
- **Only recent images are guaranteed to exist.** The Artifact Registry cleanup policy
  from `deploy/gcp/bootstrap.sh` keeps the two most recent image versions and deletes
  older ones after a day. The previous revision's image is normally still there.
  Older revisions may not be able to start new instances if their image is gone.
  Check before relying on one:

  ```bash
  gcloud artifacts docker images list "us-central1-docker.pkg.dev/$CLOUDSDK_CORE_PROJECT/fastapi-assistant/app" --include-tags
  ```

  The multi-arch originals stay in GHCR as `ghcr.io/<owner>/<repo>:sha-<12>`. To
  restore one, re-run the Release for that commit (section 3), which copies the
  image into Artifact Registry again.
- **The pin lasts until you undo it.** A revision named in `--to-revisions` keeps its
  traffic until you change it. `deploy/cloudrun/service.yaml` has no traffic block,
  so after the next Release, run step 4 again and do not assume where traffic went.
  When the fix is live, return to normal with:

  ```bash
  gcloud run services update-traffic fastapi-assistant --region us-central1 --to-latest
  ```

## 3. Redeploy an older commit through Release

**3a. Preferred: revert on `main`.** This keeps `main` equal to production.

```bash
git switch main && git pull --ff-only
git switch -c revert/<short-name>
git revert <bad-commit-sha>            # one commit per bad change; resolve conflicts if any
git push -u origin HEAD
gh pr create --fill --base main
# after review and merge:
gh run list --workflow release.yml --limit 1
gh run watch <run-id>
```

The Release builds, scans, syncs secrets and deploys Cloud Run. It finishes by polling
`/health` and `/ready` on the new revision. It does **not** deploy the Worker while
edge deploys are manual (audit F-10). If the revert touched
`deploy/cloudflare-worker/` or `web/`, also follow RUNBOOK 7.7.

**3b. Faster but temporary: re-run an earlier successful Release.**

```bash
gh run list --workflow release.yml --branch main --status success --limit 10
gh run rerun <run-id-of-the-last-good-release>
gh run watch <run-id-of-the-last-good-release>
```

Things to know about a re-run:

- **It uses that run's commit**, including the workflow file and `model-release.json`
  as they were then.
- **Secrets are synced from today's GitHub secrets**, not the old values.
- **Only runs on `main` can deploy.** Workload Identity Federation accepts only GitHub
  tokens for this repository on `refs/heads/main`. A `gh workflow run --ref <other branch>`
  therefore fails at the Google authentication step.
- **It rebuilds and rescans the image.** If a fixable HIGH or CRITICAL CVE has been
  published since that run, the Trivy gate fails the re-run. Use section 2 instead.
- **GitHub allows re-runs only for a limited time** after the original run (check
  GitHub's current limit). For older commits, use 3a.
- **`main` still contains the bad change**, so follow up with 3a.

## 4. Model rollback

The model is defined by `model-release.json`. It pins the Hugging Face repository, the
file, an exact commit `revision` and the file's `sha256`. Rolling back the file
therefore restores byte-identical weights, as long as that revision still exists on
the Hub. Retraining is never needed.

1. **Immediate.** Send traffic to the revision that served the previous model
   (section 2). Its `model-version` label shows which model it serves.
2. **Durable.** Revert the release PR's commit on `main` (section 3a). It restores
   `model-release.json` and, for a trained release, `eval/baseline.json`.
3. The PR triggers the "Model eval gate" workflow. It fails if the PR's model scores
   below the model on `main`. When rolling back from a model that scored higher but
   behaves badly in production, the gate can fail. If that check is required by
   branch protection, the owner must decide whether to override it. Keep step 1 in
   place meanwhile.
4. **Worker cache.** Only matters once Upstash is configured: redeploy the Worker
   with `--var MODEL_VERSION:<restored model_version>` (RUNBOOK 7.7). Cached answers
   are namespaced by that value, and only the blocked CI edge job bumps it
   automatically.
5. **Verify.** `curl -sS "$EDGE_URL/health"` shows the restored `model_version`, and
   `assistant_build_info` on `/metrics` agrees.

## 5. Worker and web UI rollback

The web UI is part of the Worker deployment (Workers Static Assets). Rolling back the
Worker rolls back the UI too.

1. **List recent deployments and their version ids.** `wrangler secret put` creates a
   deployment too, so expect "secret change" entries between code deploys.

   ```bash
   wr deployments list
   wr deployments status          # the version serving now
   wr versions view <version-id>  # inspect a candidate
   ```

2. **Roll back to an explicit version.**

   ```bash
   wr rollback <version-id> --message "rollback: <reason>"
   ```

   Always pass the version id. Without it, wrangler picks the previous deployment,
   which may be a secret-change deployment rather than the code you want.

3. **Read the secrets warning.** If wrangler reports that secrets have changed since
   the target version, the target may not use the current `ORIGIN_SHARED_SECRET`.
   - Before an origin-secret rotation finished: cancel, and fix forward instead (next
     step).
   - If you confirm: check immediately for 403s (RUNBOOK check 4).

4. **Fix forward when a rollback is not suitable.** Check out the last good commit and
   deploy it as in RUNBOOK 7.7. Pass the `ORIGINS` and `MODEL_VERSION` vars, and the
   commit in `--message`. Secrets carry over to new versions.

5. **Secrets after a rollback.** After a rollback, `wrangler secret put` refuses to
   change secrets until the latest uploaded version is deployed again. This is
   wrangler's guard against deploying code by accident. Either:
   - redeploy (step 4) and then set the secret; or
   - use `wrangler versions secret put` followed by `wrangler versions deploy`.

## 6. Secret rotation rollback

Use this only when a rotation broke traffic. Never use it to bring back a value that
leaked.

GitHub and Worker secrets cannot be read back. Secret Manager keeps earlier versions
until they are destroyed (RUNBOOK 7.3), so it is the place to recover a previous value
from.

```bash
gcloud secrets versions list origin-shared-secret          # find the last good version number
gcloud secrets versions access <N> --secret origin-shared-secret | gh secret set ORIGIN_SHARED_SECRET
gcloud secrets versions access <N> --secret origin-shared-secret | wr secret put ORIGIN_SHARED_SECRET
gh workflow run release.yml --ref main                      # Release syncs GitHub -> Secret Manager -> Cloud Run
```

How this works:

- **The value never appears on screen.** It goes from Secret Manager straight into
  `gh` and `wrangler` through a pipe.
- **Release adds the restored value as a new latest version.** It compares the GitHub
  secret with Secret Manager's `latest` and adds a version when they differ.
- **Other secrets work the same way**, with these differences:
  - `api-key` and `metrics-token` have no Worker copy;
  - for `api-key`, also restore `SMOKE_API_KEY` if it held the same key.

## 7. Verify after any rollback

1. **Run RUNBOOK section 4, checks 1 to 6.** The important results:
   - check 2: `/health` shows the expected `model_version` and `load_state: "ready"`;
   - check 4: an authenticated `/generate` returns 200;
   - check 5: a direct origin call returns 403.
2. **Confirm what is serving:**
   - `gcloud run services describe fastapi-assistant --region us-central1 --format='yaml(status.traffic)'`;
   - `wr deployments status`.
3. **Watch the logs for 15 minutes:**
   - `wr tail --format pretty`: no new `upstream_failures`, 403s or 5xx;
   - the application error query in RUNBOOK 5.3 returns nothing new.
4. **Trigger an uptime run and confirm it passes:**
   `gh workflow run uptime.yml`, then `gh run list --workflow uptime.yml --limit 1`.
5. **Record it in the incident log:**
   - what was rolled back, to which revision or version, and when;
   - the follow-up on `main` (section 3a).
