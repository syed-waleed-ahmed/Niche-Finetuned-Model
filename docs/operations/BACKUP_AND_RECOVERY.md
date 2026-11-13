# Backup and recovery

This document covers:

- what state the system has and where it lives;
- what protects each piece;
- the recovery targets;
- how to recover, up to rebuilding the whole stack from an empty cloud account.

Commands assume the session setup in [RUNBOOK.md](RUNBOOK.md) section 2.

The service is designed to be close to stateless. Code, configuration and the model
pin live in git. Weights live on the Hugging Face Hub. Running instances keep nothing
that needs a backup. The state that matters is the secrets, and, once enabled, the
Neon database.

## 1. State inventory

| State | Authoritative copy | History and backup | If it is lost |
|---|---|---|---|
| Source, workflows, deploy configuration, `model-release.json`, `eval/baseline.json`, datasets (`data/*.jsonl`), Grafana dashboard and alerts (`deploy/grafana/`) | The GitHub repository, `main` | Full git history; every clone is a copy. Keep a periodic mirror (section 4.1). | Restore from a mirror or any up-to-date clone |
| Container images | GHCR: multi-arch, tagged `sha-<12>` and `latest`. Artifact Registry: amd64, the two most recent versions. | Rebuildable from git. Dependencies are hash-pinned, but each build applies the Debian security updates available that day, so a rebuild is not byte-identical. | Re-run Release for the commit |
| Model weights (GGUF) | The Hugging Face repository and commit named in `model-release.json`. Today this is a third-party repository (`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF`) that the owner does not control. | Pinned by commit revision and SHA-256. Any copy with the same SHA-256 is the same model. No copy is kept by this project today. | Cold starts fail; running instances keep serving until they stop (section 4.3) |
| Fine-tuned adapter and GGUF (`v2026.11.13-2b04940` and later releases) | The owner's Hugging Face namespace (`HF_NAMESPACE`), tagged per release | Hugging Face repository history; Kaggle kernel output; the `release` workflow artifact (short retention) | Retrain from git (data and code are versioned) with the "Train on Kaggle" workflow. The credentials are configured, and the first release was made this way (audit F-04). |
| Documentation index for retrieval | Built at image build time from FastAPI docs `0.162.5` (`DEFAULT_FASTAPI_REF`) and baked into the image | Rebuildable: `fastapi-assistant rag build` | Rebuilt by the next image build |
| Secrets: `ORIGIN_SHARED_SECRET`, `API_KEY`, `METRICS_TOKEN`, `SMOKE_API_KEY`; optional `DATABASE_URL`, `SENTRY_DSN`, `UPSTASH_*` | GitHub Actions secrets. They are write-only: they cannot be read back. | Google Secret Manager keeps every version that Release synced until it is destroyed; readable by authorized accounts. The Worker holds its own write-only copy of `ORIGIN_SHARED_SECRET`. `setup_secrets.sh` keeps local copies in `~/.config/fastapi-assistant/` (`metrics_token`, and `api_key` in non-interactive mode). | Recover from Secret Manager, or generate new values (section 4.4) |
| CI credentials: the `CLOUDFLARE_API_TOKEN` and `KAGGLE_API_TOKEN` GitHub secrets, and `HF_TOKEN` attached to the Kaggle kernel | GitHub and Kaggle. They are write-only: they cannot be read back. | None. They are not synced to Secret Manager. | Issue a new token at the vendor (Cloudflare, Kaggle, Hugging Face) and set it again. Nothing that serves traffic depends on them. |
| GitHub repository variables: `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`, `EDGE_ORIGINS`, `EDGE_URL`, `CLOUDFLARE_ACCOUNT_ID`, `KAGGLE_USERNAME`, `HF_NAMESPACE`, optional `CLOUD_RUN_MAX_INSTANCES` | GitHub | `bootstrap.sh` prints the GCP values again when re-run. Keep a private record: `gh variable list` (the output contains the project id; store it privately). | Re-create with `gh variable set` |
| GCP infrastructure: Artifact Registry repository and cleanup policy, service accounts, Workload Identity Federation, IAM bindings, budget | `deploy/gcp/bootstrap.sh`, safe to re-run | The script is the backup | Re-run `bootstrap.sh` |
| Cloud Run service | `deploy/cloudrun/service.yaml`, rendered by Release | Revisions (see [ROLLBACK.md](ROLLBACK.md)) | Re-run Release |
| Edge Worker and web UI | `deploy/cloudflare-worker/`, `web/`, plus deploy-time vars `ORIGINS` and `MODEL_VERSION` | Cloudflare keeps recent versions (`wr deployments list`) | Re-run Release, which deploys the Worker from git; or deploy by hand (RUNBOOK 7.7) |
| Neon Postgres (not configured today): API key hashes, usage events, feedback, optional pgvector index | Neon | Neon's point-in-time restore, within the restore window of the plan in use [vendor, re-check]. Optionally, scheduled `pg_dump` exports kept elsewhere. | Restore with Neon (section 4.5). Keys can be re-issued. |
| Upstash Redis (not configured today): rate-limit counters, answer cache, verified-key markers | Upstash | None needed: all of it is disposable and rebuilt by traffic | Nothing to restore; the Worker runs without it |
| Logs | Cloud Logging (the default bucket keeps logs for a limited period [vendor, re-check]); Workers Logs (a few days on the Free plan [vendor, re-check]); GitHub Actions run logs (time-limited retention [vendor, re-check]) | None | Export anything needed for an incident review promptly |
| Metrics | Process memory on each Cloud Run instance only | None: nothing scrapes `/metrics` | Lost when an instance stops. Not recoverable. |

## 2. Recovery targets

These are **targets, not measurements**. No recovery drill has been run against this
deployment, so the time each step takes is unknown. Run a drill (section 5) and
replace these with measured figures.

- **RPO** (recovery point objective): how much data may be lost.
- **RTO** (recovery time objective): how long the service may be down.

| Scenario | RPO target | RTO target | Main dependency |
|---|---|---|---|
| Bad deploy (code, model or Worker) | None: no data involved | 15 minutes (revision or Worker rollback) | [ROLLBACK.md](ROLLBACK.md) |
| Secret leaked or lost | None: values are regenerated | 1 hour, plus the time to give clients the new key | `setup_secrets.sh` and Release |
| Hugging Face repository or revision unavailable | None, if a verified copy of the GGUF exists | Running instances are unaffected; new instances are blocked until a copy is published (section 4.3) | A copy of the weights you control |
| Cloudflare account or Worker lost | None | 1 hour (the edge URL changes if the account changes) | Git and Secret Manager |
| GCP project lost | None for code; secrets survive in GitHub | 4 hours | Billing account, `bootstrap.sh`, Release |
| Everything (new accounts) | Neon data since the last export | 1 working day | Section 4.7 |
| Neon data corruption or loss (when enabled) | Within Neon's point-in-time restore window | 1 hour | Neon console |
| Upstash lost (when enabled) | Not applicable (disposable) | None: the service keeps working without it | |

## 3. Routine protection

| Task | When | How |
|---|---|---|
| Mirror the repository | Weekly, and before risky repository operations | Section 4.1 |
| Record repository variables privately | After any change to them | `gh variable list` to a private file |
| Keep a verified copy of the deployed GGUF | When `model-release.json` changes | Download the pinned revision, check the SHA-256 (RUNBOOK 7.6), store it somewhere you control |
| Keep Secret Manager versions for one rotation cycle, then destroy them | After each rotation | RUNBOOK 7.3 |
| Export Neon data (once enabled) | On a schedule chosen by the owner | `pg_dump` with the Neon connection string, kept outside Neon |
| Rehearse a restore | Twice a year, or after major changes | Section 5 |

## 4. Recovery procedures

### 4.1 Repository

- **Mirror:**

  ```bash
  git clone --mirror <repository URL> fastapi-assistant.git
  ```

  Refresh it later with `git -C fastapi-assistant.git remote update`.
- **Restore into a new, empty repository:**

  ```bash
  git -C fastapi-assistant.git push --mirror <new repository URL>
  ```

After restoring into a repository with a different owner or name:

1. **Recreate the variables and secrets.** Secrets come from section 4.4 (with
   `REPO=<owner>/<name>` set for `setup_secrets.sh`, because the script has a default
   repository).
2. **Re-point Workload Identity Federation.** `bootstrap.sh` does not change an
   existing provider, so update its condition, then re-run `bootstrap.sh` with the new
   `GITHUB_REPO` to bind the deployer account:

   ```bash
   gcloud iam workload-identity-pools providers update-oidc github --location global \
     --workload-identity-pool github \
     --attribute-condition "assertion.repository == '<owner>/<name>' && assertion.ref == 'refs/heads/main'"
   ```

3. **Remove the old repository's binding.** Run
   `gcloud iam service-accounts get-iam-policy` on the deployer account, then
   `gcloud iam service-accounts remove-iam-policy-binding` for the old repository's
   principal set.

### 4.2 Images

Nothing to restore by hand. Re-run Release for the commit you want (ROLLBACK.md,
section 3). It rebuilds the image, pushes it to GHCR and copies the amd64 image into
Artifact Registry.

### 4.3 Model weights unavailable on the Hugging Face Hub

**Symptoms.** A cold start logs `Fetching ...`, then `Failed to load model` with a
download or not-found error in `exc_info`. New revisions fail their startup probe.
Instances that are already running keep serving until Cloud Run stops them.

1. Keep traffic where it is. Do not deploy, because every deploy starts new instances.
2. Find a copy whose SHA-256 equals `gguf.sha256` in `model-release.json`. Possible
   sources:
   - your verified offline copy;
   - the Hugging Face cache on a workstation that has served the model
     (`~/.cache/huggingface/hub`).

   Check it with `sha256sum`, or `"$PY" -c "from pathlib import Path; from fastapi_assistant.hub import sha256_file; print(sha256_file(Path('<file>')))"`.
3. Publish it where the service can fetch it, then pin the new location in
   `model-release.json`:
   - Normal route: upload it to a Hugging Face repository the owner controls, under
     `HF_NAMESPACE` (needs a Hugging Face write token for that account). Change
     `repo_id` and `revision`, and keep `sha256` the same. Merge, and Release deploys
     it.
   - Alternative: baking the file into the image needs Dockerfile and Release changes
     that are not prepared, and costs Artifact Registry storage (see
     [COST_MANAGEMENT.md](COST_MANAGEMENT.md)).
4. Keeping the same `sha256` proves the replacement is the same model. Never change
   the hash to fit a file you have not verified.

### 4.4 Secrets

- **A secret leaked.** Rotate it: RUNBOOK 7.1 (all) or 7.2 (one). Then destroy the old
  Secret Manager versions (RUNBOOK 7.3).
- **GitHub secrets lost** (for example, a new repository) **but Secret Manager
  intact.** Copy the values across without displaying them:

  ```bash
  gcloud secrets versions access latest --secret origin-shared-secret | gh secret set ORIGIN_SHARED_SECRET
  gcloud secrets versions access latest --secret api-key       | gh secret set API_KEY
  gcloud secrets versions access latest --secret api-key       | gh secret set SMOKE_API_KEY
  gcloud secrets versions access latest --secret metrics-token | gh secret set METRICS_TOKEN
  ```

  Clients keep working, because nothing changed.
- **Secret Manager lost** (for example, a new project) **but GitHub secrets intact.**
  Run Release. Its sync step creates each missing secret and adds the GitHub value.
  The Worker already holds the matching origin secret.
- **Both lost.** Run `bash scripts/setup_secrets.sh` (it generates everything new),
  then Release, then give clients the new API key.
- **Adding Upstash or Neon later.** Re-running `setup_secrets.sh` without options
  rotates every generated secret. Run `bash scripts/setup_secrets.sh --storage-only`
  instead, then Release: it asks only for those values and rotates nothing. By hand,
  the equivalent is:
  - Upstash: `gh secret set` for `UPSTASH_REDIS_REST_URL` and
    `UPSTASH_REDIS_REST_TOKEN` (Release uploads both to the Worker);
  - Neon: `gh secret set DATABASE_URL`, then `"$PY" -m fastapi_assistant db migrate`
    with `DATABASE_URL` exported.

  Then run Release.

### 4.5 Neon database (once enabled)

1. Use Neon's point-in-time restore (console, project, branch restore) to a moment
   before the damage, within the plan's restore window [vendor, re-check].
2. If the restore produces a new branch or endpoint, update the connection string:
   `gh secret set DATABASE_URL`, then Release.
3. Apply migrations: `"$PY" -m fastapi_assistant db migrate`. It is idempotent, and
   records applied versions in `schema_migrations`.
4. **Re-revoke keys.** Keys revoked after the restore point are active again. Revoke
   them again with `keys revoke`, using the incident log or `keys list`.
5. Usage events and feedback written after the restore point are lost.

While the database is down, the static `API_KEY` keeps working. A database key lookup
raises an error and the request fails with a server error instead of being admitted
(fail closed). Usage writes fail and are logged.

### 4.6 Cloudflare Worker or account

- **Same account.** Re-run Release (`gh workflow run release.yml --ref main`). Its
  `deploy-edge` job deploys the Worker from git and uploads `ORIGIN_SHARED_SECRET`
  from the GitHub secret. If the Worker was deleted and that job fails, re-create the
  secret from Secret Manager first (this also creates the Worker), then re-run Release
  or deploy by hand (RUNBOOK 7.7):

  ```bash
  gcloud secrets versions access latest --secret origin-shared-secret | wr secret put ORIGIN_SHARED_SECRET
  ```

- **New account.**
  1. `wr login`.
  2. Set the secret as above.
  3. In the new account, create an API token from the "Edit Cloudflare Workers"
     template. Store it with `gh secret set CLOUDFLARE_API_TOKEN`, and the account id
     with `gh variable set CLOUDFLARE_ACCOUNT_ID --body <account id>`, so Release
     deploys to the new account.
  4. Re-run Release, or deploy by hand (RUNBOOK 7.7).
  5. The Worker gets a new `workers.dev` URL. Update it with
     `gh variable set EDGE_URL --body <new URL>`, and tell API clients.

### 4.7 Rebuild the whole stack from scratch

Use this after losing the GCP project, or to stand up a fresh copy. Allow for some
downtime.

**Prerequisites:**

- the owner's Google Cloud billing account;
- a Cloudflare account, with the `CLOUDFLARE_API_TOKEN` secret and the
  `CLOUDFLARE_ACCOUNT_ID` variable set so Release can deploy the Worker (section 4.6);
- admin access to the GitHub repository;
- `gcloud`, `gh`, Node 22, and a Python 3.11 virtualenv with the project installed
  (`pip install -e .`);
- bash (Git Bash on Windows).

1. **Project and billing.** Create a project and link billing (Cloud console, or
   `gcloud projects create <id>` and `gcloud billing projects link <id> --billing-account <account>`).
2. **Bootstrap Google Cloud.** It creates the APIs, Artifact Registry with its cleanup
   policy, the runtime and deployer service accounts, Workload Identity Federation
   limited to `main`, the Secret Manager roles, and the budget alert.

   ```bash
   PROJECT_ID=<id> BILLING_ACCOUNT=<account> GITHUB_REPO=<owner>/<name> REGION=us-central1 \
     bash deploy/gcp/bootstrap.sh
   ```

3. **Repository variables.** Store the four printed values with
   `gh variable set GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WIF_PROVIDER` and
   `GCP_DEPLOY_SA` (each with `--body <value>`).
   Optionally set `gh variable set CLOUD_RUN_MAX_INSTANCES --body 2`.
4. **Secrets.**
   - If the GitHub secrets survived, skip this step: Release recreates Secret Manager.
   - Otherwise:

     ```bash
     (cd deploy/cloudflare-worker && npm ci)
     gh auth login
     wr login
     bash scripts/setup_secrets.sh
     ```

     It generates the origin secret, metrics token and API key, sets the Worker and
     GitHub secrets, and creates the Worker if it does not exist yet. Set
     `REPO=<owner>/<name>` first if the repository is not the script's default.
5. **Deploy the origin.**

   ```bash
   gh workflow run release.yml --ref main
   ```

   Then watch it with `gh run watch`. The `Deploy Cloud Run` job ends by polling
   `/health` and `/ready`. The first cold start downloads the model. This run also
   deploys the Worker, still with the previous (or an empty) `EDGE_ORIGINS`; step 7
   corrects that.
6. **Point the edge at the origin:**

   ```bash
   gh variable set EDGE_ORIGINS --body "$(gcloud run services describe fastapi-assistant --region us-central1 --format='value(status.url)')"
   ```

7. **Deploy the Worker and web UI.** Re-run Release
   (`gh workflow run release.yml --ref main`) so `deploy-edge` uses the new
   `EDGE_ORIGINS`, or deploy by hand (RUNBOOK 7.7). If the Worker URL changed, set
   `gh variable set EDGE_URL --body <worker URL>`. That enables the Uptime workflow
   and the edge smoke test.
8. **Verify.** Run RUNBOOK section 4, checks 1 to 6, then
   `gh workflow run uptime.yml`.
9. **Give API clients the new key** (and the new URL if it changed).

## 5. Restore drill

A drill turns the targets in section 2 into measurements. Suggested scope, in a
separate GCP project and on a separate Worker name, so production is untouched:

1. Time steps 1 to 8 of section 4.7.
2. Time a revision rollback and a Worker rollback in the drill environment.
3. With Neon enabled: restore a branch to a point in time, and confirm that the key
   revocations were re-applied.
4. Record the measured times, and update section 2.
5. Delete the drill project and Worker afterwards, so they do not accrue cost.
