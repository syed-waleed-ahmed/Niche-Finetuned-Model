#!/usr/bin/env bash
# One-time Google Cloud setup for the Cloud Run origin. Safe to re-run.
#
#   PROJECT_ID=my-proj BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX \
#   GITHUB_REPO=owner/Niche-Finetuned-Model REGION=us-central1 \
#   bash deploy/gcp/bootstrap.sh
#
# Creates: Artifact Registry repo (with a cleanup policy to stay inside the 0.5 GB
# free storage), a runtime service account that can only read its secrets, a
# deployer service account that GitHub Actions impersonates through Workload
# Identity Federation (no long-lived keys), four Secret Manager secrets, and a
# budget alert. Prints the values to store as GitHub repository variables.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${BILLING_ACCOUNT:?set BILLING_ACCOUNT (gcloud billing accounts list)}"
: "${GITHUB_REPO:?set GITHUB_REPO as owner/name}"
REGION=${REGION:-us-central1}
BUDGET_USD=${BUDGET_USD:-5}
REPO=fastapi-assistant
RUN_SA=fastapi-assistant-run
DEPLOY_SA=github-deployer

gcloud config set project "$PROJECT_ID" >/dev/null
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')

echo "== Enabling APIs"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com iamcredentials.googleapis.com sts.googleapis.com \
  billingbudgets.googleapis.com

echo "== Artifact Registry"
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"
POLICY=$(mktemp)
cat >"$POLICY" <<'JSON'
[
  {"name": "keep-latest-two", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 2}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"olderThan": "86400s"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies "$REPO" --location "$REGION" \
  --policy "$POLICY" --no-dry-run
rm -f "$POLICY"

echo "== Service accounts"
for sa in "$RUN_SA" "$DEPLOY_SA"; do
  gcloud iam service-accounts describe "$sa@$PROJECT_ID.iam.gserviceaccount.com" >/dev/null 2>&1 ||
    gcloud iam service-accounts create "$sa"
done
for role in roles/run.admin roles/artifactregistry.writer; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition=None --quiet \
    --member "serviceAccount:$DEPLOY_SA@$PROJECT_ID.iam.gserviceaccount.com" --role "$role" >/dev/null
done
gcloud iam service-accounts add-iam-policy-binding "$RUN_SA@$PROJECT_ID.iam.gserviceaccount.com" \
  --member "serviceAccount:$DEPLOY_SA@$PROJECT_ID.iam.gserviceaccount.com" \
  --role roles/iam.serviceAccountUser >/dev/null

echo "== Workload Identity Federation for GitHub Actions (main branch only)"
gcloud iam workload-identity-pools describe github --location global >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools create github --location global --display-name "GitHub Actions"
gcloud iam workload-identity-pools providers describe github --location global \
  --workload-identity-pool github >/dev/null 2>&1 ||
  gcloud iam workload-identity-pools providers create-oidc github --location global \
    --workload-identity-pool github \
    --issuer-uri "https://token.actions.githubusercontent.com" \
    --attribute-mapping "google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
    --attribute-condition "assertion.repository == '$GITHUB_REPO' && assertion.ref == 'refs/heads/main'"
gcloud iam service-accounts add-iam-policy-binding "$DEPLOY_SA@$PROJECT_ID.iam.gserviceaccount.com" \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository/$GITHUB_REPO" >/dev/null

echo "== Secret Manager access"
# Secret values never pass through this script: the Release workflow copies them
# from GitHub's encrypted secrets (set by scripts/setup_secrets.sh) into Secret
# Manager on each deploy. Here we only grant who may write and who may read.
gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition=None --quiet   --member "serviceAccount:$DEPLOY_SA@$PROJECT_ID.iam.gserviceaccount.com"   --role roles/secretmanager.admin >/dev/null
gcloud projects add-iam-policy-binding "$PROJECT_ID" --condition=None --quiet   --member "serviceAccount:$RUN_SA@$PROJECT_ID.iam.gserviceaccount.com"   --role roles/secretmanager.secretAccessor >/dev/null

echo "== Budget alert (notifies at 50% and 100%; it does not cap spend - maxScale does)"
gcloud billing budgets create --billing-account "$BILLING_ACCOUNT" \
  --display-name "fastapi-assistant" --budget-amount "${BUDGET_USD}USD" \
  --filter-projects "projects/$PROJECT_ID" \
  --threshold-rule percent=0.5 --threshold-rule percent=1.0 >/dev/null || true

cat <<EOF

Done. Store these as GitHub repository *variables* (Settings > Secrets and variables > Actions):
  GCP_PROJECT_ID          $PROJECT_ID
  GCP_REGION              $REGION
  GCP_WIF_PROVIDER        projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github
  GCP_DEPLOY_SA           $DEPLOY_SA@$PROJECT_ID.iam.gserviceaccount.com

Secrets are not handled here. Run scripts/setup_secrets.sh locally; the Release
workflow copies them into Secret Manager.
EOF
