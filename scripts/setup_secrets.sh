#!/usr/bin/env bash
# One-time secret setup. Run it yourself from the repository root (Git Bash on
# Windows), after `wrangler login` (deploy/cloudflare-worker) and `gh auth login`:
#
#   bash scripts/setup_secrets.sh
#
# It generates the internal secrets, asks (with hidden input) for the values only
# you can copy from the provider dashboards, and stores each one where it is used:
#
#   ORIGIN_SHARED_SECRET     generated  -> Worker secret + GitHub secret
#   METRICS_TOKEN            generated  -> GitHub secret (+ a private local file for Grafana)
#   UPSTASH_REDIS_REST_URL   you paste  -> Worker secret + GitHub secret
#   UPSTASH_REDIS_REST_TOKEN you paste  -> Worker secret + GitHub secret
#   DATABASE_URL             you paste  -> GitHub secret; also runs migrations and
#                                          issues API keys (SMOKE_API_KEY -> GitHub)
#
# The release workflow copies the GitHub secrets into Google Secret Manager for
# Cloud Run. No value is printed except your personal API key, shown once.
set -euo pipefail

REPO=${REPO:-syed-waleed-ahmed/Niche-Finetuned-Model}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKER_DIR="$ROOT/deploy/cloudflare-worker"
# Call wrangler through node: .cmd shims break on paths containing "&".
WRANGLER=(node "$WORKER_DIR/node_modules/wrangler/bin/wrangler.js")
if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then PY="$ROOT/.venv/Scripts/python.exe"
elif [ -x "$ROOT/.venv/bin/python" ]; then PY="$ROOT/.venv/bin/python"
else PY=python3; fi
PRIVATE_DIR="$HOME/.config/fastapi-assistant"

generate() { "$PY" -c "import secrets, sys; sys.stdout.write(secrets.token_hex($1))"; }

ask() {  # ask "Prompt" -> value on stdout, input hidden
  local value
  read -r -s -p "$1: " value
  echo >&2
  printf '%s' "$value"
}

to_worker() { (cd "$WORKER_DIR" && printf '%s' "$2" | "${WRANGLER[@]}" secret put "$1" >/dev/null) && echo "  Worker secret $1 set"; }
to_github() { printf '%s' "$2" | gh secret set "$1" -R "$REPO" && echo "  GitHub secret $1 set"; }

echo "== Checking logins"
gh auth status -h github.com >/dev/null || { echo "Run: gh auth login"; exit 1; }
(cd "$WORKER_DIR" && "${WRANGLER[@]}" whoami >/dev/null 2>&1) || { echo "Run: cd deploy/cloudflare-worker && npx wrangler login"; exit 1; }

echo "== Origin secret (generated)"
origin_secret=$(generate 32)
to_worker ORIGIN_SHARED_SECRET "$origin_secret"
to_github ORIGIN_SHARED_SECRET "$origin_secret"
unset origin_secret

echo "== Metrics token (generated)"
mkdir -p "$PRIVATE_DIR" && chmod 700 "$PRIVATE_DIR" 2>/dev/null || true
metrics_token=$(generate 24)
to_github METRICS_TOKEN "$metrics_token"
printf '%s' "$metrics_token" > "$PRIVATE_DIR/metrics_token" && chmod 600 "$PRIVATE_DIR/metrics_token" 2>/dev/null || true
echo "  Saved for the Grafana scrape job in $PRIVATE_DIR/metrics_token"
unset metrics_token

echo "== Upstash Redis (console.upstash.com > fastapi-assistant-edge > REST API)"
upstash_url=$(ask "UPSTASH_REDIS_REST_URL (https://...upstash.io)")
upstash_token=$(ask "UPSTASH_REDIS_REST_TOKEN")
case "$upstash_url" in https://*) ;; *) echo "That URL should start with https://"; exit 1 ;; esac
to_worker UPSTASH_REDIS_REST_URL "$upstash_url"
to_worker UPSTASH_REDIS_REST_TOKEN "$upstash_token"
to_github UPSTASH_REDIS_REST_URL "$upstash_url"
to_github UPSTASH_REDIS_REST_TOKEN "$upstash_token"
unset upstash_url upstash_token

echo "== Neon Postgres (console.neon.tech > neon-pink-tree > Connect > database fastapi_assistant)"
database_url=$(ask "DATABASE_URL (postgresql://...sslmode=require)")
case "$database_url" in postgres://*|postgresql://*) ;; *) echo "That should start with postgresql://"; exit 1 ;; esac
to_github DATABASE_URL "$database_url"
export DATABASE_URL="$database_url"
unset database_url
echo "  Applying migrations"
"$PY" -m fastapi_assistant db migrate
echo "  Issuing the CI smoke-test key"
smoke_key=$("$PY" -m fastapi_assistant keys create smoke-test | tail -1)
to_github SMOKE_API_KEY "$smoke_key"
unset smoke_key
echo "  Issuing your personal key (shown once; paste it into the web UI's Settings)"
"$PY" -m fastapi_assistant keys create owner | tail -1
unset DATABASE_URL

echo
echo "Done. Re-run the Release workflow (or merge the PR) to deploy with these secrets."
