#!/usr/bin/env bash
# One-time secret setup. Run it yourself from the repository root (Git Bash on
# Windows), after `wrangler login` (deploy/cloudflare-worker) and `gh auth login`:
#
#   bash scripts/setup_secrets.sh
#
# Everything you can paste is optional: press Enter to skip, and re-run later to
# add it. With nothing pasted you still get a working, locked-down deployment.
#
#   generated:
#     ORIGIN_SHARED_SECRET     -> Worker secret + GitHub secret (only the edge may call origins)
#     API_KEY                  -> GitHub secret; your key, printed once
#     METRICS_TOKEN            -> GitHub secret + private local file (for Grafana)
#   optional, pasted (hidden input):
#     UPSTASH_REDIS_REST_URL/TOKEN -> Worker + GitHub secrets (edge cache + rate limits)
#     DATABASE_URL (Neon)          -> GitHub secret; migrates the database and enables
#                                     per-user keys, usage, and feedback
#   SMOKE_API_KEY -> GitHub secret used by the post-deploy smoke test
#
# The Release workflow copies the GitHub secrets into Google Secret Manager for Cloud
# Run. No value is printed except your API key.
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

ask() {  # ask "Prompt" -> value on stdout, input hidden, may be empty
  local value
  read -r -s -p "$1 (Enter to skip): " value
  echo >&2
  printf '%s' "$value"
}

to_worker() { (cd "$WORKER_DIR" && printf '%s' "$2" | "${WRANGLER[@]}" secret put "$1" >/dev/null) && echo "  Worker secret $1 set"; }
to_github() { printf '%s' "$2" | gh secret set "$1" -R "$REPO" && echo "  GitHub secret $1 set"; }

echo "== Checking logins"
gh auth status -h github.com >/dev/null 2>&1 || { echo "Run: gh auth login"; exit 1; }
(cd "$WORKER_DIR" && "${WRANGLER[@]}" whoami 2>/dev/null | grep -q "logged in") ||
  { echo "Run: cd deploy/cloudflare-worker && node ./node_modules/wrangler/bin/wrangler.js login"; exit 1; }
mkdir -p "$PRIVATE_DIR" && chmod 700 "$PRIVATE_DIR" 2>/dev/null || true

echo "== Origin secret (generated)"
origin_secret=$(generate 32)
to_worker ORIGIN_SHARED_SECRET "$origin_secret"
to_github ORIGIN_SHARED_SECRET "$origin_secret"
unset origin_secret

echo "== Metrics token (generated)"
metrics_token=$(generate 24)
to_github METRICS_TOKEN "$metrics_token"
printf '%s' "$metrics_token" > "$PRIVATE_DIR/metrics_token" && chmod 600 "$PRIVATE_DIR/metrics_token" 2>/dev/null || true
echo "  Saved for the Grafana scrape job in $PRIVATE_DIR/metrics_token"
unset metrics_token

echo "== Your API key (generated)"
api_key="fa_$(generate 32)"
to_github API_KEY "$api_key"
smoke_key="$api_key"

echo "== Upstash Redis, optional (console.upstash.com > fastapi-assistant-edge > REST API)"
upstash_url=$(ask "UPSTASH_REDIS_REST_URL (https://...upstash.io)")
if [ -n "$upstash_url" ]; then
  case "$upstash_url" in https://*) ;; *) echo "That URL should start with https://"; exit 1 ;; esac
  upstash_token=$(ask "UPSTASH_REDIS_REST_TOKEN")
  [ -n "$upstash_token" ] || { echo "The token is required with the URL"; exit 1; }
  to_worker UPSTASH_REDIS_REST_URL "$upstash_url"
  to_worker UPSTASH_REDIS_REST_TOKEN "$upstash_token"
  to_github UPSTASH_REDIS_REST_URL "$upstash_url"
  to_github UPSTASH_REDIS_REST_TOKEN "$upstash_token"
  unset upstash_token
else
  echo "  Skipped: the edge runs without cache and rate limits until you add it."
fi
unset upstash_url

echo "== Neon Postgres, optional (console.neon.tech > neon-pink-tree > Connect > database fastapi_assistant)"
database_url=$(ask "DATABASE_URL (postgresql://...sslmode=require)")
if [ -n "$database_url" ]; then
  case "$database_url" in postgres://*|postgresql://*) ;; *) echo "That should start with postgresql://"; exit 1 ;; esac
  to_github DATABASE_URL "$database_url"
  export DATABASE_URL="$database_url"
  echo "  Applying migrations"
  "$PY" -m fastapi_assistant db migrate
  echo "  Issuing a database key for the CI smoke test"
  smoke_key=$("$PY" -m fastapi_assistant keys create smoke-test | tail -1)
  unset DATABASE_URL
else
  echo "  Skipped: the service uses your static API key only (no per-user keys or usage)."
fi
unset database_url
to_github SMOKE_API_KEY "$smoke_key"
unset smoke_key

echo
echo "Your API key (shown once; paste it into the web UI's Settings):"
echo "  $api_key"
unset api_key
echo
echo "Done. Tell Claude 'secrets done' (or re-run the Release workflow) to deploy."
