#!/usr/bin/env bash
# The HTTP API with curl. Reads the key from the environment, never the command line
# history:
#
#   export ASSISTANT_URL=https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev
#   export ASSISTANT_API_KEY=fa_...
#   bash examples/curl.sh
set -euo pipefail

URL=${ASSISTANT_URL:-http://localhost:8000}
URL=${URL%/}
: "${ASSISTANT_API_KEY:?Set ASSISTANT_API_KEY}"
auth=(-H "X-API-Key: ${ASSISTANT_API_KEY}")
json=(-H "Content-Type: application/json")

echo "== Health (no key needed)"
curl -fsS "$URL/health"
echo

echo "== Native API: one JSON answer"
curl -fsS "${auth[@]}" "${json[@]}" -X POST "$URL/generate" \
  -d '{"question": "How do I declare a path parameter?", "max_new_tokens": 96}'
echo

echo "== Native API: server-sent events (-N disables buffering)"
curl -fsS -N "${auth[@]}" "${json[@]}" -X POST "$URL/generate" \
  -d '{"question": "How do I return a 404?", "max_new_tokens": 48, "stream": true}'
echo

echo "== OpenAI-compatible: list models (Bearer auth works too)"
curl -fsS -H "Authorization: Bearer ${ASSISTANT_API_KEY}" "$URL/v1/models"
echo

echo "== OpenAI-compatible: chat completion"
curl -fsS "${auth[@]}" "${json[@]}" -X POST "$URL/v1/chat/completions" \
  -d '{"model": "fastapi-assistant", "max_tokens": 96,
       "messages": [{"role": "user", "content": "What does Depends() do?"}]}'
echo

echo "== OpenAI-compatible: streamed chat completion"
curl -fsS -N "${auth[@]}" "${json[@]}" -X POST "$URL/v1/chat/completions" \
  -d '{"model": "fastapi-assistant", "max_tokens": 48, "stream": true,
       "messages": [{"role": "user", "content": "How do I add CORS?"}]}'
echo

echo "== Errors are JSON with a stable shape: a missing key is 401"
code=$(curl -s -o /dev/null -w '%{http_code}' "${json[@]}" -X POST "$URL/generate" -d '{"question": "hi"}')
echo "HTTP $code"
