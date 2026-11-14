#!/usr/bin/env bash
# Does an abandoned non-streaming request free the generation slot, end to end?
#
#   bash scripts/check_abandoned_request.sh [BASE_URL]
#   ASSISTANT_API_KEY=fa_... bash scripts/check_abandoned_request.sh https://my-edge.example
#
# It sends four requests, so it is a check, not a load test:
#   1. a tiny warm-up;
#   2. a long generation, timed to its end (about GENERATION_TIMEOUT_S);
#   3. the same long generation, dropped by the client after 2 s;
#   4. at once, a tiny request, timed.
# With one generation slot, request 4 waits behind request 3. If the abandoned
# generation was cancelled, it returns in a few seconds; if not, it takes about as long
# as request 2. Run it when nobody else is using the service, or the queue skews it.
set -u

URL=${1:-${ASSISTANT_URL:-https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev}}
URL=${URL%/}
auth=()
[ -n "${ASSISTANT_API_KEY:-}" ] && auth=(-H "X-API-Key: ${ASSISTANT_API_KEY}")
LONG='{"question":"Write a complete FastAPI application with ten different endpoints covering path parameters, query parameters, request bodies, dependencies, background tasks, file uploads, WebSockets, middleware, error handlers and testing. Explain every endpoint in detail.","max_new_tokens":1024,"temperature":0}'
TINY='{"question":"What is FastAPI?","max_new_tokens":8,"temperature":0}'

post() {  # post BODY MAX_SECONDS OUTPUT_FILE -> "status seconds"
  curl -s -o "$3" -w "%{http_code} %{time_total}" --max-time "$2" -X POST "$URL/generate" \
    "${auth[@]}" -H 'Content-Type: application/json' -d "$1"
}

echo "service:  $(curl -s "$URL/health")"
echo "warm-up:  $(post "$TINY" 120 /dev/null)"
out=$(mktemp)
long=$(post "$LONG" 150 "$out")
echo "long generation, to its end:              $long  $(grep -o '"finish_reason":"[a-z]*"' "$out" | head -1)"
rm -f "$out"
sleep 2
post "$LONG" 2 /dev/null > /dev/null   # abandoned by the client after 2 s
tiny=$(post "$TINY" 150 /dev/null)
echo "tiny request behind an abandoned long one: $tiny"

long_s=${long#* }
tiny_s=${tiny#* }
if awk -v t="$tiny_s" -v l="$long_s" 'BEGIN { exit !(t < l / 4) }'; then
  echo "RESULT: the abandoned generation was cancelled (the slot was free after ${tiny_s} s)."
else
  echo "RESULT: the abandoned generation kept the slot (${tiny_s} s, against ${long_s} s for a full one)."
  exit 1
fi
