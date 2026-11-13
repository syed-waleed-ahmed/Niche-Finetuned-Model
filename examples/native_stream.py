"""Call the native /generate endpoint with httpx, including server-sent events.

    pip install httpx
    export ASSISTANT_URL=https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev
    export ASSISTANT_API_KEY=fa_...
    python examples/native_stream.py "What does Depends() do?"

The native API adds two things the OpenAI shape does not carry: opt-in retrieval
(``use_retrieval``) and the documentation ``sources`` used for the answer.
"""

from __future__ import annotations

import json
import os
import sys

import httpx


def main() -> int:
    base_url = os.environ.get("ASSISTANT_URL", "http://localhost:8000").rstrip("/")
    api_key = os.environ.get("ASSISTANT_API_KEY")
    if not api_key:
        print("Set ASSISTANT_API_KEY", file=sys.stderr)
        return 2
    question = " ".join(sys.argv[1:]) or "What does Depends() do in FastAPI?"
    headers = {"X-API-Key": api_key}
    payload = {"question": question, "max_new_tokens": 128, "use_retrieval": True}

    with httpx.Client(base_url=base_url, headers=headers, timeout=120) as client:
        # Blocking JSON.
        r = client.post("/generate", json=payload)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
            return 1
        body = r.json()
        print("--- blocking ---")
        print(body["answer"])
        for source in body.get("sources", []):
            print(f"  source: {source['heading']}  {source['url']}")
        print(f"[request_id={body['request_id']} usage={body['usage']}]")

        # Streaming: `event: token` frames, then one `event: done` (or `event: error`).
        print("\n--- streaming ---")
        event = None
        with client.stream("POST", "/generate", json={**payload, "stream": True}) as s:
            if s.status_code != 200:
                print(f"HTTP {s.status_code}: {s.read().decode()}", file=sys.stderr)
                return 1
            for line in s.iter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[5:])
                    if event == "token":
                        print(data["delta"], end="", flush=True)
                    elif event == "done":
                        print(f"\n[finish={data['finish_reason']} usage={data['usage']}]")
                    elif event == "error":
                        print(f"\nerror: {data['message']}", file=sys.stderr)
                        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
