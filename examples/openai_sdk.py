"""Call the assistant with the official OpenAI Python SDK (blocking and streaming).

    pip install "openai>=1.40"
    export ASSISTANT_URL=https://fastapi-assistant-edge.fastapi-assistant-edge.workers.dev
    export ASSISTANT_API_KEY=fa_...        # never hard-code a key
    python examples/openai_sdk.py "How do I declare a path parameter?"

The service is OpenAI-compatible under /v1, so any OpenAI client works by changing
the base URL. It serves one model; the ``model`` field is accepted and ignored.
"""

from __future__ import annotations

import os
import sys

from openai import APIStatusError, OpenAI


def main() -> int:
    base_url = os.environ.get("ASSISTANT_URL", "http://localhost:8000").rstrip("/")
    api_key = os.environ.get("ASSISTANT_API_KEY")
    if not api_key:
        print("Set ASSISTANT_API_KEY", file=sys.stderr)
        return 2
    question = " ".join(sys.argv[1:]) or "How do I declare a path parameter in FastAPI?"

    client = OpenAI(base_url=f"{base_url}/v1", api_key=api_key, timeout=120, max_retries=2)
    messages = [{"role": "user", "content": question}]

    print("models:", [m.id for m in client.models.list().data])

    # Blocking: one JSON response with usage.
    try:
        response = client.chat.completions.create(
            model="fastapi-assistant", messages=messages, max_tokens=128, temperature=0.2
        )
    except APIStatusError as exc:
        # 400 context_length_exceeded, 401, 413, 429 (with Retry-After), 503 ...
        print(f"HTTP {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    print("\n--- blocking ---")
    print(response.choices[0].message.content)
    print(f"[finish={response.choices[0].finish_reason} usage={response.usage}]")

    # Streaming: text arrives as server-sent events, token by token.
    print("\n--- streaming ---")
    stream = client.chat.completions.create(
        model="fastapi-assistant", messages=messages, max_tokens=128, stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
