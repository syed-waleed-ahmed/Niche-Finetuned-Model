"""Interactive terminal chat for local experimentation."""

from __future__ import annotations

from .config import Settings, get_settings
from .inference import AssistantEngine


def interactive_chat(engine: AssistantEngine | None = None, settings: Settings | None = None) -> None:
    """Run a simple REPL that answers FastAPI questions until the user exits."""
    settings = settings or get_settings()
    engine = engine or AssistantEngine(settings)

    print("FastAPI Niche Assistant — ask a question (type 'exit' or 'quit' to leave).")
    print("Loading model (the first response may take a moment)...\n")
    engine.load()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        print("\n...thinking...")
        answer = engine.generate(question)
        print(f"\nAssistant:\n{answer}\n")

    print("Goodbye.")
