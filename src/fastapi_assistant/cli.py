"""Interactive terminal chat for local experimentation."""

from __future__ import annotations

from .config import Settings, get_settings
from .engines import Engine, GenerationParams, create_engine
from .prompts import build_messages


def interactive_chat(engine: Engine | None = None, settings: Settings | None = None) -> None:
    """Run a simple REPL that answers FastAPI questions until the user exits."""
    settings = settings or get_settings()
    engine = engine or create_engine(settings)
    params = GenerationParams(
        max_new_tokens=settings.model_max_new_tokens,
        temperature=settings.model_temperature,
        top_p=settings.model_top_p,
        repetition_penalty=settings.model_repetition_penalty,
    )

    print("FastAPI Niche Assistant - ask a question (type 'exit' or 'quit' to leave).")
    print(f"Loading model ({engine.backend} backend; the first load may take a moment)...\n")
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

        print("\nAssistant:")
        for event in engine.stream(build_messages(question), params):
            print(event.delta, end="", flush=True)
        print("\n")

    print("Goodbye.")
