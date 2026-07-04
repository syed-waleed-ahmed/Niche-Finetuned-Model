from .inference import generate_answer


def interactive_chat() -> None:
    print("FastAPI Niche Model - Ask a question (type 'exit' to quit):")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        print("\nModel is thinking...")
        answer = generate_answer(question)
        print(f"\nAssistant:\n{answer}")
