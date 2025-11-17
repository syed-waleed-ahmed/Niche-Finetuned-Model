from src.inference import generate_answer


def main():
    print("FastAPI Niche Model – Ask a question (type 'exit' to quit):")
    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        print("\nModel is thinking...")
        answer = generate_answer(question)
        print(f"\nAssistant:\n{answer}")


if __name__ == "__main__":
    main()
