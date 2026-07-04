import argparse

from src.api import app
from src.cli import interactive_chat
from src.config import API_HOST, API_PORT


def serve_api() -> None:
    import uvicorn

    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FastAPI niche assistant")
    parser.add_argument("--serve", action="store_true", help="Run the FastAPI service instead of the CLI")
    args = parser.parse_args(argv)

    if args.serve:
        serve_api()
    else:
        interactive_chat()


if __name__ == "__main__":
    main()