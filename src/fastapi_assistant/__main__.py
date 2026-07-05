"""Console entrypoint.

Usage::

    python -m fastapi_assistant            # interactive CLI
    python -m fastapi_assistant --serve    # run the HTTP API
    python -m fastapi_assistant --train    # fine-tune the LoRA adapter
"""

from __future__ import annotations

import argparse

from .config import get_settings
from .logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastapi-assistant",
        description="LoRA fine-tuned TinyLlama assistant specialized in FastAPI.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--serve", action="store_true", help="Run the FastAPI HTTP service.")
    mode.add_argument("--train", action="store_true", help="Fine-tune the LoRA adapter.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    if args.train:
        from .training import train

        train(settings)
    elif args.serve:
        import uvicorn

        from .api import app

        uvicorn.run(app, host=settings.api_host, port=settings.api_port, log_config=None)
    else:
        from .cli import interactive_chat

        interactive_chat(settings=settings)


if __name__ == "__main__":
    main()
