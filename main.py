"""Backwards-compatible entrypoint.

Allows running the app directly from a clone without installing the package::

    python main.py            # interactive CLI
    python main.py --serve    # run the HTTP API
    python main.py --train    # fine-tune the LoRA adapter

For installed usage, prefer ``python -m fastapi_assistant`` or the
``fastapi-assistant`` console script.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src-layout package is importable when running from source.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fastapi_assistant.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
