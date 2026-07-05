"""Small cross-version compatibility helpers for the transformers library."""

from __future__ import annotations

from transformers import __version__ as _TRANSFORMERS_VERSION


def _major_version() -> int:
    try:
        return int(_TRANSFORMERS_VERSION.split(".")[0])
    except (ValueError, IndexError):  # pragma: no cover - defensive
        return 0


def dtype_kwargs(dtype) -> dict:
    """Return the correct ``from_pretrained`` dtype kwarg for the installed version.

    transformers 5.x renamed ``torch_dtype`` to ``dtype`` and deprecated the old
    name. Selecting the right key keeps the code warning-free on new versions while
    remaining compatible with 4.x.
    """
    key = "dtype" if _major_version() >= 5 else "torch_dtype"
    return {key: dtype}
