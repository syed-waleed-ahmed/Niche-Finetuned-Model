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


def warmup_kwargs(ratio: float) -> dict:
    """Return the ``TrainingArguments`` kwarg for a warmup *ratio*.

    transformers 5.x removed ``warmup_ratio``; ``warmup_steps`` now accepts a float
    in [0, 1) that is interpreted as a ratio of total steps.
    """
    key = "warmup_steps" if _major_version() >= 5 else "warmup_ratio"
    return {key: ratio}
