"""Paired bootstrap of two evaluation reports over the same items.

    python scripts/paired_bootstrap.py eval/reference/gate1-finetune-qwen.json \
        eval/reference/gate1-base-qwen.json

The first report minus the second, per item: the mean difference and a 95%
interval from 10,000 resamples of the items. The interval covers which items
were sampled, not training variance: every report here comes from one
training run with one seed. Standard library only.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from statistics import fmean

RESAMPLES = 10_000
METRICS = ("score", "keyword_recall", "code_valid", "rouge_l")


def load(path: str) -> dict[str, dict]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return {item["id"]: item for item in report["items"]}


def interval(deltas: list[float], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(fmean(rng.choices(deltas, k=len(deltas))) for _ in range(RESAMPLES))
    return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def main(first_path: str, second_path: str) -> None:
    first, second = load(first_path), load(second_path)
    ids = sorted(set(first) & set(second))
    print(f"{Path(first_path).name} minus {Path(second_path).name}, {len(ids)} shared items")
    for metric in METRICS:
        pairs = [(first[i][metric], second[i][metric]) for i in ids]
        deltas = [a - b for a, b in pairs if a is not None and b is not None]
        if not deltas:
            continue
        low, high = interval(deltas)
        better = sum(d > 1e-9 for d in deltas)
        worse = sum(d < -1e-9 for d in deltas)
        print(
            f"  {metric:<15} n={len(deltas):<4} difference {fmean(deltas):+.4f}"
            f"  95% CI [{low:+.4f}, {high:+.4f}]  better {better}, worse {worse}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python scripts/paired_bootstrap.py FIRST.json SECOND.json")
    main(sys.argv[1], sys.argv[2])
