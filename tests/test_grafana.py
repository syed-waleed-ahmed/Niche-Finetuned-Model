"""The bundled Grafana dashboard and alert rules must match the metrics the service exports.

Nothing imports these files in CI, so a renamed metric or a typo in a query would only
show up as an empty panel in production. These tests compare every query against the
real metric registry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from prometheus_client import Counter, Gauge, Histogram, Info

from fastapi_assistant.metrics import Metrics

GRAFANA = Path(__file__).resolve().parent.parent / "deploy" / "grafana"
# Labels that do not come from the service: histogram buckets, and the label the
# scrape job adds to tell replicas apart.
EXTERNAL_LABELS = {"le", "replica"}


def _exported() -> dict[str, set[str]]:
    """Every series name the service can export, with its label names."""
    series: dict[str, set[str]] = {}
    for metric in vars(Metrics()).values():
        if not isinstance(metric, (Counter, Gauge, Histogram, Info)):
            continue
        name, labels = metric._name, set(metric._labelnames)
        if isinstance(metric, Counter):
            series[f"{name}_total"] = labels
        elif isinstance(metric, Histogram):
            for suffix in ("_bucket", "_sum", "_count"):
                series[f"{name}{suffix}"] = labels
        elif isinstance(metric, Info):
            series[f"{name}_info"] = labels
        else:
            series[name] = labels
    return series


def _dashboard() -> dict:
    return json.loads((GRAFANA / "dashboard.json").read_text(encoding="utf-8"))


def _dashboard_queries() -> list[tuple[str, str]]:
    return [
        (panel["title"], target["expr"])
        for panel in _dashboard()["panels"]
        for target in panel["targets"]
    ]


def _alert_queries() -> list[tuple[str, str]]:
    yaml = pytest.importorskip("yaml")
    rules = yaml.safe_load((GRAFANA / "alerts.yaml").read_text(encoding="utf-8"))
    return [(rule["alert"], rule["expr"]) for group in rules["groups"] for rule in group["rules"]]


def _check(where: str, expr: str, exported: dict[str, set[str]]) -> None:
    used = set(re.findall(r"\bassistant_\w+", expr))
    assert used, f"{where}: the query names no service metric: {expr}"
    unknown = used - exported.keys()
    assert not unknown, f"{where}: the service does not export {sorted(unknown)}"
    allowed = EXTERNAL_LABELS.union(*(exported[name] for name in used))
    grouped = {
        label.strip()
        for group in re.findall(r"\bby\s*\(([^)]*)\)", expr)
        for label in group.split(",")
    }
    selected = set(re.findall(r"[{,]\s*(\w+)\s*(?:=~|!~|!=|=)", expr))
    bad = (grouped | selected) - allowed
    assert not bad, f"{where}: labels {sorted(bad)} do not exist on {sorted(used)}"


def test_exported_series_are_found():
    exported = _exported()
    assert exported["assistant_queue_wait_seconds_bucket"] == set()
    assert exported["assistant_time_to_first_token_seconds_bucket"] == {"endpoint"}
    assert "assistant_prompt_too_long_total" in exported


def test_dashboard_queries_use_exported_metrics_and_labels():
    exported = _exported()
    queries = _dashboard_queries()
    assert queries
    for title, expr in queries:
        _check(f"panel {title!r}", expr, exported)


def test_alert_rules_use_exported_metrics_and_labels():
    exported = _exported()
    queries = _alert_queries()
    assert queries
    for name, expr in queries:
        _check(f"alert {name}", expr, exported)


def test_a_misspelt_metric_or_label_is_caught():
    exported = _exported()
    with pytest.raises(AssertionError, match="does not export"):
        _check("x", "rate(assistant_queue_wait_second_bucket[5m])", exported)
    with pytest.raises(AssertionError, match="do not exist"):
        _check("x", "sum by (route) (rate(assistant_queue_wait_seconds_bucket[5m]))", exported)


def test_dashboard_panels_have_unique_ids_and_do_not_overlap():
    panels = _dashboard()["panels"]
    ids = [panel["id"] for panel in panels]
    assert len(ids) == len(set(ids))
    cells: set[tuple[int, int]] = set()
    for panel in panels:
        pos = panel["gridPos"]
        assert pos["x"] + pos["w"] <= 24, f"{panel['title']} is wider than the 24-column grid"
        area = {
            (x, y)
            for x in range(pos["x"], pos["x"] + pos["w"])
            for y in range(pos["y"], pos["y"] + pos["h"])
        }
        assert not area & cells, f"{panel['title']} overlaps another panel"
        cells |= area
        assert panel["datasource"] == {"type": "prometheus", "uid": "${datasource}"}
        assert all(target["datasource"] == panel["datasource"] for target in panel["targets"])
        refs = [target["refId"] for target in panel["targets"]]
        assert len(refs) == len(set(refs))


def test_dashboard_shows_the_queue_first_token_and_prompt_metrics():
    shown = " ".join(expr for _, expr in _dashboard_queries())
    for series in (
        "assistant_queue_wait_seconds_bucket",
        "assistant_time_to_first_token_seconds_bucket",
        "assistant_prompt_too_long_total",
    ):
        assert series in shown
