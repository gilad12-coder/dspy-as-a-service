"""Console + JSON renderers for scenario results.

The console renderer is intentionally plain ASCII so terminal log scrapers
parse it cleanly. The JSON renderer mirrors :class:`ScenarioResult.as_dict`
so downstream tools can diff runs over time.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

from .metrics import ScenarioResult


def print_result(result: ScenarioResult) -> None:
    """Render one :class:`ScenarioResult` as a fixed-width text block.

    Args:
        result: Snapshot to print to stdout.
    """
    bar = "=" * 60
    err_pct = (result.errors / result.total * 100.0) if result.total else 0.0
    print(f"\n{bar}\n  {result.name}\n{bar}")
    print(f"  Requests:   {result.total}")
    print(f"  Errors:     {result.errors} ({err_pct:.1f}%)")
    print(f"  Throughput: {result.rps:.1f} rps over {result.duration_seconds:.1f}s")
    print(
        f"  Latency:    "
        f"p50={result.latency_p50_ms:.0f}ms  "
        f"p95={result.latency_p95_ms:.0f}ms  "
        f"p99={result.latency_p99_ms:.0f}ms  "
        f"max={result.latency_max_ms:.0f}ms  "
        f"mean={result.latency_mean_ms:.0f}ms"
    )
    print(f"  Codes:      {result.status_codes}")
    if result.extras:
        print(f"  Extras:     {result.extras}")


def write_json_report(results: Iterable[ScenarioResult], out_path: Path) -> Path:
    """Serialize the run's results to a single JSON file.

    Args:
        results: Iterable of :class:`ScenarioResult` snapshots in run order.
        out_path: Destination file path. Parent directory is created if missing.

    Returns:
        The same path, returned so callers can echo it to the operator.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenarios": [r.as_dict() for r in results],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return out_path


def write_markdown_report(results: Iterable[ScenarioResult], out_path: Path) -> Path:
    """Render a Markdown summary table alongside the JSON.

    Easier to skim than the raw JSON when reviewing a run on GitHub.

    Args:
        results: Iterable of scenario results in run order.
        out_path: Destination file path.

    Returns:
        The same path, returned so callers can echo it to the operator.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    rows.append("| Scenario | Reqs | Errors | RPS | p50 | p95 | p99 | Max |")
    rows.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    rows.extend(
        f"| {r.name} | {r.total} | {r.errors} | "
        f"{r.rps:.1f} | {r.latency_p50_ms:.0f}ms | "
        f"{r.latency_p95_ms:.0f}ms | {r.latency_p99_ms:.0f}ms | "
        f"{r.latency_max_ms:.0f}ms |"
        for r in results
    )
    body = "# Skynet Load Test Report\n\n"
    body += f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n\n"
    body += "\n".join(rows) + "\n\n"
    for r in results:
        if r.extras:
            body += f"## {r.name} — extras\n\n"
            body += "```\n" + json.dumps(r.extras, indent=2) + "\n```\n\n"
    out_path.write_text(body)
    return out_path
