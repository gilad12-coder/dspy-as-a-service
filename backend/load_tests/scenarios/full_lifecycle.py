"""Full lifecycle scenario.

Submits N optimizations, then polls each ``GET /optimizations/{id}/summary``
until the worker reaches a terminal state (``success`` / ``failed`` /
``cancelled``). The headline number is the submit-to-complete wall clock,
not just submit-side latency — that is the metric the user feels.

The mock LM completes in ~5 ms per call so each optimization wraps in
seconds rather than minutes, even with GEPA's ``auto=light`` schedule. End-
to-end latency mostly reflects: queue wait, worker subprocess spin-up,
storage churn, and the notify-once contention guarded by the new
``claim_completion_notification`` flow.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass

import httpx

from ..lib import db as db_inspector
from ..lib.auth import auth_headers
from ..lib.metrics import ScenarioMetrics, ScenarioResult
from ..lib.payloads import run_payload
from ..lib.reporter import print_result

_TERMINAL_STATUSES = frozenset({"success", "failed", "cancelled"})


@dataclass
class LifecycleConfig:
    """Knob set for one lifecycle run."""

    api_base_url: str
    mock_lm_url: str
    usernames: list[str]
    num_jobs: int
    submission_concurrency: int
    poll_interval_seconds: float
    completion_timeout_seconds: float


async def _submit_one(
    client: httpx.AsyncClient,
    *,
    api_base_url: str,
    mock_lm_url: str,
    username: str,
    seq: int,
) -> tuple[str | None, float, int]:
    """Submit one optimization and return the assigned id + submit latency.

    Args:
        client: Shared httpx async client.
        api_base_url: Backend base URL.
        mock_lm_url: Mock LM base URL.
        username: Submitter for the run.
        seq: Sequence number used to name the optimization.

    Returns:
        A ``(optimization_id, latency_seconds, status_code)`` tuple.
        ``optimization_id`` is ``None`` when the submit failed.
    """
    body = run_payload(username=username, mock_lm_url=mock_lm_url, name=f"lifecycle-{seq}")
    t0 = time.monotonic()
    try:
        response = await client.post(
            f"{api_base_url}/run",
            json=body,
            headers=auth_headers(username),
            timeout=30.0,
        )
    except (httpx.HTTPError, OSError):
        return None, time.monotonic() - t0, 0

    latency = time.monotonic() - t0
    if response.status_code != 201:
        return None, latency, response.status_code

    try:
        optimization_id = response.json().get("optimization_id")
    except ValueError:
        optimization_id = None
    if not isinstance(optimization_id, str):
        return None, latency, response.status_code
    return optimization_id, latency, response.status_code


async def _poll_until_terminal(
    client: httpx.AsyncClient,
    *,
    api_base_url: str,
    optimization_id: str,
    username: str,
    poll_interval: float,
    deadline: float,
) -> tuple[str | None, float]:
    """Poll ``GET /optimizations/{id}/summary`` until terminal or deadline.

    Args:
        client: Shared httpx async client.
        api_base_url: Backend base URL.
        optimization_id: Optimization to follow.
        username: Owner used for the bearer token.
        poll_interval: Sleep between polls.
        deadline: ``time.monotonic()`` value past which the poll gives up.

    Returns:
        ``(terminal_status, wall_clock_seconds)``. ``terminal_status`` is
        ``None`` if the deadline elapsed before terminal state.
    """
    headers = auth_headers(username)
    start = time.monotonic()
    while time.monotonic() < deadline:
        try:
            response = await client.get(
                f"{api_base_url}/optimizations/{optimization_id}/summary",
                headers=headers,
                timeout=10.0,
            )
        except (httpx.HTTPError, OSError):
            await asyncio.sleep(poll_interval)
            continue
        if response.status_code == 200:
            try:
                status = (response.json() or {}).get("status")
            except ValueError:
                status = None
            if isinstance(status, str) and status in _TERMINAL_STATUSES:
                return status, time.monotonic() - start
        await asyncio.sleep(poll_interval)
    return None, time.monotonic() - start


async def run(config: LifecycleConfig) -> ScenarioResult:
    """Submit + follow ``config.num_jobs`` optimizations end-to-end.

    Args:
        config: Per-run knobs.

    Returns:
        A :class:`ScenarioResult` whose latencies are submit→terminal
        wall-clock seconds (not just HTTP round-trip).
    """
    db_inspector.truncate_test_users(config.usernames)
    db_inspector.fund_test_users(config.usernames)

    metrics = ScenarioMetrics("full_lifecycle")
    submit_failures = 0
    submit_latencies_ms: list[float] = []
    terminal_counts: dict[str, int] = {"success": 0, "failed": 0, "cancelled": 0, "timeout": 0}

    sem = asyncio.Semaphore(config.submission_concurrency)

    async with httpx.AsyncClient(http2=False) as client:

        async def _one(seq: int) -> None:
            """Submit one optimization then follow it through to terminal."""
            nonlocal submit_failures
            user = config.usernames[seq % len(config.usernames)]
            async with sem:
                optimization_id, submit_latency, status_code = await _submit_one(
                    client,
                    api_base_url=config.api_base_url,
                    mock_lm_url=config.mock_lm_url,
                    username=user,
                    seq=seq,
                )
            submit_latencies_ms.append(submit_latency * 1000.0)
            if optimization_id is None:
                submit_failures += 1
                metrics.record(status_code=status_code or 0, latency_seconds=submit_latency)
                return

            deadline = time.monotonic() + config.completion_timeout_seconds
            terminal, total = await _poll_until_terminal(
                client,
                api_base_url=config.api_base_url,
                optimization_id=optimization_id,
                username=user,
                poll_interval=config.poll_interval_seconds,
                deadline=deadline,
            )
            if terminal is None:
                terminal_counts["timeout"] += 1
                metrics.record(status_code=599, latency_seconds=total)
            else:
                terminal_counts[terminal] += 1
                # Use 200 (terminal observed) and record submit+wait wall clock
                # as the latency so percentiles reflect the user-felt time.
                metrics.record(status_code=200, latency_seconds=submit_latency + total)

        await asyncio.gather(*[_one(i) for i in range(config.num_jobs)])

    result = metrics.finish()
    result.extras["submit_failures"] = submit_failures
    submit_sorted = sorted(submit_latencies_ms)
    if submit_sorted:
        result.extras["submit_only_p50_ms"] = round(submit_sorted[len(submit_sorted) // 2], 1)
        result.extras["submit_only_p95_ms"] = round(
            submit_sorted[min(len(submit_sorted) - 1, int(len(submit_sorted) * 0.95))],
            1,
        )
    result.extras["terminal_counts"] = terminal_counts
    result.extras["completion_timeout_seconds"] = config.completion_timeout_seconds
    return result


def default_config(api_base_url: str, mock_lm_url: str) -> LifecycleConfig:
    """Return the canonical knob set used by the orchestrator.

    Args:
        api_base_url: Backend base URL.
        mock_lm_url: Mock LM base URL.

    Returns:
        A :class:`LifecycleConfig` calibrated for the local stack: 48
        jobs in flight up to 16 at a time, 2 s poll, 5 min wall-clock cap.
    """
    return LifecycleConfig(
        api_base_url=api_base_url,
        mock_lm_url=mock_lm_url,
        usernames=[f"load-user-{i}" for i in range(8)] + ["load-user-lifecycle"],
        num_jobs=48,
        submission_concurrency=16,
        poll_interval_seconds=2.0,
        completion_timeout_seconds=300.0,
    )


async def main(api_base_url: str, mock_lm_url: str) -> ScenarioResult:
    """Run the scenario with default knobs and print the console report.

    Args:
        api_base_url: Backend base URL.
        mock_lm_url: Mock LM base URL.

    Returns:
        The :class:`ScenarioResult` produced by :func:`run`.
    """
    result = await run(default_config(api_base_url, mock_lm_url))
    print_result(result)
    return result


if __name__ == "__main__":
    asyncio.run(
        main(
            os.environ.get("LOAD_TEST_API_URL", "http://127.0.0.1:58000"),
            os.environ.get("LOAD_TEST_MOCK_LM_URL", "http://mock-lm:9000/v1"),
        )
    )
