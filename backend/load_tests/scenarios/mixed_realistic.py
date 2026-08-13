"""Mixed production-readiness scenario for hundreds of concurrent users.

Each virtual user ramps in once, authenticates once, and then follows a
stateful journey through the same API families the product drives in parallel:
optimization submission, dashboard reads, analytics, dataset profiling and
validation, semantic Explore search, production frontend delivery, model
discovery, job summaries, and long-lived SSE progress streams.
The mock LM keeps the run free of provider cost while real API replicas,
PgBouncer, Redis, Postgres, and workers remain in the measured path.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from ..lib import db as db_inspector
from ..lib.auth import auth_headers
from ..lib.metrics import ScenarioMetrics, ScenarioResult
from ..lib.payloads import CANONICAL_COLUMN_MAPPING, grid_payload, run_payload
from ..lib.reporter import print_result
from ..lib.slo import ScenarioSLO, apply_slo

_PROFILE_ROWS = [{"q": f"question-{index}", "a": "yes"} for index in range(24)]
_MIXED_SLO = ScenarioSLO(
    max_error_rate_percent=0.5,
    max_p95_ms=1500.0,
    max_p99_ms=3000.0,
    min_rps=40.0,
)


@dataclass(frozen=True)
class MixedRealisticConfig:
    """Knobs for the production-scale mixed journey."""

    api_base_url: str
    mock_lm_url: str
    frontend_base_url: str
    virtual_users: int
    submitting_users: int
    sse_connections: int
    ramp_seconds: float
    soak_seconds: float
    completion_timeout_seconds: float
    think_time_min_seconds: float
    think_time_max_seconds: float

    def __post_init__(self) -> None:
        """Reject workload shapes that cannot represent the requested test."""
        if self.virtual_users < 1:
            raise ValueError("virtual_users must be at least 1")
        if not 0 <= self.submitting_users <= self.virtual_users:
            raise ValueError("submitting_users must be between 0 and virtual_users")
        if not 0 <= self.sse_connections <= self.submitting_users:
            raise ValueError("sse_connections must be between 0 and submitting_users")
        if self.ramp_seconds < 0 or self.soak_seconds <= 0:
            raise ValueError("ramp_seconds must be non-negative and soak_seconds must be positive")
        if self.completion_timeout_seconds <= 0:
            raise ValueError("completion_timeout_seconds must be positive")
        if not 0 <= self.think_time_min_seconds <= self.think_time_max_seconds:
            raise ValueError("think times must be non-negative and ordered")


class _JourneyMetrics:
    """Record both aggregate and per-operation request latency."""

    def __init__(self) -> None:
        """Create empty aggregate and operation collectors."""
        self.overall = ScenarioMetrics("mixed_realistic")
        self._operations: dict[str, ScenarioMetrics] = {}
        self.semantic_responses = 0
        self.semantic_fallbacks = 0

    def record(self, operation: str, *, status_code: int, latency_seconds: float) -> None:
        """Record one request in both metric views.

        Args:
            operation: Stable journey action label.
            status_code: HTTP status, or zero after a transport failure.
            latency_seconds: Client-observed request duration.
        """
        self.overall.record(status_code=status_code, latency_seconds=latency_seconds)
        operation_metrics = self._operations.setdefault(operation, ScenarioMetrics(operation))
        operation_metrics.record(status_code=status_code, latency_seconds=latency_seconds)

    def operation_results(self) -> dict[str, dict[str, float | int]]:
        """Return compact per-operation percentile summaries.

        Returns:
            Operation names mapped to request, error, and latency statistics.
        """
        results: dict[str, dict[str, float | int]] = {}
        for name, metrics in sorted(self._operations.items()):
            result = metrics.finish()
            results[name] = {
                "requests": result.total,
                "errors": result.errors,
                "p50_ms": round(result.latency_p50_ms, 1),
                "p95_ms": round(result.latency_p95_ms, 1),
                "p99_ms": round(result.latency_p99_ms, 1),
                "max_ms": round(result.latency_max_ms, 1),
            }
        return results


async def _request(
    client: httpx.AsyncClient,
    journey_metrics: _JourneyMetrics,
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    operation: str,
    json_body: dict[str, Any] | None = None,
    request_timeout: float = 30.0,
) -> httpx.Response | None:
    """Issue and measure one journey request without aborting the scenario.

    Args:
        client: Shared connection-pooled HTTP client.
        journey_metrics: Aggregate and per-operation metric collector.
        method: HTTP method.
        url: Absolute endpoint URL.
        headers: Cached bearer headers for the virtual user.
        operation: Stable label recorded in the report.
        json_body: Optional JSON request body.
        request_timeout: Per-request timeout in seconds.

    Returns:
        The response, or ``None`` after a transport-level failure.
    """
    started = time.monotonic()
    try:
        response = await client.request(
            method,
            url,
            headers=headers,
            json=json_body,
            timeout=request_timeout,
        )
    except (httpx.HTTPError, OSError):
        journey_metrics.record(operation, status_code=0, latency_seconds=time.monotonic() - started)
        return None
    journey_metrics.record(
        operation,
        status_code=response.status_code,
        latency_seconds=time.monotonic() - started,
    )
    return response


async def _submit_job(
    client: httpx.AsyncClient,
    journey_metrics: _JourneyMetrics,
    config: MixedRealisticConfig,
    *,
    username: str,
    headers: dict[str, str],
    sequence: int,
) -> str | None:
    """Submit one run or grid-search job for a virtual user.

    Args:
        client: Shared HTTP client.
        journey_metrics: Aggregate and per-operation metric collector.
        config: URLs and workload knobs.
        username: Job owner.
        headers: Cached bearer headers.
        sequence: User index used to split run and grid-search traffic.

    Returns:
        The optimization id from a 201 response, or ``None``.
    """
    is_grid = sequence % 5 == 0
    operation = "submit_grid" if is_grid else "submit_run"
    endpoint = "/grid-search" if is_grid else "/run"
    body = (
        grid_payload(username=username, mock_lm_url=config.mock_lm_url, name=f"mixed-grid-{sequence}")
        if is_grid
        else run_payload(username=username, mock_lm_url=config.mock_lm_url, name=f"mixed-run-{sequence}")
    )
    response = await _request(
        client,
        journey_metrics,
        method="POST",
        url=f"{config.api_base_url}{endpoint}",
        headers=headers,
        operation=operation,
        json_body=body,
    )
    if response is None or response.status_code != 201:
        return None
    try:
        optimization_id = response.json().get("optimization_id")
    except ValueError:
        return None
    return optimization_id if isinstance(optimization_id, str) else None


async def _sample_sse(
    client: httpx.AsyncClient,
    journey_metrics: _JourneyMetrics,
    *,
    api_base_url: str,
    optimization_id: str,
    headers: dict[str, str],
    deadline: float,
) -> tuple[bool, int, float]:
    """Open one job stream during the mixed workload and measure its TTFB.

    Args:
        client: Shared HTTP client.
        journey_metrics: Aggregate and per-operation metric collector.
        api_base_url: Backend URL.
        optimization_id: Owned job to stream.
        headers: Cached bearer headers.
        deadline: Scenario end time.

    Returns:
        ``(opened, event_lines, ttfb_ms)``.
    """
    stream_headers = dict(headers)
    stream_headers["Accept"] = "text/event-stream"
    started = time.monotonic()
    event_lines = 0
    try:
        timeout = max(deadline - started + 5.0, 10.0)
        async with client.stream(
            "GET",
            f"{api_base_url}/optimizations/{optimization_id}/stream",
            headers=stream_headers,
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=10.0),
        ) as response:
            ttfb_ms = (time.monotonic() - started) * 1000.0
            journey_metrics.record(
                "sse_stream",
                status_code=response.status_code,
                latency_seconds=ttfb_ms / 1000.0,
            )
            if response.status_code != 200:
                return False, 0, ttfb_ms
            try:
                async for line in response.aiter_lines():
                    if line.startswith(("data:", "event:")):
                        event_lines += 1
                    if time.monotonic() >= deadline:
                        break
            except httpx.ReadTimeout:
                pass
            return True, event_lines, ttfb_ms
    except (httpx.HTTPError, OSError):
        elapsed = time.monotonic() - started
        journey_metrics.record("sse_stream", status_code=0, latency_seconds=elapsed)
        return False, 0, elapsed * 1000.0


def _dataset_profile_body() -> dict[str, Any]:
    """Build the small representative dataset-profiler payload.

    Returns:
        JSON body accepted by ``POST /datasets/profile``.
    """
    return {
        "dataset": list(_PROFILE_ROWS),
        "column_mapping": CANONICAL_COLUMN_MAPPING,
        "seed": 42,
    }


async def _browse_once(
    client: httpx.AsyncClient,
    journey_metrics: _JourneyMetrics,
    config: MixedRealisticConfig,
    *,
    username: str,
    headers: dict[str, str],
    optimization_id: str | None,
    rng: random.Random,
) -> None:
    """Run one weighted foreground action from a virtual-user journey.

    Args:
        client: Shared HTTP client.
        journey_metrics: Aggregate and per-operation metric collector.
        config: URLs and workload knobs.
        username: Current virtual user.
        headers: Cached bearer headers.
        optimization_id: User-owned job when their submission succeeded.
        rng: Deterministic per-user random source.
    """
    action = rng.choices(
        (
            "dashboard",
            "analytics",
            "dataset_profile",
            "dataset_validate",
            "models",
            "semantic_search",
            "frontend",
        ),
        weights=(32, 14, 10, 10, 10, 14, 10),
        k=1,
    )[0]
    method = "GET"
    json_body: dict[str, Any] | None = None
    request_headers = headers

    if action == "dashboard":
        dashboard_action = rng.choice(("list", "counts", "sidebar", "summary"))
        if dashboard_action == "list":
            url = f"{config.api_base_url}/optimizations?limit=20&offset=0"
        elif dashboard_action == "counts":
            url = f"{config.api_base_url}/optimizations/counts"
        elif dashboard_action == "sidebar":
            url = f"{config.api_base_url}/optimizations/sidebar"
        elif optimization_id:
            url = f"{config.api_base_url}/optimizations/{optimization_id}/summary"
        else:
            dashboard_action = "list"
            url = f"{config.api_base_url}/optimizations?limit=20&offset=0"
        operation = f"dashboard_{dashboard_action}"
    elif action == "analytics":
        url = f"{config.api_base_url}/analytics/dashboard?username={username}"
        operation = "analytics_dashboard"
    elif action == "dataset_profile":
        method = "POST"
        url = f"{config.api_base_url}/datasets/profile"
        operation = "dataset_profile"
        json_body = _dataset_profile_body()
    elif action == "dataset_validate":
        method = "POST"
        url = f"{config.api_base_url}/datasets/validate"
        operation = "dataset_validate"
        json_body = {
            "row_count": len(_PROFILE_ROWS),
            "fractions": {"train": 0.7, "val": 0.15, "test": 0.15},
        }
    elif action == "models":
        url = f"{config.api_base_url}/models"
        operation = "models_catalog"
    elif action == "semantic_search":
        method = "POST"
        url = f"{config.api_base_url}/dashboard/search"
        operation = "semantic_search"
        json_body = {
            "query": "classification quality",
            "sort": "relevance",
            "page": 1,
            "size": 20,
            "owner_username": username,
        }
    else:
        frontend_path = rng.choice(("/login", "/api/auth/session"))
        url = f"{config.frontend_base_url}{frontend_path}"
        operation = "frontend_login" if frontend_path == "/login" else "frontend_session"
        request_headers = {}

    response = await _request(
        client,
        journey_metrics,
        method=method,
        url=url,
        headers=request_headers,
        operation=operation,
        json_body=json_body,
    )
    if action != "semantic_search" or response is None or response.status_code != 200:
        return
    try:
        search_type = response.json().get("search_type")
    except ValueError:
        search_type = None
    if search_type == "semantic":
        journey_metrics.semantic_responses += 1
    else:
        journey_metrics.semantic_fallbacks += 1


async def run(config: MixedRealisticConfig) -> ScenarioResult:
    """Drive the ramped mixed workload and attach release-gate SLOs.

    Args:
        config: Production-readiness user, duration, and journey knobs.

    Returns:
        Measured scenario result with workload shape, queue state, and SLO verdict.
    """
    usernames = [f"load-mixed-{index}" for index in range(config.virtual_users)]
    db_inspector.truncate_test_users(usernames)
    db_inspector.fund_test_users(usernames[: config.submitting_users])

    journey_metrics = _JourneyMetrics()
    accepted_submissions = 0
    active_users = 0
    sse_results: list[tuple[bool, int, float]] = []
    started = time.monotonic()
    deadline = started + config.ramp_seconds + config.soak_seconds

    async def _virtual_user(index: int) -> None:
        """Run one browser-like user with an independent connection pool.

        Args:
            index: Stable user index controlling ramp position and journey seed.
        """
        nonlocal accepted_submissions, active_users
        if config.ramp_seconds > 0:
            await asyncio.sleep(config.ramp_seconds * index / max(config.virtual_users - 1, 1))

        username = usernames[index]
        headers = auth_headers(username)
        rng = random.Random(10_000 + index)
        optimization_id: str | None = None
        sse_task: asyncio.Task[tuple[bool, int, float]] | None = None
        limits = httpx.Limits(max_connections=2, max_keepalive_connections=2)
        async with httpx.AsyncClient(http2=False, limits=limits) as client:
            if index < config.submitting_users:
                optimization_id = await _submit_job(
                    client,
                    journey_metrics,
                    config,
                    username=username,
                    headers=headers,
                    sequence=index,
                )
                if optimization_id is not None:
                    accepted_submissions += 1
                    if index < config.sse_connections:
                        sse_task = asyncio.create_task(
                            _sample_sse(
                                client,
                                journey_metrics,
                                api_base_url=config.api_base_url,
                                optimization_id=optimization_id,
                                headers=headers,
                                deadline=deadline,
                            ),
                        )

            became_active = False
            while time.monotonic() < deadline:
                await _browse_once(
                    client,
                    journey_metrics,
                    config,
                    username=username,
                    headers=headers,
                    optimization_id=optimization_id,
                    rng=rng,
                )
                if not became_active:
                    active_users += 1
                    became_active = True
                await asyncio.sleep(rng.uniform(config.think_time_min_seconds, config.think_time_max_seconds))
            if sse_task is not None:
                sse_results.append(await sse_task)

    await asyncio.gather(*(_virtual_user(index) for index in range(config.virtual_users)))

    result = journey_metrics.overall.finish()
    operation_results = journey_metrics.operation_results()
    opened_streams = sum(1 for opened, _, _ in sse_results if opened)
    sse_events = sum(events for _, events, _ in sse_results)
    sse_ttfb_values = sorted(ttfb for _, _, ttfb in sse_results)
    queue_drain_started = time.monotonic()
    queue_drain_deadline = queue_drain_started + config.completion_timeout_seconds
    while True:
        status_counts = db_inspector.count_job_statuses(usernames[: config.submitting_users])
        non_terminal_jobs = sum(
            status_counts.get(status, 0)
            for status in ("pending", "validating", "running", "paused")
        )
        if non_terminal_jobs == 0 or time.monotonic() >= queue_drain_deadline:
            break
        await asyncio.sleep(1.0)
    queue_drain_seconds = time.monotonic() - queue_drain_started

    result.extras.update(
        {
            "virtual_users": config.virtual_users,
            "active_users": active_users,
            "ramp_seconds": config.ramp_seconds,
            "soak_seconds": config.soak_seconds,
            "completion_timeout_seconds": config.completion_timeout_seconds,
            "queue_drain_seconds": round(queue_drain_seconds, 3),
            "think_time_seconds": [config.think_time_min_seconds, config.think_time_max_seconds],
            "submissions_attempted": config.submitting_users,
            "submissions_accepted": accepted_submissions,
            "sse_connections_requested": min(config.sse_connections, config.submitting_users),
            "sse_connections_opened": opened_streams,
            "sse_event_lines": sse_events,
            "sse_ttfb_p95_ms": (
                round(sse_ttfb_values[min(len(sse_ttfb_values) - 1, int(len(sse_ttfb_values) * 0.95))], 1)
                if sse_ttfb_values
                else None
            ),
            "operation_counts": {name: int(operation["requests"]) for name, operation in operation_results.items()},
            "operation_latency": operation_results,
            "semantic_search_responses": journey_metrics.semantic_responses,
            "semantic_search_fallbacks": journey_metrics.semantic_fallbacks,
            "semantic_search_fallback_percent": round(
                100.0
                * journey_metrics.semantic_fallbacks
                / max(journey_metrics.semantic_responses + journey_metrics.semantic_fallbacks, 1),
                3,
            ),
            "job_status_counts_at_end": status_counts,
        },
    )

    extra_violations: list[str] = []
    if active_users != config.virtual_users:
        extra_violations.append(f"only {active_users}/{config.virtual_users} virtual users became active")
    if accepted_submissions != config.submitting_users:
        extra_violations.append(
            f"only {accepted_submissions}/{config.submitting_users} submissions were accepted",
        )
    expected_streams = min(config.sse_connections, config.submitting_users)
    if opened_streams != expected_streams:
        extra_violations.append(f"only {opened_streams}/{expected_streams} SSE streams opened")
    failed_jobs = status_counts.get("failed", 0) + status_counts.get("cancelled", 0)
    if failed_jobs:
        extra_violations.append(f"{failed_jobs} submitted jobs ended failed or cancelled")
    non_terminal_jobs = sum(status_counts.get(status, 0) for status in ("pending", "validating", "running", "paused"))
    if non_terminal_jobs:
        extra_violations.append(f"{non_terminal_jobs} submitted jobs did not reach a terminal state")
    if status_counts.get("success", 0) < accepted_submissions:
        extra_violations.append(
            f"only {status_counts.get('success', 0)}/{accepted_submissions} accepted submissions completed successfully",
        )
    if journey_metrics.semantic_responses == 0:
        extra_violations.append("no Explore search response confirmed semantic pgvector execution")
    semantic_attempts = journey_metrics.semantic_responses + journey_metrics.semantic_fallbacks
    semantic_fallback_percent = 100.0 * journey_metrics.semantic_fallbacks / max(semantic_attempts, 1)
    if semantic_fallback_percent > 5.0:
        extra_violations.append(
            f"{semantic_fallback_percent:.1f}% of Explore searches fell back from semantic mode",
        )
    frontend_requests = sum(
        int(operation_results.get(name, {}).get("requests", 0))
        for name in ("frontend_login", "frontend_session")
    )
    if frontend_requests == 0:
        extra_violations.append("the production Next.js server received no mixed-journey traffic")
    apply_slo(result, _MIXED_SLO, extra_violations=extra_violations)
    return result


def default_config(api_base_url: str, mock_lm_url: str) -> MixedRealisticConfig:
    """Build the target-scale defaults, with environment overrides for CI.

    Args:
        api_base_url: Backend load-balancer URL.
        mock_lm_url: Internal mock-provider URL used by worker subprocesses.

    Returns:
        Two hundred virtual users ramped over 20 seconds and held for a
        60-second soak, with 48 submissions and 24 concurrent SSE streams.
    """
    return MixedRealisticConfig(
        api_base_url=api_base_url,
        mock_lm_url=mock_lm_url,
        frontend_base_url=os.environ.get("LOAD_TEST_FRONTEND_URL", "http://127.0.0.1:53001"),
        virtual_users=int(os.environ.get("LOAD_TEST_MIXED_USERS", "200")),
        submitting_users=int(os.environ.get("LOAD_TEST_MIXED_SUBMIT_USERS", "48")),
        sse_connections=int(os.environ.get("LOAD_TEST_MIXED_SSE_CONNECTIONS", "24")),
        ramp_seconds=float(os.environ.get("LOAD_TEST_MIXED_RAMP_SECONDS", "20")),
        soak_seconds=float(os.environ.get("LOAD_TEST_MIXED_SOAK_SECONDS", "60")),
        completion_timeout_seconds=float(os.environ.get("LOAD_TEST_MIXED_COMPLETION_TIMEOUT_SECONDS", "120")),
        think_time_min_seconds=float(os.environ.get("LOAD_TEST_MIXED_THINK_MIN_SECONDS", "0.25")),
        think_time_max_seconds=float(os.environ.get("LOAD_TEST_MIXED_THINK_MAX_SECONDS", "1.25")),
    )


async def main(api_base_url: str, mock_lm_url: str) -> ScenarioResult:
    """Run the target-scale mixed scenario and print its report.

    Args:
        api_base_url: Backend load-balancer URL.
        mock_lm_url: Mock-provider URL.

    Returns:
        The completed, SLO-gated scenario result.
    """
    result = await run(default_config(api_base_url, mock_lm_url))
    print_result(result)
    return result


if __name__ == "__main__":
    asyncio.run(
        main(
            os.environ.get("LOAD_TEST_API_URL", "http://127.0.0.1:58000"),
            os.environ.get("LOAD_TEST_MOCK_LM_URL", "http://mock-lm:9000/v1"),
        ),
    )
