"""Dashboard read scenario.

Drives the read endpoints the dashboard frontend depends on
(``/optimizations``, ``/optimizations/counts``, ``/optimizations/sidebar``,
``GET /optimizations/{id}/summary``) at sustained concurrency, then briefly
opens an SSE stream against ``/optimizations/{id}/stream`` to check that
streaming routes do not regress while the read tier is hot.

Why this matters: in production every active user keeps a dashboard tab
open, so the read tier handles orders of magnitude more requests per
submission than the write tier. A regression here is the most likely
cause of "the UI hangs" reports even when submission latency looks fine.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass

import httpx

from ..lib import db as db_inspector
from ..lib.auth import auth_headers
from ..lib.metrics import ScenarioResult
from ..lib.payloads import run_payload
from ..lib.reporter import print_result
from ..lib.runner import RequestSpec, run_load


@dataclass
class DashboardReadConfig:
    """Knob set for one dashboard-read run."""

    api_base_url: str
    usernames: list[str]
    seed_jobs_per_user: int
    read_requests: int
    concurrency: int
    sse_sample_duration_seconds: float


async def _seed_jobs(
    client: httpx.AsyncClient,
    *,
    api_base_url: str,
    mock_lm_url: str,
    usernames: list[str],
    per_user: int,
) -> list[tuple[str, str]]:
    """Submit a small batch of jobs so the read endpoints see realistic rows.

    Args:
        client: Shared httpx async client.
        api_base_url: Backend base URL.
        mock_lm_url: Mock LM base URL embedded in seed payloads.
        usernames: Submitters to round-robin through.
        per_user: How many jobs each user submits.

    Returns:
        ``(optimization_id, owner_username)`` pairs for every successfully
        created job. Returning the owner lets the read mix address each id
        with the bearer token that owns it; non-admin callers hitting
        somebody else's job get a 404 by design.
    """
    seeded: list[tuple[str, str]] = []
    for user in usernames:
        for seq in range(per_user):
            body = run_payload(username=user, mock_lm_url=mock_lm_url, name=f"seed-{user}-{seq}")
            try:
                response = await client.post(
                    f"{api_base_url}/run",
                    json=body,
                    headers=auth_headers(user),
                    timeout=30.0,
                )
            except (httpx.HTTPError, OSError):
                continue
            if response.status_code != 201:
                continue
            try:
                optimization_id = response.json().get("optimization_id")
            except ValueError:
                continue
            if isinstance(optimization_id, str):
                seeded.append((optimization_id, user))
    return seeded


def _build_read_specs(
    *,
    api_base_url: str,
    usernames: list[str],
    seeded_jobs: list[tuple[str, str]],
    n: int,
) -> list[RequestSpec]:
    """Mix the four read endpoints in proportions matching real dashboard load.

    Args:
        api_base_url: Backend base URL.
        usernames: Users to rotate Authorization tokens across (for the
            list/counts/sidebar reads — every authenticated user sees those
            tabs in the dashboard).
        seeded_jobs: ``(optimization_id, owner)`` pairs from the seeding
            step. Summary calls draw from this list and authenticate as
            the owner so a non-admin caller can read their own job.
        n: Total read requests to generate.

    Returns:
        A list of :class:`RequestSpec` weighted: 40% list, 20% counts,
        10% sidebar, 30% summary-by-id. Matches what the React app does
        when the user lands on /dashboard and idles for a minute.
    """
    rng = random.Random(1337)
    specs: list[RequestSpec] = []
    weights = [
        ("list", 40),
        ("counts", 20),
        ("sidebar", 10),
        ("summary", 30),
    ]
    bag = [kind for kind, weight in weights for _ in range(weight)]
    for i in range(n):
        kind = bag[i % len(bag)]
        user = usernames[i % len(usernames)]
        if kind == "list":
            url = f"{api_base_url}/optimizations?limit=20&offset=0"
        elif kind == "counts":
            url = f"{api_base_url}/optimizations/counts"
        elif kind == "sidebar":
            url = f"{api_base_url}/optimizations/sidebar"
        else:
            if not seeded_jobs:
                continue
            optimization_id, user = rng.choice(seeded_jobs)
            url = f"{api_base_url}/optimizations/{optimization_id}/summary"
        specs.append(RequestSpec(method="GET", url=url, username=user, timeout=15.0))
    return specs


async def _sample_sse(
    client: httpx.AsyncClient,
    *,
    api_base_url: str,
    optimization_id: str,
    username: str,
    duration: float,
) -> tuple[int, float]:
    """Open the per-job SSE stream and count events received in ``duration``.

    Args:
        client: Shared httpx async client.
        api_base_url: Backend base URL.
        optimization_id: Optimization id to follow.
        username: Owner used for the bearer token.
        duration: Wall-clock seconds to keep the stream open.

    Returns:
        ``(event_lines, ttfb_seconds)`` — how many SSE lines arrived and
        the time-to-first-byte for the stream.
    """
    headers = auth_headers(username)
    headers["Accept"] = "text/event-stream"
    deadline = time.monotonic() + duration
    events = 0
    ttfb = 0.0
    t0 = time.monotonic()
    try:
        async with client.stream(
            "GET",
            f"{api_base_url}/optimizations/{optimization_id}/stream",
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=duration + 5.0, write=10.0, pool=10.0),
        ) as response:
            if response.status_code != 200:
                return events, time.monotonic() - t0
            async for raw_line in response.aiter_lines():
                if ttfb == 0.0:
                    ttfb = time.monotonic() - t0
                if raw_line.startswith(("data:", "event:")):
                    events += 1
                if time.monotonic() >= deadline:
                    break
    except (httpx.HTTPError, OSError):
        return events, ttfb or (time.monotonic() - t0)
    return events, ttfb


async def run(config: DashboardReadConfig, *, mock_lm_url: str) -> ScenarioResult:
    """Seed a few jobs, hammer the read endpoints, then sample the SSE stream.

    Args:
        config: Per-run knobs.
        mock_lm_url: Mock LM URL used while seeding rows.

    Returns:
        A :class:`ScenarioResult` summarising the read tier with SSE
        sample stats attached as extras.
    """
    db_inspector.fund_test_users(config.usernames)
    async with httpx.AsyncClient(http2=False) as client:
        seeded_jobs = await _seed_jobs(
            client,
            api_base_url=config.api_base_url,
            mock_lm_url=mock_lm_url,
            usernames=config.usernames,
            per_user=config.seed_jobs_per_user,
        )

        specs = _build_read_specs(
            api_base_url=config.api_base_url,
            usernames=config.usernames,
            seeded_jobs=seeded_jobs,
            n=config.read_requests,
        )
        result = await run_load(
            name="dashboard_read",
            specs=specs,
            concurrency=config.concurrency,
            client=client,
        )

        sse_events = 0
        sse_ttfb_ms = 0.0
        if seeded_jobs:
            sse_optimization_id, sse_owner = seeded_jobs[0]
            sse_events, sse_ttfb = await _sample_sse(
                client,
                api_base_url=config.api_base_url,
                optimization_id=sse_optimization_id,
                username=sse_owner,
                duration=config.sse_sample_duration_seconds,
            )
            sse_ttfb_ms = round(sse_ttfb * 1000.0, 1)

    result.extras["seeded_jobs"] = len(seeded_jobs)
    result.extras["sse_event_lines"] = sse_events
    result.extras["sse_ttfb_ms"] = sse_ttfb_ms
    return result


def default_config(api_base_url: str) -> DashboardReadConfig:
    """Return the canonical knob set used by the orchestrator.

    Args:
        api_base_url: Backend base URL.

    Returns:
        A :class:`DashboardReadConfig` calibrated for the local stack:
        16 seed rows, 2 000 read requests, 32 in flight, 5 s SSE sample.
    """
    return DashboardReadConfig(
        api_base_url=api_base_url,
        usernames=[f"load-user-{i}" for i in range(4)] + ["load-user-dashboard"],
        seed_jobs_per_user=3,
        read_requests=2000,
        concurrency=32,
        sse_sample_duration_seconds=5.0,
    )


async def main(api_base_url: str, mock_lm_url: str) -> ScenarioResult:
    """Run the scenario with default knobs and print the console report.

    Args:
        api_base_url: Backend base URL.
        mock_lm_url: Mock LM URL (used for the seeding step only).

    Returns:
        The :class:`ScenarioResult` produced by :func:`run`.
    """
    result = await run(default_config(api_base_url), mock_lm_url=mock_lm_url)
    print_result(result)
    return result


if __name__ == "__main__":
    asyncio.run(
        main(
            os.environ.get("LOAD_TEST_API_URL", "http://127.0.0.1:58000"),
            os.environ.get("LOAD_TEST_MOCK_LM_URL", "http://mock-lm:9000/v1"),
        )
    )
