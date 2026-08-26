"""Async HTTP driver used by every load-test scenario.

The driver owns the httpx client + concurrency semaphore so scenarios can
focus on workload shape, not boilerplate. ``run_load`` accepts a generator
of request specs and returns a :class:`ScenarioResult` once every coroutine
has finished.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from .auth import auth_headers
from .metrics import ScenarioMetrics, ScenarioResult


@dataclass
class RequestSpec:
    """One HTTP request the scenario wants the driver to execute."""

    method: str
    url: str
    username: str
    json_body: dict[str, Any] | None = None
    headers: dict[str, str] | None = None
    timeout: float = 30.0


async def _execute(
    client: httpx.AsyncClient,
    metrics: ScenarioMetrics,
    sem: asyncio.Semaphore,
    spec: RequestSpec,
    on_response: Callable[[httpx.Response], Awaitable[None]] | None,
) -> None:
    """Issue one request honouring the concurrency semaphore.

    Failures from the transport (DNS, connection refused, timeouts) are
    treated as ``status_code=0`` so they count as errors without crashing
    the scenario.

    Args:
        client: Shared httpx async client.
        metrics: Scenario metric collector.
        sem: Concurrency semaphore controlling in-flight requests.
        spec: The request to execute.
        on_response: Optional async hook invoked with successful responses
            so scenarios can stash response bodies (e.g. job ids) for
            follow-up steps.
    """
    headers = dict(spec.headers or {})
    headers.update(auth_headers(spec.username))
    async with sem:
        t0 = time.monotonic()
        try:
            response = await client.request(
                spec.method,
                spec.url,
                headers=headers,
                json=spec.json_body,
                timeout=spec.timeout,
            )
            metrics.record(status_code=response.status_code, latency_seconds=time.monotonic() - t0)
            if on_response is not None:
                await on_response(response)
        except (httpx.HTTPError, OSError):
            metrics.record(status_code=0, latency_seconds=time.monotonic() - t0)


async def run_load(
    *,
    name: str,
    specs: AsyncIterator[RequestSpec] | list[RequestSpec],
    concurrency: int,
    on_response: Callable[[httpx.Response], Awaitable[None]] | None = None,
    client: httpx.AsyncClient | None = None,
) -> ScenarioResult:
    """Run ``specs`` against the cluster with bounded concurrency.

    Args:
        name: Scenario name shown in the report.
        specs: Either a pre-built list of :class:`RequestSpec` or an async
            iterator that yields them lazily.
        concurrency: Maximum in-flight requests.
        on_response: Optional async callback for each successful response.
        client: Optional pre-built httpx async client; one is created and
            closed automatically when omitted.

    Returns:
        The :class:`ScenarioResult` snapshot once every request has settled.
    """
    metrics = ScenarioMetrics(name)
    sem = asyncio.Semaphore(concurrency)
    own_client = client is None
    real_client = client or httpx.AsyncClient(http2=False)

    try:
        tasks: list[asyncio.Task[None]] = []
        if isinstance(specs, list):
            tasks.extend(
                asyncio.create_task(_execute(real_client, metrics, sem, spec, on_response))
                for spec in specs
            )
        else:
            tasks.extend(
                [
                    asyncio.create_task(_execute(real_client, metrics, sem, spec, on_response))
                    async for spec in specs
                ]
            )
        if tasks:
            await asyncio.gather(*tasks)
    finally:
        if own_client:
            await real_client.aclose()

    return metrics.finish()
