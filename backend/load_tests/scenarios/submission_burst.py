"""Submission burst scenario.

Fires a synchronous burst of optimization submissions across multiple
load-test users to measure how the queue + storage layer behaves under a
sustained ingress spike. The companion idempotency sub-test verifies that
the partial unique index on ``(username, idempotency_key)`` deduplicates
concurrent retries no matter how many pods race the insert.

Why this matters: the submission router is the only path that takes the
full per-user quota lock, validates the payload synchronously, and writes
a row before returning 201. If it cannot sustain a realistic spike, every
downstream optimization stalls behind the API tier.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable
from dataclasses import dataclass

import httpx

from ..lib import db as db_inspector
from ..lib.metrics import ScenarioResult
from ..lib.payloads import grid_payload, run_payload
from ..lib.reporter import print_result
from ..lib.runner import RequestSpec, run_load


@dataclass
class BurstConfig:
    """Knob set for one burst run."""

    api_base_url: str
    mock_lm_url: str
    usernames: list[str]
    run_requests: int
    grid_requests: int
    idempotent_retries: int
    concurrency: int


def _round_robin_specs(
    *,
    api_base_url: str,
    mock_lm_url: str,
    usernames: list[str],
    n_runs: int,
    n_grids: int,
) -> list[RequestSpec]:
    """Build a flat list of POST specs distributed across load-test users.

    Args:
        api_base_url: Backend base URL (no trailing slash).
        mock_lm_url: Mock LM base URL passed into every payload.
        usernames: Load-test usernames to rotate through.
        n_runs: Number of ``POST /run`` requests to generate.
        n_grids: Number of ``POST /grid-search`` requests to generate.

    Returns:
        A list of :class:`RequestSpec` with users rotated round-robin so
        no single quota / connection bucket dominates the spike.
    """
    specs: list[RequestSpec] = []
    for i in range(n_runs):
        user = usernames[i % len(usernames)]
        specs.append(
            RequestSpec(
                method="POST",
                url=f"{api_base_url}/run",
                username=user,
                json_body=run_payload(username=user, mock_lm_url=mock_lm_url, name=f"burst-run-{i}"),
            )
        )
    for i in range(n_grids):
        user = usernames[i % len(usernames)]
        specs.append(
            RequestSpec(
                method="POST",
                url=f"{api_base_url}/grid-search",
                username=user,
                json_body=grid_payload(username=user, mock_lm_url=mock_lm_url, name=f"burst-grid-{i}"),
            )
        )
    return specs


def _idempotent_specs(
    *,
    api_base_url: str,
    mock_lm_url: str,
    username: str,
    idempotency_key: str,
    retries: int,
) -> list[RequestSpec]:
    """Build ``retries`` identical ``POST /run`` specs that share one key.

    Args:
        api_base_url: Backend base URL.
        mock_lm_url: Mock LM base URL.
        username: Submitter who owns the deduped job.
        idempotency_key: Shared ``Idempotency-Key`` header value.
        retries: Number of concurrent retries to fire.

    Returns:
        ``retries`` :class:`RequestSpec` items pointing at the same
        endpoint with the same body and idempotency header.
    """
    body = run_payload(username=username, mock_lm_url=mock_lm_url, name="burst-idem")
    return [
        RequestSpec(
            method="POST",
            url=f"{api_base_url}/run",
            username=username,
            json_body=body,
            headers={"Idempotency-Key": idempotency_key},
        )
        for _ in range(retries)
    ]


async def _harvest_optimization_ids(responses: list[httpx.Response]) -> set[str]:
    """Return the unique ``optimization_id`` values across successful responses.

    Args:
        responses: Captured 2xx responses from the idempotent burst.

    Returns:
        The deduped set of optimization ids the backend returned. Should
        be ``{single_id}`` when idempotency is honoured.
    """
    seen: set[str] = set()
    for response in responses:
        try:
            payload = response.json()
        except ValueError:
            continue
        optimization_id = payload.get("optimization_id")
        if isinstance(optimization_id, str):
            seen.add(optimization_id)
    return seen


async def run(config: BurstConfig) -> ScenarioResult:
    """Drive a submission burst + idempotency sub-test and return one result.

    Args:
        config: Per-run knobs (URLs, user pool, request counts, concurrency).

    Returns:
        A :class:`ScenarioResult` with ``extras`` populated by the
        idempotency invariant counts so the reporter surfaces them.
    """
    db_inspector.truncate_test_users(config.usernames)
    db_inspector.fund_test_users(config.usernames)

    specs = _round_robin_specs(
        api_base_url=config.api_base_url,
        mock_lm_url=config.mock_lm_url,
        usernames=config.usernames,
        n_runs=config.run_requests,
        n_grids=config.grid_requests,
    )

    captured: list[httpx.Response] = []

    async def _capture(response: httpx.Response) -> None:
        """Stash 201 responses so the idempotency assertion can read job ids."""
        if response.status_code == 201:
            captured.append(response)

    async with httpx.AsyncClient(http2=False) as client:
        result = await run_load(
            name="submission_burst",
            specs=specs,
            concurrency=config.concurrency,
            on_response=_capture,
            client=client,
        )

        idempotent_user = config.usernames[0]
        idempotency_key = "burst-idempotency-key"
        idempotent_specs = _idempotent_specs(
            api_base_url=config.api_base_url,
            mock_lm_url=config.mock_lm_url,
            username=idempotent_user,
            idempotency_key=idempotency_key,
            retries=config.idempotent_retries,
        )
        idempotent_responses: list[httpx.Response] = []

        async def _capture_idem(response: httpx.Response) -> None:
            """Capture every response so we can count distinct ids returned."""
            idempotent_responses.append(response)

        idem_result = await run_load(
            name="submission_burst_idempotent",
            specs=idempotent_specs,
            concurrency=config.idempotent_retries,
            on_response=_capture_idem,
            client=client,
        )

    unique_ids = await _harvest_optimization_ids(idempotent_responses)
    db_unique = db_inspector.count_jobs_by_idempotency_key(idempotent_user, idempotency_key)

    result.extras["bulk_submitted"] = len(specs)
    result.extras["successful_submissions"] = len(captured)
    result.extras["idempotent_retries_sent"] = config.idempotent_retries
    result.extras["idempotent_retries_p95_ms"] = idem_result.latency_p95_ms
    result.extras["idempotent_unique_response_ids"] = len(unique_ids)
    result.extras["idempotent_db_row_count"] = db_unique

    return result


def _iter_default_usernames() -> Iterable[str]:
    """Yield the default load-test usernames declared in the compose env.

    Returns:
        Eight rotated usernames matching the ``QUOTA_OVERRIDES`` entries
        the docker-compose stack wires unlimited quota for.
    """
    return (f"load-user-{i}" for i in range(8))


def default_config(api_base_url: str, mock_lm_url: str) -> BurstConfig:
    """Return the canonical knob set used by the orchestrator.

    Args:
        api_base_url: Backend base URL discovered by the orchestrator.
        mock_lm_url: Mock LM base URL injected into every payload.

    Returns:
        A :class:`BurstConfig` calibrated for a single-host local stack:
        200 mixed submissions, 50 idempotent retries, 32 in flight.
    """
    return BurstConfig(
        api_base_url=api_base_url,
        mock_lm_url=mock_lm_url,
        usernames=list(_iter_default_usernames()),
        run_requests=160,
        grid_requests=40,
        idempotent_retries=50,
        concurrency=32,
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
