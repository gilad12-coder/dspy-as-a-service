"""Failure injection scenario.

Drives the failure recovery path the orphan sweeper + idempotency layer
were added for. The scenario submits a fleet of jobs, hard-kills one of
the three backend replicas while at least one job is in flight, waits
for the lease to expire and the sweeper to requeue, restarts the pod,
then asserts: every submitted job reaches a terminal state, the orphan
count drops back to zero, and an idempotent re-submit of the original
key still returns the same optimization id (idempotency survives a pod
crash).

The killed-pod path is the most painful production failure mode: it
catches missing leases, ungraceful subprocess teardown, lost completion
notifies, and partial DB writes all in one shot.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from ..lib import db as db_inspector
from ..lib.auth import auth_headers
from ..lib.metrics import ScenarioMetrics, ScenarioResult
from ..lib.payloads import run_payload
from ..lib.reporter import print_result

_TERMINAL_STATUSES = frozenset({"success", "failed", "cancelled"})


@dataclass
class FailureInjectionConfig:
    """Knob set for one failure-injection run."""

    api_base_url: str
    mock_lm_url: str
    username: str
    num_jobs: int
    compose_file: str
    compose_project: str
    target_replica: str
    lease_expiry_grace_seconds: float
    completion_timeout_seconds: float


def _docker_compose_cmd(config: FailureInjectionConfig, *args: str) -> list[str]:
    """Build a ``docker compose`` argv pointed at the load-test stack.

    Args:
        config: Scenario knobs carrying the compose file + project name.
        *args: Subcommand and its flags.

    Returns:
        The full ``docker compose ...`` argv ready for :mod:`subprocess`.
    """
    return ["docker", "compose", "-p", config.compose_project, "-f", config.compose_file, *args]


def _kill_replica(config: FailureInjectionConfig) -> bool:
    """Send SIGKILL to the targeted replica via ``docker kill``.

    The compose CLI does not expose per-replica ``kill`` so we resolve the
    container id first via ``docker compose ps`` and then ``docker kill``
    it directly.

    Args:
        config: Scenario knobs carrying compose + replica identifiers.

    Returns:
        ``True`` when the kill succeeded, ``False`` when no matching
        container could be resolved (still useful so the scenario logs
        clearly instead of crashing).
    """
    ps = subprocess.run(
        _docker_compose_cmd(config, "ps", "-q", config.target_replica.split("-")[0]),
        check=False,
        capture_output=True,
        text=True,
    )
    ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not ids:
        return False
    # When replicas: 3 is used the ids list is ordered, target_replica
    # passes the desired index (e.g. "api-2" picks index 1).
    parts = config.target_replica.split("-")
    if len(parts) == 2 and parts[1].isdigit():
        idx = int(parts[1]) - 1
        target_id = ids[idx] if 0 <= idx < len(ids) else ids[0]
    else:
        target_id = ids[0]
    subprocess.run(
        ["docker", "kill", "--signal=KILL", target_id],
        check=False,
        capture_output=True,
        text=True,
    )
    return True


def _restart_replica(config: FailureInjectionConfig) -> None:
    """Bring all replicas back up after the kill.

    ``docker compose up -d --no-deps`` is idempotent: it recreates the
    killed container without touching the surviving pods, mock LM, or
    Postgres.

    Args:
        config: Scenario knobs carrying compose identifiers.
    """
    service = config.target_replica.split("-")[0]
    subprocess.run(
        _docker_compose_cmd(config, "up", "-d", "--no-deps", service),
        check=False,
        capture_output=True,
        text=True,
    )


async def _submit(
    client: httpx.AsyncClient,
    *,
    api_base_url: str,
    mock_lm_url: str,
    username: str,
    idempotency_key: str | None,
    name_suffix: str,
) -> str | None:
    """Issue one ``POST /run`` and return its optimization id.

    Args:
        client: Shared httpx async client.
        api_base_url: Backend base URL.
        mock_lm_url: Mock LM base URL embedded in the payload.
        username: Submitter for the run.
        idempotency_key: Optional ``Idempotency-Key`` header value.
        name_suffix: Human-readable suffix used in the optimization name.

    Returns:
        The optimization id from the 201 response, or ``None`` on failure.
    """
    headers = auth_headers(username)
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    body = run_payload(username=username, mock_lm_url=mock_lm_url, name=f"failure-{name_suffix}")
    try:
        response = await client.post(
            f"{api_base_url}/run",
            json=body,
            headers=headers,
            timeout=30.0,
        )
    except (httpx.HTTPError, OSError):
        return None
    if response.status_code != 201:
        return None
    try:
        optimization_id = response.json().get("optimization_id")
    except ValueError:
        return None
    return optimization_id if isinstance(optimization_id, str) else None


async def _await_terminal(
    client: httpx.AsyncClient,
    *,
    api_base_url: str,
    username: str,
    optimization_ids: list[str],
    deadline: float,
) -> dict[str, str]:
    """Poll the summary endpoint until every id is terminal or deadline hits.

    Args:
        client: Shared httpx async client.
        api_base_url: Backend base URL.
        username: Bearer-token subject (the load-test user owns every job).
        optimization_ids: Ids to follow to terminal.
        deadline: ``time.monotonic()`` value past which polling gives up.

    Returns:
        A dict mapping optimization id → terminal status (or ``"timeout"``).
    """
    headers = auth_headers(username)
    pending = set(optimization_ids)
    final: dict[str, str] = {}
    while pending and time.monotonic() < deadline:
        for optimization_id in list(pending):
            try:
                response = await client.get(
                    f"{api_base_url}/optimizations/{optimization_id}/summary",
                    headers=headers,
                    timeout=10.0,
                )
            except (httpx.HTTPError, OSError):
                continue
            if response.status_code != 200:
                continue
            try:
                status = (response.json() or {}).get("status")
            except ValueError:
                status = None
            if isinstance(status, str) and status in _TERMINAL_STATUSES:
                final[optimization_id] = status
                pending.discard(optimization_id)
        if pending:
            await asyncio.sleep(2.0)
    for optimization_id in pending:
        final[optimization_id] = "timeout"
    return final


async def run(config: FailureInjectionConfig) -> ScenarioResult:
    """Drive submit → kill → recovery and return one scenario result.

    Args:
        config: Per-run knobs.

    Returns:
        A :class:`ScenarioResult` whose ``extras`` capture the orphan-
        recovery + idempotency invariants the scenario asserts on.
    """
    db_inspector.truncate_test_users([config.username])
    if shutil.which("docker") is None:
        raise RuntimeError(
            "docker CLI is required for the failure_injection scenario."
        )

    metrics = ScenarioMetrics("failure_injection")
    idempotency_key = "failure-injection-key"

    async with httpx.AsyncClient(http2=False) as client:
        idempotent_id = await _submit(
            client,
            api_base_url=config.api_base_url,
            mock_lm_url=config.mock_lm_url,
            username=config.username,
            idempotency_key=idempotency_key,
            name_suffix="idem-1",
        )
        idempotent_retry_id = await _submit(
            client,
            api_base_url=config.api_base_url,
            mock_lm_url=config.mock_lm_url,
            username=config.username,
            idempotency_key=idempotency_key,
            name_suffix="idem-retry",
        )

        worker_jobs: list[str] = []
        for seq in range(config.num_jobs):
            t0 = time.monotonic()
            optimization_id = await _submit(
                client,
                api_base_url=config.api_base_url,
                mock_lm_url=config.mock_lm_url,
                username=config.username,
                idempotency_key=None,
                name_suffix=f"chaos-{seq}",
            )
            metrics.record(
                status_code=201 if optimization_id else 500,
                latency_seconds=time.monotonic() - t0,
            )
            if optimization_id:
                worker_jobs.append(optimization_id)

        # Give the worker pool a beat to claim several rows before we kill
        # one of the pods — otherwise the kill is a no-op chaos test.
        await asyncio.sleep(3.0)

        killed = _kill_replica(config)
        orphans_before_sweep = 0
        if killed:
            # Wait long enough for the worker lease (60 s default) to
            # expire and the sweeper to discover the orphan row.
            await asyncio.sleep(config.lease_expiry_grace_seconds)
            orphans_before_sweep = db_inspector.count_orphaned_jobs()

        _restart_replica(config)

        deadline = time.monotonic() + config.completion_timeout_seconds
        terminal_map = await _await_terminal(
            client,
            api_base_url=config.api_base_url,
            username=config.username,
            optimization_ids=worker_jobs,
            deadline=deadline,
        )

        # After every job reaches a terminal state the orphan list must be
        # empty — anything else means the sweeper or the worker forgot a row.
        orphans_after_recovery = db_inspector.count_orphaned_jobs()

        post_crash_retry_id = await _submit(
            client,
            api_base_url=config.api_base_url,
            mock_lm_url=config.mock_lm_url,
            username=config.username,
            idempotency_key=idempotency_key,
            name_suffix="idem-post-crash",
        )

    result = metrics.finish()
    terminal_counts = {"success": 0, "failed": 0, "cancelled": 0, "timeout": 0}
    for status in terminal_map.values():
        terminal_counts[status] = terminal_counts.get(status, 0) + 1
    result.extras["pod_killed"] = killed
    result.extras["target_replica"] = config.target_replica
    result.extras["orphans_before_sweep"] = orphans_before_sweep
    result.extras["orphans_after_recovery"] = orphans_after_recovery
    result.extras["terminal_counts"] = terminal_counts
    result.extras["jobs_submitted"] = len(worker_jobs)
    result.extras["idempotent_initial_id"] = idempotent_id
    result.extras["idempotent_retry_matches_initial"] = (
        idempotent_id is not None and idempotent_retry_id == idempotent_id
    )
    result.extras["idempotent_post_crash_matches_initial"] = (
        idempotent_id is not None and post_crash_retry_id == idempotent_id
    )
    result.extras["idempotent_db_row_count"] = db_inspector.count_jobs_by_idempotency_key(
        config.username, idempotency_key
    )
    return result


def default_config(api_base_url: str, mock_lm_url: str) -> FailureInjectionConfig:
    """Return the canonical knob set used by the orchestrator.

    Args:
        api_base_url: Backend base URL.
        mock_lm_url: Mock LM base URL.

    Returns:
        A :class:`FailureInjectionConfig` calibrated for the local stack:
        12 chaos jobs, target the second api replica, 70 s lease wait,
        4 min total completion timeout.
    """
    return FailureInjectionConfig(
        api_base_url=api_base_url,
        mock_lm_url=mock_lm_url,
        username="load-user-failure",
        num_jobs=12,
        compose_file=os.environ.get(
            "LOAD_TEST_COMPOSE_FILE",
            str(Path(__file__).parent / ".." / "docker-compose.loadtest.yml"),
        ),
        compose_project=os.environ.get("LOAD_TEST_COMPOSE_PROJECT", "skynet-loadtest"),
        target_replica=os.environ.get("LOAD_TEST_TARGET_REPLICA", "api-2"),
        lease_expiry_grace_seconds=70.0,
        completion_timeout_seconds=240.0,
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
