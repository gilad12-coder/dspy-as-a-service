"""Tests for the shared helpers used by multiple domain routers."""

from __future__ import annotations

import base64
import pickle
from collections.abc import Generator
from datetime import UTC, datetime

import dspy
import pytest
from fastapi import HTTPException

from ...constants import TQDM_REMAINING_KEY
from ...models import OptimizationStatus
from ...models.artifacts import ProgramArtifact, ReactOverlay
from ...service_gateway.optimization.tool_overlay import hash_tool_schema
from ...service_gateway.react_compat import REACT_CLASS
from ...storage.usage import StorageUsage
from ..auth import AuthenticatedUser
from ..errors import DomainError

# noinspection PyProtectedMember
from ..routers import _helpers as _helpers_mod

# noinspection PyProtectedMember
from ..routers._helpers import (
    _materialize_program,
    _stable_hash,
    build_summary,
    clear_program_cache,
    enforce_storage_quota,
    load_program,
    strip_api_key,
)
from ..routers.constants import (
    TERMINAL_STATUSES,
    VALID_OPTIMIZATION_TYPES,
    VALID_STATUSES,
)
from .mocks import load_fixture

_TEST_USER = AuthenticatedUser(username="alice", role="admin", groups=("skynet-admins",))


def test_strip_api_key_removes_nested() -> None:
    """``strip_api_key`` removes the nested ``api_key`` while preserving siblings."""
    out = strip_api_key({"name": "gpt", "extra": {"api_key": "sk-secret", "region": "us"}})
    assert "api_key" not in out["extra"]
    assert out["extra"] == {"region": "us"}
    assert out["name"] == "gpt"


def test_strip_api_key_passthrough_without_extra() -> None:
    """A model config without ``extra`` round-trips unchanged."""
    out = strip_api_key({"name": "gpt"})
    assert out == {"name": "gpt"}


def test_strip_api_key_does_not_mutate_input() -> None:
    """``strip_api_key`` returns a new dict and never mutates its input."""
    src = {"name": "gpt", "extra": {"api_key": "sk-x"}}
    strip_api_key(src)
    assert src == {"name": "gpt", "extra": {"api_key": "sk-x"}}


def test_valid_statuses_match_enum() -> None:
    """The whitelisted status set matches the enum exactly."""
    assert {s.value for s in OptimizationStatus} == VALID_STATUSES


def test_terminal_statuses_are_finite() -> None:
    """Only ``success``/``failed``/``cancelled``/``paused`` are considered terminal."""
    assert {
        OptimizationStatus.success,
        OptimizationStatus.failed,
        OptimizationStatus.cancelled,
        OptimizationStatus.paused,
    } == TERMINAL_STATUSES


def test_valid_job_types_covers_run_and_grid() -> None:
    """The whitelisted optimization types cover the public run/grid options."""
    assert {"run", "grid_search"} == VALID_OPTIMIZATION_TYPES


def test_build_summary_on_pending_job_without_result() -> None:
    """A pending job without result data still produces a valid summary."""
    now = datetime.now(UTC)
    job_data = {
        "optimization_id": "abc123",
        "status": "pending",
        "created_at": now.isoformat(),
        "started_at": None,
        "completed_at": None,
        "payload_overview": {
            "job_type": "run",
            "module_name": "predict",
            "optimizer_name": "gepa",
            "model_name": "gpt-4o-mini",
            "username": "alice",
        },
        "result": None,
        "latest_metrics": {},
        "progress_count": 0,
        "log_count": 0,
    }
    summary = build_summary(job_data)
    assert summary.optimization_id == "abc123"
    assert summary.status.value == "pending"
    assert summary.baseline_test_metric is None
    assert summary.metric_improvement is None


def test_build_summary_computes_improvement() -> None:
    """The summary reports baseline, optimized, and metric improvement values."""
    now = datetime.now(UTC)
    fixture_result = load_fixture("jobs/success_single_gepa.detail.json")["result"]
    job_data = {
        "optimization_id": "job42",
        "status": "success",
        "created_at": now.isoformat(),
        "started_at": now.isoformat(),
        "completed_at": now.isoformat(),
        "payload_overview": {
            "job_type": "run",
            "module_name": "predict",
            "optimizer_name": "gepa",
            "username": "bob",
        },
        "result": fixture_result,
        "latest_metrics": {},
        "progress_count": 0,
        "log_count": 0,
    }
    summary = build_summary(job_data)
    assert summary.baseline_test_metric == 52.89
    assert summary.optimized_test_metric == 75.9
    assert summary.metric_improvement == pytest.approx(23.01, abs=1e-2)


def test_build_summary_grid_search_uses_best_pair_metrics() -> None:
    """A grid-search summary surfaces best-pair metrics and counters."""
    now = datetime.now(UTC).isoformat()
    fixture_grid_result = load_fixture("jobs/success_grid.detail.json")["grid_result"]
    job_data = {
        "optimization_id": "gs1",
        "status": "success",
        "created_at": now,
        "started_at": now,
        "completed_at": now,
        "payload_overview": {
            "optimization_type": "grid_search",
            "module_name": fixture_grid_result["module_name"],
            "optimizer_name": fixture_grid_result["optimizer_name"],
            "username": "carol",
        },
        "result": fixture_grid_result,
        "latest_metrics": {},
        "progress_count": 0,
        "log_count": 0,
    }
    summary = build_summary(job_data)

    best_pair = fixture_grid_result["best_pair"]
    assert summary.baseline_test_metric == best_pair["baseline_test_metric"]
    assert summary.optimized_test_metric == best_pair["optimized_test_metric"]
    assert summary.metric_improvement == pytest.approx(best_pair["metric_improvement"], abs=1e-6)
    gen = best_pair["generation_model"]
    ref = best_pair["reflection_model"]
    assert summary.best_pair_label == f"{gen} + {ref}"
    assert summary.completed_pairs == fixture_grid_result["completed_pairs"]
    assert summary.failed_pairs == fixture_grid_result["failed_pairs"]


def test_build_summary_grid_search_live_counters_from_latest_metrics() -> None:
    """A running grid search surfaces live pair counters from ``latest_metrics``."""
    now = datetime.now(UTC).isoformat()
    job_data = {
        "optimization_id": "gs2",
        "status": "running",
        "created_at": now,
        "started_at": now,
        "completed_at": None,
        "payload_overview": {"optimization_type": "grid_search"},
        "result": None,
        "latest_metrics": {"completed_so_far": 2, "failed_so_far": 1},
        "progress_count": 0,
        "log_count": 0,
    }
    summary = build_summary(job_data)

    assert summary.completed_pairs == 2
    assert summary.failed_pairs == 1


def test_build_summary_estimated_remaining_set_for_running_job() -> None:
    """A running job exposes a formatted ``estimated_remaining`` value."""
    now = datetime.now(UTC).isoformat()
    job_data = {
        "optimization_id": "r1",
        "status": "running",
        "created_at": now,
        "started_at": now,
        "completed_at": None,
        "payload_overview": {},
        "result": None,
        "latest_metrics": {TQDM_REMAINING_KEY: 90},
        "progress_count": 0,
        "log_count": 0,
    }
    summary = build_summary(job_data)

    assert summary.estimated_remaining == "00:01:30"


def test_build_summary_estimated_remaining_absent_for_finished_job() -> None:
    """A finished job suppresses ``estimated_remaining`` even when metrics linger."""
    now = datetime.now(UTC).isoformat()
    fixture_result = load_fixture("jobs/success_single_gepa.detail.json")["result"]
    job_data = {
        "optimization_id": "r2",
        "status": "success",
        "created_at": now,
        "started_at": now,
        "completed_at": now,
        "payload_overview": {},
        "result": fixture_result,
        "latest_metrics": {TQDM_REMAINING_KEY: 999},
        "progress_count": 0,
        "log_count": 0,
    }
    summary = build_summary(job_data)

    assert summary.estimated_remaining is None


class _StorageJobStore:
    """Store stub returning a canned storage total and budget for any user."""

    def __init__(self, *, used: int, quota: int) -> None:
        """Capture the canned used-bytes total and byte budget.

        Args:
            used: Bytes ``compute_user_storage`` reports for any user.
            quota: Budget ``get_effective_user_storage_quota`` reports.
        """
        self._used = used
        self._quota = quota

    def compute_user_storage(self, username: str) -> StorageUsage:
        """Return the canned usage for any user.

        Args:
            username: Ignored; present for signature compatibility.

        Returns:
            A :class:`StorageUsage` with the canned total and empty breakdown.
        """
        return StorageUsage(total=self._used, breakdown={})

    def get_effective_user_storage_quota(self, username: str) -> int:
        """Return the canned budget for any user.

        Args:
            username: Ignored; present for signature compatibility.

        Returns:
            The fixed byte budget captured during construction.
        """
        return self._quota


def test_enforce_storage_quota_allows_write_under_budget() -> None:
    """A write that stays under the budget passes without raising."""
    enforce_storage_quota(_StorageJobStore(used=100, quota=1000), "alice", incoming_bytes=200)


def test_enforce_storage_quota_allows_write_landing_exactly_on_budget() -> None:
    """A write landing exactly on the budget is allowed — the boundary is inclusive."""
    enforce_storage_quota(_StorageJobStore(used=800, quota=1000), "alice", incoming_bytes=200)


def test_enforce_storage_quota_rejects_write_over_budget() -> None:
    """A write that would cross the budget is rejected with a coded 409."""
    with pytest.raises(HTTPException) as exc:
        enforce_storage_quota(_StorageJobStore(used=900, quota=1000), "alice", incoming_bytes=200)
    assert exc.value.status_code == 409
    assert exc.value.code == "user.storage.quota_exceeded"
    assert exc.value.params["quota_mb"] == pytest.approx(1000 / (1024 * 1024), abs=0.1)


def _make_run_result_b64(obj: object) -> str:
    """Pickle ``obj`` and return a base64-encoded string.

    Args:
        obj: Object to serialise.

    Returns:
        A base64-encoded pickle string suitable for embedding in result JSON.
    """
    return base64.b64encode(pickle.dumps(obj)).decode()


def _run_job_data(optimization_id: str, program_obj: object) -> dict:
    """Build a minimal run-job dict with a pickled program artifact.

    Args:
        optimization_id: Job id to embed in the dict.
        program_obj: Program-like object to pickle into the artifact.

    Returns:
        A dict shaped like a stored run-job record.
    """
    now = datetime.now(UTC).isoformat()
    artifact_b64 = _make_run_result_b64(program_obj)
    return {
        "optimization_id": optimization_id,
        "status": "success",
        "created_at": now,
        "started_at": now,
        "completed_at": now,
        "payload_overview": {"optimization_type": "run"},
        "payload": {},
        "result": {
            "module_name": "MyModule",
            "optimizer_name": "GEPA",
            "metric_name": "accuracy",
            "split_counts": {"train": 70, "val": 15, "test": 15},
            "baseline_test_metric": 0.5,
            "optimized_test_metric": 0.8,
            "program_artifact": {"program_pickle_base64": artifact_b64},
        },
        "latest_metrics": {},
        "message": None,
    }


class _JobStoreWithDelete:
    """Tiny in-memory store with seed/get/delete methods for cache tests."""

    def __init__(self) -> None:
        """Initialise an empty job table."""
        self._jobs: dict = {}

    def seed(self, optimization_id: str, job_data: dict) -> None:
        """Store ``job_data`` under ``optimization_id``.

        Args:
            optimization_id: Job id to seed under.
            job_data: Job dict to store.
        """
        self._jobs[optimization_id] = job_data

    def get_job(self, optimization_id: str) -> dict:
        """Return a defensive copy of the stored job.

        Args:
            optimization_id: Job id to look up.

        Returns:
            A shallow copy of the stored job dict.

        Raises:
            KeyError: If the job was never seeded or was deleted.
        """
        if optimization_id not in self._jobs:
            raise KeyError(optimization_id)
        return dict(self._jobs[optimization_id])

    def delete_job(self, optimization_id: str) -> None:
        """Remove ``optimization_id`` from the store if present.

        Args:
            optimization_id: Job id to delete (no-op if missing).
        """
        self._jobs.pop(optimization_id, None)


@pytest.fixture(autouse=True)
def _clear_program_cache_helpers() -> Generator[None, None, None]:
    """Reset the in-process program cache around every test in this file.

    Yields:
        ``None`` once the cache is cleared; the cache is cleared again on teardown.
    """
    clear_program_cache()
    yield
    clear_program_cache()


def test_load_program_deleted_job_raises_404_before_cache() -> None:
    """``load_program`` raises 404 when the job is missing from the store."""
    store = _JobStoreWithDelete()
    # Job is never seeded — get_job will raise KeyError.
    with pytest.raises(HTTPException) as exc_info:
        load_program(store, "missing-job", _TEST_USER)
    assert exc_info.value.status_code == 404


def test_load_program_deleted_job_raises_404_even_when_cache_has_entry() -> None:
    """The cache is not a stale-fallback; deleting from the store still raises 404.

    ``load_program`` always calls ``get_job`` BEFORE consulting the cache. So
    deleting a job from the store raises 404 on the NEXT call even if the
    program was cached from a prior call. The cache does NOT act as a
    "serve from stale" fallback -- it only avoids re-deserializing the pickle.
    """
    store = _JobStoreWithDelete()
    job_data = _run_job_data("job-then-deleted", object())
    store.seed("job-then-deleted", job_data)

    # First call succeeds and populates the cache.
    _program, _, _ = load_program(store, "job-then-deleted", _TEST_USER)
    assert "job-then-deleted" in _helpers_mod._program_cache

    # Delete the job from the store.
    store.delete_job("job-then-deleted")

    # Second call still raises 404 because get_job is called first.
    with pytest.raises(HTTPException) as exc_info:
        load_program(store, "job-then-deleted", _TEST_USER)
    assert exc_info.value.status_code == 404


def test_load_program_cache_avoids_repeated_pickle_deserialisation() -> None:
    """The cache returns the same instance across calls, avoiding repeated unpickling.

    When a job exists in the store and the program is already cached,
    ``load_program`` returns the SAME (identity-equal) object on every call,
    proving the cache is used and ``pickle.loads`` is not called again.
    """
    store = _JobStoreWithDelete()
    store.seed("job-cache-hit", _run_job_data("job-cache-hit", object()))

    first_call_program, _, _ = load_program(store, "job-cache-hit", _TEST_USER)
    second_call_program, _, _ = load_program(store, "job-cache-hit", _TEST_USER)

    # Both calls must return the same object instance (from cache).
    assert first_call_program is second_call_program


def test_load_program_call_exception_propagates_and_cache_entry_is_retained() -> None:
    """A raising program propagates to the caller and the cache entry is retained.

    When the cached program's ``__call__`` raises, the exception propagates to
    the caller. The cache entry itself is NOT evicted -- ``load_program`` does
    not call the program; the cache is a plain dict and is only written on
    first load. This test documents that calling the program is the caller's
    responsibility.
    """

    # Inject a raising callable directly into the module-level cache.
    class _RaisingProgram:
        """Picklable program stub whose ``__call__`` always raises."""

        def __call__(self, *args, **kwargs):
            """Raise ``RuntimeError`` to simulate a program-level failure."""
            raise RuntimeError("program exploded")

    raiser = _RaisingProgram()
    _helpers_mod._program_cache["direct-inject"] = raiser

    # Simulate the caller invoking the cached program — the RuntimeError propagates.
    with pytest.raises(RuntimeError, match="program exploded"):
        _helpers_mod._program_cache["direct-inject"]()

    # The cache entry must still be present after the failed call.
    assert "direct-inject" in _helpers_mod._program_cache


_REACT_SIGNATURE_CODE = (
    "import dspy\n"
    "class ServeSig(dspy.Signature):\n"
    "    '''Answer the question.'''\n"
    "    question: str = dspy.InputField()\n"
    "    answer: str = dspy.OutputField()\n"
)


def _canned_react_tool() -> dspy.Tool:
    """Build a single real ``dspy.Tool`` for serve-react rebuild fixtures.

    A real tool (not a stub) is required so :func:`hash_tool_schema` produces a
    deterministic digest the drift check can compare against.

    Returns:
        A ``dspy.Tool`` named ``echo`` with a stable schema.
    """

    def echo(query: str) -> str:
        """Echo the query back unchanged."""
        return query

    return dspy.Tool(echo, name="echo", desc="Echo the query back unchanged.")


def _react_artifact(tool: dspy.Tool, schema_hashes: dict[str, str]) -> ProgramArtifact:
    """Build a ``ProgramArtifact`` carrying a react overlay and a real program state.

    The program state comes from a freshly dumped ``ReActV2`` so ``load_state``
    round-trips exactly as the serve path expects.

    Args:
        tool: The roster tool the seed program was built around.
        schema_hashes: Persisted ``{tool_name: sha256}`` snapshot to embed on
            the overlay — drives the serve-time drift check.

    Returns:
        A ``ProgramArtifact`` with ``program_state_json`` and ``react_overlay``.
    """
    signature_cls = _helpers_mod.load_signature_from_code(_REACT_SIGNATURE_CODE)
    seed = REACT_CLASS(signature_cls, tools=[tool], max_iters=4)
    return ProgramArtifact(
        program_state_json=seed.dump_state(),
        react_overlay=ReactOverlay(
            tool_descriptions={"echo": "GEPA-optimized echo wording."},
            tool_arg_descriptions={"echo": {"query": "GEPA-optimized arg wording."}},
            tool_schema_hashes=schema_hashes,
            max_iters=4,
            tool_source={"kind": "live_mcp", "mcp_url": "http://mcp.local"},
        ),
    )


def test_materialize_react_program_rebuilds_reactv2_with_overlays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_materialize_program`` re-sources a react artifact into a ``ReActV2`` with overlays.

    The live MCP roster is monkeypatched to a canned tool whose schema matches
    the persisted snapshot, so the drift check passes and the optimized per-tool
    and per-arg wording is re-applied onto the rebuilt program's tools.
    """
    tool = _canned_react_tool()
    schema_hashes = {"echo": hash_tool_schema(tool)}
    artifact = _react_artifact(tool, schema_hashes)
    overview = {"signature_code": _REACT_SIGNATURE_CODE}

    canned = _canned_react_tool()
    monkeypatch.setattr(
        _helpers_mod,
        "resolve_react_tools",
        lambda *_a, **_k: ([canned], {"echo": hash_tool_schema(canned)}),
    )

    program = _materialize_program(artifact, overview)

    assert isinstance(program, REACT_CLASS)
    assert canned.desc == "GEPA-optimized echo wording."
    assert canned.args["query"]["description"] == "GEPA-optimized arg wording."


def test_materialize_react_program_renames_roster_from_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persisted ``tool_names`` overlay renames the served roster after drift+overlays.

    Drift-check and desc/arg overlays still key on the canonical ``echo`` name
    (the schema hash is canonical), then the proposed display name is applied so
    ``ReActV2`` re-keys its tool map under the agent-facing name.
    """
    tool = _canned_react_tool()
    schema_hashes = {"echo": hash_tool_schema(tool)}
    artifact = _react_artifact(tool, schema_hashes)
    artifact.react_overlay.tool_names = {"echo": "search_records"}
    overview = {"signature_code": _REACT_SIGNATURE_CODE}

    canned = _canned_react_tool()
    monkeypatch.setattr(
        _helpers_mod,
        "resolve_react_tools",
        lambda *_a, **_k: ([canned], {"echo": hash_tool_schema(canned)}),
    )

    program = _materialize_program(artifact, overview)

    assert isinstance(program, REACT_CLASS)
    # Drift+desc/arg overlays applied under the canonical name before the rename.
    assert canned.desc == "GEPA-optimized echo wording."
    assert canned.name == "search_records"
    assert "search_records" in program.tools


def test_materialize_react_program_none_tool_names_keeps_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``tool_names`` is unset the served roster keeps canonical names (back-compat)."""
    tool = _canned_react_tool()
    schema_hashes = {"echo": hash_tool_schema(tool)}
    artifact = _react_artifact(tool, schema_hashes)
    overview = {"signature_code": _REACT_SIGNATURE_CODE}

    canned = _canned_react_tool()
    monkeypatch.setattr(
        _helpers_mod,
        "resolve_react_tools",
        lambda *_a, **_k: ([canned], {"echo": hash_tool_schema(canned)}),
    )

    program = _materialize_program(artifact, overview)

    assert canned.name == "echo"
    assert "echo" in program.tools


def test_materialize_react_program_drift_raises_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persisted schema hash that mismatches the live roster fails serve with 409.

    The overlay records a hash that no live tool produces, so
    ``_assert_tool_set_matches`` raises ``ToolSchemaDriftError`` which the serve
    path surfaces as a 409 ``optimization.tool_schema_drift`` ``DomainError``.
    """
    tool = _canned_react_tool()
    artifact = _react_artifact(tool, {"echo": "stale-hash-from-an-older-mcp-surface"})
    overview = {"signature_code": _REACT_SIGNATURE_CODE}

    canned = _canned_react_tool()
    monkeypatch.setattr(
        _helpers_mod,
        "resolve_react_tools",
        lambda *_a, **_k: ([canned], {"echo": hash_tool_schema(canned)}),
    )

    with pytest.raises(DomainError) as exc_info:
        _materialize_program(artifact, overview)
    assert exc_info.value.status_code == 409
    assert exc_info.value.code == "optimization.tool_schema_drift"


def test_load_program_react_cache_key_includes_roster_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``load_program`` folds the react roster hash into the cache key.

    The key is ``f"{optimization_id}:{stable_hash(tool_schema_hashes)}"`` so a
    re-sourced or drifted roster never serves a stale cached program; the bare
    ``optimization_id`` must not appear as a key on its own.
    """
    tool = _canned_react_tool()
    schema_hashes = {"echo": hash_tool_schema(tool)}
    artifact = _react_artifact(tool, schema_hashes)

    canned = _canned_react_tool()
    monkeypatch.setattr(
        _helpers_mod,
        "resolve_react_tools",
        lambda *_a, **_k: ([canned], {"echo": hash_tool_schema(canned)}),
    )

    now = datetime.now(UTC).isoformat()
    store = _JobStoreWithDelete()
    store.seed(
        "react-job",
        {
            "optimization_id": "react-job",
            "status": "success",
            "created_at": now,
            "started_at": now,
            "completed_at": now,
            "payload_overview": {
                "optimization_type": "run",
                "signature_code": _REACT_SIGNATURE_CODE,
            },
            "payload": {},
            "result": {
                "module_name": "ReActV2",
                "optimizer_name": "GEPA",
                "metric_name": "general",
                "split_counts": {"train": 70, "val": 15, "test": 15},
                "program_artifact": artifact.model_dump(),
            },
            "latest_metrics": {},
            "message": None,
        },
    )

    program, _, _ = load_program(store, "react-job", _TEST_USER)

    expected_key = f"react-job:{_stable_hash(schema_hashes)}"
    assert isinstance(program, REACT_CLASS)
    assert expected_key in _helpers_mod._program_cache
    assert "react-job" not in _helpers_mod._program_cache
