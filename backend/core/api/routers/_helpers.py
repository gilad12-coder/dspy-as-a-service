"""Shared helpers used by the domain routers.

Contains the small number of functions, constants, and caches that were
closures inside ``create_app`` but are needed by more than one extracted
router. Kept under a leading underscore to signal "package-internal".
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import pickle
from collections import OrderedDict
from collections.abc import AsyncIterable, AsyncIterator
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from ...billing import StripeBillingService
from ...billing.metering import meter_llm_run
from ...config import settings
from ...constants import (
    OPTIMIZATION_TYPE_GRID_SEARCH,
    OPTIMIZATION_TYPE_RUN,
    PAYLOAD_OVERVIEW_COMPILE_KWARGS,
    PAYLOAD_OVERVIEW_MODEL_NAME,
    PAYLOAD_OVERVIEW_MODULE_KWARGS,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS,
    PAYLOAD_OVERVIEW_SEED,
    PAYLOAD_OVERVIEW_SHUFFLE,
    PAYLOAD_OVERVIEW_SIGNATURE_CODE,
    PAYLOAD_OVERVIEW_SPLIT_FRACTIONS,
    PAYLOAD_OVERVIEW_TASK_FINGERPRINT,
    PAYLOAD_OVERVIEW_TOOL_SOURCE,
    PAYLOAD_OVERVIEW_WORKFLOW,
)
from ...exceptions import ServiceError
from ...models import (
    GridSearchResponse,
    OptimizationStatus,
    OptimizationSummaryResponse,
    PairResult,
    ProgramArtifact,
    ReactOverlay,
    RunResponse,
    WorkflowSpec,
)
from ...models.serve import WorkflowNodeTrace
from ...registry import ResolverError, resolve_module_factory
from ...service_gateway.optimization.data import load_signature_from_code
from ...service_gateway.optimization.retrying_react import RetryingReActV2
from ...service_gateway.optimization.tool_overlay import (
    ToolSchemaDriftError,
    _apply_bundle_tool_overrides,
    _apply_tool_name_overrides,
    _assert_tool_set_matches,
)
from ...service_gateway.optimization.training_ground.run_react import (
    resolve_react_tools,
)
from ...service_gateway.optimization.workflow import build_workflow_program, workflow_tool_users
from ..auth import AuthenticatedUser, is_admin
from ..converters import (
    compute_elapsed,
    extract_estimated_remaining,
    job_owner,
    overview_to_base_fields,
    parse_overview,
    parse_timestamp,
    status_to_job_status,
)
from ..errors import DomainError
from ..response_limits import AGENT_MAX_TEXT, truncate_text
from ..sharing_access import (
    MEMBER_ROLES,
    ShareRole,
    get_grant,
    list_grants_for_user,
    role_rank,
)
from .constants import TERMINAL_STATUSES

logger = logging.getLogger(__name__)

_USER_FACING_OPTIMIZATION_TYPES = frozenset(
    {OPTIMIZATION_TYPE_RUN, OPTIMIZATION_TYPE_GRID_SEARCH}
)

class _BoundedProgramCache(OrderedDict[str, Any]):
    """OrderedDict-backed LRU for deserialized DSPy programs.

    A plain dict here grew without bound — every served optimization pinned
    its compiled module in process memory until the API restarted. This
    bounded variant evicts the least-recently-used entry once
    ``settings.program_cache_max_entries`` is exceeded so a long-lived API
    process can't be DoS'd by serving many distinct optimizations.
    """

    def __init__(self) -> None:
        """Initialise an empty cache; capacity is read lazily from settings."""
        super().__init__()

    @property
    def _max_entries(self) -> int:
        """Resolve the live cache ceiling from application settings.

        Reading on every mutation lets test fixtures override the cap by
        tweaking ``settings.program_cache_max_entries`` between cases without
        re-importing this module.
        """
        return max(int(settings.program_cache_max_entries), 1)

    def __setitem__(self, key: str, value: Any) -> None:
        """Insert ``value`` at most-recently-used and evict if past capacity.

        Args:
            key: Cache key (optimization id or pair-scoped key).
            value: Deserialized DSPy program object.
        """
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._max_entries:
            self.popitem(last=False)

    def __getitem__(self, key: str) -> Any:
        """Return the cached program and mark it most-recently-used.

        Args:
            key: Cache key to look up.

        Returns:
            The previously stored program object.

        Raises:
            KeyError: When ``key`` is not present in the cache.
        """
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value


# Cache deserialized programs to avoid repeated pickle loads.
# Keyed by optimization_id (for single runs + grid-search best pair) or
# f"{optimization_id}_pair_{pair_index}" (for per-pair serving). Bounded by
# ``settings.program_cache_max_entries`` to cap process resident memory.
_program_cache: _BoundedProgramCache = _BoundedProgramCache()


def clear_program_cache() -> None:
    """Clear the module-level deserialized-program cache.

    Intended for test teardown; production code should not call this.
    """
    _program_cache.clear()


def enforce_llm_credits(job_store, username: str) -> None:
    """Refuse an interactive LLM turn for an account with no spendable credits.

    The turn-surface twin of the submit gate: agent chats, interview turns and
    tagging predictions spend managed tokens, so a depleted account is stopped
    before the LLM call rather than billed into the negative. The free grant
    means a brand-new account always passes. A store with no SQL engine
    (legacy/in-memory) skips the gate, matching the submit path.

    Args:
        job_store: Job-store instance whose ORM engine backs the billing tables.
        username: Account attempting the turn.

    Raises:
        DomainError: 402 when the account has no spendable credits.
    """
    engine = getattr(job_store, "engine", None)
    if engine is None or not username:
        return
    if StripeBillingService(engine=engine).spendable_credits(username) > 0:
        return
    raise DomainError("billing.insufficient_credits", status=402)


async def stream_with_llm_metering(
    source: AsyncIterable[dict[str, Any]],
    *,
    job_store,
    username: str,
    description: str,
    usage_sink: list,
) -> AsyncIterator[dict[str, Any]]:
    """Mirror SSE events and meter the turn's LLM usage when the stream ends.

    Wraps a turn's event stream; on teardown — normal completion, error, or
    the client dropping the SSE connection mid-turn — it debits and
    Stripe-meters whatever usage the LMs in ``usage_sink`` accumulated. The
    sink is harvested in ``finally`` (not on the ``done`` event) because the
    frontend routinely tears streams down early, and a ``MeteredLM``'s running
    totals are current at any point. The synchronous billing write is
    offloaded to a worker thread, mirroring the persistence wrapper.

    Args:
        source: Upstream async event stream for one turn.
        job_store: Job-store whose engine backs the billing tables; a store
            without an engine streams unmetered.
        username: Account the turn is billed to.
        description: Ledger-row label for the charge (e.g. ``"Agent chat"``).
        usage_sink: List the run function appends its ``MeteredLM``(s) to.

    Yields:
        The upstream events, unchanged.
    """
    try:
        async for event in source:
            yield event
    finally:
        engine = getattr(job_store, "engine", None)
        if engine is not None and usage_sink:
            try:
                await asyncio.to_thread(
                    meter_llm_run, engine, username, list(usage_sink), description=description
                )
            except Exception:
                logger.exception("LLM turn metering failed for %s", username)


# Idle gap after which the SSE serializer emits a comment line to keep the
# connection alive through proxy/LB idle timeouts (commonly 30-60s).
SSE_KEEPALIVE_SECONDS = 15.0


async def sse_from_events(
    source: AsyncIterable[dict[str, Any]],
) -> AsyncIterator[str]:
    """Serialize an ``{event, data}`` async iterable as Server-Sent Events.

    Every SSE route in this codebase shares the same formatting contract:
    each yielded mapping has an ``event`` name and a ``data`` dict that is
    JSON-encoded (``ensure_ascii=False``) and emitted as
    ``"event: <name>\\ndata: <json>\\n\\n"``. Centralising it here lets
    route handlers stay free of nested ``event_generator`` closures.

    A ``: keep-alive`` comment is emitted whenever the source stays quiet for
    ``SSE_KEEPALIVE_SECONDS`` — a model with a long time-to-first-token
    otherwise leaves the connection idle long enough for an intermediary
    (ingress/LB idle timeout) to silently drop it mid-turn.

    Args:
        source: Async iterable yielding ``{"event": str, "data": dict}`` mappings.

    Yields:
        SSE-formatted strings ready for ``StreamingResponse``.
    """
    iterator = aiter(source)
    next_event: asyncio.Task | None = None
    try:
        while True:
            if next_event is None:
                next_event = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait({next_event}, timeout=SSE_KEEPALIVE_SECONDS)
            if not done:
                yield ": keep-alive\n\n"
                continue
            try:
                event = next_event.result()
            except StopAsyncIteration:
                next_event = None
                return
            next_event = None
            name = event["event"]
            payload = json.dumps(event["data"], ensure_ascii=False, default=str)
            yield f"event: {name}\ndata: {payload}\n\n"
    finally:
        if next_event is not None:
            next_event.cancel()


def strip_api_key(d: dict) -> dict:
    """Return a shallow copy of *d* with ``extra.api_key`` removed.

    Args:
        d: A model-settings dict that may contain an ``extra.api_key`` field.

    Returns:
        A copy of ``d`` with the API key scrubbed from ``extra``.
    """
    result = dict(d)
    extra = result.get("extra")
    if isinstance(extra, dict) and "api_key" in extra:
        result["extra"] = {k: v for k, v in extra.items() if k != "api_key"}
    return result


def stable_seed(optimization_id: str) -> int:
    """Derive a process-stable 31-bit RNG seed from an optimization id.

    Python's built-in :func:`hash` is salted per process via
    ``PYTHONHASHSEED``, so the same id maps to a different int in every
    worker. Read paths (``/dataset``, ``/test-results``,
    ``/pair/{i}/test-results``) recompute the train/val/test split with
    this seed when the payload's stored seed is missing — using
    :func:`hash` there silently desyncs splits across workers and breaks
    UI index remapping. SHA256 truncated to 31 bits keeps the same numeric
    shape but is byte-stable across processes.

    Args:
        optimization_id: Optimization identifier to derive a seed from.

    Returns:
        A non-negative ``int`` strictly less than ``2 ** 31``.
    """
    digest = hashlib.sha256(optimization_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


def compute_task_fingerprint(
    signature_code: str,
    metric_code: str,
    dataset: list[dict[str, Any]],
) -> str:
    """Return a stable SHA256 fingerprint identifying the ML task.

    Two jobs share a fingerprint iff their signature source, metric source,
    and dataset content are byte-identical. Used by the frontend to gate
    apples-to-apples run comparisons.

    Args:
        signature_code: Source code of the user's DSPy signature.
        metric_code: Source code of the user's metric function.
        dataset: The full list of dataset rows used for the optimization.

    Returns:
        A hex-encoded SHA256 digest derived from the three inputs.
    """
    dataset_blob = json.dumps(dataset, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    dataset_hash = hashlib.sha256(dataset_blob.encode("utf-8")).hexdigest()
    task_blob = f"{signature_code}\x00{metric_code}\x00{dataset_hash}"
    return hashlib.sha256(task_blob.encode("utf-8")).hexdigest()


def compute_compare_fingerprint(optimization_id: str, overview: dict[str, Any]) -> str | None:
    """Return a fingerprint identifying compareability (same task + same split).

    Two jobs share a ``compare_fingerprint`` iff they share a ``task_fingerprint``
    AND evaluate on byte-identical train/val/test splits. ``task_fingerprint``
    alone only proves the signature/metric/dataset match; jobs with mismatching
    seeds, shuffle flags, or split fractions land on different test rows and
    must not be compared row-by-row.

    The effective seed mirrors the fallback in ``/dataset`` and ``/test-results``:
    use the stored seed when present, otherwise derive ``stable_seed(optimization_id)``
    — the same value those endpoints used to compute the actual split.

    Args:
        optimization_id: Optimization id; used as the seed fallback.
        overview: Parsed ``payload_overview`` dict for the job.

    Returns:
        A hex-encoded SHA256 digest, or ``None`` when the base ``task_fingerprint``
        is missing (legacy job; can't be compared either way).
    """
    task_fp = overview.get(PAYLOAD_OVERVIEW_TASK_FINGERPRINT)
    if not isinstance(task_fp, str) or not task_fp:
        return None
    stored_seed = overview.get(PAYLOAD_OVERVIEW_SEED)
    effective_seed = stored_seed if stored_seed is not None else stable_seed(optimization_id)
    shuffle = overview.get(PAYLOAD_OVERVIEW_SHUFFLE, True)
    fractions = overview.get(PAYLOAD_OVERVIEW_SPLIT_FRACTIONS) or {}
    if hasattr(fractions, "model_dump"):
        fractions = fractions.model_dump()
    fractions_blob = json.dumps(fractions, sort_keys=True, separators=(",", ":"))
    blob = f"{task_fp}\x00{effective_seed}\x00{bool(shuffle)}\x00{fractions_blob}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _grant_role(job_store, optimization_id: str, username: str) -> ShareRole | None:
    """Resolve a member grant's role for ``username`` on an optimization.

    Opens a short-lived session on the job store's SQLAlchemy engine to read
    the ``optimization_share_grants`` row. Job stores without an ``engine``
    (the in-memory/local store used in tests and offline mode) carry no grant
    table, so this returns ``None`` and access falls back to owner/admin only.

    Args:
        job_store: Job-store whose ``engine`` backs the grant table.
        optimization_id: Optimization the grant would apply to.
        username: Caller's username (compared case-insensitively by the query).

    Returns:
        The grant's :class:`ShareRole`, or ``None`` when there is no grant or
        the store exposes no grant-bearing engine.
    """
    engine = getattr(job_store, "engine", None)
    if engine is None:
        return None
    with Session(engine) as session:
        grant = get_grant(session, optimization_id, username)
    if grant is not None and grant.role in MEMBER_ROLES:
        return ShareRole(grant.role)
    return None


def load_job_with_role(
    job_store,
    optimization_id: str,
    user: AuthenticatedUser,
    *,
    include_payload: bool = True,
) -> tuple[dict[str, Any], ShareRole]:
    """Load a job row and resolve the caller's effective role on it.

    Access is granted to the owner, any admin, or an invited member (Google-
    Drive-style grants — see :mod:`core.api.sharing_access`). A caller with no
    ownership and no grant gets 404 (not 403): not leaking existence keeps
    unauthorized observers from confirming an ID exists.

    Args:
        job_store: Job-store the row is read from.
        optimization_id: Optimization id to load.
        user: Authenticated caller.
        include_payload: When ``False``, the row's ``payload`` JSONB is never
            materialized — the status-poll endpoint passes ``False`` because
            its response carries no raw payload.

    Returns:
        ``(job_row, effective_role)`` where ``effective_role`` is
        :attr:`ShareRole.owner` for the creator/admin or the member grant's tier.

    Raises:
        DomainError: 404 when the id is unknown or the caller has no access.
    """
    try:
        # Narrow through get_job_no_payload so store doubles that predate the
        # kwarg fall back to the full read instead of a TypeError.
        if include_payload:
            job_data = job_store.get_job(optimization_id)
        else:
            job_data = get_job_no_payload(job_store, optimization_id)
    except KeyError:
        raise DomainError("optimization.not_found", status=404, optimization_id=optimization_id) from None
    overview = parse_overview(job_data)
    optimization_type = (
        overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE)
        or job_data.get("optimization_type")
        or OPTIMIZATION_TYPE_RUN
    )
    if optimization_type not in _USER_FACING_OPTIMIZATION_TYPES:
        raise DomainError("optimization.not_found", status=404, optimization_id=optimization_id)
    if is_admin(user):
        return job_data, ShareRole.owner
    owner = job_owner(job_data)
    username = user.username.strip().lower()
    if owner is not None and owner == username:
        return job_data, ShareRole.owner
    role = _grant_role(job_store, optimization_id, username)
    if role is not None:
        return job_data, role
    raise DomainError("optimization.not_found", status=404, optimization_id=optimization_id)


def load_job_for_user(
    job_store,
    optimization_id: str,
    user: AuthenticatedUser,
) -> dict[str, Any]:
    """Load a job row, enforcing access (owner, admin, or invited member).

    The viewer-floor gate: any caller with access (owner / admin / a grant of
    any tier) gets the row; everyone else 404s. Routes that mutate require a
    higher tier — see :func:`require_role_at_least`.

    Args:
        job_store: Job-store the row is read from.
        optimization_id: Optimization id to load.
        user: Authenticated caller.

    Returns:
        The raw job-row mapping.

    Raises:
        DomainError: 404 when the id is unknown or the caller has no access.
    """
    job_data, _role = load_job_with_role(job_store, optimization_id, user)
    return job_data


def require_role_at_least(
    job_store,
    optimization_id: str,
    user: AuthenticatedUser,
    minimum: ShareRole,
) -> tuple[dict[str, Any], ShareRole]:
    """Load a job row, requiring the caller's effective role to meet ``minimum``.

    A caller with access but below the required tier (e.g. a viewer hitting a
    delete route) gets 403 — they already know the run exists, so hiding it as
    404 would be misleading. A caller with no access at all still 404s (via
    :func:`load_job_with_role`).

    Args:
        job_store: Job-store the row is read from.
        optimization_id: Optimization id to load.
        user: Authenticated caller.
        minimum: The lowest :class:`ShareRole` permitted to proceed.

    Returns:
        ``(job_row, effective_role)`` for a caller meeting the tier.

    Raises:
        DomainError: 404 when unknown/inaccessible; 403 when the caller has
            access but a role below ``minimum``.
    """
    job_data, role = load_job_with_role(job_store, optimization_id, user)
    if role_rank(role) < role_rank(minimum):
        raise DomainError(
            "optimization.insufficient_role",
            status=403,
            optimization_id=optimization_id,
            required=str(minimum),
            role=str(role),
        )
    return job_data, role


def grant_roles_for(
    job_store, optimization_ids: list[str], username: str
) -> dict[str, str]:
    """Batch-resolve the caller's grant roles across many optimizations.

    One query for the whole id set (see :func:`list_grants_for_user`); stores
    without a grant-bearing ``engine`` return ``{}``.

    Args:
        job_store: Job-store whose ``engine`` backs the grant table.
        optimization_ids: Ids to scope the lookup to.
        username: Caller's username.

    Returns:
        ``{optimization_id: role}`` for the caller's intersecting grants.
    """
    engine = getattr(job_store, "engine", None)
    if engine is None:
        return {}
    with Session(engine) as session:
        return list_grants_for_user(session, optimization_ids, username)


def get_job_no_payload(job_store, optimization_id: str) -> dict[str, Any]:
    """Read a job row without materializing its payload when the store can.

    Bulk ownership/metadata paths only read the overview and owner, so the
    (potentially multi-MB) payload JSONB is skipped on stores that support the
    narrowing kwarg; older store doubles fall back to the full read.

    Args:
        job_store: Job-store to read from.
        optimization_id: Row to fetch.

    Returns:
        The job row (``payload`` reported as ``None`` when deferred).

    Raises:
        KeyError: When the job does not exist.
    """
    try:
        return job_store.get_job(optimization_id, include_payload=False)
    except TypeError:
        return job_store.get_job(optimization_id)


def filter_ids_at_least(
    job_store,
    optimization_ids: list[str],
    user: AuthenticatedUser,
    minimum: ShareRole,
) -> tuple[list[str], list[str]]:
    """Split ids into ``(allowed, denied)`` by whether the caller meets ``minimum``.

    The bulk-action counterpart of :func:`require_role_at_least`: admins keep
    every id; otherwise an id is allowed when the caller is its owner or holds a
    grant whose tier is at least ``minimum``. Unknown ids and below-tier ids land
    in ``denied`` so the caller can mark them skipped.

    Args:
        job_store: Job-store used to read each row's owner.
        optimization_ids: Already-deduplicated list of ids.
        user: Authenticated caller.
        minimum: The lowest :class:`ShareRole` permitted per id.

    Returns:
        ``(allowed, denied)``.
    """
    if is_admin(user):
        return list(optimization_ids), []
    username = user.username.strip().lower()
    min_rank = role_rank(minimum)
    granted = grant_roles_for(job_store, optimization_ids, username)
    allowed: list[str] = []
    denied: list[str] = []
    for oid in optimization_ids:
        try:
            job_data = get_job_no_payload(job_store, oid)
        except KeyError:
            denied.append(oid)
            continue
        owner = job_owner(job_data)
        if owner is not None and owner == username:
            role: ShareRole | None = ShareRole.owner
        else:
            grant_role = granted.get(oid)
            role = ShareRole(grant_role) if grant_role in MEMBER_ROLES else None
        if role is not None and role_rank(role) >= min_rank:
            allowed.append(oid)
        else:
            denied.append(oid)
    return allowed, denied


def enforce_user_quota(job_store, username: str) -> None:
    """Raise if ``username`` is at or over their job quota.

    Live DB overrides take precedence over static config. Admins and users
    with an explicit ``None`` override bypass the check entirely.

    Args:
        job_store: The job store used to count the user's existing jobs.
        username: The user whose quota should be enforced.

    Raises:
        DomainError: When the user already has at least ``quota`` jobs (HTTP 409).
    """
    live_quota_resolver = getattr(job_store, "get_effective_user_quota", None)
    quota = live_quota_resolver(username) if callable(live_quota_resolver) else settings.get_user_quota(username)
    if quota is None:
        return
    current = job_store.count_jobs(username=username)
    if current >= quota:
        raise DomainError("quota.reached", status=409, quota=quota)


def _mb(num_bytes: int) -> float:
    """Return ``num_bytes`` as megabytes rounded to one decimal for messages."""
    return round(num_bytes / (1024 * 1024), 1)


def enforce_storage_quota(job_store, username: str, incoming_bytes: int) -> None:
    """Raise if persisting ``incoming_bytes`` would exceed the storage budget.

    The unified per-user storage total (jobs, datasets, logs, agent chats,
    staged uploads, embeddings) plus the incoming write is compared against the
    user's effective byte budget. This is the single gate that supersedes the
    legacy per-job count cap and the per-library dataset quota.

    Args:
        job_store: Store exposing ``get_effective_user_storage_quota`` and
            ``compute_user_storage``.
        username: The user the budget belongs to.
        incoming_bytes: Size of the not-yet-persisted write (payload or dataset).

    Raises:
        DomainError: ``user.storage.quota_exceeded`` (HTTP 409) when the user's
            total would cross their budget.
    """
    quota = job_store.get_effective_user_storage_quota(username)
    used = job_store.compute_user_storage(username).total
    if used + incoming_bytes <= quota:
        return
    raise DomainError(
        "user.storage.quota_exceeded",
        status=409,
        used_mb=_mb(used),
        quota_mb=_mb(quota),
        incoming_mb=_mb(incoming_bytes),
    )


def is_resumable(job_store: Any, job_data: dict) -> bool:
    """Return whether a stopped run can be resumed in place from its checkpoint.

    True only for the narrow case the Resume affordance targets: a terminal
    ``failed``/``cancelled`` run that still has attempts left under
    ``job_max_attempts`` and a saved GEPA checkpoint. The cheap, indexed
    checkpoint lookup is gated behind the status/attempts checks so list pages
    query only the few candidate rows, never the whole page.

    Args:
        job_store: The job store used to test for a saved checkpoint.
        job_data: Raw job row from the store.

    Returns:
        ``True`` when the run should offer Resume rather than Restart.
    """
    optimization_id = _resume_candidate_id(job_data)
    if not optimization_id:
        return False
    return _has_resumable_state(job_store, optimization_id)


def _resume_candidate_id(job_data: dict) -> str | None:
    """Apply the cheap (no-query) resume gates and return the job id when they pass.

    Shared by :func:`is_resumable` (per-row) and :func:`resumable_id_flags`
    (batched) so both paths gate identically before the checkpoint lookup.

    Args:
        job_data: Raw job row from the store.

    Returns:
        The optimization id when the row is a resume candidate, else ``None``.
    """
    status = status_to_job_status(job_data.get("status", "pending"))
    if status not in {OptimizationStatus.failed, OptimizationStatus.cancelled, OptimizationStatus.paused}:
        return None
    # A grid is resumed per pair (in its results), not via a whole-job button, so
    # the top-level flag stays False for grids — see ``grid_resumable_pairs``.
    if parse_overview(job_data).get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE) == OPTIMIZATION_TYPE_GRID_SEARCH:
        return None
    # A manual pause is user-driven, not failure recovery, so it is exempt from the
    # attempt cap; only failed/cancelled (auto-recovery) runs are bounded by it.
    if status != OptimizationStatus.paused and int(job_data.get("attempts") or 0) >= settings.job_max_attempts:
        return None
    return job_data.get("optimization_id") or None


def resumable_id_flags(job_store: Any, rows: list[dict]) -> set[str]:
    """Return the ids in ``rows`` whose runs can offer Resume, in one batch.

    The batched counterpart of :func:`is_resumable` for list pages: the cheap
    gates run in Python, then a single ``IN (...)`` existence query covers all
    candidates instead of one probe per row (up to page-size round trips).
    Stores without the batch method fall back to the per-row probe with
    identical results.

    Args:
        job_store: The job store used to test for saved checkpoints.
        rows: Raw job rows from the store.

    Returns:
        The set of resumable optimization ids.
    """
    candidates = [oid for row in rows if (oid := _resume_candidate_id(row))]
    if not candidates:
        return set()
    batch = getattr(job_store, "resumable_state_ids", None)
    if callable(batch):
        return set(batch(candidates))
    return {oid for oid in candidates if _has_resumable_state(job_store, oid)}


def is_pausable(job_store: Any, job_data: dict) -> bool:
    """Return whether a running optimization can be manually paused.

    True only while the run is actively ``running`` (not a grid) and already has
    saved optimizer state, so a pause is guaranteed to be resumable — pausing
    before the first checkpoint would otherwise strand the run. The checkpoint
    lookup is gated behind the status/type checks so it runs for the single
    candidate row only.

    Args:
        job_store: The job store used to test for a saved checkpoint.
        job_data: Raw job row from the store.

    Returns:
        ``True`` when the run should offer a Pause control.
    """
    status = status_to_job_status(job_data.get("status", "pending"))
    if status != OptimizationStatus.running:
        return False
    if parse_overview(job_data).get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE) == OPTIMIZATION_TYPE_GRID_SEARCH:
        return False
    optimization_id = job_data.get("optimization_id")
    if not optimization_id:
        return False
    return _has_resumable_state(job_store, optimization_id)


def _has_resumable_state(job_store: Any, optimization_id: str) -> bool:
    """Return whether saved state exists to resume from (single checkpoint or any grid pair).

    A single run is resumable from its GEPA checkpoint; a grid is resumable when
    any pair has an in-flight checkpoint OR any pair already finished (so the rest
    can be re-run while finished pairs are kept).

    Args:
        job_store: The job store to query.
        optimization_id: The job id to test.

    Returns:
        ``True`` when there is state to resume from.
    """
    has_checkpoint = getattr(job_store, "has_gepa_checkpoint", None)
    if callable(has_checkpoint) and has_checkpoint(optimization_id):
        return True
    has_pairs = getattr(job_store, "has_grid_pair_results", None)
    return bool(callable(has_pairs) and has_pairs(optimization_id))


def grid_resumable_pairs(job_store: Any, optimization_id: str) -> list[int]:
    """Return the grid pair indices that have a saved checkpoint to resume from.

    Drives the per-pair control in the grid results: a failed pair whose index is
    here crashed mid-GEPA and offers Resume; a failed pair not here failed without
    state and offers Restart. Successful pairs have no checkpoint (dropped when
    they finished), so they never appear.

    Args:
        job_store: The job store to query.
        optimization_id: The grid job id.

    Returns:
        Sorted pair indices with a checkpoint (empty when none or unsupported).
    """
    lister = getattr(job_store, "list_gepa_checkpoints", None)
    if not callable(lister):
        return []
    return sorted(cp.pair_index for cp in lister(optimization_id) if cp.pair_index >= 0)


def build_summary(job_data: dict) -> OptimizationSummaryResponse:
    """Build a compact dashboard-card summary from a raw job dict.

    Handles both single-run and grid-search jobs. For grid search the best pair
    is used as the representative result. Live pair counters fall back to
    ``latest_metrics`` while the sweep is still in progress.

    Args:
        job_data: Raw job row from the job store.

    Returns:
        A populated :class:`OptimizationSummaryResponse` for dashboard rendering.
    """
    created_at = parse_timestamp(job_data.get("created_at")) or datetime.now(UTC)
    started_at = parse_timestamp(job_data.get("started_at"))
    completed_at = parse_timestamp(job_data.get("completed_at"))
    overview = parse_overview(job_data)
    job_status = status_to_job_status(job_data.get("status", "pending"))
    optimization_type = overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE, OPTIMIZATION_TYPE_RUN)

    est_remaining = None
    if job_status not in TERMINAL_STATUSES:
        est_remaining = extract_estimated_remaining(job_data)

    result_data = job_data.get("result")
    latest_metrics = job_data.get("latest_metrics", {})
    baseline = None
    optimized = None
    completed_pairs = None
    failed_pairs = None
    best_pair_label = None

    if isinstance(result_data, dict):
        if optimization_type == OPTIMIZATION_TYPE_GRID_SEARCH:
            best_pair = result_data.get("best_pair")
            if isinstance(best_pair, dict):
                baseline = best_pair.get("baseline_test_metric")
                optimized = best_pair.get("optimized_test_metric")
                gen = best_pair.get("generation_model", "")
                ref = best_pair.get("reflection_model", "")
                best_pair_label = f"{gen} + {ref}"
            completed_pairs = result_data.get("completed_pairs")
            failed_pairs = result_data.get("failed_pairs")
        else:
            baseline = result_data.get("baseline_test_metric")
            optimized = result_data.get("optimized_test_metric")

    # For grid search, pull live counters from latest_metrics if result not yet available
    if optimization_type == OPTIMIZATION_TYPE_GRID_SEARCH:
        if completed_pairs is None:
            live_completed = latest_metrics.get("completed_so_far")
            completed_pairs = live_completed if isinstance(live_completed, int) else 0
        if failed_pairs is None:
            live_failed = latest_metrics.get("failed_so_far")
            failed_pairs = live_failed if isinstance(live_failed, int) else 0

    metric_improvement = None
    if baseline is not None and optimized is not None:
        metric_improvement = round(optimized - baseline, 6)

    elapsed_str, elapsed_secs = compute_elapsed(
        created_at, started_at, completed_at, job_data.get("accumulated_runtime_seconds") or 0.0
    )

    optimization_id = job_data["optimization_id"]
    return OptimizationSummaryResponse(
        optimization_id=optimization_id,
        status=job_status,
        message=job_data.get("message"),
        created_at=created_at,
        started_at=started_at,
        completed_at=completed_at,
        elapsed=elapsed_str,
        elapsed_seconds=elapsed_secs,
        estimated_remaining=est_remaining,
        **overview_to_base_fields(overview),
        compare_fingerprint=compute_compare_fingerprint(optimization_id, overview),
        split_fractions=overview.get(PAYLOAD_OVERVIEW_SPLIT_FRACTIONS),
        shuffle=overview.get(PAYLOAD_OVERVIEW_SHUFFLE),
        seed=overview.get(PAYLOAD_OVERVIEW_SEED),
        optimizer_kwargs=overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS, {}),
        compile_kwargs=overview.get(PAYLOAD_OVERVIEW_COMPILE_KWARGS, {}),
        latest_metrics=latest_metrics,
        progress_count=job_data.get("progress_count", 0),
        log_count=job_data.get("log_count", 0),
        stored_bytes=job_data.get("stored_bytes", 0),
        baseline_test_metric=baseline,
        optimized_test_metric=optimized,
        metric_improvement=metric_improvement,
        completed_pairs=completed_pairs,
        failed_pairs=failed_pairs,
        best_pair_label=best_pair_label,
        summary_text=job_data.get("summary_text"),
    )


def _artifact_has_payload(artifact: ProgramArtifact | None) -> bool:
    """Return whether the artifact carries either JSON state or a legacy pickle.

    Args:
        artifact: The artifact pulled off a ``RunResponse`` or ``PairResult``.

    Returns:
        ``True`` when the artifact has something we can materialize a program
        from; ``False`` otherwise.
    """
    if artifact is None:
        return False
    return artifact.program_state_json is not None or bool(artifact.program_pickle_base64)


def _materialize_program(artifact: ProgramArtifact, overview: dict) -> Any:
    """Materialize a runnable DSPy program from an artifact.

    The state-JSON path is preferred: reconstruct the module shell from the
    stored ``signature_code`` / ``module_name`` / ``module_kwargs`` and apply
    the saved state via ``program.load_state``. The pickle path is retained
    only as a fallback for artifacts written before the JSON migration.

    Args:
        artifact: The artifact carrying either ``program_state_json`` (new
            jobs) or ``program_pickle_base64`` (legacy jobs).
        overview: Parsed payload-overview dict from the job row; supplies
            ``signature_code``, ``module_name``, and ``module_kwargs`` for
            the JSON path.

    Returns:
        A live DSPy program object ready for inference.

    Raises:
        DomainError: 409 when the artifact lacks both payload variants, when
            ``signature_code`` is missing from the overview for a JSON
            artifact, or when module reconstruction fails.
    """
    workflow_spec = workflow_spec_from_overview(overview)
    if workflow_spec is not None:
        return _materialize_workflow_program(artifact, overview, workflow_spec)

    if artifact.react_overlay is not None:
        return _materialize_react_program(artifact, overview)

    if artifact.program_state_json is not None:
        signature_code = overview.get(PAYLOAD_OVERVIEW_SIGNATURE_CODE)
        if not signature_code:
            # Pre-migration overviews don't carry signature_code. Force the
            # legacy pickle path if it's present; otherwise fail loudly so
            # the caller can prompt a re-run instead of producing nonsense.
            if artifact.program_pickle_base64:
                return _legacy_pickle_load(artifact.program_pickle_base64)
            raise DomainError("optimization.no_signature_code_for_reload", status=409)

        module_name = overview.get(PAYLOAD_OVERVIEW_MODULE_NAME) or "predict"
        module_kwargs = dict(overview.get(PAYLOAD_OVERVIEW_MODULE_KWARGS, {}))
        try:
            signature_cls = load_signature_from_code(signature_code)
            module_factory, auto_signature = resolve_module_factory(module_name)
        except ResolverError as exc:
            raise DomainError(
                "optimization.module_reconstruction_failed", status=409, error=str(exc)
            ) from exc

        if auto_signature or "signature" not in module_kwargs:
            module_kwargs["signature"] = signature_cls
        program = module_factory(**module_kwargs)
        program.load_state(artifact.program_state_json)
        return program

    return _legacy_pickle_load(artifact.program_pickle_base64)


def workflow_spec_from_overview(overview: dict) -> WorkflowSpec | None:
    """Parse the persisted workflow spec off a payload overview, if any.

    Args:
        overview: Parsed payload-overview dict from the job row.

    Returns:
        The parsed ``WorkflowSpec``, or ``None`` for non-workflow runs.
    """
    payload = overview.get(PAYLOAD_OVERVIEW_WORKFLOW)
    if not payload:
        return None
    return WorkflowSpec.model_validate(payload)


def sanitize_node_traces(raw_traces: list[dict[str, Any]]) -> list[WorkflowNodeTrace]:
    """Convert raw workflow trace dicts into response-safe trace models.

    Node port values are arbitrary Python objects produced by user code;
    anything that is not JSON-native is stringified so the response always
    serializes, and long strings are capped like serve outputs.

    Args:
        raw_traces: Trace dicts collected by ``capture_node_traces``.

    Returns:
        Response-ready ``WorkflowNodeTrace`` models in execution order.
    """
    return [
        WorkflowNodeTrace(
            node_id=trace["node_id"],
            kind=trace["kind"],
            name=trace["name"],
            inputs=sanitize_port_values(trace.get("inputs") or {}),
            outputs=sanitize_port_values(trace["outputs"]) if trace.get("outputs") is not None else None,
            elapsed_ms=trace.get("elapsed_ms", 0.0),
            error=trace.get("error"),
        )
        for trace in raw_traces
    ]


def sanitize_port_values(values: dict[str, Any]) -> dict[str, Any]:
    """Make one trace value dict JSON-safe and size-capped.

    Args:
        values: Raw port-name to value mapping from a node trace.

    Returns:
        A new dict with non-JSON-native values stringified and long
        strings truncated.
    """
    sanitized: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, str):
            sanitized[key] = truncate_text(value, AGENT_MAX_TEXT)
        elif value is None or isinstance(value, (bool, int, float)):
            sanitized[key] = value
        else:
            sanitized[key] = truncate_text(str(value), AGENT_MAX_TEXT)
    return sanitized


def _materialize_workflow_program(
    artifact: ProgramArtifact, overview: dict, workflow_spec: WorkflowSpec
) -> Any:
    """Rebuild a served workflow program from its spec and state JSON.

    The shell is reconstructed with the same deterministic builder the run
    path used, so the saved per-node predictor state keys line up 1:1.
    Tool-using graphs re-source their roster from the overview's scrubbed
    ``tool_source``; a ``dataset_snapshot`` source cannot be re-sourced
    without the original dataset and is rejected until snapshot specs are
    persisted for workflows.

    Args:
        artifact: The artifact carrying ``program_state_json``.
        overview: Parsed payload-overview dict.
        workflow_spec: The workflow spec parsed from the overview.

    Returns:
        A live ``WorkflowProgram`` ready for inference.

    Raises:
        DomainError: 409 when the artifact lacks state JSON, the graph needs
            tools it cannot re-source, or reconstruction fails.
    """
    if artifact.program_state_json is None:
        raise DomainError("optimization.no_program_artifact_scoped", status=409)
    tool_source = overview.get(PAYLOAD_OVERVIEW_TOOL_SOURCE)
    if workflow_tool_users(workflow_spec):
        if not tool_source:
            raise DomainError(
                "optimization.module_reconstruction_failed",
                status=409,
                error="workflow uses tools but no tool_source was persisted.",
            )
        if tool_source.get("kind") == "dataset_snapshot":
            raise DomainError(
                "optimization.module_reconstruction_failed",
                status=409,
                error="serving a workflow with dataset_snapshot tools is not supported yet.",
            )
    try:
        program, _schema_hashes = build_workflow_program(workflow_spec, tool_source=tool_source)
    except (ServiceError, ValueError) as exc:
        raise DomainError("optimization.module_reconstruction_failed", status=409, error=str(exc)) from exc
    program.load_state(artifact.program_state_json)
    return program


def _materialize_react_program(artifact: ProgramArtifact, overview: dict) -> Any:
    """Rebuild a served ``ReActV2`` program from a persisted react artifact.

    Unlike the scalar JSON path, a react program owns a tool roster that the
    state dump deliberately drops (``save_program=False``). We re-source that
    roster from ``react_overlay.tool_source`` — the same resolver the run path
    uses — drift-check it against the schema-hash snapshot taken at training
    time, re-apply the optimized per-tool wording, then build ``ReActV2`` and
    load the stored state.

    Args:
        artifact: The artifact carrying ``program_state_json`` and a
            ``react_overlay`` (signature reconstruction reads ``overview``).
        overview: Parsed payload-overview dict; supplies ``signature_code``.

    Returns:
        A live ``dspy.ReActV2`` program ready for inference.

    Raises:
        DomainError: 409 when ``signature_code`` is missing, when module/tool
            reconstruction fails, or when the live tool schema has drifted
            from the snapshot recorded at training time.
    """
    react_overlay = artifact.react_overlay
    signature_code = overview.get(PAYLOAD_OVERVIEW_SIGNATURE_CODE)
    if not signature_code:
        raise DomainError("optimization.no_signature_code_for_reload", status=409)

    try:
        signature_cls = load_signature_from_code(signature_code)
        # live_mcp re-sources with auth None: the MCP auth header is a secret
        # and is intentionally never persisted on the overlay (see
        # _persist_react_program), so serve falls back to the settings-default
        # URL with no header rather than replaying a stored credential.
        tools, _live_hashes = resolve_react_tools(
            react_overlay.tool_source, signature_cls, settings
        )
    except (ResolverError, ValueError) as exc:
        raise DomainError(
            "optimization.module_reconstruction_failed", status=409, error=str(exc)
        ) from exc

    try:
        # Strict: a served run must materialise against the exact tool surface
        # it was optimised against — any added/removed tool, not just a hash
        # mismatch, is drift. Mirrors the chat driver so both react-serve
        # surfaces gate identically.
        _assert_tool_set_matches(react_overlay.tool_schema_hashes, tools, strict=True)
    except ToolSchemaDriftError as exc:
        raise DomainError(
            "optimization.tool_schema_drift", status=409, error=str(exc)
        ) from exc

    _apply_bundle_tool_overrides(
        tools,
        tool_descriptions=react_overlay.tool_descriptions,
        tool_arg_descriptions=react_overlay.tool_arg_descriptions,
    )
    # Rename to the proposed display names last (canonical drift-check + overlays
    # above key on the original names). None preserves pre-rename behavior.
    _apply_tool_name_overrides(tools, react_overlay.tool_names)

    program = RetryingReActV2(
        signature_cls, tools=tools, max_iters=react_overlay.max_iters
    )
    program.load_state(artifact.program_state_json)
    return program


def _legacy_pickle_load(program_pickle_base64: str | None) -> Any:
    """Deserialize a legacy pickle artifact.

    Retained for artifacts written before the JSON migration. New jobs use
    the state-JSON path in :func:`_materialize_program` and never reach this
    branch. The same trust boundary applies: artifact bytes were produced by
    our own worker, never accepted from API input.

    Args:
        program_pickle_base64: Base64-encoded ``program.pkl`` bytes from a
            legacy artifact.

    Returns:
        The deserialized DSPy program.

    Raises:
        DomainError: 409 when the pickle payload is empty (caller forgot to
            guard via :func:`_artifact_has_payload`).
    """
    if not program_pickle_base64:
        raise DomainError("optimization.no_program_artifact_scoped", status=409)
    program_bytes = base64.b64decode(program_pickle_base64)
    # pickle.loads is RCE if an attacker can write to the jobs table. The
    # artifact bytes are produced by our own worker and never accepted from
    # API input, so today the only attacker who reaches this branch already
    # has DB write — at which point they own the process anyway. New jobs
    # don't take this path; this exists only to serve already-finished jobs
    # that predate the JSON migration.
    return pickle.loads(program_bytes)


def _stable_hash(payload: Any) -> str:
    """Return a process-stable short SHA256 over a JSON-able payload.

    Used to fold a react program's tool-roster identity into its cache key.
    Built-in ``hash`` is salted per process via ``PYTHONHASHSEED``, so two
    workers would derive different keys for the same roster; canonical JSON +
    SHA256 keeps the key byte-stable across processes.

    Args:
        payload: Any JSON-serializable value (here, the schema-hash map).

    Returns:
        A 16-hex-char prefix of the SHA256 digest — enough entropy to
        distinguish rosters without bloating the cache key.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def load_program(
    job_store,
    optimization_id: str,
    user: AuthenticatedUser,
) -> tuple[Any, RunResponse, dict]:
    """Load and cache an optimized program from a completed job.

    For grid-search jobs, loads the best pair's program automatically and
    synthesizes a ``RunResponse`` from the grid result envelope. Serving spends
    the owner's API key, so this requires editor-tier access (owner / admin /
    editor-or-owner grant): a viewer-tier member gets 403, a caller with no
    access 404s — see :func:`require_role_at_least`.

    Args:
        job_store: The job store to read the job row from.
        optimization_id: The optimization to load.
        user: Authenticated caller; must hold editor-tier access to serve.

    Returns:
        A ``(program, RunResponse, overview)`` tuple where ``program`` is the
        deserialized DSPy module, ``RunResponse`` is the synthesized result,
        and ``overview`` is the parsed payload-overview dict.

    Raises:
        DomainError: 404 when the job is unknown or inaccessible; 403 when the
            caller's role is below editor; 409 when the job is not in a success
            state, has no result, or lacks a serialized program artifact.
    """
    job_data, _role = require_role_at_least(job_store, optimization_id, user, ShareRole.editor)

    overview = parse_overview(job_data)
    optimization_type = overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE, OPTIMIZATION_TYPE_RUN)

    status = status_to_job_status(job_data.get("status", "pending"))
    if status != OptimizationStatus.success:
        raise DomainError(
            "optimization.not_success_status_for_serve",
            status=409,
            params={"status": status.value},
        )

    result_data = job_data.get("result")
    if not result_data or not isinstance(result_data, dict):
        raise DomainError("optimization.no_result", status=409)

    if optimization_type == OPTIMIZATION_TYPE_GRID_SEARCH:
        grid_result = GridSearchResponse.model_validate(result_data)
        if not grid_result.best_pair:
            raise DomainError("grid_search.no_best_pair", status=409)
        artifact = grid_result.best_pair.program_artifact
        if not _artifact_has_payload(artifact):
            raise DomainError("grid_search.no_best_program_artifact", status=409)
        result = RunResponse(
            module_name=grid_result.module_name,
            optimizer_name=grid_result.optimizer_name,
            metric_name=grid_result.metric_name,
            split_counts=grid_result.split_counts,
            baseline_test_metric=grid_result.best_pair.baseline_test_metric,
            optimized_test_metric=grid_result.best_pair.optimized_test_metric,
            metric_improvement=grid_result.best_pair.metric_improvement,
            program_artifact=artifact,
        )
        overview[PAYLOAD_OVERVIEW_MODEL_NAME] = grid_result.best_pair.generation_model
    else:
        result = RunResponse.model_validate(result_data)
        artifact = result.program_artifact
        if not _artifact_has_payload(artifact):
            raise DomainError("optimization.no_program_artifact_scoped", status=409)

    # Fold the tool-roster identity into the key for react programs so a
    # re-sourced or drifted roster never serves a stale cached program. The
    # non-react key stays the bare optimization_id.
    cache_key = optimization_id
    if artifact.react_overlay is not None:
        cache_key = f"{optimization_id}:{_stable_hash(artifact.react_overlay.tool_schema_hashes)}"

    if cache_key not in _program_cache:
        _program_cache[cache_key] = _materialize_program(artifact, overview)

    return _program_cache[cache_key], result, overview


def load_react_chat_inputs(
    job_store,
    optimization_id: str,
    user: AuthenticatedUser,
) -> tuple[type, str, ReactOverlay, dict]:
    """Load a react run's signature + state + overlay for the live chat driver.

    Mirrors :func:`load_program`'s access/status guards but deliberately does
    **not** materialize or cache a program. The chat driver builds a fresh
    ``ReActV2`` per turn bound to a live MCP session that closes when the turn
    ends; a shared, cached program with a dead-session roster would be both
    wrong (tools can't execute) and unsafe (concurrent turns would race on its
    mutable ``tools`` map). Also enforces that the run is a react run served
    from a live-MCP tool source — the only shape that supports live tool calls.

    Args:
        job_store: The job store to read the row from.
        optimization_id: The optimization to chat against.
        user: Authenticated caller; chat is editor-tier (it spends the owner's
            key), so the caller must hold editor-tier access.

    Returns:
        ``(signature_cls, program_state_json, react_overlay, overview)`` — the
        pieces :func:`~...service_gateway.agents.react_serve.run_react_chat`
        needs to assemble a fresh program.

    Raises:
        DomainError: 404 when unknown/inaccessible; 403 when the caller's role
            is below editor; 409 when not in a success state, has no result, is
            not a react run, was not served from a live-MCP source, or is
            missing its signature code.
    """
    job_data, _role = require_role_at_least(job_store, optimization_id, user, ShareRole.editor)
    overview = parse_overview(job_data)
    optimization_type = overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE, OPTIMIZATION_TYPE_RUN)

    status = status_to_job_status(job_data.get("status", "pending"))
    if status != OptimizationStatus.success:
        raise DomainError(
            "optimization.not_success_status_for_serve",
            status=409,
            params={"status": status.value},
        )

    result_data = job_data.get("result")
    if not result_data or not isinstance(result_data, dict):
        raise DomainError("optimization.no_result", status=409)
    # React runs are always run-type (grid search is hidden for react in the
    # wizard), so a grid envelope here means this isn't a chattable react run.
    if optimization_type == OPTIMIZATION_TYPE_GRID_SEARCH:
        raise DomainError("serve.chat_not_react", status=409)

    result = RunResponse.model_validate(result_data)
    artifact = result.program_artifact
    if not _artifact_has_payload(artifact) or artifact.react_overlay is None:
        raise DomainError("serve.chat_not_react", status=409)

    react_overlay = artifact.react_overlay
    tool_source = react_overlay.tool_source or {}
    if not isinstance(tool_source, dict) or tool_source.get("kind") != "live_mcp":
        raise DomainError("serve.chat_requires_live_mcp", status=409)

    signature_code = overview.get(PAYLOAD_OVERVIEW_SIGNATURE_CODE)
    if not signature_code:
        raise DomainError("optimization.no_signature_code_for_reload", status=409)
    signature_cls = load_signature_from_code(signature_code)

    return signature_cls, artifact.program_state_json, react_overlay, overview


def load_pair_program(
    job_store,
    optimization_id: str,
    pair_index: int,
    user: AuthenticatedUser,
) -> tuple[Any, PairResult, dict]:
    """Load and cache the compiled program for a specific grid-search pair.

    Serving spends the owner's key, so this requires editor-tier access (owner /
    admin / editor-or-owner grant): a viewer-tier member gets 403, a caller with
    no access 404s — see :func:`require_role_at_least`.

    Args:
        job_store: The job store to read the job row from.
        optimization_id: The grid-search optimization to load.
        pair_index: The index of the pair within the grid sweep.
        user: Authenticated caller; must hold editor-tier access to serve.

    Returns:
        A ``(program, PairResult, overview)`` tuple where ``program`` is the
        deserialized DSPy module for the pair, ``PairResult`` describes the
        pair's outcome, and ``overview`` is the parsed payload-overview dict.

    Raises:
        DomainError: 404 when the job or pair index is unknown or the caller
            cannot access the job; 403 when the caller's role is below editor;
            409 when the job is not a successful grid search, the pair failed,
            or the program artifact is missing.
    """
    job_data, _role = require_role_at_least(job_store, optimization_id, user, ShareRole.editor)

    overview = parse_overview(job_data)
    optimization_type = overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE, OPTIMIZATION_TYPE_RUN)

    if optimization_type != OPTIMIZATION_TYPE_GRID_SEARCH:
        raise DomainError("grid_search.pair_submission_grid_only", status=409)

    status = status_to_job_status(job_data.get("status", "pending"))
    if status != OptimizationStatus.success:
        raise DomainError(
            "optimization.not_success_status_for_serve",
            status=409,
            params={"status": status.value},
        )

    result_data = job_data.get("result")
    if not result_data or not isinstance(result_data, dict):
        raise DomainError("optimization.no_result", status=409)

    grid_result = GridSearchResponse.model_validate(result_data)

    pair = None
    for pr in grid_result.pair_results:
        if pr.pair_index == pair_index:
            pair = pr
            break
    if pair is None:
        raise DomainError(
            "grid_search.pair_position_missing",
            status=404,
            pair_index=pair_index,
        )

    if pair.error:
        raise DomainError(
            "grid_search.pair_failed_error",
            status=409,
            pair_index=pair_index,
            error=pair.error,
        )

    artifact = pair.program_artifact
    if not _artifact_has_payload(artifact):
        raise DomainError(
            "grid_search.pair_no_artifact",
            status=409,
            pair_index=pair_index,
        )

    cache_key = f"{optimization_id}_pair_{pair_index}"
    if cache_key not in _program_cache:
        _program_cache[cache_key] = _materialize_program(artifact, overview)

    return _program_cache[cache_key], pair, overview
