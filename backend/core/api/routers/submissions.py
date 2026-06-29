"""Routes for submitting optimization jobs. [PUBLIC DEV API]

``POST /run`` — single optimization run.
``POST /grid-search`` — sweep over (generation, reflection) model pairs.

Both endpoints are part of the public dev surface and are listed in
``_SCALAR_PUBLIC_PATHS`` (see ``backend/core/api/app.py``).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, cast
from uuid import uuid4

from fastapi import APIRouter, Depends, Header
from sqlalchemy.orm import Session

from ...billing import ProviderKeyVault, StripeBillingService, provider_slug_for_model
from ...billing.model_access import is_model_locked
from ...constants import (
    OPTIMIZATION_TYPE_GRID_SEARCH,
    OPTIMIZATION_TYPE_RUN,
    PAYLOAD_OVERVIEW_COLUMN_MAPPING,
    PAYLOAD_OVERVIEW_COMPILE_KWARGS,
    PAYLOAD_OVERVIEW_DATASET_FILENAME,
    PAYLOAD_OVERVIEW_DATASET_ROWS,
    PAYLOAD_OVERVIEW_DESCRIPTION,
    PAYLOAD_OVERVIEW_GENERATION_MODELS,
    PAYLOAD_OVERVIEW_IS_PRIVATE,
    PAYLOAD_OVERVIEW_MODEL_NAME,
    PAYLOAD_OVERVIEW_MODEL_SETTINGS,
    PAYLOAD_OVERVIEW_MODULE_KWARGS,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS,
    PAYLOAD_OVERVIEW_OPTIMIZER_NAME,
    PAYLOAD_OVERVIEW_REFLECTION_MODEL,
    PAYLOAD_OVERVIEW_REFLECTION_MODELS,
    PAYLOAD_OVERVIEW_SEED,
    PAYLOAD_OVERVIEW_SHUFFLE,
    PAYLOAD_OVERVIEW_SIGNATURE_CODE,
    PAYLOAD_OVERVIEW_SOURCE_DATASET_ID,
    PAYLOAD_OVERVIEW_SPLIT_FRACTIONS,
    PAYLOAD_OVERVIEW_TASK_FINGERPRINT,
    PAYLOAD_OVERVIEW_TASK_MODEL,
    PAYLOAD_OVERVIEW_TOKEN_SOURCE,
    PAYLOAD_OVERVIEW_TOTAL_PAIRS,
    PAYLOAD_OVERVIEW_USERNAME,
    TOKEN_SOURCE_BYOK,
    TOKEN_SOURCE_MANAGED,
)
from ...i18n import t
from ...i18n_keys import I18nKey
from ...models import (
    GridSearchRequest,
    OptimizationStatus,
    OptimizationSubmissionResponse,
    RunRequest,
)
from ...models.common import ModelConfig, OptimizationType
from ...models.submissions import _OptimizationRequestBase
from ...notifications import notify_job_started
from ...registry import RegistryError
from ...service_gateway import ServiceError
from ...service_gateway.safe_exec import validate_signature_code
from ...storage.dataset_library import DatasetLibraryStore, PostgresDatasetBlobStore
from ...storage.usage import json_byte_size
from ...worker.engine import get_worker
from ..auth import AuthenticatedUser, get_authenticated_user
from ..dataset_access import resolve_effective_role
from ..errors import DomainError
from ..model_catalog import get_catalog_cached
from ._helpers import compute_task_fingerprint, enforce_storage_quota, stable_seed, strip_api_key

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]
IdempotencyKeyHeader = Annotated[
    str | None,
    Header(
        alias="Idempotency-Key",
        description=(
            "Optional client-supplied dedup key. When the same key is reused "
            "for the same authenticated submitter, the original submission "
            "response is returned and no new job is enqueued."
        ),
        max_length=128,
    ),
]


def _existing_submission_response(job_store, optimization_id: str) -> OptimizationSubmissionResponse | None:
    """Rehydrate a previous submission's response from the persisted overview.

    Used when an ``Idempotency-Key`` header matches a prior submission so the
    retry returns the same shape it did the first time without re-enqueuing.

    Args:
        job_store: Source of the persisted job + payload overview.
        optimization_id: Identifier matched via :meth:`find_job_by_idempotency_key`.

    Returns:
        The rebuilt response, or ``None`` if the job vanished between the
        lookup and the rehydration (treated as no-dedup-hit by callers).
    """
    try:
        job = job_store.get_job(optimization_id)
    except KeyError:
        return None
    overview = job.get("payload_overview") or {}
    optimization_type = overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE) or OPTIMIZATION_TYPE_RUN
    status_value = job.get("status") or OptimizationStatus.pending.value
    try:
        status_enum = OptimizationStatus(status_value)
    except ValueError:
        status_enum = OptimizationStatus.pending
    return OptimizationSubmissionResponse(
        optimization_id=optimization_id,
        optimization_type=cast(OptimizationType, optimization_type),
        status=status_enum,
        created_at=job.get("created_at") or datetime.now(UTC),
        name=overview.get(PAYLOAD_OVERVIEW_NAME, ""),
        username=overview.get(PAYLOAD_OVERVIEW_USERNAME, ""),
        module_name=overview.get(PAYLOAD_OVERVIEW_MODULE_NAME, ""),
        optimizer_name=overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_NAME, ""),
    )


def _catalog_models_as_configs() -> list[ModelConfig]:
    """Return every available catalog model wrapped as a ``ModelConfig``.

    Used by the grid-search route to expand ``use_all_available_*`` flags
    into concrete model lists.

    Returns:
        A list of ``ModelConfig`` instances, one per available catalog model.

    Raises:
        DomainError: 400 ``submit.no_models_available`` when the catalog
            reports no available models (usually no provider API keys configured).
    """
    catalog = get_catalog_cached()
    configs = [ModelConfig(name=entry.value) for entry in catalog.models]
    if not configs:
        raise DomainError("submit.no_models_available", status=400)
    return configs


def _enforce_vision_capability(
    *,
    signature_code: str,
    candidate_models: list[ModelConfig],
) -> None:
    """Reject submissions whose signature has dspy.Image inputs but a non-vision model.

    Parses ``signature_code`` once via the safe-exec subprocess, then — only when
    ``dspy.Image`` typed inputs are present — looks up every candidate model in
    the catalog and requires ``supports_vision`` for each one.

    Args:
        signature_code: User-provided DSPy Signature source.
        candidate_models: Models that would receive image-bearing inputs.

    Raises:
        DomainError: 400 ``submission.vision_required`` listing offending
            models when any candidate lacks vision support.
    """
    intro = validate_signature_code(signature_code)
    image_fields = list(intro.image_input_fields)
    if not image_fields:
        return

    catalog = get_catalog_cached()
    vision_supported: dict[str, bool] = {entry.value: entry.supports_vision for entry in catalog.models}

    offenders = sorted(
        {
            cfg.normalized_identifier()
            for cfg in candidate_models
            if not vision_supported.get(cfg.normalized_identifier(), False)
        }
    )
    if offenders:
        raise DomainError(
            I18nKey.SUBMISSION_VISION_REQUIRED,
            status=400,
            fields=", ".join(image_fields),
            model=", ".join(offenders),
        )


def _materialize_staged_dataset(
    payload: _OptimizationRequestBase,
    *,
    job_store,
    username: str,
) -> str | None:
    """Inline a staged dataset into ``payload.dataset`` ahead of validation.

    Agent-driven submits arrive with ``staged_dataset_id`` set instead of an
    inline ``dataset`` so the model never has to ferry tens of thousands of
    rows through its tool arguments. This helper loads the persisted rows
    and swaps them onto ``payload.dataset``; eviction happens in
    :func:`_evict_staged_dataset`, called only after the job has been
    successfully created. Keeping the row alive across validation lets a
    failed submit (e.g. missing reflection_model_config) be retried without
    forcing the user to re-upload — historically the eager delete here
    silently consumed the staged row on the first 422 and the next attempt
    hit 400 ``staged_dataset_not_found``.

    Args:
        payload: The validated request body (mutated in place).
        job_store: Backend exposing ``get_staged_dataset``.
        username: Authenticated submitter; staging rows are scoped to one owner.

    Returns:
        The staged id when one was consumed (caller passes it to
        :func:`_evict_staged_dataset` after the job lands), or ``None`` for
        inline payloads.

    Raises:
        DomainError: 400 ``submission.staged_dataset_not_found`` when the id is
            unknown or owned by another user.
    """
    staged_id = payload.staged_dataset_id
    if not staged_id:
        return None
    rows = job_store.get_staged_dataset(staged_id, username)
    if not rows:
        raise DomainError(
            I18nKey.SUBMISSION_STAGED_DATASET_NOT_FOUND,
            status=400,
            staged_dataset_id=staged_id,
        )
    payload.dataset = rows
    payload.staged_dataset_id = None
    return staged_id


def _evict_staged_dataset(job_store, staged_id: str | None, username: str) -> None:
    """Drop the staged row after the job has been committed; never raises."""
    if not staged_id:
        return
    try:
        job_store.delete_staged_dataset(staged_id, username)
    except Exception:
        logger.warning("Failed to evict staged dataset %s after consumption", staged_id, exc_info=True)


def _materialize_library_dataset(
    payload: _OptimizationRequestBase,
    *,
    job_store,
    user: AuthenticatedUser,
) -> str | None:
    """Inline a personal-library dataset into ``payload.dataset`` by reference.

    The submit-wizard consumer path sends ``source_dataset_id`` instead of inline
    rows so the browser never re-uploads a file already saved to the library.
    This helper resolves the caller's access (viewer-or-above on the dataset),
    loads the saved rows onto ``payload.dataset``, and clears the reference so the
    persisted payload carries the exact rows the run used — the link back to the
    dataset survives in the payload overview, not the payload. Unlike a staged
    dataset, the library entry is never evicted: it is a durable, navigable source
    every run that used it points at.

    Args:
        payload: The validated request body (mutated in place).
        job_store: Backend whose ``engine`` carries the dataset tables.
        user: Authenticated submitter; access is resolved against their grants.

    Returns:
        The source dataset id when one was consumed (recorded in the overview by
        the caller), or ``None`` for inline/staged payloads.

    Raises:
        DomainError: 404 ``dataset.library.not_found`` when the caller cannot
            reach the dataset or its rows are missing.
    """
    source_id = payload.source_dataset_id
    if not source_id:
        return None
    with Session(job_store.engine) as session:
        role = resolve_effective_role(session, source_id, user)
    if role is None:
        raise DomainError("dataset.library.not_found", status=404)
    store = DatasetLibraryStore(job_store.engine, PostgresDatasetBlobStore(job_store.engine))
    rows = store.get_rows(source_id)
    if not rows:
        raise DomainError("dataset.library.not_found", status=404)
    payload.dataset = rows
    payload.source_dataset_id = None
    return source_id


def _expand_catalog_grid_payload(payload: GridSearchRequest) -> None:
    """Populate generation/reflection model lists from the catalog when flagged.

    Replaces ``payload.generation_models`` and/or ``payload.reflection_models``
    with every available catalog model when the matching
    ``use_all_available_*`` flag is set. When neither flag is set this is a
    no-op.

    Args:
        payload: The grid-search request to mutate in place.

    Raises:
        DomainError: 400 when expansion is requested but no models are
            available.
    """
    if not (payload.use_all_available_generation_models or payload.use_all_available_reflection_models):
        return
    expanded = _catalog_models_as_configs()
    if payload.use_all_available_generation_models:
        payload.generation_models = expanded
    if payload.use_all_available_reflection_models:
        payload.reflection_models = expanded


def _enforce_credit_balance(job_store, username: str, token_source: str) -> None:
    """Block a depleted account from starting a managed run at submit time.

    Reads the account's spendable credits (free grant + purchased balance, with
    the rolling grant reset applied) and refuses the submission when nothing is
    left. The free grant means a brand-new account always passes; this only fires
    once both the grant and any purchased balance are exhausted. A BYOK run bills
    the user's own provider key, so it spends no credits and is exempt — the gate
    fires for managed runs only.

    Args:
        job_store: Job-store instance whose ORM engine backs the billing tables.
        username: Account attempting the submission.
        token_source: ``"managed"`` or ``"byok"``; BYOK runs skip the gate.

    Raises:
        DomainError: 402 when a managed account has no spendable credits.
    """
    if token_source == TOKEN_SOURCE_BYOK:
        return
    engine = getattr(job_store, "engine", None)
    if engine is None or not username:
        return
    service = StripeBillingService(engine=engine)
    if service.spendable_credits(username) <= 0:
        raise DomainError("billing.insufficient_credits", status=402)


def _enforce_frontier_lock(
    job_store, username: str, token_source: str, model_values: list[str]
) -> None:
    """Refuse a managed run on a frontier model the account can't yet unlock.

    Mirrors the wizard's client-side lock server-side so it is enforced, not
    advisory: in managed mode an account without a purchased balance (or active
    Premium) may run mini models on its free grant but not the frontier
    task/reflection models. BYOK runs bill the user's own key and never lock, so
    they are exempt. A no-op when the store exposes no SQL engine (legacy/in-memory).

    Args:
        job_store: Job-store instance whose ORM engine backs the billing tables.
        username: Account attempting the submission.
        token_source: ``"managed"`` or ``"byok"``; BYOK is exempt.
        model_values: Fully-qualified model ids the run would execute on.

    Raises:
        DomainError: 402 ``billing.frontier_locked`` listing the locked models.
    """
    if token_source != TOKEN_SOURCE_MANAGED:
        return
    engine = getattr(job_store, "engine", None)
    if engine is None or not username:
        return
    service = StripeBillingService(engine=engine)
    if service.frontier_unlocked(username):
        return
    locked = sorted({m for m in model_values if m and is_model_locked(m, token_source, False)})
    if locked:
        raise DomainError("billing.frontier_locked", status=402, model=", ".join(locked))


def _enforce_byok_connections(
    job_store, username: str, token_source: str, model_values: list[str]
) -> None:
    """Refuse a BYOK run when the account has no saved key for a model's provider.

    In BYOK mode every model authenticates with the user's own provider key,
    resolved from the encrypt-at-rest vault at run time. If the account saved no
    connection for a model's provider, the run would have nothing to authenticate
    with, so reject it at submit with a clear, translated error rather than
    letting the job fail mid-run. Managed runs are exempt (they spend platform
    credits). Models with no ``provider/`` prefix are skipped — there is no
    provider to resolve a key for. A no-op when the store exposes no SQL engine.

    Args:
        job_store: Job-store instance whose ORM engine backs the billing tables.
        username: Account attempting the submission.
        token_source: ``"managed"`` or ``"byok"``; managed is exempt.
        model_values: Fully-qualified model ids the run would execute on.

    Raises:
        DomainError: 400 ``billing.byok_missing_connection`` listing the providers
            the account has no saved connection for.
    """
    if token_source != TOKEN_SOURCE_BYOK:
        return
    engine = getattr(job_store, "engine", None)
    if engine is None or not username:
        return
    vault = ProviderKeyVault(engine=engine)
    providers = {provider_slug_for_model(m) for m in model_values if m}
    missing = sorted(
        p for p in providers if p is not None and not vault.has_connection(username, p)
    )
    if missing:
        raise DomainError("billing.byok_missing_connection", status=400, provider=", ".join(missing))


def create_submissions_router(*, service, job_store) -> APIRouter:
    """Build the submissions router.

    Args:
        service: Optimization service used for synchronous validation.
        job_store: Job-store instance used to persist new submissions.

    Returns:
        A FastAPI ``APIRouter`` exposing the ``/run`` and ``/grid-search``
        endpoints.
    """
    router = APIRouter()

    @router.post(
        "/run",
        response_model=OptimizationSubmissionResponse,
        status_code=201,
        summary="Submit a single DSPy optimization run",
        tags=["agent"],
    )
    def submit_job(
        payload: RunRequest,
        current_user: AuthenticatedUserDep,
        idempotency_key: IdempotencyKeyHeader = None,
    ) -> OptimizationSubmissionResponse:
        """Queue one end-to-end DSPy optimization for background execution.

        Validates synchronously, persists a job row, and enqueues the
        payload. Returns HTTP 201 immediately; poll
        ``/optimizations/{id}/summary`` or stream via
        ``/optimizations/{id}/stream`` for progress. Security:
        ``model_settings.api_key`` is stripped from the persisted overview;
        the persisted owner is the authenticated caller, not whatever the
        client posted.

        Args:
            payload: The run-request body validated by FastAPI.
            current_user: Authenticated submitter resolved from the bearer token.

        Returns:
            An ``OptimizationSubmissionResponse`` carrying the assigned id
            and ``pending`` status.

        Raises:
            DomainError: 400 (validation), 409 (quota), 422 (malformed body).
        """
        payload.username = current_user.username

        normalized_key = (idempotency_key or "").strip() or None
        if normalized_key:
            existing_id = job_store.find_job_by_idempotency_key(payload.username, normalized_key)
            if existing_id:
                cached = _existing_submission_response(job_store, existing_id)
                if cached is not None:
                    logger.info(
                        "Idempotent retry hit: returning existing %s for user=%s key=%s",
                        existing_id,
                        payload.username,
                        normalized_key,
                    )
                    return cached

        staged_id = _materialize_staged_dataset(payload, job_store=job_store, username=payload.username)
        source_dataset_id = _materialize_library_dataset(payload, job_store=job_store, user=current_user)

        try:
            service.validate_payload(payload)
        except (ServiceError, RegistryError) as exc:
            # Log the resolver/validation detail server-side only; the client
            # gets a stable code without the internal registry surface leaked.
            logger.warning("Payload validation failed: %s", exc)
            raise DomainError("submission.validation_failed", status=400) from exc

        _enforce_vision_capability(
            signature_code=payload.signature_code,
            candidate_models=[payload.model_settings],
        )

        enforce_storage_quota(
            job_store,
            payload.username,
            incoming_bytes=json_byte_size(payload.model_dump(mode="json", by_alias=True)),
        )

        _enforce_credit_balance(job_store, payload.username, payload.token_source)
        _run_model_values = [
            cfg.normalized_identifier()
            for cfg in (payload.model_settings, payload.reflection_model_settings, payload.task_model_settings)
            if cfg is not None
        ]
        _enforce_frontier_lock(job_store, payload.username, payload.token_source, _run_model_values)
        _enforce_byok_connections(job_store, payload.username, payload.token_source, _run_model_values)

        optimization_id = str(uuid4())
        task_fingerprint = compute_task_fingerprint(payload.signature_code, payload.metric_code, payload.dataset)
        # Derive the default seed from the task fingerprint (not the optimization id)
        # so submissions of the same task share train/val/test splits — a prerequisite
        # for the compare flow to line up per-row test results across deduplicated runs.
        if payload.seed is None:
            payload.seed = stable_seed(task_fingerprint)

        job_store.create_job(optimization_id, username=payload.username, idempotency_key=normalized_key)
        job_store.set_payload_overview(
            optimization_id,
            {
                PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE: OPTIMIZATION_TYPE_RUN,
                PAYLOAD_OVERVIEW_NAME: payload.name,
                PAYLOAD_OVERVIEW_DESCRIPTION: payload.description,
                PAYLOAD_OVERVIEW_USERNAME: payload.username,
                PAYLOAD_OVERVIEW_MODULE_NAME: payload.module_name,
                PAYLOAD_OVERVIEW_MODULE_KWARGS: dict(payload.module_kwargs),
                PAYLOAD_OVERVIEW_SIGNATURE_CODE: payload.signature_code,
                PAYLOAD_OVERVIEW_OPTIMIZER_NAME: payload.optimizer_name,
                PAYLOAD_OVERVIEW_MODEL_NAME: payload.model_settings.normalized_identifier(),
                PAYLOAD_OVERVIEW_MODEL_SETTINGS: strip_api_key(payload.model_settings.model_dump()),
                PAYLOAD_OVERVIEW_REFLECTION_MODEL: (
                    payload.reflection_model_settings.normalized_identifier()
                    if payload.reflection_model_settings
                    else None
                ),
                PAYLOAD_OVERVIEW_TASK_MODEL: (
                    payload.task_model_settings.normalized_identifier() if payload.task_model_settings else None
                ),
                PAYLOAD_OVERVIEW_COLUMN_MAPPING: payload.column_mapping.model_dump(),
                PAYLOAD_OVERVIEW_DATASET_ROWS: len(payload.dataset),
                PAYLOAD_OVERVIEW_DATASET_FILENAME: payload.dataset_filename,
                PAYLOAD_OVERVIEW_SPLIT_FRACTIONS: payload.split_fractions.model_dump(),
                PAYLOAD_OVERVIEW_SHUFFLE: payload.shuffle,
                PAYLOAD_OVERVIEW_SEED: payload.seed,
                PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS: dict(payload.optimizer_kwargs),
                PAYLOAD_OVERVIEW_COMPILE_KWARGS: dict(payload.compile_kwargs),
                PAYLOAD_OVERVIEW_TASK_FINGERPRINT: task_fingerprint,
                PAYLOAD_OVERVIEW_TOKEN_SOURCE: payload.token_source,
                PAYLOAD_OVERVIEW_IS_PRIVATE: payload.is_private,
                PAYLOAD_OVERVIEW_SOURCE_DATASET_ID: source_dataset_id,
            },
        )
        _evict_staged_dataset(job_store, staged_id, payload.username)

        current_worker = get_worker(job_store, service=service)
        current_worker.submit_job(optimization_id, payload)

        logger.info(
            "Enqueued job %s for module=%s optimizer=%s",
            optimization_id,
            payload.module_name,
            payload.optimizer_name,
        )

        notify_job_started(
            optimization_id=optimization_id,
            username=payload.username,
            optimization_type=OPTIMIZATION_TYPE_RUN,
            optimizer_name=payload.optimizer_name,
            module_name=payload.module_name,
            model_name=payload.model_settings.normalized_identifier(),
        )

        return OptimizationSubmissionResponse(
            optimization_id=optimization_id,
            optimization_type=cast(OptimizationType, OPTIMIZATION_TYPE_RUN),
            status=OptimizationStatus.pending,
            created_at=datetime.now(UTC),
            name=payload.name,
            username=payload.username,
            module_name=payload.module_name,
            optimizer_name=payload.optimizer_name,
        )

    @router.post(
        "/grid-search",
        response_model=OptimizationSubmissionResponse,
        status_code=201,
        summary="Submit a grid search over model pairs",
        tags=["agent"],
    )
    def submit_grid_search(
        payload: GridSearchRequest,
        current_user: AuthenticatedUserDep,
        idempotency_key: IdempotencyKeyHeader = None,
    ) -> OptimizationSubmissionResponse:
        """Queue a sweep over ``(generation_model, reflection_model)`` pairs.

        Runs one optimization per Cartesian pair serially inside a single
        job. When ``use_all_available_generation_models`` or
        ``use_all_available_reflection_models`` is set, the server replaces
        the matching list with every model currently available in the
        catalog before validation or enqueue. Poll
        ``/optimizations/{id}/summary`` for per-pair progress. The persisted
        owner is the authenticated caller, not whatever the client posted.

        Args:
            payload: The grid-search request body validated by FastAPI.
            current_user: Authenticated submitter resolved from the bearer token.

        Returns:
            An ``OptimizationSubmissionResponse`` carrying the assigned id
            and ``pending`` status.

        Raises:
            DomainError: 400 (validation/empty catalog), 409 (quota), 422
                (malformed).
        """
        payload.username = current_user.username
        _expand_catalog_grid_payload(payload)

        normalized_key = (idempotency_key or "").strip() or None
        if normalized_key:
            existing_id = job_store.find_job_by_idempotency_key(payload.username, normalized_key)
            if existing_id:
                cached = _existing_submission_response(job_store, existing_id)
                if cached is not None:
                    logger.info(
                        "Idempotent grid-search retry hit: returning existing %s for user=%s key=%s",
                        existing_id,
                        payload.username,
                        normalized_key,
                    )
                    return cached

        staged_id = _materialize_staged_dataset(payload, job_store=job_store, username=payload.username)
        source_dataset_id = _materialize_library_dataset(payload, job_store=job_store, user=current_user)

        if hasattr(service, "validate_grid_search_payload"):
            try:
                service.validate_grid_search_payload(payload)
            except (ServiceError, RegistryError) as exc:
                # Log the detail server-side only; don't leak it to the client.
                logger.warning("Grid search validation failed: %s", exc)
                raise DomainError("submission.validation_failed", status=400) from exc

        _enforce_vision_capability(
            signature_code=payload.signature_code,
            candidate_models=list(payload.generation_models),
        )

        enforce_storage_quota(
            job_store,
            payload.username,
            incoming_bytes=json_byte_size(payload.model_dump(mode="json", by_alias=True)),
        )

        _enforce_credit_balance(job_store, payload.username, payload.token_source)
        _grid_model_values = [
            cfg.normalized_identifier()
            for cfg in (*payload.generation_models, *payload.reflection_models)
        ]
        _enforce_frontier_lock(job_store, payload.username, payload.token_source, _grid_model_values)
        _enforce_byok_connections(job_store, payload.username, payload.token_source, _grid_model_values)

        optimization_id = str(uuid4())
        if payload.seed is None:
            payload.seed = stable_seed(optimization_id)
        total_pairs = len(payload.generation_models) * len(payload.reflection_models)

        task_fingerprint = compute_task_fingerprint(payload.signature_code, payload.metric_code, payload.dataset)

        job_store.create_job(optimization_id, username=payload.username, idempotency_key=normalized_key)
        job_store.set_payload_overview(
            optimization_id,
            {
                PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE: OPTIMIZATION_TYPE_GRID_SEARCH,
                PAYLOAD_OVERVIEW_NAME: payload.name,
                PAYLOAD_OVERVIEW_DESCRIPTION: payload.description,
                PAYLOAD_OVERVIEW_USERNAME: payload.username,
                PAYLOAD_OVERVIEW_MODULE_NAME: payload.module_name,
                PAYLOAD_OVERVIEW_MODULE_KWARGS: dict(payload.module_kwargs),
                PAYLOAD_OVERVIEW_SIGNATURE_CODE: payload.signature_code,
                PAYLOAD_OVERVIEW_OPTIMIZER_NAME: payload.optimizer_name,
                PAYLOAD_OVERVIEW_COLUMN_MAPPING: payload.column_mapping.model_dump(),
                PAYLOAD_OVERVIEW_DATASET_ROWS: len(payload.dataset),
                PAYLOAD_OVERVIEW_DATASET_FILENAME: payload.dataset_filename,
                PAYLOAD_OVERVIEW_SPLIT_FRACTIONS: payload.split_fractions.model_dump(),
                PAYLOAD_OVERVIEW_SHUFFLE: payload.shuffle,
                PAYLOAD_OVERVIEW_SEED: payload.seed,
                PAYLOAD_OVERVIEW_OPTIMIZER_KWARGS: dict(payload.optimizer_kwargs),
                PAYLOAD_OVERVIEW_COMPILE_KWARGS: dict(payload.compile_kwargs),
                PAYLOAD_OVERVIEW_TOTAL_PAIRS: total_pairs,
                PAYLOAD_OVERVIEW_GENERATION_MODELS: [m.model_dump() for m in payload.generation_models],
                PAYLOAD_OVERVIEW_REFLECTION_MODELS: [m.model_dump() for m in payload.reflection_models],
                PAYLOAD_OVERVIEW_TASK_FINGERPRINT: task_fingerprint,
                PAYLOAD_OVERVIEW_TOKEN_SOURCE: payload.token_source,
                PAYLOAD_OVERVIEW_IS_PRIVATE: payload.is_private,
                PAYLOAD_OVERVIEW_SOURCE_DATASET_ID: source_dataset_id,
            },
        )
        _evict_staged_dataset(job_store, staged_id, payload.username)

        current_worker = get_worker(job_store, service=service)
        current_worker.submit_job(optimization_id, payload)

        logger.info(
            "Enqueued grid search %s: %d pairs, module=%s optimizer=%s",
            optimization_id,
            total_pairs,
            payload.module_name,
            payload.optimizer_name,
        )

        notify_job_started(
            optimization_id=optimization_id,
            username=payload.username,
            optimization_type=OPTIMIZATION_TYPE_GRID_SEARCH,
            optimizer_name=payload.optimizer_name,
            module_name=payload.module_name,
            model_name=t("optimization.pairs_label", count=total_pairs),
        )

        return OptimizationSubmissionResponse(
            optimization_id=optimization_id,
            optimization_type=cast(OptimizationType, OPTIMIZATION_TYPE_GRID_SEARCH),
            status=OptimizationStatus.pending,
            created_at=datetime.now(UTC),
            name=payload.name,
            username=payload.username,
            module_name=payload.module_name,
            optimizer_name=payload.optimizer_name,
        )

    return router
