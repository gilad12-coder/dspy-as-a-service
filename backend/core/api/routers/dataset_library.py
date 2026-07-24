"""Routes for the personal dataset library — save, list, read, edit, share-aware.

A library dataset is a single saved file: lean metadata plus the row bytes
behind the :class:`DatasetBlobStore` seam (see
:mod:`core.storage.dataset_library`). These endpoints are the CRUD surface the
three producer doorways (tagger, upload page, optimization Data tab) and the
submit-wizard consumer call.

Access is tiered by :mod:`core.api.dataset_access`: a caller reaches a dataset
as its owner (the creator or an admin) or through a share grant. Reads
(metadata, rows) need ``viewer``; editing rows needs ``editor``; renaming,
deleting, and cloning-quota are owner-only — sharing management lives in
:mod:`core.api.routers.dataset_share`. The listing folds in datasets shared with
the caller alongside their own.

Saving gates on three limits, in order: the per-file compressed cap
(``settings.dataset_max_file_bytes`` → 413), content-hash dedupe (a byte-
identical re-save returns the existing entry, charged nothing), and the unified
per-user storage budget (``user.storage.quota_exceeded`` → 409, shared with
optimizations and every other surface — see
:func:`core.api.routers._helpers.enforce_storage_quota`). The budget counts the
uncompressed ``byte_size``; the per-file cap still measures the compressed
``stored_bytes``.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...config import settings
from ...constants import (
    PAYLOAD_OVERVIEW_DATASET_FILENAME,
    PAYLOAD_OVERVIEW_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE,
    PAYLOAD_OVERVIEW_SOURCE_DATASET_ID,
)
from ...models import (
    BulkDeleteByIdsRequest,
    BulkDeleteByIdsResponse,
    BulkDeleteByIdsSkipped,
)
from ...storage.dataset_library import (
    DatasetLibraryStore,
    DatasetRecord,
    PostgresDatasetBlobStore,
    StagedDataset,
)
from ..auth import AuthenticatedUser, get_authenticated_user
from ..converters import parse_overview
from ..dataset_access import (
    ShareRole,
    list_grants_for_user_all,
    require_role,
)
from ..errors import DomainError
from ._helpers import enforce_storage_quota, load_job_for_user

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]

# Upper bound on how many of the caller's visible runs the reverse-link endpoint
# scans for a matching ``source_dataset_id``. The payload-overview has no indexed
# column to filter on, so the match is a list-then-filter in Python (the pattern
# the analytics routes use); this caps that scan. A user with more than this many
# runs that used one dataset would see the link truncate — acceptable for v1, and
# the natural upgrade is a promoted indexed column.
_REVERSE_LINK_SCAN_LIMIT = 1000


class SaveDatasetRequest(BaseModel):
    """Request body for ``POST /datasets/library`` — rows plus saved schema."""

    name: str = Field(min_length=1, max_length=255)
    source: str = Field(default="upload", max_length=32)
    dataset: list[dict[str, Any]] = Field(min_length=1, max_length=200_000)
    column_schema: dict[str, Any] = Field(default_factory=dict)


class RenameDatasetRequest(BaseModel):
    """Request body for ``PATCH /datasets/library/{id}`` — the new display name."""

    name: str = Field(min_length=1, max_length=255)


class EditRowsRequest(BaseModel):
    """Request body for ``PUT /datasets/library/{id}/rows`` — replacement rows."""

    rows: list[dict[str, Any]] = Field(min_length=1, max_length=200_000)
    column_schema: dict[str, Any] | None = None


class DatasetSummary(BaseModel):
    """One library dataset's metadata, as returned to the client.

    ``owner_username`` and ``role`` let the client tell an owned entry from one
    shared with the caller and gate its actions (rename/delete/share are owner-
    only, row edits are editor+).
    """

    id: str
    name: str
    source: str
    row_count: int
    column_count: int
    byte_size: int
    content_hash: str
    owner_username: str
    role: str
    created_at: str
    updated_at: str


class UsageMeter(BaseModel):
    """Aggregate library storage used by the caller against their quota."""

    used_bytes: int
    quota_bytes: int


class SaveDatasetResponse(BaseModel):
    """Envelope for ``POST /datasets/library`` — the entry and whether it deduped."""

    dataset: DatasetSummary
    deduplicated: bool


class DatasetListResponse(BaseModel):
    """Envelope for ``GET /datasets/library`` — the caller's entries and usage."""

    datasets: list[DatasetSummary]
    usage: UsageMeter


class DatasetRowsResponse(BaseModel):
    """Envelope for ``GET /datasets/library/{id}/rows`` — columns, rows, schema.

    ``column_schema`` carries the saved roles/kinds/order so the submit wizard
    can pre-fill its column step when the dataset is picked from the library.
    """

    id: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    column_schema: dict[str, Any]


class DeleteDatasetResponse(BaseModel):
    """Envelope for ``DELETE /datasets/library/{id}`` — definitive delete flag."""

    deleted: bool


class SaveFromOptimizationRequest(BaseModel):
    """Optional body for ``POST /datasets/library/from-optimization/{id}``.

    ``name`` overrides the name derived from the run's stored dataset filename or
    optimization name; omit it to accept that default.
    """

    name: str | None = Field(default=None, min_length=1, max_length=255)


class DatasetOptimizationRef(BaseModel):
    """One optimization that was submitted from a library dataset (lean ref)."""

    optimization_id: str
    name: str | None = None
    status: str | None = None
    optimization_type: str | None = None
    username: str | None = None
    created_at: str | None = None


class DatasetOptimizationsResponse(BaseModel):
    """Envelope for ``GET /datasets/library/{id}/optimizations`` — the reverse link."""

    optimizations: list[DatasetOptimizationRef]


def _summary(record: DatasetRecord, role: ShareRole) -> DatasetSummary:
    """Project a :class:`DatasetRecord` onto the client-facing summary.

    Args:
        record: The stored dataset metadata.
        role: The caller's effective role on the dataset (``owner`` for their
            own entries, the grant tier for shared ones).

    Returns:
        The :class:`DatasetSummary` for the API response.
    """
    return DatasetSummary(
        id=record.id,
        name=record.name,
        source=record.source,
        row_count=record.row_count,
        column_count=record.column_count,
        byte_size=record.byte_size,
        content_hash=record.content_hash,
        owner_username=record.owner_username,
        role=str(role),
        created_at=record.created_at.isoformat(),
        updated_at=record.updated_at.isoformat(),
    )


def _columns_of(record: DatasetRecord, rows: list[dict[str, Any]]) -> list[str]:
    """Resolve a dataset's column order for the rows response.

    Args:
        record: The dataset metadata; its saved ``column_order`` wins when set.
        rows: The decompressed rows, used as the fallback column source.

    Returns:
        The ordered column names.
    """
    order = record.column_schema.get("column_order")
    if isinstance(order, list) and order:
        return [str(col) for col in order]
    return list(rows[0].keys()) if rows else []


def _mapping_columns(side: Any) -> list[str]:
    """Extract dataset column names from one side of a stored column mapping.

    A run's ``column_mapping`` stores ``inputs``/``outputs`` as
    ``{signature_field: column_name}`` dicts; legacy rows stored bare lists of
    column names. Both shapes are tolerated so saving a run's dataset never 500s
    on an old row.

    Args:
        side: The ``inputs`` or ``outputs`` value from the stored mapping.

    Returns:
        The dataset column names bound on that side, or ``[]`` for an unusable
        shape.
    """
    if isinstance(side, dict):
        return [str(col) for col in side.values()]
    if isinstance(side, list):
        return [str(col) for col in side]
    return []


def _schema_from_run_payload(payload: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a library column schema from a run's payload for the save-run path.

    Maps the run's ``column_mapping`` onto per-column roles (inputs → ``input``,
    outputs → ``output``), preserves the submitted ``column_order``, and defaults
    every column's kind to ``text`` — enough for the submit wizard to pre-fill its
    column step when the saved dataset is later picked.

    Args:
        payload: The run's stored payload (its ``column_mapping`` / ``column_order``).
        rows: The run's dataset rows, used to recover column order when unstored.

    Returns:
        A ``{column_order, column_roles, column_kinds}`` schema dict.
    """
    raw_mapping = payload.get("column_mapping") or {}
    column_roles: dict[str, str] = dict.fromkeys(_mapping_columns(raw_mapping.get("inputs")), "input")
    for col in _mapping_columns(raw_mapping.get("outputs")):
        column_roles[col] = "output"
    order = payload.get("column_order")
    if not (isinstance(order, list) and order):
        order = list(rows[0].keys()) if rows else []
    column_order = [str(col) for col in order]
    return {
        "column_order": column_order,
        "column_roles": column_roles,
        "column_kinds": dict.fromkeys(column_order, "text"),
    }


def create_dataset_library_router(*, job_store) -> APIRouter:
    """Build the personal dataset-library CRUD router.

    Args:
        job_store: Storage backend whose ``engine`` carries the ``datasets`` and
            ``dataset_blobs`` tables; used to construct the library store.

    Returns:
        A configured :class:`APIRouter` exposing the owner-scoped library
        endpoints under ``/datasets/library``.
    """
    store = DatasetLibraryStore(job_store.engine, PostgresDatasetBlobStore(job_store.engine))

    def _require(dataset_id: str, caller: AuthenticatedUser, minimum: ShareRole) -> tuple[DatasetRecord, ShareRole]:
        """Load a dataset the caller may reach at ``minimum`` tier, or raise.

        Args:
            dataset_id: Id of the requested dataset.
            caller: Authenticated caller.
            minimum: Lowest :class:`ShareRole` the route requires.

        Returns:
            A ``(record, role)`` pair — the dataset metadata and the caller's
            effective role (at least ``minimum``).

        Raises:
            DomainError: 404 when the caller has no access to the dataset; 403
                when their tier is below ``minimum``.
        """
        with Session(job_store.engine) as session:
            role = require_role(session, dataset_id, caller, minimum)
        record = store.get_dataset(dataset_id)
        if record is None:
            raise DomainError("dataset.library.not_found", status=404)
        return record, role

    def _save_gated(
        *, owner: str, name: str, source: str, rows: list[dict[str, Any]], column_schema: dict[str, Any]
    ) -> tuple[DatasetRecord, bool]:
        """Stage rows and persist them as a new owned entry, gating size/quota.

        Shared by the save and clone paths: serialize/compress once, reject an
        over-cap file (413), return an existing byte-identical entry instead of
        storing a copy (dedupe), then reject a save that would push the owner
        over their unified storage budget (409) before committing.

        Args:
            owner: Lowercased owner the entry is saved under.
            name: Display name for the entry.
            source: Producing surface recorded on the entry.
            rows: The rows to store.
            column_schema: Saved column roles/kinds/order.

        Returns:
            A ``(record, deduplicated)`` pair — ``deduplicated`` is ``True`` when
            an existing identical entry was returned instead of a new one.

        Raises:
            DomainError: 413 over the per-file cap; 409 over the unified storage
                budget.
        """
        staged: StagedDataset = store.stage(rows, column_schema)
        if staged.stored_bytes > settings.dataset_max_file_bytes:
            raise DomainError(
                "dataset.library.too_large",
                status=413,
                max_mb=round(settings.dataset_max_file_bytes / (1024 * 1024), 1),
            )
        existing = store.find_by_hash(owner, staged.digest)
        if existing is not None:
            return existing, True
        enforce_storage_quota(job_store, owner, incoming_bytes=staged.byte_size)
        record = store.commit_staged(owner_username=owner, name=name, source=source, staged=staged)
        return record, False

    router = APIRouter()

    @router.post(
        "/datasets/library",
        response_model=SaveDatasetResponse,
        summary="Save a dataset to the caller's personal library",
    )
    def save_dataset(
        payload: SaveDatasetRequest,
        current_user: AuthenticatedUserDep,
    ) -> SaveDatasetResponse:
        """Save rows as a new library entry, gating on size, dedupe, and quota.

        Args:
            payload: The dataset name, source, rows, and saved column schema.
            current_user: Authenticated owner of the new entry.

        Returns:
            A :class:`SaveDatasetResponse` carrying the stored summary and
            whether an existing byte-identical entry was returned instead.

        Raises:
            DomainError: 413 when the compressed file exceeds the per-file cap;
                409 when storing it would exceed the unified storage budget.
        """
        record, deduped = _save_gated(
            owner=current_user.username,
            name=payload.name,
            source=payload.source,
            rows=payload.dataset,
            column_schema=payload.column_schema,
        )
        return SaveDatasetResponse(dataset=_summary(record, ShareRole.owner), deduplicated=deduped)

    @router.get(
        "/datasets/library",
        response_model=DatasetListResponse,
        operation_id="list_datasets_for_agent",
        summary="List the caller's saved datasets (and those shared with them)",
        tags=["agent"],
    )
    def list_datasets(
        current_user: AuthenticatedUserDep,
        include_shared: bool = True,
    ) -> DatasetListResponse:
        """Return the caller's datasets plus, optionally, those shared with them.

        Owned entries come first (newest first) at the ``owner`` tier, then —
        when ``include_shared`` — datasets the caller holds a share grant on, at
        their granted tier. The usage meter reflects the caller's unified storage
        total across all of their Skynet data against their budget (shared
        datasets cost the owner, not the viewer).

        Args:
            current_user: Authenticated caller whose library is listed.
            include_shared: Whether to fold in datasets shared with the caller.

        Returns:
            A :class:`DatasetListResponse` with one summary per entry and the
            compressed-bytes usage meter for the "X of Y used" display.
        """
        owned = store.list_datasets(current_user.username)
        summaries = [_summary(r, ShareRole.owner) for r in owned]
        if include_shared:
            owned_ids = {r.id for r in owned}
            with Session(job_store.engine) as session:
                shared_roles = list_grants_for_user_all(session, current_user.username)
            shared_ids = [did for did in shared_roles if did not in owned_ids]
            for record in store.list_datasets_by_ids(shared_ids):
                if record.owner_username == current_user.username:
                    continue
                summaries.append(_summary(record, ShareRole(shared_roles[record.id])))
        usage = UsageMeter(
            used_bytes=job_store.compute_user_storage(current_user.username).total,
            quota_bytes=job_store.get_effective_user_storage_quota(current_user.username),
        )
        return DatasetListResponse(datasets=summaries, usage=usage)

    @router.get(
        "/datasets/library/{dataset_id}",
        response_model=DatasetSummary,
        summary="Fetch one saved dataset's metadata",
    )
    def get_dataset(
        dataset_id: str,
        current_user: AuthenticatedUserDep,
    ) -> DatasetSummary:
        """Return one library dataset's metadata (viewer+).

        Args:
            dataset_id: Id of the requested dataset.
            current_user: Authenticated caller.

        Returns:
            The :class:`DatasetSummary` for the entry, carrying the caller's role.

        Raises:
            DomainError: 404 when the dataset is unknown or the caller has no
                access to it.
        """
        record, role = _require(dataset_id, current_user, ShareRole.viewer)
        return _summary(record, role)

    @router.get(
        "/datasets/library/{dataset_id}/rows",
        response_model=DatasetRowsResponse,
        summary="Fetch a saved dataset's rows for preview or wizard pre-fill",
    )
    def get_dataset_rows(
        dataset_id: str,
        current_user: AuthenticatedUserDep,
    ) -> DatasetRowsResponse:
        """Return a dataset's decompressed rows, column order, and saved schema (viewer+).

        Args:
            dataset_id: Id of the requested dataset.
            current_user: Authenticated caller.

        Returns:
            A :class:`DatasetRowsResponse` with the ordered columns, rows, and
            saved column schema for wizard pre-fill.

        Raises:
            DomainError: 404 when the dataset is unknown, the caller has no
                access, or its row bytes are missing.
        """
        record, _role = _require(dataset_id, current_user, ShareRole.viewer)
        rows = store.get_rows(dataset_id)
        if rows is None:
            raise DomainError("dataset.library.not_found", status=404)
        return DatasetRowsResponse(
            id=dataset_id,
            columns=_columns_of(record, rows),
            rows=rows,
            row_count=len(rows),
            column_schema=record.column_schema,
        )

    @router.put(
        "/datasets/library/{dataset_id}/rows",
        response_model=DatasetSummary,
        summary="Replace a saved dataset's rows (editor+)",
    )
    def edit_dataset_rows(
        dataset_id: str,
        payload: EditRowsRequest,
        current_user: AuthenticatedUserDep,
    ) -> DatasetSummary:
        """Replace a dataset's rows in place, re-hashing and recompressing (editor+).

        The new bytes overwrite the existing blob (no new entry, no dedupe), so
        an edit re-points the live link every optimization holds at the updated
        rows. Owner quota is not re-checked here — an edit replaces bytes the
        owner already holds rather than adding a file.

        Args:
            dataset_id: Id of the dataset to edit.
            payload: The replacement rows and optional new column schema.
            current_user: Authenticated editor (or owner/admin).

        Returns:
            The updated :class:`DatasetSummary`.

        Raises:
            DomainError: 404 when unknown or inaccessible; 403 when the caller is
                below the editor tier.
        """
        _record, role = _require(dataset_id, current_user, ShareRole.editor)
        updated = store.update_rows(dataset_id, rows=payload.rows, column_schema=payload.column_schema)
        if updated is None:
            raise DomainError("dataset.library.not_found", status=404)
        return _summary(updated, role)

    @router.post(
        "/datasets/library/{dataset_id}/clone",
        response_model=SaveDatasetResponse,
        summary="Clone a dataset into the caller's own library (viewer+)",
    )
    def clone_dataset(
        dataset_id: str,
        current_user: AuthenticatedUserDep,
    ) -> SaveDatasetResponse:
        """Copy a dataset the caller can view into a new entry they own (viewer+).

        The clone carries the source rows and saved schema into a fresh entry
        owned by the caller, gated on the caller's own per-file cap and quota. A
        caller who already holds a byte-identical entry gets it back instead of a
        second copy.

        Args:
            dataset_id: Id of the source dataset.
            current_user: Authenticated caller (viewer+ on the source).

        Returns:
            A :class:`SaveDatasetResponse` for the caller-owned clone.

        Raises:
            DomainError: 404 when unknown/inaccessible or its bytes are missing;
                413 over the caller's per-file cap; 409 over their quota.
        """
        record, _role = _require(dataset_id, current_user, ShareRole.viewer)
        rows = store.get_rows(dataset_id)
        if rows is None:
            raise DomainError("dataset.library.not_found", status=404)
        clone, deduped = _save_gated(
            owner=current_user.username,
            name=record.name,
            source=record.source,
            rows=rows,
            column_schema=record.column_schema,
        )
        return SaveDatasetResponse(dataset=_summary(clone, ShareRole.owner), deduplicated=deduped)

    @router.patch(
        "/datasets/library/{dataset_id}",
        response_model=DatasetSummary,
        summary="Rename a saved dataset (owner only)",
    )
    def rename_dataset(
        dataset_id: str,
        payload: RenameDatasetRequest,
        current_user: AuthenticatedUserDep,
    ) -> DatasetSummary:
        """Rename one library dataset (owner/admin only).

        Args:
            dataset_id: Id of the dataset to rename.
            payload: The new display name.
            current_user: Authenticated owner.

        Returns:
            The updated :class:`DatasetSummary`.

        Raises:
            DomainError: 404 when unknown/inaccessible; 403 when the caller is
                not the owner/admin.
        """
        _record, role = _require(dataset_id, current_user, ShareRole.owner)
        record = store.rename_dataset(dataset_id, payload.name)
        if record is None:
            raise DomainError("dataset.library.not_found", status=404)
        return _summary(record, role)

    @router.delete(
        "/datasets/library/{dataset_id}",
        response_model=DeleteDatasetResponse,
        summary="Delete a saved dataset and its bytes (owner only)",
    )
    def delete_dataset(
        dataset_id: str,
        current_user: AuthenticatedUserDep,
    ) -> DeleteDatasetResponse:
        """Delete one library dataset and its stored bytes (owner/admin only).

        The cascading foreign keys drop the dataset's share links and grants
        with it, so deleting an owned dataset also tears down its sharing.

        Args:
            dataset_id: Id of the dataset to delete.
            current_user: Authenticated owner.

        Returns:
            A :class:`DeleteDatasetResponse` with ``deleted`` set to ``True``.

        Raises:
            DomainError: 404 when unknown/inaccessible; 403 when the caller is
                not the owner/admin.
        """
        _record, _role = _require(dataset_id, current_user, ShareRole.owner)
        store.delete_dataset(dataset_id)
        return DeleteDatasetResponse(deleted=True)

    @router.post(
        "/datasets/library/bulk-delete",
        response_model=BulkDeleteByIdsResponse,
        summary="Delete many saved datasets in a single request (owner only)",
    )
    def bulk_delete_datasets(
        body: BulkDeleteByIdsRequest, current_user: AuthenticatedUserDep
    ) -> BulkDeleteByIdsResponse:
        """Delete a batch of owned library datasets and report per-id outcomes.

        Duplicate ids are deduplicated. An id the caller cannot reach as its
        owner — unknown, shared-but-not-owned, or below ``owner`` tier — lands in
        ``skipped`` rather than failing the whole request; a per-id store failure
        is likewise reported as skipped instead of raising.

        Args:
            body: The bulk-delete request body carrying the dataset ids.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A :class:`BulkDeleteByIdsResponse` listing deleted and skipped ids.
        """
        deleted: list[str] = []
        skipped: list[BulkDeleteByIdsSkipped] = []
        seen: set[str] = set()
        for dataset_id in body.ids:
            if dataset_id in seen:
                continue
            seen.add(dataset_id)
            try:
                _require(dataset_id, current_user, ShareRole.owner)
            except DomainError:
                skipped.append(BulkDeleteByIdsSkipped(id=dataset_id, reason="not_found"))
                continue
            try:
                store.delete_dataset(dataset_id)
            except Exception as exc:
                skipped.append(BulkDeleteByIdsSkipped(id=dataset_id, reason=f"error: {exc}"))
                continue
            deleted.append(dataset_id)
        return BulkDeleteByIdsResponse(deleted=deleted, skipped=skipped)

    @router.get(
        "/datasets/library/{dataset_id}/optimizations",
        response_model=DatasetOptimizationsResponse,
        summary="List optimizations submitted from a saved dataset (viewer+)",
    )
    def list_dataset_optimizations(
        dataset_id: str,
        current_user: AuthenticatedUserDep,
    ) -> DatasetOptimizationsResponse:
        """Return the runs the caller can see that were submitted from this dataset.

        The dataset→optimization half of the live link. Access is gated at
        ``viewer`` on the dataset, then scoped to runs the caller owns or holds a
        grant on — a viewer never learns of private runs they could not otherwise
        see. Matching is by the ``source_dataset_id`` recorded in each run's
        payload overview at submit time.

        Args:
            dataset_id: Id of the dataset whose runs are listed.
            current_user: Authenticated caller (viewer+ on the dataset).

        Returns:
            A :class:`DatasetOptimizationsResponse`, newest run first.

        Raises:
            DomainError: 404 when the dataset is unknown or inaccessible.
        """
        _require(dataset_id, current_user, ShareRole.viewer)
        visible = job_store.list_jobs_visible_to(current_user.username, limit=_REVERSE_LINK_SCAN_LIMIT)
        refs: list[DatasetOptimizationRef] = []
        for job in visible:
            overview = parse_overview(job)
            if overview.get(PAYLOAD_OVERVIEW_SOURCE_DATASET_ID) != dataset_id:
                continue
            refs.append(
                DatasetOptimizationRef(
                    optimization_id=job["optimization_id"],
                    name=overview.get(PAYLOAD_OVERVIEW_NAME),
                    status=job.get("status"),
                    optimization_type=overview.get(PAYLOAD_OVERVIEW_OPTIMIZATION_TYPE),
                    username=job.get("username"),
                    created_at=job.get("created_at"),
                )
            )
        return DatasetOptimizationsResponse(optimizations=refs)

    @router.post(
        "/datasets/library/from-optimization/{optimization_id}",
        response_model=SaveDatasetResponse,
        summary="Save a run's dataset to the caller's library (viewer+ on the run)",
    )
    def save_dataset_from_optimization(
        optimization_id: str,
        current_user: AuthenticatedUserDep,
        payload: SaveFromOptimizationRequest | None = None,
    ) -> SaveDatasetResponse:
        """Save the exact rows a run used as a new library entry the caller owns.

        The Data-tab producer doorway. Access is gated at ``viewer`` on the run
        (owner / admin / grantee); the saved dataset is owned by the caller and
        gated on their own per-file cap and quota. The run's column mapping and
        order are carried onto the saved schema so the dataset pre-fills the
        submit wizard when picked later. A caller who already holds a byte-
        identical entry gets it back instead of a second copy.

        Args:
            optimization_id: Id of the run whose dataset is saved.
            current_user: Authenticated caller (viewer+ on the run).
            payload: Optional name override; defaults to the run's dataset
                filename or optimization name.

        Returns:
            A :class:`SaveDatasetResponse` for the caller-owned entry.

        Raises:
            DomainError: 404 when the run is unknown/inaccessible or carries no
                dataset; 413 over the caller's per-file cap; 409 over their quota.
        """
        job_data = load_job_for_user(job_store, optimization_id, current_user)
        run_payload = job_data.get("payload")
        rows = run_payload.get("dataset") if isinstance(run_payload, dict) else None
        if not rows or not isinstance(rows, list):
            raise DomainError("optimization.dataset_unavailable", status=404)
        overview = parse_overview(job_data)
        name = (
            (payload.name if payload else None)
            or overview.get(PAYLOAD_OVERVIEW_DATASET_FILENAME)
            or overview.get(PAYLOAD_OVERVIEW_NAME)
            or optimization_id
        )
        record, deduped = _save_gated(
            owner=current_user.username,
            name=name,
            source="optimization",
            rows=rows,
            column_schema=_schema_from_run_payload(run_payload, rows),
        )
        return SaveDatasetResponse(dataset=_summary(record, ShareRole.owner), deduplicated=deduped)

    return router
