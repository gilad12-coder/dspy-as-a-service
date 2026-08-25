"""Routes for dataset profiling, split-plan recommendations, and demo samples. [INTERNAL]

``POST /datasets/profile`` — inspect the uploaded rows and return a
``DatasetProfile`` plus a recommended ``SplitPlan``. Purely advisory:
the caller either accepts the plan's fractions on the subsequent
``/run`` or ``/grid-search`` submission, or overrides them manually.

``GET /datasets/samples`` / ``POST /datasets/samples/{id}/stage`` —
curated demo datasets the agent can stage into the wizard in one step
so non-technical users can try the product end-to-end with zero setup.

``POST /datasets/column-roles`` — validate a proposed column-role map
and return a ``wizard_state`` patch so the agent can configure column
roles on the user's behalf.

``POST /datasets/request-upload`` — agent-callable signal that the chat
UI should render an inline upload card so the user can attach a dataset
file. Stateless; the upload itself is handled client-side.

All endpoints are hidden from the public Scalar reference (none are in
``_SCALAR_PUBLIC_PATHS``) — they exist to support the in-app wizard.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ...i18n import SAMPLE_DATASETS
from ...models import (
    ProfileDatasetRequest,
    ProfileDatasetResponse,
    ValidateDatasetRequest,
    ValidateDatasetResponse,
)
from ...service_gateway.datasets.planner import recommend_split
from ...service_gateway.datasets.profiler import profile_dataset
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError


class SampleDatasetSummary(BaseModel):
    """Metadata for one curated sample in the ``GET /datasets/samples`` list."""

    sample_id: str
    name: str
    description: str
    task_type: str
    row_count: int
    columns: list[str]


class SampleDatasetListResponse(BaseModel):
    """Envelope for ``GET /datasets/samples`` — the catalog of bundled samples."""

    samples: list[SampleDatasetSummary]


class SampleDatasetStageResponse(BaseModel):
    """Envelope for ``POST /datasets/samples/{id}/stage`` — rows + wizard patch."""

    sample_id: str
    name: str
    description: str
    dataset: list[dict[str, Any]]
    dataset_filename: str
    wizard_state: dict[str, Any]


class ColumnRolesRequest(BaseModel):
    """Request body for ``POST /datasets/column-roles``."""

    dataset_columns: list[str] = Field(min_length=1, max_length=100)
    column_roles: dict[str, str] = Field(
        description="Map of dataset column name → role: 'input', 'output', or 'ignore'.",
    )
    job_name: str | None = Field(default=None, max_length=200)


class ColumnRolesResponse(BaseModel):
    """Envelope for ``POST /datasets/column-roles`` — validated roles + wizard patch."""

    wizard_state: dict[str, Any]


class RequestUserDatasetRequest(BaseModel):
    """Request body for ``POST /datasets/request-upload``."""

    prompt: str = Field(
        default="",
        max_length=400,
        description="Short Hebrew sentence explaining why a dataset is needed.",
    )


class RequestUserDatasetResponse(BaseModel):
    """Envelope for ``POST /datasets/request-upload`` — UI-trigger marker."""

    awaiting_upload: bool
    prompt: str


class RequestLibraryDatasetResponse(BaseModel):
    """Envelope for ``POST /datasets/request-library-pick`` — UI-trigger marker."""

    awaiting_selection: bool
    prompt: str


class StageDatasetForAgentRequest(BaseModel):
    """Request body for ``POST /datasets/stage-for-agent``."""

    dataset: list[dict[str, Any]] = Field(min_length=1, max_length=200_000)
    dataset_filename: str = Field(min_length=1, max_length=255)


class StageDatasetForAgentResponse(BaseModel):
    """Envelope for ``POST /datasets/stage-for-agent`` — opaque staged id."""

    staged_dataset_id: str
    row_count: int


class StagedDatasetResponse(BaseModel):
    """Rehydration payload for ``GET /datasets/staged/{id}`` — rows + columns."""

    staged_dataset_id: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int


_VALID_COLUMN_ROLES = {"input", "output", "ignore"}

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


def create_datasets_router(*, job_store) -> APIRouter:
    """Build the datasets router.

    Args:
        job_store: Backend used by ``POST /datasets/stage-for-agent`` to persist
            wizard-parsed rows so the generalist agent can submit jobs without
            inlining the dataset into its tool arguments.

    Returns:
        A configured :class:`APIRouter` exposing the dataset profile, sample,
        and column-roles endpoints.
    """
    router = APIRouter()

    @router.post(
        "/datasets/profile",
        response_model=ProfileDatasetResponse,
        summary="Profile a dataset and recommend a split plan",
        tags=["agent"],
    )
    def profile(
        payload: ProfileDatasetRequest,
        current_user: AuthenticatedUserDep,
    ) -> ProfileDatasetResponse:
        """Return a ``DatasetProfile`` and a recommended ``SplitPlan``.

        The profiler walks the rows once to compute size, duplicate count,
        and primary-target stats. The planner then picks train/val/test
        fractions tuned to the size tier. Users can accept the plan or
        override ``split_fractions`` when submitting the actual optimization.

        Agent callers profiling a staged dataset pass ``staged_dataset_id``
        instead of inline rows (the rows are too large to ferry through tool
        arguments); the rows are rehydrated from staging here. Rehydration is
        read-only — unlike submit, profiling never evicts the staged dataset,
        so the user can profile it repeatedly before committing.

        Errors: 400 (empty dataset), 404 (unknown staged id), 422 (malformed body).

        Args:
            payload: Profiling request carrying either inline rows or a
                ``staged_dataset_id``, plus the column mapping and seed.
            current_user: Authenticated caller; staged rows are scoped to them.

        Returns:
            A :class:`ProfileDatasetResponse` containing the profile and the
            recommended split plan.

        Raises:
            DomainError: 404 when ``staged_dataset_id`` matches no staged
                dataset owned by this user.
        """
        rows = payload.dataset
        if payload.staged_dataset_id and not rows:
            rows = job_store.get_staged_dataset(payload.staged_dataset_id, current_user.username)
            if not rows:
                raise DomainError("dataset.staged.not_found", status=404)
        dataset_profile = profile_dataset(rows, payload.column_mapping)
        plan = recommend_split(dataset_profile, seed=payload.seed)
        return ProfileDatasetResponse(profile=dataset_profile, plan=plan)

    @router.post(
        "/datasets/validate",
        response_model=ValidateDatasetResponse,
        summary="Pre-submit validation that the chosen split is runnable",
        tags=["agent"],
    )
    def validate(payload: ValidateDatasetRequest) -> ValidateDatasetResponse:
        """Block submissions whose split leaves zero held-out examples.

        Mirrors ``POST /validate-code``: a small synchronous preflight the
        submit wizard runs before letting the user advance past the
        dataset step. A split with ``val=0`` and ``test=0`` produces no
        held-out data for evaluation, so the optimization can't measure
        improvement and must be rejected here rather than failing later.

        Args:
            payload: Request containing the dataset row count and the
                chosen train/val/test fractions.

        Returns:
            A :class:`ValidateDatasetResponse` with ``valid``, ``errors``,
            and ``warnings`` lists.
        """
        errors: list[str] = []
        warnings: list[str] = []
        total = payload.row_count
        if total <= 0:
            errors.append("Dataset is empty.")
        else:
            val_count = int(total * payload.fractions.val)
            test_count = int(total * payload.fractions.test)
            if val_count + test_count == 0:
                errors.append(
                    "Dataset is too small to run optimization: chosen split has 0 validation and 0 test rows."
                )
        return ValidateDatasetResponse(valid=not errors, errors=errors, warnings=warnings)

    @router.get(
        "/datasets/samples",
        response_model=SampleDatasetListResponse,
        summary="List curated sample datasets the agent can stage",
        tags=["agent"],
    )
    def list_sample_datasets() -> SampleDatasetListResponse:
        """Return the catalog of bundled demo datasets without the rows.

        The catalog is small and static — suited for first-run demos where
        the user doesn't have a dataset of their own yet. Use
        ``POST /datasets/samples/{sample_id}/stage`` to prefill the wizard.

        Returns:
            A :class:`SampleDatasetListResponse` containing one summary per sample.
        """
        summaries = [
            SampleDatasetSummary(
                sample_id=sample_id,
                name=sample["name"],
                description=sample["description"],
                task_type=sample["task_type"],
                row_count=len(sample["rows"]),
                columns=list(sample["input_columns"]) + list(sample["output_columns"]),
            )
            for sample_id, sample in SAMPLE_DATASETS.items()
        ]
        return SampleDatasetListResponse(samples=summaries)

    @router.post(
        "/datasets/samples/{sample_id}/stage",
        response_model=SampleDatasetStageResponse,
        summary="Stage a sample dataset into the submit wizard",
        tags=["agent"],
    )
    def stage_sample_dataset(sample_id: str) -> SampleDatasetStageResponse:
        """Return the sample rows plus a ``wizard_state`` patch ready to submit.

        Populates dataset_ready, columns_configured, dataset_columns,
        column_roles, signature_code, metric_code, and a default job_name
        so a non-technical user can hit "run" without writing code.

        Args:
            sample_id: Identifier of the bundled sample dataset.

        Returns:
            A :class:`SampleDatasetStageResponse` with rows, filename, and
            wizard-state patch.

        Raises:
            DomainError: 404 when ``sample_id`` is unknown.
        """
        sample = SAMPLE_DATASETS.get(sample_id)
        if sample is None:
            raise DomainError(
                "dataset.sample_unknown",
                status=404,
                sample_id=sample_id,
            )

        columns = list(sample["input_columns"]) + list(sample["output_columns"])
        column_roles: dict[str, str] = {}
        for col in sample["input_columns"]:
            column_roles[col] = "input"
        for col in sample["output_columns"]:
            column_roles[col] = "output"

        wizard_state: dict[str, Any] = {
            "dataset_ready": True,
            "columns_configured": True,
            "dataset_columns": columns,
            "column_roles": column_roles,
            "signature_code": sample["signature_code"],
            "metric_code": sample["metric_code"],
            "job_name": sample["name"],
        }

        return SampleDatasetStageResponse(
            sample_id=sample_id,
            name=sample["name"],
            description=sample["description"],
            dataset=sample["rows"],
            dataset_filename=sample["dataset_filename"],
            wizard_state=wizard_state,
        )

    @router.post(
        "/datasets/column-roles",
        response_model=ColumnRolesResponse,
        summary="Validate a column-role map and project it into the wizard",
        tags=["agent"],
    )
    def set_column_roles(req: ColumnRolesRequest) -> ColumnRolesResponse:
        """Validate a proposed column-role map and return a ``wizard_state`` patch.

        Every column in ``column_roles`` must be listed in
        ``dataset_columns`` and have role ``input``, ``output``, or
        ``ignore``. At least one input and one output are required.
        On success the response contains a wizard_state patch with
        ``columns_configured`` set to true, the role map, and an
        optional ``job_name``.

        Args:
            req: Request body with the dataset columns, role map, and an
                optional job name.

        Returns:
            A :class:`ColumnRolesResponse` with the wizard-state patch.

        Raises:
            DomainError: 422 when a column is unknown, a role is invalid, or
                no input/output role is supplied.
        """
        columns_set = set(req.dataset_columns)
        unknown = [col for col in req.column_roles if col not in columns_set]
        if unknown:
            raise DomainError(
                "dataset.column_roles_unknown",
                status=422,
                unknown=sorted(unknown),
            )
        bad_roles = {col: role for col, role in req.column_roles.items() if role not in _VALID_COLUMN_ROLES}
        if bad_roles:
            raise DomainError(
                "dataset.column_roles_invalid",
                status=422,
                bad=bad_roles,
            )
        inputs = [col for col, role in req.column_roles.items() if role == "input"]
        outputs = [col for col, role in req.column_roles.items() if role == "output"]
        if not inputs:
            raise DomainError("dataset.column_roles_need_input", status=422)
        if not outputs:
            raise DomainError("dataset.column_roles_need_output", status=422)

        wizard_state: dict[str, Any] = {
            "dataset_columns": req.dataset_columns,
            "column_roles": dict(req.column_roles),
            "columns_configured": True,
        }
        if req.job_name is not None and req.job_name.strip():
            wizard_state["job_name"] = req.job_name.strip()

        return ColumnRolesResponse(wizard_state=wizard_state)

    @router.post(
        "/datasets/request-upload",
        response_model=RequestUserDatasetResponse,
        summary="Ask the user to upload a dataset; the chat panel renders an upload card",
        tags=["agent"],
    )
    def request_user_dataset(req: RequestUserDatasetRequest) -> RequestUserDatasetResponse:
        """Signal the chat UI to render an inline dataset-upload card.

        Stateless: the endpoint exists only so the agent can call a named
        tool that the frontend recognizes via its ``tool_start`` SSE event.
        The actual file parsing, role mapping, and wizard hydration happen
        client-side in the rendered card. The agent should call this once
        and then wait for the user to attach the file in the chat.

        Args:
            req: Optional prompt describing why the dataset is needed.

        Returns:
            A :class:`RequestUserDatasetResponse` carrying the prompt back
            so it can render inside the upload card.
        """
        return RequestUserDatasetResponse(
            awaiting_upload=True,
            prompt=req.prompt.strip(),
        )

    @router.post(
        "/datasets/request-library-pick",
        response_model=RequestLibraryDatasetResponse,
        summary="Ask the user to pick a saved dataset; the chat panel renders a library picker",
        operation_id="request_user_dataset_from_library",
        tags=["agent"],
    )
    def request_user_dataset_from_library(
        req: RequestUserDatasetRequest,
    ) -> RequestLibraryDatasetResponse:
        """Signal the chat UI to render an inline saved-dataset picker.

        Stateless twin of :func:`request_user_dataset` for the by-reference
        path: the agent calls this named tool so the frontend renders a picker
        over the caller's dataset library. The chosen dataset's rows, column
        roles, and ``source_dataset_id`` are hydrated into the wizard
        client-side; the run then submits against the library dataset by
        reference (durable) rather than a staged upload (evicted after the run).

        Args:
            req: Optional prompt describing why the dataset is needed.

        Returns:
            A :class:`RequestLibraryDatasetResponse` echoing the prompt so it
            can render inside the picker card.
        """
        return RequestLibraryDatasetResponse(
            awaiting_selection=True,
            prompt=req.prompt.strip(),
        )

    @router.post(
        "/datasets/stage-for-agent",
        response_model=StageDatasetForAgentResponse,
        summary="Persist parsed wizard rows so the agent can submit by id",
        operation_id="stage_dataset_for_agent",
    )
    def stage_dataset_for_agent(
        req: StageDatasetForAgentRequest,
        current_user: AuthenticatedUserDep,
    ) -> StageDatasetForAgentResponse:
        """Persist a parsed dataset under the caller's identity and return an opaque id.

        The wizard calls this immediately after a successful upload+parse so a
        later ``POST /run`` from the generalist agent can pass
        ``staged_dataset_id`` instead of inlining the rows. Each staged row is
        scoped to the authenticated submitter and is evicted on first use.

        Args:
            req: The parsed dataset rows and original filename.
            current_user: Authenticated submitter resolved from the bearer token.

        Returns:
            A :class:`StageDatasetForAgentResponse` carrying the staged id and
            the number of rows persisted.
        """
        staged_id = job_store.stage_dataset(
            username=current_user.username,
            dataset_filename=req.dataset_filename,
            rows=req.dataset,
        )
        return StageDatasetForAgentResponse(
            staged_dataset_id=staged_id,
            row_count=len(req.dataset),
        )

    @router.get(
        "/datasets/staged/{staged_dataset_id}",
        response_model=StagedDatasetResponse,
        summary="Fetch staged dataset rows by id so the wizard can rehydrate",
        operation_id="get_staged_dataset",
    )
    def get_staged_dataset(
        staged_dataset_id: str,
        current_user: AuthenticatedUserDep,
    ) -> StagedDatasetResponse:
        """Return the rows a chat-side upload staged so the wizard can mirror it.

        The shared wizard state carries only the opaque ``staged_dataset_id``
        (the rows are too large to thread through it). When the wizard sees an
        id it isn't already showing — e.g. the user attached a file in the
        agent panel — it calls this to materialise the same dataset preview.
        Column order is derived from the first row so it matches what the user
        confirmed.

        Args:
            staged_dataset_id: Id previously returned by
                ``POST /datasets/stage-for-agent``.
            current_user: Authenticated caller; staged rows are scoped to them.

        Returns:
            A :class:`StagedDatasetResponse` with the rows, derived columns,
            and row count.

        Raises:
            DomainError: 404 when no staged dataset matches the id for this user.
        """
        rows = job_store.get_staged_dataset(staged_dataset_id, current_user.username)
        if rows is None:
            raise DomainError("dataset.staged.not_found", status=404)
        columns = list(rows[0].keys()) if rows else []
        return StagedDatasetResponse(
            staged_dataset_id=staged_dataset_id,
            columns=columns,
            rows=rows,
            row_count=len(rows),
        )

    return router
