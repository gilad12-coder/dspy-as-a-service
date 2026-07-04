"""Route for pre-submission workflow dry runs. [internal]

``POST /workflows/dry-run`` executes an *unoptimized* workflow graph once on
a caller-supplied sample input so the canvas can verify wiring, signatures,
and transforms before the user spends optimization credits. Node failures
are an expected, renderable outcome — they return 200 with the failing node
id and the traces collected up to the failure, never a 5xx.

Trust boundary: like the code_validation router and the code agent, this
endpoint execs user-authored signature/transform code in the API process
(deep validation still runs in safe_exec subprocesses first). Authenticated
callers already hold arbitrary-code-execution rights against the gateway;
do not expose this endpoint beyond that audience without sandboxing.
"""

from __future__ import annotations

import logging
from typing import Annotated

import dspy
from fastapi import APIRouter, Depends

from ...exceptions import ServiceError
from ...models.submissions import WorkflowDryRunRequest, WorkflowDryRunResponse
from ...service_gateway.language_models import build_language_model
from ...service_gateway.optimization.workflow import (
    WorkflowNodeExecutionError,
    build_workflow_program,
    capture_node_traces,
    validate_workflow,
)
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ._helpers import sanitize_node_traces, sanitize_port_values

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


def create_workflows_router() -> APIRouter:
    """Build the workflows router.

    Returns:
        A FastAPI ``APIRouter`` exposing the ``/workflows/dry-run`` endpoint.
    """
    router = APIRouter()

    @router.post(
        "/workflows/dry-run",
        response_model=WorkflowDryRunResponse,
        summary="Execute an unoptimized workflow once on a sample input",
        tags=["Workflows"],
    )
    def workflow_dry_run(req: WorkflowDryRunRequest, current_user: AuthenticatedUserDep) -> WorkflowDryRunResponse:
        """Run one test execution of a workflow graph before submission.

        Deep-validates the graph, builds the composed program, and executes
        it on the supplied inputs under the requested model. Billing follows
        the same semantics as serve calls (the model config carries the
        token source).

        Args:
            req: The workflow spec, sample inputs, model config, and
                optional tool source.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``WorkflowDryRunResponse`` with final outputs and per-node
            traces; on node failure, ``error`` and ``failed_node_id`` are
            set and ``outputs`` is ``None``.

        Raises:
            DomainError: 400 when the graph fails validation, required
                inputs are missing, or the program cannot be built.
        """
        _ = current_user
        try:
            validate_workflow(req.workflow)
        except ServiceError as exc:
            raise DomainError("workflow.validation_failed", status=400, error=str(exc)) from exc

        input_fields = req.workflow.input_field_names()
        missing = [name for name in input_fields if name not in req.inputs]
        if missing:
            raise DomainError(
                "serve.missing_inputs",
                status=400,
                missing=missing,
                input_fields=input_fields,
            )
        filtered_inputs = {name: req.inputs[name] for name in input_fields}

        try:
            program, _schema_hashes = build_workflow_program(
                req.workflow, tool_source=req.tool_source, dataset=None
            )
        except ServiceError as exc:
            raise DomainError("workflow.validation_failed", status=400, error=str(exc)) from exc

        lm = build_language_model(req.model_settings)
        model_used = req.model_settings.normalized_identifier()

        with capture_node_traces() as raw_traces:
            try:
                with dspy.context(lm=lm):
                    prediction = program(**filtered_inputs)
            except WorkflowNodeExecutionError as exc:
                logger.info("Workflow dry run failed at node %s: %s", exc.node_id, exc)
                return WorkflowDryRunResponse(
                    outputs=None,
                    node_traces=sanitize_node_traces(raw_traces),
                    model_used=model_used,
                    error=str(exc),
                    failed_node_id=exc.node_id,
                )

        outputs = {name: getattr(prediction, name, None) for name in req.workflow.output_field_names()}
        return WorkflowDryRunResponse(
            outputs=sanitize_port_values(outputs),
            node_traces=sanitize_node_traces(raw_traces),
            model_used=model_used,
        )

    return router
