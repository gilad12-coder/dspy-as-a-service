"""Route for pre-submission workflow dry runs. [internal]

``POST /workflows/dry-run`` executes an *unoptimized* workflow graph once on
a caller-supplied sample input so the canvas can verify wiring, signatures,
and transforms before the user spends optimization credits. Node failures
are an expected, renderable outcome — they return 200 with the failing node
id and the traces collected up to the failure, never a 5xx.

``POST /workflows/dry-run/stream`` is the same execution surfaced as
Server-Sent Events: ``token`` events per output-field chunk, then one
``final`` event shaped like ``WorkflowDryRunResponse`` so the dialog can
show the answer as it generates.

Trust boundary: like the code_validation router and the code agent, this
endpoint execs user-authored signature/transform code in the API process
(deep validation still runs in safe_exec subprocesses first). Authenticated
callers already hold arbitrary-code-execution rights against the gateway;
do not expose this endpoint beyond that audience without sandboxing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any, AsyncIterator

import dspy
from dspy.streaming import StreamListener, StreamResponse
from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

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
from ._helpers import sanitize_node_traces, sanitize_port_values, sse_from_events

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


def _prepare_dry_run(req: WorkflowDryRunRequest) -> tuple[Any, dict[str, Any], Any, str]:
    """Validate the request and build everything a dry-run execution needs.

    Shared by the blocking and streaming routes so both enforce identical
    graph validation, input filtering, and model resolution.

    Args:
        req: The workflow spec, sample inputs, model config, and optional
            tool source.

    Returns:
        ``(program, filtered_inputs, lm, model_used)``.

    Raises:
        DomainError: 400 when the graph fails validation, required inputs
            are missing, or the program cannot be built.
    """
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
    return program, filtered_inputs, lm, req.model_settings.normalized_identifier()


async def _stream_dry_run_events(
    *,
    program: Any,
    lm: Any,
    filtered_inputs: dict[str, Any],
    output_fields: list[str],
    model_used: str,
) -> AsyncIterator[dict[str, Any]]:
    """Yield ``{event, data}`` dicts for a streamed workflow dry run.

    Mirrors ``serve._stream_program_inference`` — ``dspy.streamify`` first,
    blocking fallback when listeners can't wire onto the composed graph —
    but keeps the dry-run contract: node traces are captured throughout and
    a node failure emits a ``final`` event with ``error``/``failed_node_id``
    (a renderable outcome, not a stream error).

    Args:
        program: Composed workflow program to invoke.
        lm: DSPy language model context to bind during inference.
        filtered_inputs: Caller inputs filtered to the input-anchor fields.
        output_fields: Output-anchor field names to listen on and collect.
        model_used: Model identifier surfaced in the ``final`` event.

    Yields:
        ``{"event": "token"|"final"|"error", "data": ...}`` dicts, where the
        ``final`` payload matches ``WorkflowDryRunResponse``.

    Raises:
        asyncio.CancelledError: Re-raised so the SSE wrapper tears the
            generator down cleanly on client disconnect.
    """
    listeners = [StreamListener(signature_field_name=field) for field in output_fields]
    final_outputs: dict[str, Any] = {}
    with capture_node_traces() as raw_traces:
        try:
            try:
                stream_program = dspy.streamify(
                    program,
                    stream_listeners=listeners,
                    async_streaming=True,
                )
                with dspy.context(lm=lm):
                    output_stream = stream_program(**filtered_inputs)
                    async for item in output_stream:
                        if isinstance(item, StreamResponse):
                            yield {
                                "event": "token",
                                "data": {"field": item.signature_field_name, "chunk": item.chunk},
                            }
                        elif isinstance(item, dspy.Prediction):
                            for field in output_fields:
                                final_outputs[field] = getattr(item, field, None)
            except (WorkflowNodeExecutionError, asyncio.CancelledError):
                raise
            except Exception:
                # streamify can't always wire listeners onto a composed
                # graph — rerun blocking so the caller still gets a final
                # answer, just without token events.
                with dspy.context(lm=lm):
                    prediction = await asyncio.to_thread(lambda: program(**filtered_inputs))
                for field in output_fields:
                    final_outputs[field] = getattr(prediction, field, None)
        except WorkflowNodeExecutionError as exc:
            logger.info("Workflow dry-run stream failed at node %s: %s", exc.node_id, exc)
            yield {
                "event": "final",
                "data": {
                    "outputs": None,
                    "node_traces": [t.model_dump() for t in sanitize_node_traces(raw_traces)],
                    "model_used": model_used,
                    "error": str(exc),
                    "failed_node_id": exc.node_id,
                },
            }
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Workflow dry-run stream failed")
            yield {"event": "error", "data": {"error": "streaming failed"}}
            return
    yield {
        "event": "final",
        "data": {
            "outputs": sanitize_port_values(final_outputs),
            "node_traces": [t.model_dump() for t in sanitize_node_traces(raw_traces)],
            "model_used": model_used,
            "error": None,
            "failed_node_id": None,
        },
    }


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
        program, filtered_inputs, lm, model_used = _prepare_dry_run(req)

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

    @router.post(
        "/workflows/dry-run/stream",
        summary="Execute an unoptimized workflow once and stream the answer as SSE",
        tags=["Workflows"],
    )
    async def workflow_dry_run_stream(
        req: WorkflowDryRunRequest, current_user: AuthenticatedUserDep
    ) -> StreamingResponse:
        """Run one workflow test execution, streaming the answer as it forms.

        Emits one ``token`` event per output-field chunk, then a terminal
        ``final`` event shaped like ``WorkflowDryRunResponse`` (including
        node traces and the failed-node contract). Falls back to blocking
        inference with no token events when ``dspy.streamify`` can't attach
        listeners to the composed graph.

        Args:
            req: The workflow spec, sample inputs, model config, and
                optional tool source.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``StreamingResponse`` with a ``text/event-stream`` body.

        Raises:
            DomainError: 400 mirroring the non-streaming route's validation.
        """
        _ = current_user
        # Graph validation + program build run safe_exec subprocesses;
        # offload so the event loop isn't blocked before the first byte.
        program, filtered_inputs, lm, model_used = await asyncio.to_thread(_prepare_dry_run, req)
        source = _stream_dry_run_events(
            program=program,
            lm=lm,
            filtered_inputs=filtered_inputs,
            output_fields=req.workflow.output_field_names(),
            model_used=model_used,
        )
        return StreamingResponse(
            sse_from_events(source),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
