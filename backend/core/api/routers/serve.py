"""Routes for inference on optimized programs (single runs and grid-search pairs). [MIXED]

Public dev surface (in ``_SCALAR_PUBLIC_PATHS``):
- ``POST /serve/{id}`` — invoke the trained program.
- ``GET /serve/{id}/info`` — input/output schema for the program.

Internal (frontend-only, hidden from public docs):
- ``POST /serve/{id}/stream`` and per-pair variants under
  ``/serve/{id}/pair/{idx}/...`` — too granular for dev integration.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any

import dspy
from dspy.streaming import StreamListener, StreamResponse
from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from ...billing.metering import meter_llm_run
from ...config import settings
from ...constants import (
    PAYLOAD_OVERVIEW_MODEL_NAME,
    PAYLOAD_OVERVIEW_MODEL_SETTINGS,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZER_NAME,
)
from ...models import ModelConfig, ServeInfoResponse, ServeRequest, ServeResponse
from ...service_gateway.agents.generalist import TrustMode, get_approval_registry
from ...service_gateway.agents.react_serve import run_react_chat
from ...service_gateway.language_models import build_language_model
from ...service_gateway.optimization.workflow import capture_node_traces
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..response_limits import AGENT_MAX_INSTRUCTIONS, AGENT_MAX_TEXT, truncate_text
from ..sharing_access import ShareRole
from ._helpers import (
    enforce_llm_credits,
    load_job_for_user,
    load_pair_program,
    load_program,
    load_react_chat_inputs,
    require_role_at_least,
    sanitize_node_traces,
    sse_from_events,
    stream_with_llm_metering,
    workflow_spec_from_overview,
)

logger = logging.getLogger(__name__)

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


class RequestUserInferenceRequest(BaseModel):
    """Request body for ``POST /serve/{id}/request-form``."""

    prompt: str = Field(
        default="",
        max_length=400,
        description="Short Hebrew sentence explaining why inference is being offered.",
    )


class RequestUserInferenceResponse(BaseModel):
    """Envelope for ``POST /serve/{id}/request-form`` — UI-trigger marker."""

    optimization_id: str
    awaiting_inputs: bool
    prompt: str


class ServeChatTurn(BaseModel):
    """One prior chat turn carried by the react-serve chat request."""

    role: str = Field(default="user")
    content: str = Field(default="", max_length=AGENT_MAX_TEXT)


class ServeChatRequest(BaseModel):
    """Request body for ``POST /serve/{id}/chat`` (live ReAct chat turn)."""

    user_message: str = Field(..., max_length=AGENT_MAX_TEXT)
    chat_history: list[ServeChatTurn] = Field(default_factory=list)
    trust_mode: TrustMode = Field(
        default="ask",
        description="'ask'/'auto_safe' confirm every tool, 'yolo' confirms none.",
    )
    model_config_override: ModelConfig | None = Field(
        default=None,
        description="Optional model override. Uses the run's model if omitted.",
    )


class ServeChatConfirmRequest(BaseModel):
    """Confirm payload for resolving a pending react-serve tool approval."""

    call_id: str
    approved: bool


class ServeChatConfirmResponse(BaseModel):
    """Ack for a react-serve approval confirm call."""

    resolved: bool


def _cap_serve_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    """Truncate string fields in a serve-response ``outputs`` dict.

    LLM predictions can run to many KB; echoing the raw output into the
    agent's context blows the window after one or two calls. Non-string
    values pass through unchanged.

    Args:
        outputs: Mapping of output-field name to predicted value.

    Returns:
        A new dict with long string values truncated.
    """
    return {
        key: truncate_text(value, AGENT_MAX_TEXT) if isinstance(value, str) else value for key, value in outputs.items()
    }


def _artifact_prompt_fields(artifact: Any) -> tuple[list[str], list[str], str | None, int]:
    """Read prompt metadata from a ``ProgramArtifact`` with ``None``-safe fallbacks.

    ``artifact.optimized_prompt`` may be unset for legacy results, so this
    helper centralises the empty-default behaviour the routes rely on.

    Args:
        artifact: A ``ProgramArtifact`` (or grid-pair equivalent).

    Returns:
        ``(input_fields, output_fields, instructions, demo_count)``.
    """
    prompt = artifact.optimized_prompt
    if prompt is None:
        return [], [], None, 0
    return (
        list(prompt.input_fields),
        list(prompt.output_fields),
        prompt.instructions,
        len(prompt.demos),
    )


# Long or multi-line example values make the usage snippet unwieldy and can
# break single-line shells, so the sample helpers drop them in favour of a
# ``<field>`` placeholder the frontend renders.
_MAX_SAMPLE_LEN = 200


def _coerce_sample_value(value: Any) -> str | None:
    """Return a snippet-safe string for a sample value, or ``None`` to skip it.

    Args:
        value: A raw input value from a demo or a dataset row.

    Returns:
        The trimmed string for short single-line strings (and stringified
        numbers/bools), or ``None`` when the value is unsuitable for inlining.
    """
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed and len(trimmed) <= _MAX_SAMPLE_LEN and "\n" not in trimmed:
            return trimmed
        return None
    if isinstance(value, (int, float)):  # bool is an int subclass; str() is fine
        return str(value)
    return None


def _collect_sample(
    source: dict[str, Any], input_fields: list[str], mapping: dict[str, Any] | None = None
) -> dict[str, str]:
    """Pick snippet-safe example values for ``input_fields`` from one row/demo.

    Reads each field directly, falling back to the column-mapping (in either
    direction) when the signature field name differs from the dataset column.

    Args:
        source: A single demo's ``inputs`` dict or a dataset row.
        input_fields: Declared signature input field names.
        mapping: Optional ``column_mapping['inputs']`` for field↔column lookup.

    Returns:
        A ``{field: value}`` map containing only the fields with usable values.
    """
    result: dict[str, str] = {}
    for field in input_fields:
        raw = source.get(field)
        if raw is None and mapping:
            for key, val in mapping.items():
                if val == field and key in source:
                    raw = source[key]
                    break
                if key == field and isinstance(val, str) and val in source:
                    raw = source[val]
                    break
        coerced = _coerce_sample_value(raw)
        if coerced is not None:
            result[field] = coerced
    return result


def _sample_inputs(
    job_store: Any,
    optimization_id: str,
    user: AuthenticatedUser,
    artifact: Any,
    input_fields: list[str],
) -> dict[str, str]:
    """Build example input values to prefill the integration snippet.

    Prefers a real example baked into the program (a demo); falls back to the
    first dataset row so optimizers that carry no demos — GEPA evolves
    instructions and ships zero — still yield copy-paste-ready values.

    Args:
        job_store: Store used to read the dataset for the fallback path.
        optimization_id: Optimization whose dataset is the fallback source.
        user: Authenticated caller; ownership is re-checked on the dataset read.
        artifact: Program artifact whose demos are the preferred source.
        input_fields: Declared signature input field names.

    Returns:
        A ``{field: value}`` map (possibly partial or empty).
    """
    prompt = getattr(artifact, "optimized_prompt", None)
    demos = list(getattr(prompt, "demos", []) or []) if prompt is not None else []
    if demos:
        demo_inputs = getattr(demos[0], "inputs", None) or {}
        sample = _collect_sample(demo_inputs, input_fields)
        if sample:
            return sample
    job_data = load_job_for_user(job_store, optimization_id, user)
    payload = job_data.get("payload") or {}
    dataset = payload.get("dataset") or []
    if not dataset or not isinstance(dataset[0], dict):
        return {}
    mapping = (payload.get("column_mapping") or {}).get("inputs") or {}
    return _collect_sample(dataset[0], input_fields, mapping)


async def _stream_program_inference(
    *,
    program: Any,
    lm: Any,
    filtered_inputs: dict[str, Any],
    input_fields: list[str],
    output_fields: list[str],
    listeners: list[StreamListener],
    model_used: str,
    error_log_context: str,
) -> Any:
    """Yield ``{event, data}`` dicts for streaming a DSPy program's inference.

    Tries ``dspy.streamify`` first and falls back to a synchronous call when
    the program / output fields aren't streamable. Emits ``token`` events
    per chunk and a terminal ``final`` event, or a single ``error`` event on
    failure. Re-raises ``asyncio.CancelledError`` so the caller can tear
    the generator down cleanly.

    Args:
        program: Compiled DSPy program to invoke.
        lm: DSPy language model context to bind during inference.
        filtered_inputs: Caller-provided inputs filtered to declared fields.
        input_fields: Declared input field names from the optimized prompt.
        output_fields: Declared output field names from the optimized prompt.
        listeners: Stream listeners (one per output field) for token fan-out.
        model_used: Identifier for the LM, surfaced in the ``final`` event.
        error_log_context: Tag included in the failure log for debuggability.

    Yields:
        ``{"event": "token"|"final"|"error", "data": ...}`` dicts.

    Raises:
        asyncio.CancelledError: Re-raised so the caller can tear the
            generator down cleanly.
    """
    try:
        final_outputs: dict[str, Any] = {}
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
                        if output_fields:
                            for field in output_fields:
                                final_outputs[field] = getattr(item, field, None)
                        else:
                            final_outputs |= {
                                key: val for key, val in item.toDict().items() if key not in filtered_inputs
                            }
            yield {
                "event": "final",
                "data": {
                    "outputs": final_outputs,
                    "input_fields": input_fields,
                    "output_fields": output_fields,
                    "model_used": model_used,
                },
            }
            return
        except Exception as stream_exc:
            with dspy.context(lm=lm):
                prediction = await asyncio.to_thread(lambda: program(**filtered_inputs))
            if output_fields:
                for field in output_fields:
                    final_outputs[field] = getattr(prediction, field, None)
            else:
                final_outputs |= {key: val for key, val in prediction.toDict().items() if key not in filtered_inputs}
            yield {
                "event": "final",
                "data": {
                    "outputs": final_outputs,
                    "input_fields": input_fields,
                    "output_fields": output_fields,
                    "model_used": model_used,
                    "streaming_fallback": True,
                    "fallback_reason": str(stream_exc),
                },
            }
            return
    except asyncio.CancelledError:
        raise
    except Exception:
        yield {"event": "error", "data": {"error": "streaming failed"}}
        logger.exception("Serve stream failed for %s", error_log_context)
        return


def create_serve_router(*, job_store) -> APIRouter:
    """Build the serve router.

    Args:
        job_store: Job-store instance backing the load/inference helpers.

    Returns:
        A FastAPI ``APIRouter`` exposing single-run and grid-pair serve routes.
    """
    router = APIRouter()

    @router.get(
        "/serve/{optimization_id}/info",
        response_model=ServeInfoResponse,
        summary="Describe an optimized program without running it",
        tags=["agent"],
    )
    def serve_info(optimization_id: str, current_user: AuthenticatedUserDep) -> ServeInfoResponse:
        """Describe a compiled optimized program without running it.

        Metadata-only — no LLM calls. 404 if unknown or inaccessible to the
        caller, 409 if not finished.

        Args:
            optimization_id: Optimization id whose artifact should be described.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``ServeInfoResponse`` listing the program's I/O fields,
            instructions, and demo count.

        Raises:
            DomainError: 404 if unknown or inaccessible, 409 if the
                optimization is not in a serveable state.
        """
        _, result, overview = load_program(job_store, optimization_id, current_user)
        artifact = result.program_artifact
        input_fields, output_fields, instructions, demo_count = _artifact_prompt_fields(artifact)
        workflow_spec = workflow_spec_from_overview(overview)
        if workflow_spec is not None:
            # The artifact prompt reflects a single predictor; a workflow's
            # servable surface is its anchor fields.
            input_fields = workflow_spec.input_field_names()
            output_fields = workflow_spec.output_field_names()

        return ServeInfoResponse(
            optimization_id=optimization_id,
            module_name=overview.get(PAYLOAD_OVERVIEW_MODULE_NAME, ""),
            optimizer_name=overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_NAME, ""),
            model_name=overview.get(PAYLOAD_OVERVIEW_MODEL_NAME, ""),
            input_fields=input_fields,
            output_fields=output_fields,
            instructions=truncate_text(instructions, AGENT_MAX_INSTRUCTIONS),
            demo_count=demo_count,
            sample_inputs=_sample_inputs(
                job_store, optimization_id, current_user, artifact, input_fields
            ),
        )

    @router.post(
        "/serve/{optimization_id}/request-form",
        response_model=RequestUserInferenceResponse,
        operation_id="request_user_inference",
        summary="Ask the user to fill an inference form; the chat panel renders an input card",
        tags=["agent"],
    )
    def request_user_inference(
        optimization_id: str,
        req: RequestUserInferenceRequest,
        current_user: AuthenticatedUserDep,
    ) -> RequestUserInferenceResponse:
        """Signal the chat UI to render an inline inference-input card.

        Stateless: the endpoint exists only so the agent can call a named
        tool that the frontend recognizes via its ``tool_start`` SSE event.
        Access is gated by the same ``load_program`` permission check used
        by ``serve_info`` / ``serve_program`` so the agent can't render a
        form for an optimization the caller doesn't own. The card itself
        hits ``/serve/{id}/info`` for the field schema and ``/serve/{id}``
        for the actual inference call — the agent never needs to invoke
        ``serve_program`` directly.

        Args:
            optimization_id: Optimization id whose form should be rendered.
            req: Optional prompt describing why inference is being offered.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A :class:`RequestUserInferenceResponse` carrying the prompt back
            so the upload card can display it.

        Raises:
            DomainError: 404 if unknown or inaccessible, 409 if the
                optimization is not in a serveable state.
        """
        load_program(job_store, optimization_id, current_user)
        return RequestUserInferenceResponse(
            optimization_id=optimization_id,
            awaiting_inputs=True,
            prompt=req.prompt.strip(),
        )

    @router.post(
        "/serve/{optimization_id}",
        response_model=ServeResponse,
        summary="Run a single inference through an optimized program",
        tags=["agent"],
    )
    def serve_program(
        optimization_id: str, req: ServeRequest, current_user: AuthenticatedUserDep
    ) -> ServeResponse:
        """Run a blocking inference call through the compiled program.

        Model resolution: ``model_config_override`` → stored job settings →
        stored model name. All ``input_fields`` must be supplied; extras are
        ignored.

        Args:
            optimization_id: Optimization id whose program should run.
            req: Inference request carrying inputs and optional model override.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``ServeResponse`` with the predicted outputs and resolved model.

        Raises:
            DomainError: 400 (bad inputs / no model), 402 (caller has no
                spendable credits), 404 (unknown or inaccessible), 409 (not
                in a serveable state).
        """
        program, result, overview = load_program(job_store, optimization_id, current_user)
        artifact = result.program_artifact

        if req.model_config_override:
            model_config = req.model_config_override
        else:
            model_settings = overview.get(PAYLOAD_OVERVIEW_MODEL_SETTINGS, {})
            model_name = overview.get(PAYLOAD_OVERVIEW_MODEL_NAME, "")
            if model_settings:
                model_config = ModelConfig.model_validate(model_settings)
            elif model_name:
                model_config = ModelConfig(name=model_name)
            else:
                raise DomainError("serve.no_model_config", status=400)

        input_fields, output_fields, _instructions, _demo_count = _artifact_prompt_fields(artifact)
        workflow_spec = workflow_spec_from_overview(overview)
        if workflow_spec is not None:
            input_fields = workflow_spec.input_field_names()
            output_fields = workflow_spec.output_field_names()

        if not input_fields:
            raise DomainError("serve.no_declared_inputs", status=400)
        missing = [f for f in input_fields if f not in req.inputs]
        if missing:
            raise DomainError(
                "serve.missing_inputs",
                status=400,
                missing=missing,
                input_fields=input_fields,
            )
        filtered_inputs = {f: req.inputs[f] for f in input_fields}

        enforce_llm_credits(job_store, current_user.username)
        lm = build_language_model(model_config)

        node_traces = None
        try:
            with dspy.context(lm=lm):
                if workflow_spec is not None:
                    with capture_node_traces() as raw_traces:
                        prediction = program(**filtered_inputs)
                    node_traces = sanitize_node_traces(raw_traces)
                else:
                    prediction = program(**filtered_inputs)
        finally:
            # A failed inference still consumed provider tokens (dspy retries
            # under the hood), so the debit rides ``finally``.
            meter_llm_run(
                getattr(job_store, "engine", None),
                current_user.username,
                [lm],
                description="Serve inference",
            )

        outputs: dict[str, Any] = {}
        if output_fields:
            for field in output_fields:
                outputs[field] = getattr(prediction, field, None)
        else:
            outputs = {key: val for key, val in prediction.toDict().items() if key not in req.inputs}

        return ServeResponse(
            optimization_id=optimization_id,
            outputs=_cap_serve_outputs(outputs),
            input_fields=input_fields,
            output_fields=output_fields,
            model_used=model_config.normalized_identifier(),
            node_traces=node_traces,
        )

    @router.post(
        "/serve/{optimization_id}/stream",
        summary="Run inference and stream partial outputs as SSE",
    )
    async def serve_program_stream(
        optimization_id: str, req: ServeRequest, current_user: AuthenticatedUserDep
    ):
        """Run inference and stream partial outputs as Server-Sent Events.

        Emits one ``token`` event per chunk, keyed by output field, then a
        terminal ``final`` event. Falls back to blocking inference with
        ``streaming_fallback=true`` if ``dspy.streamify`` can't set up
        listeners. Same input validation and model resolution as the
        non-streaming endpoint.

        Args:
            optimization_id: Optimization id whose program should run.
            req: Inference request carrying inputs and optional model override.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A streaming ``StreamingResponse`` with ``text/event-stream`` body.

        Raises:
            DomainError: 400, 402, 404 (including inaccessible to caller),
                or 409 mirroring the non-streaming route.
        """
        # Offload the synchronous DB read + program deserialization so it does
        # not block the single event loop (and every other in-flight request /
        # SSE stream) before the first byte is produced.
        program, result, overview = await asyncio.to_thread(
            load_program, job_store, optimization_id, current_user
        )
        artifact = result.program_artifact

        if req.model_config_override:
            model_config = req.model_config_override
        else:
            model_settings = overview.get(PAYLOAD_OVERVIEW_MODEL_SETTINGS, {})
            model_name = overview.get(PAYLOAD_OVERVIEW_MODEL_NAME, "")
            if model_settings:
                model_config = ModelConfig.model_validate(model_settings)
            elif model_name:
                model_config = ModelConfig(name=model_name)
            else:
                raise DomainError("serve.no_model_config", status=400)

        input_fields, output_fields, _instructions, _demo_count = _artifact_prompt_fields(artifact)
        workflow_spec = workflow_spec_from_overview(overview)
        if workflow_spec is not None:
            # Workflows serve their anchor fields. Per-output-field token
            # listeners still attach; if streamify can't wire them onto the
            # composed program the generator's fallback runs blocking
            # inference and emits a single final event.
            input_fields = workflow_spec.input_field_names()
            output_fields = workflow_spec.output_field_names()

        if not input_fields:
            raise DomainError("serve.no_declared_inputs", status=400)
        missing = [f for f in input_fields if f not in req.inputs]
        if missing:
            raise DomainError(
                "serve.missing_inputs",
                status=400,
                missing=missing,
                input_fields=input_fields,
            )
        filtered_inputs = {f: req.inputs[f] for f in input_fields}

        await asyncio.to_thread(enforce_llm_credits, job_store, current_user.username)
        lm = build_language_model(model_config)
        model_used = model_config.normalized_identifier()
        listeners = [StreamListener(signature_field_name=f) for f in output_fields]
        source = _stream_program_inference(
            program=program,
            lm=lm,
            filtered_inputs=filtered_inputs,
            input_fields=input_fields,
            output_fields=output_fields,
            listeners=listeners,
            model_used=model_used,
            error_log_context=f"job {optimization_id}",
        )
        metered = stream_with_llm_metering(
            source,
            job_store=job_store,
            username=current_user.username,
            description="Serve inference",
            usage_sink=[lm],
        )
        return StreamingResponse(
            sse_from_events(metered),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get(
        "/serve/{optimization_id}/pair/{pair_index}/info",
        response_model=ServeInfoResponse,
        summary="Describe the program for one grid-search pair",
        tags=["agent"],
    )
    def serve_pair_info(
        optimization_id: str, pair_index: int, current_user: AuthenticatedUserDep
    ) -> ServeInfoResponse:
        """Describe the program for one grid-search pair without running it.

        Same shape as ``GET /serve/{id}/info`` but scoped to a specific
        grid-search pair; ``model_name`` is the pair's generation model.

        Args:
            optimization_id: Grid-search optimization id.
            pair_index: Index of the pair in the grid-search result.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``ServeInfoResponse`` describing the pair's compiled program.

        Raises:
            DomainError: 404 (unknown / inaccessible), 409 (not finished or
                pair failed).
        """
        _program, pair, overview = load_pair_program(job_store, optimization_id, pair_index, current_user)
        artifact = pair.program_artifact
        input_fields, output_fields, instructions, demo_count = _artifact_prompt_fields(artifact)

        return ServeInfoResponse(
            optimization_id=optimization_id,
            module_name=overview.get(PAYLOAD_OVERVIEW_MODULE_NAME, ""),
            optimizer_name=overview.get(PAYLOAD_OVERVIEW_OPTIMIZER_NAME, ""),
            model_name=pair.generation_model,
            input_fields=input_fields,
            output_fields=output_fields,
            instructions=truncate_text(instructions, AGENT_MAX_INSTRUCTIONS),
            demo_count=demo_count,
            sample_inputs=_sample_inputs(
                job_store, optimization_id, current_user, artifact, input_fields
            ),
        )

    @router.post(
        "/serve/{optimization_id}/pair/{pair_index}",
        response_model=ServeResponse,
        summary="Run inference through one grid-search pair",
    )
    def serve_pair_program(
        optimization_id: str,
        pair_index: int,
        req: ServeRequest,
        current_user: AuthenticatedUserDep,
    ) -> ServeResponse:
        """Run inference through one grid-search pair's compiled program.

        Default model is the pair's generation model; override with
        ``model_config_override``. All ``input_fields`` must be supplied;
        extras are ignored.

        Args:
            optimization_id: Grid-search optimization id.
            pair_index: Index of the pair in the grid-search result.
            req: Inference request carrying inputs and optional model override.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A ``ServeResponse`` with the predicted outputs and resolved model.

        Raises:
            DomainError: 400 (bad inputs), 402 (caller has no spendable
                credits), 404 (unknown / inaccessible), 409 (not finished
                or pair failed).
        """
        program, pair, _overview = load_pair_program(job_store, optimization_id, pair_index, current_user)
        artifact = pair.program_artifact

        model_config = req.model_config_override or ModelConfig(name=pair.generation_model)

        input_fields, output_fields, _instructions, _demo_count = _artifact_prompt_fields(artifact)

        if not input_fields:
            raise DomainError("serve.no_declared_inputs", status=400)
        missing = [f for f in input_fields if f not in req.inputs]
        if missing:
            raise DomainError(
                "serve.missing_inputs",
                status=400,
                missing=missing,
                input_fields=input_fields,
            )
        filtered_inputs = {f: req.inputs[f] for f in input_fields}

        enforce_llm_credits(job_store, current_user.username)
        lm = build_language_model(model_config)

        try:
            with dspy.context(lm=lm):
                prediction = program(**filtered_inputs)
        finally:
            # A failed inference still consumed provider tokens (dspy retries
            # under the hood), so the debit rides ``finally``.
            meter_llm_run(
                getattr(job_store, "engine", None),
                current_user.username,
                [lm],
                description="Serve inference",
            )

        outputs: dict[str, Any] = {}
        if output_fields:
            for field in output_fields:
                outputs[field] = getattr(prediction, field, None)
        else:
            outputs = {key: val for key, val in prediction.toDict().items() if key not in req.inputs}

        return ServeResponse(
            optimization_id=optimization_id,
            outputs=_cap_serve_outputs(outputs),
            input_fields=input_fields,
            output_fields=output_fields,
            model_used=model_config.normalized_identifier(),
        )

    @router.post(
        "/serve/{optimization_id}/pair/{pair_index}/stream",
        summary="Stream inference from one grid-search pair as SSE",
    )
    async def serve_pair_program_stream(
        optimization_id: str,
        pair_index: int,
        req: ServeRequest,
        current_user: AuthenticatedUserDep,
    ):
        """Stream inference from one grid-search pair as Server-Sent Events.

        Same ``token`` -> ``final`` event shape as the single-run stream
        endpoint, same blocking fallback behavior when ``dspy.streamify``
        can't set up listeners.

        Args:
            optimization_id: Grid-search optimization id.
            pair_index: Index of the pair in the grid-search result.
            req: Inference request carrying inputs and optional model override.
            current_user: Authenticated caller resolved from the bearer token.

        Returns:
            A streaming ``StreamingResponse`` with ``text/event-stream`` body.

        Raises:
            DomainError: 400 (bad inputs), 402 (caller has no spendable
                credits), 404 (unknown / inaccessible), 409 (not finished
                or pair failed).
        """
        # Offload the synchronous DB read + program deserialization so it does
        # not block the single event loop before the first byte is produced.
        program, pair, _overview = await asyncio.to_thread(
            load_pair_program, job_store, optimization_id, pair_index, current_user
        )
        artifact = pair.program_artifact

        model_config = req.model_config_override or ModelConfig(name=pair.generation_model)

        input_fields, output_fields, _instructions, _demo_count = _artifact_prompt_fields(artifact)

        if not input_fields:
            raise DomainError("serve.no_declared_inputs", status=400)
        missing = [f for f in input_fields if f not in req.inputs]
        if missing:
            raise DomainError(
                "serve.missing_inputs",
                status=400,
                missing=missing,
                input_fields=input_fields,
            )
        filtered_inputs = {f: req.inputs[f] for f in input_fields}

        await asyncio.to_thread(enforce_llm_credits, job_store, current_user.username)
        lm = build_language_model(model_config)
        model_used = model_config.normalized_identifier()
        listeners = [StreamListener(signature_field_name=f) for f in output_fields]
        source = _stream_program_inference(
            program=program,
            lm=lm,
            filtered_inputs=filtered_inputs,
            input_fields=input_fields,
            output_fields=output_fields,
            listeners=listeners,
            model_used=model_used,
            error_log_context=f"job {optimization_id} pair {pair_index}",
        )
        metered = stream_with_llm_metering(
            source,
            job_store=job_store,
            username=current_user.username,
            description="Serve inference",
            usage_sink=[lm],
        )
        return StreamingResponse(
            sse_from_events(metered),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post(
        "/serve/{optimization_id}/chat",
        summary="Stream a live ReAct chat turn against an optimized agent",
    )
    async def serve_chat_stream(
        optimization_id: str,
        req: ServeChatRequest,
        current_user: AuthenticatedUserDep,
        authorization: str | None = Header(default=None),
    ) -> StreamingResponse:
        """Stream one chat turn against a served, optimized ReActV2 agent as SSE.

        The agent's tools execute live against the MCP server the run was
        optimized against; each call is approval-gated by ``trust_mode`` (every
        tool except in ``yolo``). Emits the generalist SSE envelope
        (``reasoning_patch``, ``tool_start`` / ``tool_end``, ``pending_approval``
        / ``approval_resolved``, ``message_patch``, ``done``, ``error``).

        Args:
            optimization_id: The react run to chat against.
            req: Chat request with the user's message, prior turns, trust mode,
                and an optional model override.
            current_user: Authenticated caller; non-admins are restricted to
                their own runs.
            authorization: Caller's bearer token, forwarded into the agent's MCP
                session so its tool calls authenticate as the same user.

        Returns:
            A streaming ``text/event-stream`` response.

        Raises:
            DomainError: 404 (unknown/inaccessible), 409 (not a success react
                run, or not served from a live-MCP source), 400 (no model),
                402 (caller has no spendable credits).
        """
        # Offload the synchronous DB read + program deserialization so it does
        # not block the single event loop before the first byte is produced.
        signature_cls, program_state_json, react_overlay, overview = await asyncio.to_thread(
            load_react_chat_inputs, job_store, optimization_id, current_user
        )

        if req.model_config_override:
            model_config = req.model_config_override
        else:
            model_settings = overview.get(PAYLOAD_OVERVIEW_MODEL_SETTINGS, {})
            model_name = overview.get(PAYLOAD_OVERVIEW_MODEL_NAME, "")
            if model_settings:
                model_config = ModelConfig.model_validate(model_settings)
            elif model_name:
                model_config = ModelConfig(name=model_name)
            else:
                raise DomainError("serve.no_model_config", status=400)

        await asyncio.to_thread(enforce_llm_credits, job_store, current_user.username)
        lm = build_language_model(model_config)
        tool_source = react_overlay.tool_source or {}
        mcp_url = tool_source.get("mcp_url") or settings.generalist_agent_mcp_url

        source = run_react_chat(
            signature_cls=signature_cls,
            program_state_json=program_state_json,
            react_overlay=react_overlay,
            user_message=req.user_message,
            trust_mode=req.trust_mode,
            lm=lm,
            model_name=model_config.normalized_identifier(),
            mcp_url=mcp_url,
            auth_header=authorization,
        )
        metered = stream_with_llm_metering(
            source,
            job_store=job_store,
            username=current_user.username,
            description="Serve chat",
            usage_sink=[lm],
        )
        return StreamingResponse(
            sse_from_events(metered),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post(
        "/serve/{optimization_id}/chat/confirm",
        response_model=ServeChatConfirmResponse,
        summary="Resolve a pending react-serve chat tool approval",
    )
    def serve_chat_confirm(
        optimization_id: str,
        req: ServeChatConfirmRequest,
        current_user: AuthenticatedUserDep,
    ) -> ServeChatConfirmResponse:
        """Resolve an outstanding tool approval from the react-serve chat.

        Approval call-ids are process-unique, so this shares the same global
        registry the generalist agent uses; ``optimization_id`` scopes the
        route and enforces the caller holds editor-tier access (chat spends the
        owner's key, so it is editor+ like the rest of the serve surface).

        Args:
            optimization_id: The react run the pending call belongs to.
            req: Confirm payload with the ``call_id`` and approval boolean.
            current_user: Authenticated caller; must hold editor-tier access.

        Returns:
            A :class:`ServeChatConfirmResponse` with ``resolved=True`` on success.

        Raises:
            DomainError: 404 when the run is unknown/inaccessible; 403 when the
                caller's role is below editor; 404 when the call id is unknown
                or already resolved.
        """
        require_role_at_least(job_store, optimization_id, current_user, ShareRole.editor)
        resolved = get_approval_registry().resolve(req.call_id, req.approved)
        if not resolved:
            raise DomainError("agent.approval.unknown_call_id", status=404)
        return ServeChatConfirmResponse(resolved=True)

    return router
