"""Generalist agent that drives the Skynet wizard via MCP tools.

A :class:`dspy.ReActV2` on top of the MCP surface exposed by
``backend/core/api/mcp_mount.py``. The agent observes the current wizard
state, chooses from a phased tool list, and streams reasoning + sub-tool
progress over the same SSE envelope used by :mod:`code_agent`.

Phased exposure (the gate):

* Always available: read-only discovery tools (``list_models``,
  ``get_registry_snapshot``, ``get_job_*``, analytics).
* Unlocked once the dataset has columns + roles: ``validate_code``,
  ``profile_datasets``.
* Unlocked once the run is NAMED and the dataset is ready:
  ``request_code_authoring`` (the Signature/Metric step). Mirrors the
  wizard's Basics → Data → Params → Code order so the wizard is populated
  and verifiable before any code exists.
* Unlocked once name + signature + metric + model are all set:
  ``submit_job``.
* Always available post-submit: ``cancel_job``, rename/pin.

Tool docstrings become the agent prompt, so we rely on the trimming in
:mod:`mcp_mount._trim_tool_spec` to keep each description ≤240 chars. Any
gating logic that would need a long description lives in the system
prompt of :class:`GeneralistSig` instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from functools import partial
from typing import Any, Literal, TypedDict

import dspy
from dspy.streaming import StatusMessageProvider
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from sqlalchemy import delete
from sqlalchemy.orm import Session

from ...config import settings
from ...exceptions import ServiceError
from ...i18n import t
from ...models import ModelConfig
from ...storage.models import AgentApprovalModel
from ..language_models import (
    apply_model_reasoning_config,
    build_language_model,
    served_model_from,
)
from ..optimization.retrying_react import RetryingReActV2
from ..optimization.training_ground.registry import hash_tool_schema
from .code import ReactReplyStream, _agent_error_payload, _format_agent_error, _reply_language
from .constants import REASONING_FIELD


def _build_generalist_lm() -> dspy.LM:
    """Construct the default LM for the generalist agent from settings.

    Reasoning configuration, by provider:

    - **Native MiniMax** (``minimax/...``): ``extra_body={"reasoning_split": true}``
      surfaces the interleaved ``<think>`` channel as ``reasoning_details``.
    - **Fireworks-hosted MiniMax** (``fireworks_ai/...``) **and OpenRouter
      MiniMax** (``openrouter/minimax/...``, the shipped default): reasoning
      streams inline in the assistant content as ``<think>…</think>`` blocks;
      no provider-side knob.
    - **OpenAI reasoning models** (``openai/gpt-5.*``, ``openai/o1|o3|o4*``):
      pass ``reasoning_effort="medium"`` so the model emits reasoning content
      that LiteLLM normalizes to ``delta.reasoning_content``. DSPy validates
      these models at init — ``temperature=1.0`` and ``max_tokens>=16000`` are
      mandatory, not optional.
    - **Everything else**: no reasoning knob; ``max_tokens=4000`` is plenty for
      a chat-style reply.

    Returns:
        A configured :class:`dspy.LM` instance for the generalist agent.
    """
    config = apply_model_reasoning_config(
        ModelConfig(
            name=settings.generalist_agent_model,
            base_url=settings.generalist_agent_base_url or None,
        )
    )
    _apply_interactive_timeout(config)
    return build_language_model(config, disable_cache=True)


def _apply_interactive_timeout(config: ModelConfig) -> None:
    """Give a chat-turn LM an interactive timeout instead of the job-scale one.

    ``build_language_model`` defaults to ``lm_request_timeout_seconds`` (sized
    for batch optimization runs) with watchdog-derived retries — on a stalled
    provider that is tens of minutes of dead air for a chat turn. ``extra``
    merges over those defaults, so seed it with the chat-scale knobs unless the
    caller pinned its own.
    """
    config.extra.setdefault("timeout", settings.agent_request_timeout_seconds)
    config.extra.setdefault("num_retries", 2)


logger = logging.getLogger(__name__)

TrustMode = Literal["ask", "auto_safe", "yolo"]

# Tools whose side-effects can destroy or create billing-bearing work.
# Always require confirmation except in YOLO mode.
_DESTRUCTIVE_TOOLS: frozenset[str] = frozenset(
    {
        "delete_job_optimizations",
        "bulk_delete_jobs_optimizations_bulk_delete_post",
        "submit_job_run_post",
        "submit_grid_search_grid_search_post",
        "cancel_job_optimizations",
        "bulk_cancel_jobs_optimizations_bulk_cancel_post",
        "clone_job_optimizations",
        "retry_job_optimizations",
    }
)

# Safe mutations — metadata toggles, local-only operations.
# Confirm in Ask mode; auto-approve in Auto-safe and YOLO.
_SAFE_MUTATIONS: frozenset[str] = frozenset(
    {
        "rename_job_optimizations",
        "toggle_pin_job_optimizations",
        "edit_code_optimizations_edit_code_post",
        "validate_code_validate_code_post",
        "profile_datasets_profile_post",
        "discover_models_models_discover_post",
        "set_column_roles_datasets_column_roles_post",
        "update_wizard_state",
        "bulk_pin_jobs_optimizations_bulk_pin_post",
    }
)


# Upper bound on how long a tool waits for the user's approval decision. The
# registry is per-process, so a confirm POST that lands on another replica (or
# a client that vanished) can never resolve the future — without a bound the
# stream hangs forever with a spinning tool pill. Long enough for a human who
# stepped away; on expiry the tool is treated as declined.
APPROVAL_TIMEOUT_SECONDS = 900.0


# How often the awaiting stream checks the durable store for a decision that
# a confirm landing on another replica persisted. The in-process future still
# resolves same-pod confirms instantly; this only bounds cross-pod latency.
_DURABLE_POLL_SECONDS = 1.5


class ApprovalRegistry:
    """``call_id → Future[bool]`` store for pending tool approvals.

    The generalist SSE stream emits a ``pending_approval`` event carrying
    a ``call_id``; the paired ``POST /optimizations/generalist-agent/confirm``
    endpoint calls :meth:`resolve_or_persist` with the same id to unblock the
    tool. The in-process future is the fast path; when the confirm lands on a
    different replica (the registry is per-process), the decision is written
    to the shared ``agent_approvals`` table and :meth:`wait_for_decision`'s
    poll loop on the streaming replica picks it up.
    """

    def __init__(self) -> None:
        """Initialize the in-memory pending-approvals map."""
        self._pending: dict[str, asyncio.Future[bool]] = {}
        self._engine = None

    def bind_engine(self, engine: Any) -> None:
        """Attach the app database engine that backs cross-replica handoff.

        Args:
            engine: SQLAlchemy engine for the shared application database.
        """
        self._engine = engine

    def register(self, call_id: str) -> asyncio.Future[bool]:
        """Register ``call_id`` and return a future the tool awaits until resolved.

        Args:
            call_id: Unique identifier for the pending tool call.

        Returns:
            A future that resolves to the user's approval decision.
        """
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[call_id] = fut
        return fut

    def resolve(self, call_id: str, approved: bool) -> bool:
        """Complete a pending approval future.

        Args:
            call_id: Identifier matching a previous :meth:`register` call.
            approved: The user's decision (True to allow, False to decline).

        Returns:
            True when a matching pending future was resolved; False if the
            id was unknown or already settled.
        """
        fut = self._pending.pop(call_id, None)
        if fut is None or fut.done():
            return False
        fut.set_result(approved)
        return True

    def cancel(self, call_id: str) -> None:
        """Cancel a still-pending approval (e.g. the stream was torn down).

        Args:
            call_id: Identifier matching a previous :meth:`register` call.
        """
        fut = self._pending.pop(call_id, None)
        if fut is not None and not fut.done():
            fut.cancel()

    def resolve_or_persist(self, call_id: str, approved: bool) -> bool:
        """Resolve locally, else persist the decision for the owning replica.

        Args:
            call_id: The pending tool call's identifier.
            approved: The user's decision.

        Returns:
            True when the decision was delivered locally or persisted; False
            only when the id is unknown here and no durable store is bound.
        """
        if self.resolve(call_id, approved):
            return True
        if self._engine is None:
            return False
        self._persist(call_id, approved)
        return True

    def _persist(self, call_id: str, approved: bool) -> None:
        """Write (or overwrite) a decision row, purging stale leftovers.

        Args:
            call_id: The pending tool call's identifier.
            approved: The user's decision.
        """
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            row = session.get(AgentApprovalModel, call_id)
            if row is None:
                session.add(AgentApprovalModel(call_id=call_id, approved=approved, created_at=now))
            else:
                row.approved = approved
            # Rows are normally consumed within seconds; anything older than
            # the approval timeout belongs to a stream that already gave up.
            cutoff = now - timedelta(seconds=APPROVAL_TIMEOUT_SECONDS)
            session.execute(delete(AgentApprovalModel).where(AgentApprovalModel.created_at < cutoff))
            session.commit()

    def _take_durable(self, call_id: str) -> bool | None:
        """Consume a persisted decision for ``call_id``, if one arrived.

        Args:
            call_id: The pending tool call's identifier.

        Returns:
            The decision, or None when no row exists yet.
        """
        with Session(self._engine) as session:
            row = session.get(AgentApprovalModel, call_id)
            if row is None:
                return None
            approved = bool(row.approved)
            session.delete(row)
            session.commit()
            return approved

    async def wait_for_decision(self, call_id: str, fut: asyncio.Future[bool]) -> bool:
        """Await the user's decision from either delivery path, bounded.

        Races the in-process future (same-replica confirm, instant) against a
        poll of the durable store (cross-replica confirm). Expires as declined
        after :data:`APPROVAL_TIMEOUT_SECONDS` so a lost confirm can never
        hang the stream forever.

        Args:
            call_id: The pending tool call's identifier.
            fut: The future returned by :meth:`register` for this call.

        Returns:
            The user's decision, or False on expiry.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + APPROVAL_TIMEOUT_SECONDS
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                self.cancel(call_id)
                return False
            try:
                # shield(): a poll-interval timeout must not cancel the shared
                # future — the next loop iteration keeps awaiting it.
                return await asyncio.wait_for(
                    asyncio.shield(fut), timeout=min(_DURABLE_POLL_SECONDS, remaining)
                )
            except TimeoutError:
                if self._engine is None:
                    continue
                decision = await asyncio.to_thread(self._take_durable, call_id)
                if decision is not None:
                    self.cancel(call_id)
                    return decision


_global_registry = ApprovalRegistry()


def get_approval_registry() -> ApprovalRegistry:
    """Return the process-wide :class:`ApprovalRegistry` singleton.

    Returns:
        The shared registry used to coordinate tool approvals.
    """
    return _global_registry


def _needs_approval(tool_name: str, trust_mode: TrustMode) -> bool:
    """Decide whether a tool call must pause for user confirmation.

    Args:
        tool_name: The MCP tool's registered name.
        trust_mode: The caller's selected trust level.

    Returns:
        True when the call should be gated behind an approval prompt.
    """
    if trust_mode == "yolo":
        return False
    if tool_name in _DESTRUCTIVE_TOOLS:
        return True
    return trust_mode == "ask" and tool_name in _SAFE_MUTATIONS


class _TurnAuthoringFlag:
    """Turn-scoped flag recording whether ``request_code_authoring`` fired.

    One instance is created per user turn in :func:`_drive_generalist_agent`
    and shared across every :class:`_ApprovalGatedTool` wrapper for that turn.
    ``request_code_authoring`` writes its authored Signature/Metric back to the
    wizard asynchronously (a later turn), so a ``submit_job_run_post`` in the
    SAME turn would ship stale code into a doomed run. The wrapper sets this
    flag when authoring is (re)requested and denies any submit that follows in
    the same turn. The prompt is the primary guard; this is the backstop.
    """

    def __init__(self) -> None:
        """Initialize the flag as not-yet-requested for this turn."""
        self.authoring_requested = False


def _serialize_tool_result(v: Any) -> Any:
    """Best-effort JSON-friendly conversion for SSE ``tool_end.result``.

    Args:
        v: The raw value returned by an MCP tool.

    Returns:
        ``v`` unchanged if already JSON-friendly, the parsed JSON value
        when ``str(v)`` decodes, or the stringified form as a last resort.
    """
    if v is None or isinstance(v, (dict, list, bool, int, float)):
        return v
    s = str(v)
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


class _ApprovalGatedTool:
    """Callable replacement for a ``dspy.Tool.func`` that enforces approval.

    Installed by :func:`_wrap_tool_with_approval`. The original async
    callable is stored on the instance so the wrapper can live at module
    scope instead of inside a closure. ``__call__`` emits ``tool_start``
    before the underlying call and ``tool_end`` after it (status
    ``ok``/``error`` with a best-effort JSON-serialized result). When the
    tool is gated by the caller's ``TrustMode`` it also emits
    ``pending_approval`` / ``approval_resolved`` events and returns
    ``"User declined"`` on refusal so the ReAct loop can reason about it.
    """

    def __init__(
        self,
        original: Callable[..., Awaitable[Any]],
        tool_name: str,
        trust_mode: TrustMode,
        registry: ApprovalRegistry,
        emit: Callable[[dict], None],
        outer_loop: asyncio.AbstractEventLoop,
        staged_dataset_id: str | None = None,
        source_dataset_id: str | None = None,
        wizard_state: WizardState | None = None,
        authoring_flag: _TurnAuthoringFlag | None = None,
        needs_approval: Callable[[str, TrustMode], bool] | None = None,
    ) -> None:
        """Capture the underlying tool and the side-channel plumbing.

        Args:
            original: The async callable to wrap.
            tool_name: Registered MCP tool name.
            trust_mode: The caller's trust level.
            registry: Approval registry used for gating.
            emit: Thread-safe callback for SSE events.
            outer_loop: The asyncio event loop where the MCP ``ClientSession``
                and approval futures live. DSPy 3.3 dispatches sync tool
                calls from worker threads with no running loop; the body
                must be marshalled back to ``outer_loop`` via
                ``run_coroutine_threadsafe`` or the MCP socket hangs
                because the session is bound to its original loop.
            staged_dataset_id: If the wizard snapshot has a staged dataset
                attached to this conversation, this wrapper auto-injects it
                into submit-tool calls that omit it. Mirrors the OpenAI /
                Anthropic Files API convention where uploaded files are
                bound to the thread and tools pick them up automatically
                instead of the LLM relaying the id on every turn.
            source_dataset_id: Id of a saved library dataset the user picked
                for this conversation; auto-injected into submit-tool calls
                that supply no dataset. The by-reference twin of
                ``staged_dataset_id`` — the two are mutually exclusive, so
                whichever the wizard carries is the one that lands.
            wizard_state: Turn-start wizard snapshot, used to validate the
                field order of ``update_wizard_state`` patches in real time
                (so the agent can't populate a later-step field before its
                earlier steps are complete). One ``update_wizard_state`` call
                per turn means the snapshot is stable for the check.
            authoring_flag: Turn-scoped flag shared across all wrappers in this
                turn. Set when ``request_code_authoring`` fires; checked to
                deny a ``submit_job_run_post`` that follows it in the same turn
                (the authored code is written back asynchronously, so it is not
                yet in this turn's snapshot).
            needs_approval: Policy deciding whether a call must pause for
                confirmation. Defaults to the wizard-tool classifier
                :func:`_needs_approval`; the react-serve driver injects a
                gate-everything-but-yolo policy for arbitrary MCP rosters.
        """
        self._original = original
        self._tool_name = tool_name
        self._trust_mode = trust_mode
        self._registry = registry
        self._emit = emit
        self._outer_loop = outer_loop
        self._staged_dataset_id = staged_dataset_id
        self._source_dataset_id = source_dataset_id
        self._wizard_state = wizard_state or {}
        self._authoring_flag = authoring_flag or _TurnAuthoringFlag()
        self._needs_approval = needs_approval or _needs_approval

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Sync entrypoint — DSPy 3.3 ``Tool.__call__`` invokes this from a worker thread.

        We don't run the async body here directly because the MCP session
        and approval futures live on ``self._outer_loop`` (the FastAPI
        request loop). Dispatching via ``run_coroutine_threadsafe`` keeps
        all asyncio work on the loop that owns the session and blocks
        the worker thread until the future resolves. ``Future.result()``
        propagates exceptions naturally so DSPy's ReAct loop sees real
        errors instead of timing out on a hung coroutine.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._async_body(*args, **kwargs), self._outer_loop
        )
        return future.result()

    async def _async_body(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped tool, emitting approval and lifecycle events.

        Returns ``"User declined"`` when the gated approval is rejected;
        re-raises ``CancelledError`` and tool-side exceptions after emitting
        a ``tool_end`` with ``status="error"``.

        Args:
            *args: Positional arguments forwarded to the underlying tool.
            **kwargs: Keyword arguments forwarded to the underlying tool.

        Returns:
            The wrapped tool's return value, or the literal string
            ``"User declined"`` when the call is rejected.

        Raises:
            asyncio.CancelledError: If the surrounding stream is cancelled.
            Exception: Re-raised after emitting a ``tool_end`` error event
                when the underlying tool raises.
        """
        call_id = uuid.uuid4().hex[:12]
        # ``request_code_authoring`` writes the authored Signature/Metric back
        # to the wizard asynchronously (a later turn), so a submit in the same
        # turn would ship stale/unauthored code. The prompt forbids this; this
        # is the runtime backstop.
        if self._tool_name in _CODE_AUTHORING_TOOLS:
            self._authoring_flag.authoring_requested = True
        submit_after_authoring = (
            self._tool_name in _SUBMIT_TOOLS
            and self._authoring_flag.authoring_requested
        )
        if self._tool_name in _SUBMIT_TOOLS and not submit_after_authoring:
            if (
                self._staged_dataset_id
                and not kwargs.get("staged_dataset_id")
                and not kwargs.get("dataset")
            ):
                kwargs["staged_dataset_id"] = self._staged_dataset_id
            # By-reference twin: a library dataset the user picked. Only inject
            # when no other dataset source is present — ``RunRequest`` requires
            # exactly one of dataset / staged_dataset_id / source_dataset_id, and
            # the two id fields are mutually exclusive by construction (the
            # wizard carries one or the other, never both).
            if (
                self._source_dataset_id
                and not kwargs.get("source_dataset_id")
                and not kwargs.get("staged_dataset_id")
                and not kwargs.get("dataset")
            ):
                kwargs["source_dataset_id"] = self._source_dataset_id
            # Privacy is private-by-default, matching the wizard. The submit
            # request models default ``is_private=False`` (public), so an agent
            # run that never set it would silently publish to Explore — force the
            # snapshot value, defaulting to private, so a run only goes public
            # when the user explicitly asked for it.
            kwargs["is_private"] = bool(self._wizard_state.get("is_private", True))
            # The wizard's description rides the same rails as privacy: set via
            # ``update_wizard_state`` (or typed in the Basics step), it lands on
            # submit even when the agent omits the arg. An agent-supplied
            # ``description`` wins. Clipped to the submit model's 280-char cap —
            # the wizard patch accepts up to 500.
            if not kwargs.get("description"):
                snapshot_desc = str(self._wizard_state.get("job_description") or "").strip()
                if snapshot_desc:
                    kwargs["description"] = snapshot_desc[:280]
            # Signature/Metric (or, for a workflow run, the authored graph) are
            # produced and validated by ``request_code_authoring`` and mirrored
            # into the wizard snapshot. The agent has historically re-typed its
            # own broken code into submit args even after a clean authoring pass
            # (3-arg metrics, unmatched braces), producing 400s that dead-ended
            # at the user. Source the program from the snapshot and discard
            # whatever the agent supplied so only validated code reaches submit.
            module_name = str(self._wizard_state.get("module_name") or "").strip().lower()
            workflow_snapshot = self._wizard_state.get("workflow")
            if module_name == "workflow" and workflow_snapshot:
                # ``RunRequest`` enforces workflow XOR signature_code; ship the
                # graph and drop any signature the agent may have supplied.
                kwargs["module_name"] = "workflow"
                kwargs["workflow"] = workflow_snapshot
                kwargs.pop("signature_code", None)
                metric_snapshot = self._wizard_state.get("metric_code")
                if metric_snapshot:
                    kwargs["metric_code"] = metric_snapshot
            else:
                for code_field in ("signature_code", "metric_code"):
                    snapshot_code = self._wizard_state.get(code_field)
                    if snapshot_code:
                        kwargs[code_field] = snapshot_code
        # Profiling a staged dataset needs the same rehydration submit relies
        # on: the rows live behind an opaque id, never inline in the model's
        # args, so without this the agent passes an empty dataset, the profile
        # comes back empty, and it loops until max_iters. Hand the backend the
        # staged id (read-only — profiling never evicts the staged rows).
        if (
            self._tool_name == "profile_datasets_profile_post"
            and self._staged_dataset_id
            and not kwargs.get("staged_dataset_id")
            and not kwargs.get("dataset")
        ):
            kwargs["staged_dataset_id"] = self._staged_dataset_id
        self._emit(
            {
                "event": "tool_start",
                "data": {
                    "id": call_id,
                    "tool": self._tool_name,
                    "reason": "",
                    "arguments": kwargs,
                },
            }
        )
        try:
            if submit_after_authoring:
                denial = (
                    "Submit blocked: request_code_authoring ran this turn, so the "
                    "authored Signature/Metric is not in the wizard yet. End the "
                    "turn with a status message and submit on a later turn once "
                    "the code is reflected in wizard_state."
                )
                self._emit(
                    {
                        "event": "tool_end",
                        "data": {
                            "id": call_id,
                            "tool": self._tool_name,
                            "status": "error",
                            "result": denial,
                        },
                    }
                )
                return denial
            if self._tool_name == "update_wizard_state":
                order_error = validate_wizard_patch_order(kwargs, self._wizard_state)
                if order_error:
                    self._emit(
                        {
                            "event": "tool_end",
                            "data": {
                                "id": call_id,
                                "tool": self._tool_name,
                                "status": "error",
                                "result": order_error,
                            },
                        }
                    )
                    return order_error
            if self._needs_approval(self._tool_name, self._trust_mode):
                fut = self._registry.register(call_id)
                self._emit(
                    {
                        "event": "pending_approval",
                        "data": {"id": call_id, "tool": self._tool_name, "arguments": kwargs},
                    }
                )
                try:
                    # Bounded wait racing the local future against the durable
                    # store — see ApprovalRegistry.wait_for_decision.
                    approved = await self._registry.wait_for_decision(call_id, fut)
                except asyncio.CancelledError:
                    self._registry.cancel(call_id)
                    self._emit(
                        {
                            "event": "tool_end",
                            "data": {
                                "id": call_id,
                                "tool": self._tool_name,
                                "status": "error",
                                "result": "cancelled",
                            },
                        }
                    )
                    raise
                self._emit(
                    {
                        "event": "approval_resolved",
                        "data": {"id": call_id, "tool": self._tool_name, "approved": approved},
                    }
                )
                if not approved:
                    self._emit(
                        {
                            "event": "tool_end",
                            "data": {
                                "id": call_id,
                                "tool": self._tool_name,
                                "status": "error",
                                "result": "User declined",
                            },
                        }
                    )
                    return "User declined"
            result = await self._original(*args, **kwargs)
            self._emit(
                {
                    "event": "tool_end",
                    "data": {
                        "id": call_id,
                        "tool": self._tool_name,
                        "status": "ok",
                        "result": _serialize_tool_result(result),
                    },
                }
            )
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit(
                {
                    "event": "tool_end",
                    "data": {
                        "id": call_id,
                        "tool": self._tool_name,
                        "status": "error",
                        "result": _format_agent_error(exc),
                    },
                }
            )
            raise


def _wrap_tool_with_approval(
    tool: dspy.Tool,
    *,
    trust_mode: TrustMode,
    registry: ApprovalRegistry,
    emit: Callable[[dict], None],
    outer_loop: asyncio.AbstractEventLoop,
    staged_dataset_id: str | None = None,
    source_dataset_id: str | None = None,
    wizard_state: WizardState | None = None,
    authoring_flag: _TurnAuthoringFlag | None = None,
    needs_approval: Callable[[str, TrustMode], bool] | None = None,
) -> dspy.Tool:
    """Replace ``tool.func`` with an approval-aware wrapper.

    Args:
        tool: The DSPy tool whose ``func`` is being wrapped in place.
        trust_mode: Caller's trust level.
        registry: Approval registry used for gating.
        emit: Thread-safe SSE event emitter.
        outer_loop: Event loop owning the MCP session; the wrapper
            marshals every tool call back to this loop via
            ``run_coroutine_threadsafe`` because DSPy 3.3 dispatches
            sync ``Tool.__call__`` from worker threads with no loop.
        staged_dataset_id: Optional staged-dataset id auto-injected into
            submit tool calls that omit it.
        source_dataset_id: Optional library-dataset id auto-injected into
            submit tool calls that supply no dataset (by-reference runs).
        wizard_state: Turn-start wizard snapshot used to validate
            ``update_wizard_state`` field ordering.
        authoring_flag: Turn-scoped flag shared across all wrappers in the
            turn, used to block a submit that follows
            ``request_code_authoring`` in the same turn.
        needs_approval: Optional gating policy override; defaults to the
            wizard-tool classifier when omitted.

    Returns:
        The same ``tool`` instance with its ``func`` replaced.
    """
    tool.func = _ApprovalGatedTool(
        original=tool.func,
        tool_name=tool.name,
        trust_mode=trust_mode,
        registry=registry,
        emit=emit,
        outer_loop=outer_loop,
        staged_dataset_id=staged_dataset_id,
        source_dataset_id=source_dataset_id,
        wizard_state=wizard_state,
        authoring_flag=authoring_flag,
        needs_approval=needs_approval,
    )
    return tool


class WizardState(TypedDict, total=False):
    """Snapshot of the wizard the agent is driving.

    Fed into ``tools_for`` to phase tool exposure. Every field is optional
    so callers can send a partial snapshot; missing fields count as "not
    ready". Mirrors a subset of the frontend ``SubmissionDraft`` state.
    """

    job_name: str
    # Run description authored in the wizard (Basics step) or patched by the
    # agent via ``update_wizard_state``; injected into submit calls that omit
    # ``description`` so an agent-driven run keeps the description the wizard
    # carries.
    job_description: str
    dataset_ready: bool
    columns_configured: bool
    signature_code: str
    metric_code: str
    model_configured: bool
    staged_dataset_id: str
    # Id of a saved library dataset the run submits by reference (durable),
    # set when the user picks from the dataset library instead of uploading.
    # Mutually exclusive with ``staged_dataset_id`` (an evicted upload).
    source_dataset_id: str
    optimizer_name: str
    module_name: str
    job_type: str
    is_private: bool
    # Authored ``WorkflowSpec`` graph for multi-module runs. Present only when
    # ``module_name == "workflow"``; carries the run's program in place of
    # ``signature_code`` (the two are mutually exclusive at submit).
    workflow: dict[str, Any]
    model_config: dict[str, Any]
    reflection_model_config: dict[str, Any]
    generation_models: list[dict[str, Any]]
    reflection_models: list[dict[str, Any]]
    use_all_generation_models: bool
    use_all_reflection_models: bool


_ALWAYS_TOOLS = frozenset(
    {
        # Agent-only model catalog: each row exposes a single canonical
        # ``name`` (provider-prefixed) — no separate ``label`` field to copy
        # by accident. The frontend keeps using ``/models``; the agent never
        # sees that one.
        "list_models_for_agent",
        "get_registry_snapshot_registry_get",
        "list_jobs_optimizations_get",
        "get_optimization_counts_optimizations_counts_get",
        "get_job_summary_optimizations",
        "get_job_logs_optimizations",
        "get_analytics_summary_analytics_summary_get",
        "get_optimizer_stats_analytics_optimizers_get",
        "get_model_stats_analytics_models_get",
        # Read-only wallet: credit balance, free grant, plan, and recent ledger —
        # lets the agent answer "how many credits do I have / what did I spend?".
        "get_wallet_for_agent",
        "serve_info_serve",
        "serve_pair_info_serve",
        # UI-trigger tool — calling it renders an inline inference-input
        # card in the chat. The card fetches the field schema via
        # /serve/{id}/info and runs the actual inference via /serve/{id};
        # the agent NEVER calls serve_program directly because it cannot
        # know the user's inputs.
        "request_user_inference",
        # Grid-search twin of request_user_inference: renders the inference card
        # for one pair. The agent picks pair_index from serve_pair_info /
        # get_grid_search_result; the card runs the pair inference client-side.
        "request_user_pair_inference",
        "discover_models_models_discover_post",
        "rename_job_optimizations",
        "toggle_pin_job_optimizations",
        # Job lifecycle tools that can run on an existing optimization at any time.
        "clone_job_optimizations",
        "retry_job_optimizations",
        "compare_jobs_optimizations_compare_post",
        "bulk_pin_jobs_optimizations_bulk_pin_post",
        # Wizard-prefill tools — safe to expose before a dataset is staged;
        # they patch wizard state, they don't consume rows.
        "set_column_roles_datasets_column_roles_post",
        # UI-trigger tool — calling it renders an inline upload card in the
        # chat. The card handles parsing + column-role confirmation
        # client-side and dispatches wizard:dataset-staged on confirm.
        "request_user_dataset_datasets_request_upload_post",
        # Saved dataset library: a read tool to surface the caller's datasets,
        # and a UI-trigger picker whose selection hydrates the wizard with a
        # source_dataset_id — a durable by-reference run, unlike the evicted
        # staged upload above.
        "list_datasets_for_agent",
        "request_user_dataset_from_library",
        # Generalized wizard patch — any editable field, partial updates.
        "update_wizard_state",
        # Semantic + structured search across every public optimization. The
        # agent uses it to surface comparable runs ("find me sentiment jobs
        # that beat 0.8 with GEPA") before the user has filled the wizard.
        "public_search_dashboard_search_post",
        # Diagnostic readouts for finished runs — per-example baseline /
        # optimized scores and full grid-result detail. Read-only and safe
        # to call once an optimization id is in scope.
        "get_test_results_optimizations",
        "get_grid_search_result_optimizations",
        "get_pair_test_results_optimizations",
        # Read-only tagger reach: list the caller's tagging sessions so the agent
        # can answer questions about them and point the user at /tagger/{id}. The
        # tagger has its own assist agent; the generalist only reads here.
        "list_tagging_sessions_for_agent",
    }
)
# Diagnostic tools unlocked the moment a dataset has columns + roles. These
# inspect the data; they do not advance the wizard, so they don't depend on
# the run being named yet.
_DATASET_READY_TOOLS = frozenset(
    {
        "validate_code_validate_code_post",
        "profile_datasets_profile_post",
    }
)
# UI-trigger tool — calling it renders an inline code-authoring card that runs
# the dedicated code agent (streaming Signature + Metric with the wizard's
# timeline) and writes the result back to the wizard. Replaces the old
# block-and-return ``edit_code`` path so the generalist never hand-writes
# signature/metric code. Gated behind a named run (see ``tools_for``) so the
# agent fills the wizard in the same order the manual wizard enforces —
# Basics (name) → Data → Params → Code — and the wizard is populated and
# verifiable before any code is authored.
_CODE_AUTHORING_TOOLS = frozenset({"request_code_authoring"})
# The submit surface is job-type-exclusive: exactly ONE submit tool is exposed
# per turn, chosen by ``job_type`` (see ``tools_for``). Exposing both at once
# historically made MiniMax oscillate between them and occasionally lapse into a
# "no submit tool" hallucinated refusal — so a single run only ever sees
# ``submit_job_run_post`` and a grid run only ever sees the grid tool; the two
# are never visible together.
_READY_TO_SUBMIT_TOOLS = frozenset({"submit_job_run_post"})
_GRID_SUBMIT_TOOLS = frozenset({"submit_grid_search_grid_search_post"})
# Every submit surface — used to scope the wrapper's argument injection (staged
# dataset, validated program, privacy) uniformly across single and grid runs.
_SUBMIT_TOOLS = _READY_TO_SUBMIT_TOOLS | _GRID_SUBMIT_TOOLS
_POST_SUBMIT_TOOLS = frozenset(
    {
        "cancel_job_optimizations",
        "bulk_cancel_jobs_optimizations_bulk_cancel_post",
        "delete_job_optimizations",
        "bulk_delete_jobs_optimizations_bulk_delete_post",
    }
)


def _name_set(state: WizardState) -> bool:
    """Return True when the run has a non-blank job name (Basics step)."""
    name_val = state.get("job_name") or state.get("name") or ""
    return bool(isinstance(name_val, str) and name_val.strip())


def _dataset_ready(state: WizardState) -> bool:
    """Return True when the dataset has columns + roles configured (Data step)."""
    return bool(state.get("columns_configured") or state.get("dataset_ready"))


# Sentinels seeded by the frontend's un-mapped code templates (see
# frontend/src/features/submit/lib/build-signature.ts and build-metric.ts).
# Before any dataset columns are mapped the signature template declares the
# ``input_field``/``output_field`` field names and the metric falls back to
# ``fields = ["output_field"]``. Those field names never match a real column
# mapping, so a submit built from them fails validation — that is the only
# not-yet-ready state we gate on. We deliberately do NOT key on the template's
# ``"Describe the task here."`` docstring: build-signature.ts keeps that
# docstring even after columns are mapped, when the same template carries real
# field names and is a valid, submittable signature.
_PLACEHOLDER_SENTINEL_FIELDS = ("input_field", "output_field")
# The no-columns metric fallback emits exactly ``fields = ["output_field"]``.
_PLACEHOLDER_METRIC_FALLBACK = 'fields = ["output_field"]'


def _is_placeholder_signature(code: str) -> bool:
    """Return True when ``code`` is the wizard's un-mapped Signature template.

    The frontend seeds a Signature whose fields are the ``input_field`` /
    ``output_field`` sentinels until the user maps dataset columns; those names
    never match a real column mapping, so a submit built from them fails
    validation. Detect that un-mapped template by its sentinel field names.
    The template's default docstring is intentionally NOT a signal: once
    columns are mapped the same template carries real field names and is a
    valid, submittable Signature even if the docstring is left unchanged.

    Args:
        code: The ``signature_code`` value from the wizard state.

    Returns:
        True if ``code`` still declares the un-mapped sentinel fields.
    """
    if not code:
        return False
    return all(field in code for field in _PLACEHOLDER_SENTINEL_FIELDS)


def _is_placeholder_metric(code: str) -> bool:
    """Return True when ``code`` is the wizard's un-edited Metric template.

    The metric template only degrades to the ``output_field`` sentinel when no
    output columns are mapped; once real columns exist the generated metric is
    genuinely valid. Match the full fallback ``fields`` literal rather than a
    bare ``output_field`` substring so a dataset that legitimately has a column
    named ``output_field`` is not misclassified as not-yet-authored.

    Args:
        code: The ``metric_code`` value from the wizard state.

    Returns:
        True if ``code`` still contains the seeded ``output_field`` fallback.
    """
    if not code:
        return False
    return _PLACEHOLDER_METRIC_FALLBACK in code


def _code_ready(state: WizardState) -> bool:
    """Return True when the Code step is authored for the chosen module.

    A non-empty value is not enough: the frontend seeds placeholder templates
    into the wizard, so the gate must reject those un-edited placeholders and
    stay locked until the user authors real code. Workflow runs carry an
    authored graph instead of a single Signature — a valid graph plus a real
    metric is the ready state for that module.
    """
    metric = state.get("metric_code") or ""
    metric_ready = bool(metric) and not _is_placeholder_metric(metric)
    if str(state.get("module_name") or "").strip().lower() == "workflow":
        workflow = state.get("workflow")
        graph_ready = bool(isinstance(workflow, dict) and workflow.get("nodes"))
        return graph_ready and metric_ready
    signature = state.get("signature_code") or ""
    if not signature or not metric_ready:
        return False
    return not _is_placeholder_signature(signature)


def _is_grid(state: WizardState) -> bool:
    """Return True when the run is configured as a grid-search sweep."""
    return str(state.get("job_type") or "run").strip().lower() == "grid_search"


def _has_model_list(value: object) -> bool:
    """Return True when ``value`` is a non-empty list of named model configs."""
    return isinstance(value, list) and any(
        isinstance(m, dict) and m.get("name") for m in value
    )


def _model_ready(state: WizardState) -> bool:
    """Return True when the Model step is complete for the chosen run type.

    A single run needs a generation model (and, for GEPA, a reflection model —
    submitting GEPA without one is a known 422). A grid-search sweep instead
    needs a non-empty generation-model list (or the sweep-all flag) and, for
    GEPA, a reflection-model list (or its sweep-all flag). GEPA is the only
    supported optimizer, so an absent ``optimizer_name`` defaults to it (the
    strict path).

    Args:
        state: Current wizard snapshot.

    Returns:
        True when the model(s) the chosen run type requires are present.
    """
    is_gepa = str(state.get("optimizer_name") or "gepa").strip().lower() == "gepa"
    if _is_grid(state):
        has_generation = bool(state.get("use_all_generation_models")) or _has_model_list(
            state.get("generation_models")
        )
        if not has_generation:
            return False
        if is_gepa:
            return bool(state.get("use_all_reflection_models")) or _has_model_list(
                state.get("reflection_models")
            )
        return True
    model_cfg = state.get("model_config") or {}
    has_generation = bool(state.get("model_configured")) or bool(
        isinstance(model_cfg, dict) and model_cfg.get("name")
    )
    if not has_generation:
        return False
    if is_gepa:
        reflection_cfg = state.get("reflection_model_config") or {}
        return bool(isinstance(reflection_cfg, dict) and reflection_cfg.get("name"))
    return True


def tools_for(state: WizardState) -> set[str]:
    """Compute the MCP tool names exposed for a given wizard snapshot.

    The generalist never sees all tools at once — it would burn context
    and invite misuse. Each phase of the wizard unlocks its own slice.

    Args:
        state: Snapshot describing what the wizard has filled in so far.

    Returns:
        The set of MCP tool names allowed for this wizard state.
    """
    allowed = set(_ALWAYS_TOOLS) | set(_POST_SUBMIT_TOOLS)
    dataset_ready = _dataset_ready(state)
    name_set = _name_set(state)
    if dataset_ready:
        allowed |= _DATASET_READY_TOOLS
        # Mirror the wizard's step order: the Code step (Signature + Metric)
        # only opens once the run is named and the dataset is in place. This
        # keeps the wizard populated + verifiable before code exists, and
        # stops the agent from authoring/submitting an unnamed run.
        if name_set:
            allowed |= _CODE_AUTHORING_TOOLS
    if (
        dataset_ready
        and name_set
        and _code_ready(state)
        and _model_ready(state)
    ):
        # Expose exactly one submit surface, matching the chosen run type, so
        # the agent never sees two submit tools at once (which made the model
        # oscillate). Grid runs sweep model lists; single runs use one pair.
        allowed |= _GRID_SUBMIT_TOOLS if _is_grid(state) else _READY_TO_SUBMIT_TOOLS
    return allowed


# Field-level ordering for ``update_wizard_state`` patches. Each editable
# field belongs to a wizard step; a field may only be set once every REQUIRED
# earlier step is populated. This mirrors the manual wizard's sequential form
# (Basics → Data → Params → Code → Model) at the field granularity, so the
# agent gets a precise, actionable error the moment it tries to skip ahead —
# rather than silently corrupting order. ``signature_code`` / ``metric_code``
# (the Code step) are intentionally absent: ``update_wizard_state`` rejects
# them outright (they are authored only via ``request_code_authoring``).
_WIZARD_STEP_LABELS = ("Basics", "Data", "Params", "Code", "Model")
_FIELD_STEP: dict[str, int] = {
    "job_name": 0,
    "job_description": 0,
    "is_private": 0,
    "column_roles": 1,
    "optimizer_name": 2,
    "module_name": 2,
    "job_type": 2,
    "split_fractions": 2,
    "split_mode": 2,
    "seed": 2,
    "shuffle": 2,
    "optimizer_kwargs": 2,
    "model_config": 4,
    "reflection_model_config": 4,
    "generation_models": 4,
    "reflection_models": 4,
    "use_all_generation_models": 4,
    "use_all_reflection_models": 4,
}
# Steps that gate later ones. Params (2) and Model (4) never block an earlier
# field, so they are not prerequisites for anything.
_PREREQ_STEPS = (0, 1, 3)
# What the agent must DO to satisfy each gating step (used in the error hint).
_STEP_FIX_HINT = {
    0: "set ``job_name`` via update_wizard_state",
    1: "attach the dataset via request_user_dataset (then confirm column roles)",
    3: "author the Signature + Metric via request_code_authoring",
}


def _step_satisfied(step: int, state: WizardState, patch: dict[str, Any]) -> bool:
    """Return True when wizard ``step`` is complete in ``state`` or this ``patch``.

    Args:
        step: Wizard step index (see ``_WIZARD_STEP_LABELS``).
        state: Current wizard snapshot.
        patch: The fields the agent is trying to set this call — a field set
            here counts toward satisfying its own step (so name + dataset can
            land in one patch).

    Returns:
        True when the step's required fields are present.
    """
    if step == 0:
        return _name_set(state) or bool(str(patch.get("job_name") or "").strip())
    if step == 1:
        if _dataset_ready(state):
            return True
        roles = patch.get("column_roles")
        if isinstance(roles, dict):
            vals = set(roles.values())
            return "input" in vals and "output" in vals
        return False
    if step == 3:
        return _code_ready(state)
    return True


def validate_wizard_patch_order(patch: dict[str, Any], state: WizardState) -> str | None:
    """Reject an ``update_wizard_state`` patch that skips wizard step order.

    Enforces the manual wizard's sequential form at the field level: a field
    may only be set once every REQUIRED earlier step is populated (in the
    current ``state`` or by this same ``patch``). Returns an actionable error
    string the ReAct loop reads as a tool observation so the agent can fix the
    order itself — the "real-time intervention" the wizard's Next-button
    gating gives a human.

    Args:
        patch: The keyword arguments of the ``update_wizard_state`` call.
        state: The turn-start wizard snapshot.

    Returns:
        ``None`` when the patch respects the order; otherwise a one-line
        English error naming the blocked field and the steps to do first.
    """
    touched = [(_FIELD_STEP[f], f) for f in patch if f in _FIELD_STEP]
    if not touched:
        return None
    target_step = max(step for step, _ in touched)
    blocked_field = next(field for step, field in touched if step == target_step)
    missing = [
        step
        for step in _PREREQ_STEPS
        if step < target_step and not _step_satisfied(step, state, patch)
    ]
    if not missing:
        return None
    steps_txt = "; ".join(f"{_WIZARD_STEP_LABELS[s]} — {_STEP_FIX_HINT[s]}" for s in missing)
    return (
        f"Out of order: ``{blocked_field}`` belongs to the "
        f"{_WIZARD_STEP_LABELS[target_step]} step, but earlier required steps are "
        f"incomplete. Do these first: {steps_txt}. Then set the "
        f"{_WIZARD_STEP_LABELS[target_step]} field(s)."
    )


class GeneralistSig(dspy.Signature):
    """Every turn ENDS with a ``submit`` tool call. No exceptions.

    The user sees ONLY the text you pass as ``submit(assistant_message=…)``.
    Reasoning, plans, and intentions are invisible until you call
    ``submit``. A turn without a ``submit`` call renders as a blank
    bubble — the user literally sees nothing and the conversation stalls.

    FORBIDDEN reasoning patterns (these all cause blank bubbles):
      • "No tools needed for a greeting" — WRONG. ``submit`` IS a tool;
        a greeting is ONE ``submit`` call with the greeting (written in
        ``reply_language``) in ``assistant_message``.
      • "Let me craft a reply" then stopping without calling ``submit`` —
        WRONG. Crafting in reasoning is invisible; the reply only exists
        when you call ``submit(assistant_message=<your text>)``.
      • "I'll respond directly" — WRONG. There is no "respond directly"
        path. Responding == calling ``submit``.

    Examples — every turn ends in submit (example replies below are shown
    in Hebrew; YOUR replies are always written in ``reply_language``):

    User says "הי" → one tool call only:
        submit(assistant_message="שלום! אני העוזר של Skynet לאופטימיזציית
        DSPy. במה תרצה/י להתחיל — להעלות dataset, לשכפל הרצה קיימת, או
        משהו אחר?")

    User says "אני רוצה להעלות דאטה סט" → two tool calls in order:
        1. request_user_dataset_datasets_request_upload_post(prompt="צרף/י
           קובץ CSV או JSON.")
        2. submit(assistant_message="הצגתי קארד להעלאה — צרף/י את הקובץ
           שלך ואמשיך משם.")

    User says "תגיש" with the wizard fully configured → two tool calls:
        1. submit_job_run_post(name="…", …)
        2. submit(assistant_message="ההגשה הוגשה. עוקב אחר ההתקדמות.")

    You are the Skynet assistant driving a DSPy optimization wizard. The
    user is typically non-technical; the UI language they chose arrives in
    ``reply_language``. Your job is to move the user toward a successful
    optimization run by calling tools — one coherent action per turn, not
    a chain of every possible step. Every turn still ends with ``submit``.

    Rules:
    * Reply in ``reply_language`` — every ``assistant_message``, status
      line, and user-facing ``prompt`` argument you write. Product terms
      (Signature, Metric, optimizer names) stay in English inside the
      localized prose.
    * Prefer calling tools over explaining. One tool call per turn is ideal.
    * Opening turn (greeting): 2–3 short sentences in ``reply_language``
      ending in a
      single targeted question. Never enumerate specific model names from
      memory — wait until the user is ready to pick a model, then call
      ``list_models_for_agent`` and use THAT result.
    * Batch ``update_wizard_state`` into one call per turn — it accepts
      every wizard field at once. Don't fire 3–7 sequential identical
      pills.
    * WIZARD ORDER — mandatory, mirrors the manual wizard. Fill the wizard
      in this sequence; earlier steps gate the tools for the later ones:
        1. Basics — set ``job_name`` (a short descriptive name in the
           user's language or English) via ``update_wizard_state``. Do this FIRST, before
           authoring code or submitting, even when the user only described
           the task in prose. NEVER leave the run unnamed.
        2. Data — call ``request_user_dataset`` so the user attaches the
           dataset and confirms column roles.
        3. Params — set ``optimizer_name`` / ``module_name`` / split if the
           user wants non-defaults (``gepa`` + ``predict`` are the
           defaults).
        4. Code — ``request_code_authoring`` becomes available ONLY after
           the run is named AND the dataset is ready. If you want to author
           code and the tool is NOT in your list this turn, the cause is a
           missing ``job_name`` (or dataset) — set it first, then it
           unlocks next turn.
        5. Model — pick the model (``model_config``; for GEPA also
           ``reflection_model_config``).
        6. Submit — ``submit_job_run_post`` unlocks once name + dataset +
           Signature + Metric + model are all present.
      Do NOT skip ahead. Authoring code or submitting before the run is
      named leaves the wizard unpopulated and unverifiable for the user.
      Field order is ENFORCED: if you set an ``update_wizard_state`` field
      whose earlier steps aren't filled yet (e.g. ``model_config`` before the
      code is authored, or ``optimizer_name`` before the dataset), the call
      is rejected with an "Out of order" error naming exactly which step to
      complete first. On an "Out of order" error, do EXACTLY this, then STOP:
        1. Do the ONE named step (e.g. "Do these first: Code" → call
           ``request_code_authoring`` once).
        2. END THE TURN with a short status line (in ``reply_language``)
           via ``submit``.
      Then OBEY these hard NEVERs on an out-of-order rejection:
        • NEVER re-fire the rejected patch. The field that was rejected
          (e.g. ``model_config``) belongs to a LATER step — do not retry it
          this turn or next turn; it unlocks on its own once the earlier
          step propagates into a future ``wizard_state`` snapshot.
        • NEVER re-request ``request_code_authoring`` just because a
          later-step field was rejected. A later-step rejection means an
          earlier step is still PROPAGATING, NOT that code is missing.
          Re-requesting authoring on a model_config rejection is the exact
          loop that doubles the turn — do not do it.
      Re-firing the rejected patch or re-requesting authoring in response to
      an out-of-order error is a forbidden loop.
    * If a tool returns an error, surface it to the user in
      ``reply_language`` and ask
      how to proceed — do not retry blindly. A 422/400 on submit is proof
      a wizard field is missing, not proof the submit tool is unavailable.
    * Never invent optimization IDs or model names. Get them from the
      discovery tools first.
    * When choosing a model, call ``list_models_for_agent`` and copy
      each row's ``name`` field verbatim into ``model_name`` /
      ``model_config.name``. Every ``name`` is already provider-prefixed
      (e.g. ``openai/gpt-4o-mini``); never strip the prefix. Obey these
      hard rules on every ``list_models_for_agent`` call:
        • ALWAYS pass a ``query`` argument — the model the user named, or
          a keyword (provider/family). E.g.
          ``list_models_for_agent(query="gpt-5.4-nano")`` or
          ``list_models_for_agent(query="claude")``.
        • NEVER call it with no query / NEVER fetch the full catalog. The
          unfiltered catalog is ~18KB and ~130 entries; reading it all
          costs ~15s of inference. A query shrinks the response to a few
          hundred bytes and returns in under a second.
        • Call it AT MOST ONCE per turn and REUSE that result for the rest
          of the turn. Do not re-call it to look up a second model — the
          first response already lists the matches.
    * When the user asks to submit in any language (e.g. "תגיש" / "תשלח" /
      "יש אישור" / "submit"): if
      ``submit_job_run_post`` is in your tool list THIS turn, call it;
      if it isn't, identify the missing wizard field and patch it via
      ``update_wizard_state`` / ``set_column_roles`` /
      ``request_user_dataset``. Never reply "אין לי גישה לכלי שליחת
      האופטימיזציה" — that's a hallucinated refusal.

    Supported backend capabilities (these are the ONLY valid values —
    never claim, suggest, or pass any others, even if DSPy supports them
    upstream):
    * Optimizer (``optimizer_name``): ``gepa`` is the only supported
      optimizer. Do not mention BootstrapFewShot, MIPRO/MIPROv2, COPRO,
      BootstrapFinetune, Ensemble, or any other DSPy optimizer — they are
      not wired into this backend.
    * Module (``module_name``): ``predict`` (dspy.Predict), ``cot``
      (dspy.ChainOfThought), and ``workflow`` are the only supported
      modules. ``workflow`` is a multi-node graph (a chain/DAG of
      signatures, Python transforms, and tool calls) that the user
      composes in the visual builder on the Code step; pick it when the
      task needs multiple LLM steps wired together. The graph itself is
      authored in the canvas (via ``request_code_authoring``), never as a
      single ``signature_code``.
    * Metric: there are no preset metrics. The user writes a metric
      function as Python source in ``metric_code`` (a callable taking
      ``(example, pred, trace=None)`` and returning a float).
    * If the user asks "which optimizers can I use?" answer GEPA only.
      If the user names an unsupported optimizer/module, tell them (in
      ``reply_language``) that it isn't wired into Skynet and offer the
      supported alternative.

    Capabilities worth knowing about:
    * Dataset uploads: when the user needs to provide a dataset (or you
      determine one is required to proceed), call ``request_user_dataset``
      with a short ``prompt`` sentence (in ``reply_language``) asking the
      user to attach a
      dataset file. That renders an upload card inline in the chat — the
      user picks the file, the panel parses it, the user confirms which
      columns are input/output, and the wizard hydrates automatically.
      Do **not** ask the user to upload in plain text; always call this
      tool so they get the rich upload affordance. After the card
      reports back via the next user message (with filename, row count,
      and the confirmed column roles), you can validate or refine the
      configuration with ``set_column_roles`` if needed. Never invent
      column names — use what the user confirms verbatim.
    * Existing jobs: ``clone_job`` duplicates a job (1–5 copies),
      ``retry_job`` re-runs a failed/cancelled one, ``compare_jobs`` gives
      a side-by-side snapshot of 2–5 optimizations, ``bulk_pin_jobs``
      toggles pin state in batch, ``bulk_cancel_jobs`` stops many
      running/pending jobs at once, ``bulk_delete_jobs`` removes many
      terminal jobs at once.
    * Column roles: ``set_column_roles`` writes a validated input/output
      map back to the wizard; prefer it over hand-editing code.
    * Any other wizard field: ``update_wizard_state`` patches any subset
      of editable fields — optimizer_name, module_name, model_config
      (teacher/student), reflection_model_config, generation_models /
      reflection_models (grid search), split_fractions, split_mode, seed,
      shuffle, optimizer_kwargs, job_name, job_description, job_type.
      Supply only the fields you want to change; everything else is left
      alone. Prefer it over the narrow per-field tools when changing one
      thing. Do NOT patch ``signature_code`` / ``metric_code`` here — they
      are authored only by ``request_code_authoring`` (see below); the
      ``update_wizard_state`` endpoint REJECTS those two fields.
    * When you NAME a run, describe it too: set ``job_description`` (one
      or two plain sentences — the task, the data, the goal, in
      ``reply_language``) in the SAME ``update_wizard_state`` patch as
      ``job_name``. It shows on the wizard's Basics step and lands as the
      run's description on submit automatically — a run you drive should
      never ship with a name but no description.
    * HARD RULE — one ``update_wizard_state`` call per turn. If you are
      patching N fields this turn, bundle them into a single ``patch``
      object on one call. Splitting "set optimizer, then set model, then
      set signature" into three separate ``update_wizard_state`` calls
      bloats the trajectory and never unlocks new tools mid-turn — the
      tool list is computed once at turn start from the snapshot you
      were handed. The unlock happens on the NEXT turn.
    * When the user picks an optimizer that needs a reflection model
      (e.g. ``gepa``), patch ``reflection_model_config`` in the SAME
      ``update_wizard_state`` call as ``model_config`` — typically
      mirroring the same ``name``. Submitting GEPA without
      ``reflection_model_config`` is a known failure mode.
    * Signature & Metric code: NEVER hand-write ``signature_code`` or
      ``metric_code`` yourself — that path is error-prone (bad class
      names, wrong metric arity) and is rejected by the wizard. Once the
      run is NAMED (``job_name`` set) and the dataset + column roles are in
      place, call ``request_code_authoring`` with a short ``goal`` (or
      empty to seed from the data). The tool stays hidden until the run is
      named — if it's missing, set ``job_name`` first. It renders an inline
      card
      that runs the dedicated code agent — the SAME one the submit wizard
      uses — which streams the Signature then the Metric as it drafts them,
      validates them, auto-fixes errors, and writes the finished code back
      into the wizard. After you call it, END your turn: the authored code
      lands in your NEXT turn's ``wizard_state`` (``signature_code`` +
      ``metric_code``), and only then does ``submit_job_run_post`` unlock.
      To refine later, call it again with a goal like "make the metric
      give partial credit for close answers".
    * NEVER call ``submit_job_run_post`` in the SAME turn as
      ``request_code_authoring``. ``request_code_authoring`` authors the
      Signature + Metric in an inline card and writes the result back to the
      wizard ASYNCHRONOUSLY — the new code is NOT in this turn's
      ``wizard_state``, so submitting now ships stale or wrong code that
      dead-ends in a doomed run. The instant you (re)request authoring —
      whether to seed code or to FIX a problem you just found in the existing
      Signature/Metric — END the turn with a short status line (in
      ``reply_language``) and
      submit ONLY on a LATER turn, once the authored code is reflected in the
      ``wizard_state`` snapshot you are handed. Requesting authoring and
      submitting in one turn is a contradiction: you cannot submit code you
      just flagged as wrong.
    * Logs: ``get_job_logs`` returns the log trail when the user is
      debugging a failed run.
    * Cross-corpus search: ``public_search`` does semantic + structured
      search over every public optimization (free-text query in any
      language, plus optional models / optimizers / optimization_types /
      date filters, sorted by relevance / recency / gain). Use it when the
      user asks to find comparable runs (free-text queries in the user's
      language, like
      "show me sentiment runs that scored above 0.8") before reaching for
      the wizard.
    * Run diagnostics: ``get_test_results`` returns per-example baseline
      and optimized test scores for a single run; ``get_grid_search_result``
      returns the full per-pair table for a finished grid search;
      ``get_pair_test_results`` zooms into one pair's per-example scores.
      Call them when the user asks why a run scored what it did or which
      examples regressed.
    * Live inference: when the user wants to try the trained program on a
      fresh input ("how would this run classify X?"), call
      ``request_user_inference`` with the ``optimization_id``. That renders
      an inline form in the chat — the user types the input values and
      the frontend runs the inference itself. Do NOT try to call any
      inference tool directly; you cannot know the user's inputs, and
      guessing them would waste an LLM call. After ``request_user_inference``
      returns, stop and wait for the next user message — the form result
      arrives as a follow-up turn.
    * Submitting an optimization: when the user asks to run / start /
      submit / launch an optimization, you submit it yourself by calling
      ``submit_job_run_post`` (single run) or
      ``submit_grid_search_grid_search_post`` (grid search). These tools
      become available only after the wizard is fully populated:
      ``job_name`` AND ``dataset_ready`` AND ``columns_configured`` AND
      ``signature_code`` AND ``metric_code`` AND a chosen model
      (``model_config.name``) must all be present in the wizard snapshot. If a prerequisite is
      missing, do NOT tell the user that you can't submit — identify
      which fields are blank from the wizard_state snapshot and either
      patch them via ``update_wizard_state`` / ``set_column_roles`` /
      ``request_user_dataset``, or ask one targeted question (in
      ``reply_language``) to
      fill the single biggest gap, then submit on the next turn. Never
      tell the user, in Hebrew or any other language, that you lack a
      submit tool — submission is always reachable once the wizard fields
      are in place, and you must drive the user there step by step rather
      than refuse. Completing the wizard is NOT submitting: setting the
      final field (typically the model) only UNLOCKS
      ``submit_job_run_post`` on your NEXT turn. On the turn you fill that
      last field, tell the user the run is ready and that you'll submit —
      do NOT report it as submitted. A run is submitted ONLY when
      ``submit_job_run_post`` returns a successful result in your
      trajectory this turn.
    * Grid search vs single run: the two submit tools are mutually
      exclusive — only the one matching ``job_type`` is exposed. The
      default ``job_type`` (``"run"``) uses a single model pair
      (``model_config`` + ``reflection_model_config``) and unlocks
      ``submit_job_run_post``. Set ``job_type`` to ``"grid_search"`` via
      update_wizard_state to sweep several models: a grid run needs model
      LISTS (``generation_models`` + ``reflection_models``, or the
      ``use_all_*`` flags) instead of the single configs, and unlocks
      ``submit_grid_search_grid_search_post`` instead. Only propose a grid
      search when the user asks to compare/sweep models.
    * Run privacy: runs are private by default (excluded from public
      Explore). Set ``is_private`` to false via update_wizard_state ONLY
      when the user explicitly asks to make the run public.
    * Dataset handoff for submit: never inline ``dataset`` rows into the
      submit tool arguments. The wizard stages the parsed rows on the
      backend after upload and surfaces a ``staged_dataset_id`` in the
      wizard_state snapshot. You do NOT need to pass ``staged_dataset_id``
      explicitly — the agent runtime auto-attaches the wizard's staged id
      to every submit call you make (the same way OpenAI/Anthropic
      Files-API attach files to a thread). Just call submit with the
      other fields; leave ``dataset``, ``username``, and
      ``staged_dataset_id`` unset. If ``staged_dataset_id`` is absent
      from the wizard snapshot when the user asks to submit, the dataset
      is not staged yet: call ``request_user_dataset`` and stop. Do NOT
      ask the user to re-upload an already-staged dataset.
    * Code handoff for submit: likewise never pass ``signature_code`` or
      ``metric_code`` into the submit call. The runtime injects the
      validated Signature/Metric authored by ``request_code_authoring``
      from the wizard snapshot, overriding anything you supply — so
      hand-typed code is discarded. Leave both unset. If they are blank
      in the snapshot the code isn't authored yet: call
      ``request_code_authoring`` and stop. Never re-type code from an
      earlier failed submit; the authored snapshot is the only source.

    CRITICAL — never fabricate tool results:
    * If ``submit_job_run_post`` (or any other tool) is NOT in your
      current tool list, you have NOT called it. Do not invent an
      optimization ID, status payload, or confirmation message.
      Fabricating a submission and reporting "the run was created
      successfully" with a made-up ``opt_xxx`` id when no such call was
      made is a critical failure.
    * The only valid optimization IDs are the ones returned by an actual
      successful ``submit_job_run_post`` / ``submit_grid_search_grid_search_post``
      tool result that appeared in your trajectory THIS TURN. If you did
      not see such a tool result, you have no ID to report.
    * If you discover mid-turn that the submit tool is unavailable
      because the wizard is incomplete, fix the wizard (via
      ``update_wizard_state`` / ``set_column_roles``) or ask the user
      one targeted question — but tell the truth about the current
      state. Do not pretend a submission happened.

    CRITICAL — never claim you lack a tool you actually have:
    * Tool availability is determined ONLY by what appears in your
      current tool list. If ``submit_job_run_post`` is in your tool
      list this turn, you DO have access to it — full stop.
    * A failure on a previous turn (e.g. an earlier ``submit_job_run_post``
      returned a 422 because ``reflection_model_config`` was missing) is
      NOT evidence that the tool is missing or unavailable. It is
      evidence of a missing wizard field. Diagnose the field from the
      previous tool result, patch it via ``update_wizard_state``, and
      call submit again on the next turn.
    * Never tell the user — in Hebrew, English, or any other language
      — that you "do not have access to the submit tool" or "the
      submit option is not exposed to me" when the tool is in fact in
      your current tool list. That is a hallucinated refusal and it
      breaks the user's trust.
    """

    wizard_state: str = dspy.InputField(desc="JSON snapshot of the current wizard state.")
    chat_history: str = dspy.InputField(desc="Prior {role, content} turns as JSON.")
    reply_language: str = dspy.InputField(
        desc="Language every user-facing string you write must be in (e.g. 'Hebrew', 'French'). "
        "Applies to assistant_message, status lines, and tool prompt arguments."
    )
    user_message: str = dspy.InputField(desc="The user's latest message.")
    assistant_message: str = dspy.OutputField(
        desc="Reply to the user, written in reply_language, summarizing what you did and what's next."
    )


class GeneralistStatusProvider(StatusMessageProvider):
    """Emit short Hebrew status messages around each tool call.

    DSPy's streamify pipes these as ``status`` chunks; the SSE wrapper in
    :func:`run_generalist_agent` forwards them as ``status_patch`` events.
    """

    def tool_start_status_message(self, instance: Any, inputs: dict[str, Any]) -> str:
        """Return the localized status line shown just before a tool call.

        Args:
            instance: The tool instance about to run.
            inputs: Keyword arguments the tool will be invoked with.

        Returns:
            Localized status text for the ``tool_start`` event.
        """
        return t("agent.status.tool_start")

    def tool_end_status_message(self, outputs: Any) -> str:
        """Return the localized status line shown after a tool call settles.

        Args:
            outputs: The value returned by the completed tool call.

        Returns:
            Localized status text for the ``tool_end`` event.
        """
        return t("agent.status.tool_end")


@asynccontextmanager
async def _mcp_session(
    mcp_url: str,
    *,
    auth_header: str | None = None,
) -> AsyncGenerator[ClientSession, None]:
    """Open a Streamable-HTTP MCP client session bound to ``mcp_url``.

    The generalist agent typically hits its own sibling-mounted MCP
    server (``http://localhost:<port>/mcp/``); taking the URL as an
    argument keeps the function testable against an out-of-process
    MCP server or a test fixture.

    Args:
        mcp_url: The HTTP endpoint of the target MCP server.
        auth_header: Verbatim ``Authorization`` header value (e.g.
            ``"Bearer <jwt>"``) to forward to the MCP server. Required when
            the target MCP mount sits behind ``get_authenticated_user``;
            ``mcp_mount.py`` forwards it through to the inner ASGI route.

    Yields:
        An initialized :class:`ClientSession` ready for ``list_tools``.
    """
    headers = {"Authorization": auth_header} if auth_header else None
    async with (
        streamablehttp_client(mcp_url, headers=headers) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


def _emit_to_queue_threadsafe(loop: asyncio.AbstractEventLoop, out_queue: asyncio.Queue[dict], ev: dict) -> None:
    """Hand ``ev`` to ``out_queue`` from any thread by scheduling ``put_nowait`` on ``loop``.

    The ReAct loop runs tool wrappers on worker threads; SSE events must
    land on the coroutine's queue from the coroutine's loop to avoid
    ``asyncio.Queue`` thread-unsafety. Binding ``loop`` and ``out_queue``
    with :func:`functools.partial` turns this into a closure-free drop-in
    emit callback.

    Args:
        loop: The event loop owning ``out_queue``.
        out_queue: Destination queue for SSE events.
        ev: The event payload to enqueue.
    """
    loop.call_soon_threadsafe(out_queue.put_nowait, ev)


async def _drive_generalist_agent(
    *,
    mcp_url: str,
    wizard_state: WizardState,
    chat_history: list[dict],
    user_message: str,
    trust_mode: TrustMode,
    registry: ApprovalRegistry,
    emit: Callable[[dict], None],
    lm: Any,
    reply_language: str,
    auth_header: str | None = None,
) -> str:
    """Open the MCP session, run the ReAct loop, and return the final assistant message.

    Streams reasoning / assistant / status chunks through ``emit`` as they
    arrive from DSPy's async streamer. The final assistant reply is
    returned so the outer coroutine can emit a terminal ``done`` event.

    Args:
        mcp_url: HTTP endpoint of the target MCP server.
        wizard_state: Snapshot of wizard state used to phase tool exposure.
        chat_history: Prior chat turns as ``{role, content}`` dicts.
        user_message: The user's latest message.
        trust_mode: Caller's trust level for tool gating.
        registry: Approval registry used for tool gating.
        emit: Thread-safe SSE event emitter.
        lm: Language model bound to the ReAct program.
        reply_language: English name of the language the agent replies in
            (e.g. ``"Hebrew"``), fed to the Signature's ``reply_language``
            input.
        auth_header: Verbatim ``Authorization`` header forwarded to the MCP
            session so tool calls hit the agent-tagged routes as the same
            user that opened the SSE stream.

    Returns:
        The full assistant reply text after the loop completes.
    """
    async with _mcp_session(mcp_url, auth_header=auth_header) as session:
        listing = await session.list_tools()
        allowed_names = tools_for(wizard_state)
        staged_id = wizard_state.get("staged_dataset_id") or None
        source_id = wizard_state.get("source_dataset_id") or None
        # The MCP session is bound to THIS loop. ``streamify`` will dispatch
        # tool calls from a worker thread (asyncify), so the wrapper has
        # to marshal each call back here via run_coroutine_threadsafe.
        outer_loop = asyncio.get_running_loop()
        # One flag per turn, shared across every wrapper, so a submit can see
        # whether request_code_authoring already fired earlier in this turn.
        authoring_flag = _TurnAuthoringFlag()
        dspy_tools = [
            _wrap_tool_with_approval(
                dspy.Tool.from_mcp_tool(session, t),
                trust_mode=trust_mode,
                registry=registry,
                emit=emit,
                outer_loop=outer_loop,
                staged_dataset_id=staged_id,
                source_dataset_id=source_id,
                wizard_state=wizard_state,
                authoring_flag=authoring_flag,
            )
            for t in listing.tools
            if t.name in allowed_names
        ]
        # Snapshot the live tool surface for downstream training-ground
        # persistence (training_ground_SPEC.md §4). The persistence wrapper
        # consumes this event and writes the recorded values into
        # ``agent_messages`` so the optimize CLI can reproduce phasing later.
        emit(
            {
                "event": "turn_metadata",
                "data": {
                    "allowed_tools": sorted(t.name for t in dspy_tools),
                    "tool_schema_hashes": {
                        tool.name: hash_tool_schema(tool) for tool in dspy_tools
                    },
                },
            }
        )
        # RetryingReActV2, not the stock class: the default generalist model is
        # minimax-class, which occasionally breaks the turn protocol and raises
        # AdapterParseError — the retrying loop resamples the turn instead of
        # failing the whole chat reply.
        react = RetryingReActV2(GeneralistSig, tools=dspy_tools, max_iters=8)
        # The user's ``assistant_message`` rides a ``submit`` tool call on ReActV2
        # or a separate ``extract`` predictor on classic ReAct; ``ReactReplyStream``
        # wires the right listeners and decodes whichever shape into reply deltas.
        # Leaving ``is_async_program`` at its default (False) lets ``streamify``
        # wrap the sync ``forward`` via ``asyncify``: on ReActV2 ``acall`` would
        # otherwise delegate to an ``aforward`` the class never defines
        # (AttributeError on the first turn), and classic ReAct's async path is
        # likewise bypassed — streaming behaviour and listeners are unchanged.
        reply_stream = ReactReplyStream(react, "assistant_message")
        program = dspy.streamify(
            react,
            stream_listeners=reply_stream.listeners(),
            status_message_provider=GeneralistStatusProvider(),
            async_streaming=True,
        )

        inputs = {
            "wizard_state": json.dumps(wizard_state, ensure_ascii=False),
            "chat_history": json.dumps(chat_history, ensure_ascii=False),
            "reply_language": reply_language,
            "user_message": user_message,
        }
        reply_text = ""
        with dspy.context(lm=lm):
            async for chunk in program(**inputs):
                if isinstance(chunk, dspy.streaming.StatusMessage):
                    emit({"event": "status_patch", "data": {"chunk": chunk.message}})
                elif isinstance(chunk, dspy.streaming.StreamResponse):
                    if chunk.signature_field_name == REASONING_FIELD:
                        emit({"event": "reasoning_patch", "data": {"chunk": chunk.chunk}})
                    else:
                        delta = reply_stream.reply_delta(chunk)
                        if delta:
                            reply_text += delta
                            emit({"event": "message_patch", "data": {"chunk": delta}})
                elif isinstance(chunk, dspy.Prediction):
                    final = getattr(chunk, "assistant_message", "") or ""
                    if final and final != reply_text:
                        reply_text = final
        return reply_text


async def run_generalist_agent(
    *,
    wizard_state: WizardState,
    chat_history: list[dict],
    user_message: str,
    trust_mode: TrustMode = "ask",
    mcp_url: str | None = None,
    model_config: ModelConfig | None = None,
    approval_registry: ApprovalRegistry | None = None,
    auth_header: str | None = None,
    locale: str | None = None,
    usage_sink: list | None = None,
) -> AsyncGenerator[dict, None]:
    """Stream generalist-agent events for one user turn.

    Emits the same SSE envelope as :func:`run_code_agent` so the frontend
    chat primitives work unchanged:

    * ``reasoning_patch`` — per-token reasoning
    * ``tool_start`` / ``tool_end`` — wrap each MCP tool call
    * ``status_patch`` — human-readable progress from ``StatusMessageProvider``
    * ``message_patch`` — per-token assistant reply
    * ``done`` — terminal event with the final assistant message and the model id used
    * ``error`` — terminal event carrying a user-facing error string

    On caller-side cancellation (SSE stream dropped) the orchestration task
    is cancelled and ``CancelledError`` is re-raised; every other
    orchestration error is caught and surfaced as an ``error`` envelope.

    Args:
        wizard_state: Snapshot of the wizard the agent is driving.
        chat_history: Prior chat turns as ``{role, content}`` dicts.
        user_message: The user's latest message.
        trust_mode: Trust level controlling which tool calls require approval.
        mcp_url: Optional override for the MCP server URL.
        model_config: Optional override for the language model configuration.
        approval_registry: Optional registry used for tool approval coordination.
        auth_header: Verbatim ``Authorization`` header from the SSE caller.
            Forwarded to the MCP session so the agent's tool calls
            authenticate against ``get_authenticated_user`` on the same
            FastAPI app — without it every agent-tagged route returns 401.
        locale: UI locale code of the client (e.g. ``he``, ``fr-CA``).
            Resolved via :func:`_reply_language`; unknown or missing falls
            back to Hebrew.
        usage_sink: Optional list the built LM is appended to, so the caller
            can meter the turn's token usage on any exit path — including a
            client disconnect where the ``done`` event never fires.

    Yields:
        SSE event dicts of shape ``{"event": str, "data": dict}``.

    Raises:
        asyncio.CancelledError: Re-raised when the stream is cancelled.
    """
    url = mcp_url or settings.generalist_agent_mcp_url
    registry = approval_registry or get_approval_registry()
    model_name = model_config.name if model_config else settings.generalist_agent_model
    try:
        if model_config:
            # A caller-chosen model runs through the same pipeline as the
            # default: platform base_url unless the config carries its own,
            # the reasoning-model knobs, and no response cache.
            override = model_config.model_copy(
                update={
                    "base_url": model_config.base_url
                    or settings.generalist_agent_base_url
                    or None
                }
            )
            override = apply_model_reasoning_config(override)
            _apply_interactive_timeout(override)
            lm = build_language_model(override, disable_cache=True)
        else:
            lm = _build_generalist_lm()
    except ServiceError as exc:
        yield {"event": "error", "data": {"error": str(exc)}}
        return
    if usage_sink is not None:
        usage_sink.append(lm)

    # The approval wrapper is called from a worker thread by DSPy, so we
    # need a thread-safe hop back to this coroutine's event loop to emit
    # SSE events onto the out-queue below.
    out_queue: asyncio.Queue[dict] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    emit: Callable[[dict], None] = partial(_emit_to_queue_threadsafe, loop, out_queue)

    drive_task = asyncio.create_task(
        _drive_generalist_agent(
            mcp_url=url,
            wizard_state=wizard_state,
            chat_history=chat_history,
            user_message=user_message,
            trust_mode=trust_mode,
            registry=registry,
            emit=emit,
            lm=lm,
            reply_language=_reply_language(locale),
            auth_header=auth_header,
        )
    )
    try:
        while not drive_task.done() or not out_queue.empty():
            getter = asyncio.create_task(out_queue.get())
            done, _pending = await asyncio.wait({drive_task, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield getter.result()
            else:
                getter.cancel()
            if drive_task in done and out_queue.empty():
                break
        reply = await drive_task
        yield {
            "event": "done",
            "data": {
                "assistant_message": reply,
                "model": model_name,
                # The concrete model behind an auto-routed turn (None when the
                # request named one explicitly); the reply footer reveals it.
                "served_model": served_model_from(lm),
            },
        }
    except asyncio.CancelledError:
        drive_task.cancel()
        raise
    except Exception as exc:
        logger.exception("generalist agent failed")
        yield {"event": "error", "data": _agent_error_payload(exc)}
