"""Version-agnostic handle on DSPy's ReAct program.

DSPy 3.3 reworked the agentic loop as ``ReActV2``: a single inner ``react``
predictor whose final answer is an argument of an internal ``submit`` tool call.
DSPy 3.2.x ships the classic ``ReAct``: the same ``react`` loop predictor plus a
separate ``extract`` predictor that emits the signature's output fields directly.

Both expose the same constructor (``signature, tools, max_iters``), an inner
``self.react`` predictor, and a ``self.tools`` dict, so the agent layer binds to
whichever class the installed DSPy provides and forks only where the two
genuinely differ — final-answer streaming (see ``react_reply_stream``).
"""

from __future__ import annotations

import contextlib
import logging

import dspy
from dspy.adapters.chat_adapter import ChatAdapter

from ..config import settings

logger = logging.getLogger(__name__)

REACT_CLASS: type[dspy.Module] = getattr(dspy, "ReActV2", None) or dspy.ReAct
"""The ReAct program class of the installed DSPy: ``ReActV2`` on 3.3+, else ``ReAct``."""


def react_uses_submit(program: dspy.Module) -> bool:
    """Report whether ``program`` carries its reply in a ``submit`` tool call.

    ReActV2 streams the final answer as a ``submit`` argument on the inner
    ``react`` predictor; classic ReAct streams it straight off a separate
    ``extract`` predictor. The presence of ``extract`` is the load-bearing
    distinction, so it is checked per-instance rather than inferred from the
    DSPy version.

    Args:
        program: A constructed ReAct/ReActV2 program (or subclass).

    Returns:
        ``True`` when the reply rides a ``submit`` tool call (ReActV2),
        ``False`` when a dedicated ``extract`` predictor emits it (classic).
    """
    return not hasattr(program, "extract")


def native_react_adapter() -> ChatAdapter:
    """Build the ChatAdapter that routes ReActV2 tools through native function calling.

    ``use_native_function_calling`` diverts a signature's ``list[dspy.Tool]``
    input and ``dspy.ToolCalls`` output onto the provider's ``tools=`` API, so
    turns come back as structured tool calls instead of DSPy's text protocol.
    It stays a no-op for signatures without a ToolCalls field, so only ReAct's
    inner predictor is affected. ``parallel_tool_calls`` is deliberately left
    unset: the adapter would otherwise inject it on every call, including
    ReActV2's forced-submit turn whose ``tool_choice`` pins the ``submit``
    function — a combination providers reject. Provider-default parallelism
    still applies on the normal (auto-``tool_choice``) turns.

    Returns:
        A ``ChatAdapter`` with native function calling enabled.
    """
    return ChatAdapter(use_native_function_calling=True)


def configure_native_tool_calling() -> None:
    """Install the native-function-calling adapter process-wide when enabled.

    Reads ``settings.react_native_tool_calling``; when on, sets a global
    ``ChatAdapter`` via ``dspy.configure`` so every ReAct run — optimization
    rollouts and all serve paths, including those built from the plain
    ``REACT_CLASS`` — resolves it through the ``main_thread_config`` fallback.
    A no-op when the flag is off. Failures are swallowed so a DSPy build that
    rejects the adapter kwargs can never abort process startup.
    """
    if not settings.react_native_tool_calling:
        return
    with contextlib.suppress(Exception):
        dspy.configure(adapter=native_react_adapter())
        logger.info("ReActV2 native function calling enabled (global ChatAdapter).")


def native_tool_calling_active() -> bool:
    """Report whether the currently active DSPy adapter uses native function calling.

    Inspects the live ``dspy.settings.adapter`` rather than the config flag, so
    reply-streaming adapts to whatever adapter is actually in force at call time
    (context override or global default).

    Returns:
        ``True`` when an adapter with ``use_native_function_calling`` is active.
    """
    adapter = getattr(dspy.settings, "adapter", None)
    return bool(getattr(adapter, "use_native_function_calling", False))


__all__ = [
    "REACT_CLASS",
    "configure_native_tool_calling",
    "native_react_adapter",
    "native_tool_calling_active",
    "react_uses_submit",
]
