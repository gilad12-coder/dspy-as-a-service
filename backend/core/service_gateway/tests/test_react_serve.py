"""Tests for the live react-serve chat driver (:mod:`...agents.react_serve`)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import dspy
import pytest

from ..agents.generalist import ApprovalRegistry, _wrap_tool_with_approval
from ..agents.react_serve import (
    _filter_tools,
    _format_react_outputs,
    _react_needs_approval,
    run_react_chat,
)


def _make_fake_tool(name: str, return_value: str = "ok") -> dspy.Tool:
    """Build a ``dspy.Tool`` whose async ``func`` returns a fixed value.

    Args:
        name: Registered tool name.
        return_value: Value the tool's ``func`` returns.

    Returns:
        A ready-to-use ``dspy.Tool``.
    """

    async def func(**_kwargs: Any) -> str:
        """Return the fixed value regardless of arguments."""
        return return_value

    return dspy.Tool(func=func, name=name, desc="t", args={}, arg_types={}, arg_desc={})


def test_react_needs_approval_gates_all_but_yolo() -> None:
    """Every tool is gated in ask/auto_safe; none in yolo."""
    assert _react_needs_approval("any_tool", "ask") is True
    assert _react_needs_approval("any_tool", "auto_safe") is True
    assert _react_needs_approval("any_tool", "yolo") is False


def test_filter_tools_passthrough_when_no_filter() -> None:
    """A ``None`` filter returns the roster unchanged."""
    tools = [_make_fake_tool("a"), _make_fake_tool("b")]
    assert _filter_tools(tools, None) == tools


def test_filter_tools_keeps_filter_order_and_skips_missing() -> None:
    """The filter selects + reorders, dropping names absent from the roster."""
    tools = [_make_fake_tool("a"), _make_fake_tool("b"), _make_fake_tool("c")]
    result = _filter_tools(tools, ["c", "missing", "a"])
    assert [t.name for t in result] == ["c", "a"]


def test_format_react_outputs_single_field_verbatim() -> None:
    """A single output field is returned without a label."""
    pred = SimpleNamespace(answer="42")
    assert _format_react_outputs(pred, ["answer"]) == "42"


def test_format_react_outputs_multi_field_labelled() -> None:
    """Multiple output fields are labelled and joined."""
    pred = SimpleNamespace(answer="42", confidence="high")
    out = _format_react_outputs(pred, ["answer", "confidence"])
    assert "answer: 42" in out
    assert "confidence: high" in out


def test_format_react_outputs_json_encodes_non_string() -> None:
    """A single non-string output field is JSON-encoded, unlabelled."""
    pred = SimpleNamespace(answer={"k": 1})
    assert _format_react_outputs(pred, ["answer"]) == '{"k": 1}'


def test_format_react_outputs_skips_none_fields() -> None:
    """``None`` output fields are dropped from the assembled reply."""
    pred = SimpleNamespace(answer="hi", note=None)
    out = _format_react_outputs(pred, ["answer", "note"])
    assert "note" not in out
    assert out == "answer: hi"


@pytest.mark.asyncio
async def test_injected_policy_gates_arbitrary_tool_in_ask() -> None:
    """The react policy gates an arbitrary MCP tool the generalist would not.

    The default classifier only gates known wizard tools, so an unknown tool
    in ``ask`` would run straight through. Injecting ``_react_needs_approval``
    makes it pause for approval — proving the policy override is wired.
    """
    events: list[dict] = []
    registry = ApprovalRegistry()
    tool = _wrap_tool_with_approval(
        _make_fake_tool("some_external_search_tool", return_value="done"),
        trust_mode="ask",
        registry=registry,
        emit=events.append,
        outer_loop=asyncio.get_running_loop(),
        needs_approval=_react_needs_approval,
    )
    call_task = asyncio.create_task(tool.func._async_body())
    for _ in range(20):
        await asyncio.sleep(0)
        if any(e["event"] == "pending_approval" for e in events):
            break
    pending = next((e for e in events if e["event"] == "pending_approval"), None)
    assert pending is not None
    assert registry.resolve(pending["data"]["id"], True) is True
    assert await call_task == "done"


@pytest.mark.asyncio
async def test_run_react_chat_pumps_emitted_events_then_done(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_react_chat`` forwards driver-emitted events and appends ``done``."""

    async def fake_drive(*, emit, **_kwargs):
        """Emit two patch events then return a final assistant message."""
        emit({"event": "reasoning_patch", "data": {"chunk": "thinking"}})
        emit({"event": "message_patch", "data": {"chunk": "hello"}})
        return "hello world"

    monkeypatch.setattr("core.service_gateway.agents.react_serve._drive_react_chat", fake_drive)

    events = [
        ev
        async for ev in run_react_chat(
            signature_cls=object,
            program_state_json="{}",
            react_overlay=SimpleNamespace(tool_source={}),
            user_message="hi",
            trust_mode="ask",
            lm=object(),
            model_name="openai/gpt-4o-mini",
            mcp_url="http://mcp.local",
        )
    ]
    names = [e["event"] for e in events]
    assert names == ["reasoning_patch", "message_patch", "done"]
    assert events[-1]["data"] == {"assistant_message": "hello world", "model": "openai/gpt-4o-mini"}


@pytest.mark.asyncio
async def test_run_react_chat_surfaces_driver_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver exception is surfaced as a terminal ``error`` event, not raised."""

    async def boom(*, emit, **_kwargs):
        """Raise to simulate the driver failing mid-turn."""
        raise RuntimeError("mcp unreachable")

    monkeypatch.setattr("core.service_gateway.agents.react_serve._drive_react_chat", boom)

    events = [
        ev
        async for ev in run_react_chat(
            signature_cls=object,
            program_state_json="{}",
            react_overlay=SimpleNamespace(tool_source={}),
            user_message="hi",
            trust_mode="ask",
            lm=object(),
            model_name="m",
            mcp_url="http://mcp.local",
        )
    ]
    assert events[-1]["event"] == "error"
    assert "error" in events[-1]["data"]
