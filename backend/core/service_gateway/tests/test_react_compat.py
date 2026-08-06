"""Tests for the ReAct compatibility + native-function-calling helpers.

``react_compat`` hides the DSPy 3.2-vs-3.3 ReAct split behind ``REACT_CLASS``
and gates the optional provider-native function-calling path behind the
``REACT_NATIVE_TOOL_CALLING`` flag. These tests exercise the native-calling
helpers without mutating the process-wide adapter, so they stay isolated from
the rest of the suite: the flag-on install path is intentionally left to the
serve-loop smoke path rather than asserted here.
"""

from __future__ import annotations

import dspy
import pytest
from dspy.adapters.chat_adapter import ChatAdapter

from core.service_gateway import react_compat
from core.service_gateway.react_compat import (
    configure_native_tool_calling,
    native_react_adapter,
    native_tool_calling_active,
)


def test_native_react_adapter_enables_native_fc_without_parallel() -> None:
    """The native adapter turns on native function calling but leaves parallel unset."""
    adapter = native_react_adapter()

    assert isinstance(adapter, ChatAdapter)
    assert adapter.use_native_function_calling is True
    # parallel_tool_calls stays unset so it is never injected on the forced-submit
    # turn, whose pinned tool_choice conflicts with parallel calling.
    assert adapter.parallel_tool_calls is None


def test_native_tool_calling_active_reflects_context_adapter() -> None:
    """The probe tracks the live adapter, honouring context overrides both ways."""
    with dspy.context(adapter=ChatAdapter(use_native_function_calling=True)):
        assert native_tool_calling_active() is True
    with dspy.context(adapter=ChatAdapter(use_native_function_calling=False)):
        assert native_tool_calling_active() is False


def test_configure_native_tool_calling_is_noop_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag off, configuring leaves the active adapter untouched."""
    monkeypatch.setattr(react_compat.settings, "react_native_tool_calling", False)
    before = native_tool_calling_active()

    configure_native_tool_calling()

    assert native_tool_calling_active() == before
