"""Unit tests for the code agent's chat edit tools.

Covers the no-LLM surface of :class:`_CodeEditSession`: the adapter-debris
strip applied to tool arguments (minimax occasionally emits a malformed
trailing field marker) and its interaction with the per-turn edit
guardrails. Validators are monkeypatched — their subprocess round-trip is
covered elsewhere.
"""

from __future__ import annotations

import pytest

from core.service_gateway.agents import code as code_module
from core.service_gateway.agents.code import _CodeEditSession, _strip_adapter_debris

_SIG = (
    "class Step(dspy.Signature):\n"
    '    """Do the step."""\n\n'
    "    q: str = dspy.InputField()\n"
    "    a: str = dspy.OutputField()\n"
)

_NEW_SIG = (
    "class Step(dspy.Signature):\n"
    '    """Do the step carefully."""\n\n'
    "    q: str = dspy.InputField()\n"
    "    a: str = dspy.OutputField()\n"
)

_MET = "def metric(gold, pred, trace=None):\n    return 1.0\n"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("code = 1\n[[ ## completed ## ]]", "code = 1"),
        ("code = 1\n[[ ## completed ## ]", "code = 1"),
        ("code = 1\n[[ ## completed ##", "code = 1"),
        ("x = a[b[0]]", "x = a[b[0]]"),
    ],
)
def test_strip_adapter_debris(raw: str, expected: str) -> None:
    """Trailing field markers are dropped, well-formed or malformed; real code survives."""
    assert _strip_adapter_debris(raw) == expected


@pytest.fixture
def events() -> list[dict]:
    """Collect the SSE events a session emits during a test."""
    return []


@pytest.fixture
def session(events: list[dict]) -> _CodeEditSession:
    """Build a session over the starter signature/metric with a capturing emitter."""
    return _CodeEditSession(signature_code=_SIG, metric_code=_MET, emit=events.append)


def test_edit_signature_strips_debris_before_validation(
    monkeypatch: pytest.MonkeyPatch, session: _CodeEditSession
) -> None:
    """A trailing malformed marker is stripped so the edit validates and applies."""
    seen: list[str] = []
    monkeypatch.setattr(
        code_module, "_validate_signature_code", lambda c: seen.append(c) or ""
    )
    observation = session.edit_signature("tighten docstring", _NEW_SIG + "[[ ## completed ## ]")
    assert "replaced and validated" in observation
    assert session.signature_code == _NEW_SIG.rstrip()
    assert seen == [_NEW_SIG.rstrip()]


def test_edit_signature_debris_only_edit_is_rejected_as_identical(
    monkeypatch: pytest.MonkeyPatch, session: _CodeEditSession
) -> None:
    """Current code plus debris is recognized as a no-op, not applied as a change."""
    monkeypatch.setattr(code_module, "_validate_signature_code", lambda c: "")
    observation = session.edit_signature("noop", _SIG + "[[ ## completed ## ]]")
    assert "identical" in observation
    assert session.signature_code == _SIG


def test_edit_metric_strips_debris_before_validation(
    monkeypatch: pytest.MonkeyPatch, session: _CodeEditSession
) -> None:
    """The metric tool applies the same strip as the signature tool."""
    seen: list[str] = []
    monkeypatch.setattr(code_module, "_validate_metric_code", lambda c: seen.append(c) or "")
    new_metric = "def metric(gold, pred, trace=None):\n    return 0.5\n"
    observation = session.edit_metric("halve score", new_metric + "[[ ## completed ## ]")
    assert "replaced and validated" in observation
    assert session.metric_code == new_metric.rstrip()
    assert seen == [new_metric.rstrip()]
