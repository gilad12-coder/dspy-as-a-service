"""Tests for the version-agnostic ReAct reply streamer.

``ReactReplyStream`` bridges the two ways a ReAct program surfaces its reply:
ReActV2 (DSPy 3.3+) carries it as a ``submit`` tool-call argument on the inner
``react`` predictor's ``tool_calls`` field, while classic ReAct (DSPy 3.2.x)
streams it straight off a separate ``extract`` predictor. These tests exercise
both branches regardless of which DSPy line is installed, by toggling the
presence of an ``extract`` attribute on a stand-in program.
"""

from __future__ import annotations

from types import SimpleNamespace

import dspy
import pytest

from core.service_gateway.agents import code as code_module
from core.service_gateway.agents.code import (
    NativeToolCallStreamListener,
    ReactReplyStream,
    _NativeSubmitArgExtractor,
    _SubmitArgExtractor,
)
from core.service_gateway.react_compat import REACT_CLASS, react_uses_submit


class _Sig(dspy.Signature):
    """Reply to the user."""

    user_message: str = dspy.InputField()
    reply: str = dspy.OutputField()


def _noop(x: str) -> str:
    """Echo the argument.

    Args:
        x: Arbitrary string.

    Returns:
        The argument unchanged.
    """
    return x


def _response(field: str, chunk: str, *, last: bool = False) -> dspy.streaming.StreamResponse:
    """Build a ``StreamResponse`` for a given field and chunk.

    Args:
        field: The ``signature_field_name`` the chunk belongs to.
        chunk: The streamed text fragment.
        last: Whether this is the field's terminal chunk.

    Returns:
        A populated ``dspy.streaming.StreamResponse``.
    """
    return dspy.streaming.StreamResponse(
        predict_name="react",
        signature_field_name=field,
        chunk=chunk,
        is_last_chunk=last,
    )


class _SubmitProgram:
    """A stand-in ReActV2 program: an inner ``react`` predictor and no ``extract``."""

    def __init__(self, react: dspy.Predict) -> None:
        """Store the inner predictor that drives the loop.

        Args:
            react: A real predictor so reasoning-listener binding succeeds.
        """
        self.react = react


def test_native_program_matches_capability_probe() -> None:
    """The streamer's submit/extract choice tracks ``react_uses_submit``."""
    program = REACT_CLASS(_Sig, tools=[_noop], max_iters=3)
    stream = ReactReplyStream(program, "reply")

    assert stream._uses_submit is react_uses_submit(program)
    listeners = stream.listeners()
    assert len(listeners) == 2
    assert dspy.streamify(program, stream_listeners=listeners, async_streaming=True) is not None


def test_extract_program_streams_reply_field_directly() -> None:
    """Classic ReAct: the reply field's chunks pass through verbatim."""
    program = REACT_CLASS(_Sig, tools=[_noop], max_iters=3)
    if react_uses_submit(program):
        program.extract = program.react  # force the classic branch on a 3.3 install

    stream = ReactReplyStream(program, "reply")

    assert stream._uses_submit is False
    assert stream.reply_delta(_response("reply", "Hel")) == "Hel"
    assert stream.reply_delta(_response("reply", "lo", last=True)) == "lo"
    assert stream.reply_delta(_response("next_thought", "ignored")) is None


def test_submit_program_decodes_partial_tool_call_json() -> None:
    """ReActV2: partial ``submit`` JSON yields the growing reply argument."""
    base = REACT_CLASS(_Sig, tools=[_noop], max_iters=3)
    stream = ReactReplyStream(_SubmitProgram(base.react), "reply")

    assert stream._uses_submit is True
    assert stream.reply_delta(_response("reply", "anything")) is None  # not the tool_calls field
    first = stream.reply_delta(_response("tool_calls", '{"tool_calls":[{"name":"submit","args":{"reply":"Hi'))
    second = stream.reply_delta(_response("tool_calls", ' there"}}]}', last=True))
    assert (first or "") + (second or "") == "Hi there"


def test_serial_submit_stream_suppresses_reply_that_races_a_tool() -> None:
    """A mixed tool turn never leaks its premature submit text to the user."""
    base = REACT_CLASS(_Sig, tools=[_noop], max_iters=3)
    base.react._serial_tool_calls = True
    stream = ReactReplyStream(_SubmitProgram(base.react), "reply")

    first = stream.reply_delta(
        _response(
            "tool_calls",
            '{"tool_calls":[{"name":"count","args":{}},{"name":"submit","args":{"reply":"Checking',
        )
    )
    second = stream.reply_delta(_response("tool_calls", ' now."}}]}', last=True))
    final = stream.reply_delta(
        _response(
            "tool_calls",
            '{"tool_calls":[{"name":"submit","args":{"reply":"You have 3."}}]}',
            last=True,
        )
    )

    assert first is None
    assert second is None
    assert final == "You have 3."


def _lm_chunk(tool_calls: list | None = None, finish: str | None = None) -> SimpleNamespace:
    """Build a stand-in LiteLLM streaming chunk.

    Args:
        tool_calls: The ``delta.tool_calls`` list, or ``None`` for an empty delta.
        finish: The choice's ``finish_reason``, or ``None`` mid-stream.

    Returns:
        An object shaped like a LiteLLM ``ModelResponseStream`` chunk.
    """
    delta = SimpleNamespace(tool_calls=tool_calls, content=None)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)])


def _tool_call(index: int, name: str | None = None, arguments: str | None = None) -> SimpleNamespace:
    """Build a stand-in streamed tool-call delta.

    Args:
        index: The tool call's stream index.
        name: The function name (present only in the call's first delta).
        arguments: The streaming ``arguments`` JSON fragment.

    Returns:
        An object shaped like one element of ``delta.tool_calls``.
    """
    return SimpleNamespace(index=index, function=SimpleNamespace(name=name, arguments=arguments))


def _drain_native(listener: NativeToolCallStreamListener, stream: ReactReplyStream, chunks: list) -> str:
    """Run raw chunks through the native listener + reply bridge, as the serve loop does.

    Args:
        listener: The native tool-call listener from ``stream.listeners()``.
        stream: The reply stream whose ``_NativeSubmitArgExtractor`` decodes the deltas.
        chunks: The sequence of stand-in LiteLLM chunks to replay.

    Returns:
        The reconstructed reply text.
    """
    out = ""
    for chunk in chunks:
        response = listener.receive(chunk)
        if response is None:
            continue
        delta = stream.reply_delta(response)
        if delta:
            out += delta
    return out


def test_submit_program_defaults_to_text_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    """With native calling inactive, ReActV2 keeps the text tool-call extractor."""
    monkeypatch.setattr(code_module, "native_tool_calling_active", lambda: False)
    base = REACT_CLASS(_Sig, tools=[_noop], max_iters=3)
    stream = ReactReplyStream(_SubmitProgram(base.react), "reply")

    assert stream._uses_submit is True
    assert stream._native is False
    assert isinstance(stream._extractor, _SubmitArgExtractor)
    assert not isinstance(stream.listeners()[0], NativeToolCallStreamListener)


def test_native_submit_program_decodes_provider_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native calling: the submit call's provider ``arguments`` reconstruct the reply.

    A parallel non-submit tool call streams on index 0 and must be ignored; the
    submit call on index 1 carries the user-visible reply.
    """
    monkeypatch.setattr(code_module, "native_tool_calling_active", lambda: True)
    base = REACT_CLASS(_Sig, tools=[_noop], max_iters=3)
    stream = ReactReplyStream(_SubmitProgram(base.react), "reply")

    assert stream._native is True
    assert isinstance(stream._extractor, _NativeSubmitArgExtractor)
    listener = stream.listeners()[0]
    assert isinstance(listener, NativeToolCallStreamListener)

    chunks = [
        _lm_chunk([_tool_call(0, name="edit_signature", arguments="")]),
        _lm_chunk([_tool_call(0, arguments='{"code":"x"}')]),
        _lm_chunk([_tool_call(1, name="submit", arguments="")]),
        _lm_chunk([_tool_call(1, arguments='{"reply":"Hi')]),
        _lm_chunk([_tool_call(1, arguments=' there"}')]),
        _lm_chunk(finish="tool_calls"),
    ]
    assert _drain_native(listener, stream, chunks) == "Hi there"


def test_native_listener_buffers_args_arriving_before_submit_name() -> None:
    """Args streamed before the ``submit`` name is seen are buffered, then flushed."""
    listener = NativeToolCallStreamListener(predict=None, allow_reuse=True)
    extractor = _NativeSubmitArgExtractor("reply")

    chunks = [
        _lm_chunk([_tool_call(0, arguments='{"reply":"par')]),
        _lm_chunk([_tool_call(0, name="submit", arguments="tial")]),
        _lm_chunk([_tool_call(0, arguments=' done"}')]),
        _lm_chunk(finish="stop"),
    ]
    out = ""
    for chunk in chunks:
        response = listener.receive(chunk)
        if response is None:
            continue
        delta = extractor.feed(response.chunk)
        if response.is_last_chunk:
            extractor.reset()
        if delta:
            out += delta
    assert out == "partial done"


def test_native_listener_resets_between_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reused listener + extractor decode a second turn cleanly after finish_reason."""
    monkeypatch.setattr(code_module, "native_tool_calling_active", lambda: True)
    base = REACT_CLASS(_Sig, tools=[_noop], max_iters=3)
    stream = ReactReplyStream(_SubmitProgram(base.react), "reply")
    listener = stream.listeners()[0]

    first = _drain_native(
        listener,
        stream,
        [
            _lm_chunk([_tool_call(0, name="submit", arguments='{"reply":"one"}')]),
            _lm_chunk(finish="stop"),
        ],
    )
    second = _drain_native(
        listener,
        stream,
        [
            _lm_chunk([_tool_call(0, name="submit", arguments='{"reply":"two"}')]),
            _lm_chunk(finish="stop"),
        ],
    )
    assert first == "one"
    assert second == "two"
