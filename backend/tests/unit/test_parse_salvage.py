"""Unit tests for ``core.service_gateway.agents.parse_salvage``."""

import dspy
from dspy.utils.exceptions import AdapterParseError

from core.service_gateway.agents.code_interview import CodeInterviewTurnSig
from core.service_gateway.agents.parse_salvage import (
    find_adapter_parse_error,
    salvage_prediction,
)

CHAT_FORMAT_RESPONSE = """[[ ## message ## ]]
מעולה. איך צריך להיראות פורמט המספר?

[[ ## options_json ## ]]
[{"label": "מספר שלם", "description": "ללא נקודה עשרונית"}]

[[ ## brief_json ## ]]
[]

[[ ## done ## ]]
false"""


def _parse_error(lm_response: str) -> AdapterParseError:
    """Build the error dspy raises when the JSON fallback cannot parse ``lm_response``."""
    return AdapterParseError(
        adapter_name="JSONAdapter",
        signature=CodeInterviewTurnSig,
        lm_response=lm_response,
    )


def test_salvages_chat_format_reply():
    """A complete chat-adapter reply is recovered into a prediction."""
    pred = salvage_prediction(_parse_error(CHAT_FORMAT_RESPONSE))
    assert isinstance(pred, dspy.Prediction)
    assert pred.message.startswith("מעולה")
    assert pred.options_json == '[{"label": "מספר שלם", "description": "ללא נקודה עשרונית"}]'
    assert pred.done == "false"


def test_salvages_error_nested_in_exception_groups():
    """streamify wraps failures in task groups; salvage unwraps them."""
    err = BaseExceptionGroup(
        "outer",
        [BaseExceptionGroup("inner", [_parse_error(CHAT_FORMAT_RESPONSE)])],
    )
    assert find_adapter_parse_error(err) is not None
    assert salvage_prediction(err) is not None


def test_returns_none_for_incomplete_reply():
    """A reply missing output fields cannot be salvaged."""
    assert salvage_prediction(_parse_error("[[ ## message ## ]]\nhello")) is None


def test_returns_none_for_unrelated_errors():
    """Non-parse failures (network, auth, …) are left to the retry path."""
    assert salvage_prediction(RuntimeError("connection reset")) is None
    assert salvage_prediction(BaseExceptionGroup("g", [ValueError("x")])) is None
