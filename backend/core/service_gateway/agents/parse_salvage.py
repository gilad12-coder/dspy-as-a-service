"""Salvage a finished interview turn whose final adapter parse failed. [INTERNAL]

minimax-class models occasionally answer in the chat adapter's
``[[ ## field ## ]]`` section format even after dspy's JSONAdapter fallback
asked for JSON, so the turn's terminal parse raises ``AdapterParseError``
although every output field is present in the raw response. ``retrying_react``
handles the same model quirk inside ReAct loops by resampling; an interview
turn has already streamed its reply tokens to the user, so a resample costs
the whole turn again — re-parsing the finished response with the chat adapter
recovers it for free. Resampling stays as the fallback for genuinely broken
responses (see the retry loops in ``code_interview`` / ``tagging``).
"""

from __future__ import annotations

import logging

import dspy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


def find_adapter_parse_error(err: BaseException) -> AdapterParseError | None:
    """Return the ``AdapterParseError`` inside ``err``, unwrapping exception groups.

    dspy's ``streamify`` runs the program in an anyio task group, so the parse
    failure surfaces wrapped in (possibly nested) ``BaseExceptionGroup``s.

    Args:
        err: The exception raised by the streamed program.

    Returns:
        The first ``AdapterParseError`` found, or ``None`` when the failure is
        something else (network, auth, …) that salvage cannot help.
    """
    if isinstance(err, AdapterParseError):
        return err
    if isinstance(err, BaseExceptionGroup):
        for sub in err.exceptions:
            found = find_adapter_parse_error(sub)
            if found is not None:
                return found
    return None


def salvage_prediction(err: BaseException) -> dspy.Prediction | None:
    """Recover a prediction from a parse failure whose response is chat-format.

    Args:
        err: The exception raised by the streamed program.

    Returns:
        A ``dspy.Prediction`` carrying the signature's output fields when the
        failed response turns out to be a complete chat-adapter reply, else
        ``None``.
    """
    parse_err = find_adapter_parse_error(err)
    if parse_err is None:
        return None
    try:
        fields = ChatAdapter().parse(parse_err.signature, parse_err.lm_response)
    except Exception:
        return None
    logger.warning(
        "salvaged a chat-format reply after a %s parse failure",
        parse_err.adapter_name,
    )
    return dspy.Prediction(**fields)
