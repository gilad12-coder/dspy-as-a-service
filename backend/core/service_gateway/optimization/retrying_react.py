"""A ``dspy.ReActV2`` variant that resamples its loop turn on parse failures.

minimax-class models occasionally break the ReActV2 turn protocol — leaking a
``</mm:think>`` reasoning tag into the output, or short-circuiting straight to a
final message — so neither the chat nor the JSON adapter can parse the turn and
``AdapterParseError`` is raised. Stock ``ReActV2`` abandons the loop on the
first such failure; during a GEPA rollout that pins the example's reward at the
failure score, and at serve time it degrades to a forced-submit reply.

This module wraps the loop's single inner predictor so a parse failure is
retried a few times, each with a fresh ``rollout_id`` that bypasses the LM
response cache and forces a genuine resample (only effective at temperature > 0;
the production student model runs at 1.0). After the retries are exhausted the
original error is re-raised, so the stock forced-submit fallback still runs
exactly as before — the retry only ever adds attempts, never changes terminal
behavior.

Both the optimization rollout (``gepa_adapter``) and the three serve builders
(``react_serve``, ``_helpers``, ``tool_overlay``) drive the program through the
synchronous ``Predict.forward`` — serve via ``dspy.streamify``'s asyncify path,
which runs the sync loop in a worker thread — so overriding ``forward`` covers
every site.
"""

from __future__ import annotations

import logging

import dspy
from dspy.utils.exceptions import AdapterParseError

from ..react_compat import REACT_CLASS

logger = logging.getLogger(__name__)

PARSE_RETRY_ATTEMPTS = 2
"""Extra resamples per inner-predict call before surfacing the parse failure.

Each retry is one more call against a slow student model, so this is kept low;
the dominant failure is a transient tag leak that a single resample clears."""

_RESAMPLE_STRIDE = 7919
"""Prime per-attempt offset so each retry lands on a distinct LM cache key."""


class RetryingPredict(dspy.Predict):
    """A ``dspy.Predict`` that resamples its LM call on an adapter parse failure."""

    def __init__(
        self,
        signature,
        *,
        parse_retries: int = PARSE_RETRY_ATTEMPTS,
        serial_tool_calls: bool = False,
        callbacks=None,
        **config,
    ):
        """Build the predictor and record how many resamples to attempt.

        Args:
            signature: The ReActV2 inner-loop signature to predict against.
            parse_retries: Extra resample attempts after the first failure.
            serial_tool_calls: Disable parallel tool calls on normal loop turns
                so a terminal submit cannot race a sibling tool result.
            callbacks: Forwarded to ``dspy.Predict``.
            **config: Default LM kwargs forwarded to ``dspy.Predict``.
        """
        super().__init__(signature, callbacks=callbacks, **config)
        self._parse_retries = parse_retries
        self._serial_tool_calls = serial_tool_calls

    def forward(self, **kwargs):
        """Predict one turn, resampling on parse failure before re-raising.

        Args:
            **kwargs: The per-call inputs ReActV2 passes to the predictor,
                optionally carrying a ``config`` dict of LM kwargs.

        Returns:
            The ``dspy.Prediction`` from the first attempt that parses.

        Raises:
            AdapterParseError: When every attempt fails to parse.
            ValueError: When every attempt raises a value error from the adapter.
        """
        attempts = self._parse_retries + 1
        last_err: Exception | None = None
        for attempt in range(attempts):
            call_kwargs = (
                kwargs if attempt == 0 else self._with_resample_id(kwargs, attempt)
            )
            call_kwargs = self._with_tool_call_policy(call_kwargs)
            try:
                prediction = super().forward(**call_kwargs)
                return self._without_racing_submit(prediction)
            except (AdapterParseError, ValueError) as err:
                last_err = err
                logger.warning(
                    "ReActV2 inner predict parse failure (attempt %d/%d): %s",
                    attempt + 1,
                    attempts,
                    err,
                )
        raise last_err  # type: ignore[misc]

    def _without_racing_submit(self, prediction: dspy.Prediction) -> dspy.Prediction:
        """Drop a terminal submit that raced another tool in the same turn.

        ReActV2 ends the loop as soon as it executes a submit. If a provider
        returns submit beside a read tool, that reply was authored before the
        read result existed and cannot answer the user's question. Keeping only
        the non-submit calls lets ReAct execute them, append their results to
        history, and ask the model for the real final answer on the next turn.

        Args:
            prediction: Parsed inner-loop prediction from DSPy.

        Returns:
            The prediction unchanged unless serial execution is enabled and it
            contains both submit and non-submit calls.
        """
        if not self._serial_tool_calls:
            return prediction
        envelope = getattr(prediction, "tool_calls", None)
        calls = getattr(envelope, "tool_calls", None)
        if not isinstance(calls, list):
            return prediction
        has_submit = any(getattr(call, "name", None) == "submit" for call in calls)
        non_submit = [call for call in calls if getattr(call, "name", None) != "submit"]
        if not has_submit or not non_submit:
            return prediction
        prediction.tool_calls = envelope.model_copy(update={"tool_calls": non_submit})
        return prediction

    def _with_tool_call_policy(self, kwargs: dict) -> dict:
        """Apply serial execution to normal turns without touching forced submit.

        DSPy's forced-submit fallback pins ``tool_choice`` to ``submit``. Some
        providers reject that option when ``parallel_tool_calls`` is also sent,
        so the serial flag is added only to ordinary ReAct loop calls.

        Args:
            kwargs: Predictor inputs and optional per-call LM config.

        Returns:
            The original kwargs unless serial execution is enabled; otherwise
            a shallow copy whose normal-turn config disables parallel calls.
        """
        if not self._serial_tool_calls:
            return kwargs
        config = dict(kwargs.get("config") or {})
        if "tool_choice" in config:
            config.pop("parallel_tool_calls", None)
        else:
            config["parallel_tool_calls"] = False
        return {**kwargs, "config": config}

    def _with_resample_id(self, kwargs: dict, attempt: int) -> dict:
        """Return ``kwargs`` with a per-attempt ``rollout_id`` to bust the cache.

        Args:
            kwargs: The original predictor call kwargs.
            attempt: 1-based retry index used to derive a distinct cache key.

        Returns:
            A shallow copy of ``kwargs`` whose ``config`` carries a fresh
            ``rollout_id`` (any caller-supplied id is kept as the offset base).
        """
        config = dict(kwargs.get("config") or {})
        config["rollout_id"] = (
            config.get("rollout_id") or 0
        ) + attempt * _RESAMPLE_STRIDE
        return {**kwargs, "config": config}


class RetryingReActV2(REACT_CLASS):
    """A ReAct program whose inner ``react`` predictor resamples on parse failures.

    Subclasses whichever base the installed DSPy provides — ``ReActV2`` on 3.3+
    or classic ``ReAct`` on 3.2.x. Both expose the same ``react`` loop predictor,
    so the resample swap is identical across versions; classic ReAct's extra
    ``extract`` predictor is left untouched (parse failures only strike the loop).
    """

    def __init__(
        self,
        signature,
        tools,
        max_iters: int = 20,
        *,
        parse_retries: int = PARSE_RETRY_ATTEMPTS,
        serial_tool_calls: bool = False,
    ):
        """Build a stock ReAct program then swap its inner predictor for a retrying one.

        The replacement keeps the SAME signature, so GEPA's named-predictor
        optimization and the ``load_state`` round-trip still see a plain
        ``react`` predictor — only its turn call gains the resample wrapper.

        Args:
            signature: Task signature, as for the base ReAct program.
            tools: Tool roster, as for the base ReAct program.
            max_iters: Loop budget, as for the base ReAct program.
            parse_retries: Extra resample attempts per inner-predict call.
            serial_tool_calls: Disable parallel tool calls on normal loop
                turns while preserving the forced-submit compatibility path.
        """
        super().__init__(signature, tools, max_iters=max_iters)
        self.react = RetryingPredict(
            self.react.signature,
            parse_retries=parse_retries,
            serial_tool_calls=serial_tool_calls,
        )


__all__ = ["PARSE_RETRY_ATTEMPTS", "RetryingPredict", "RetryingReActV2"]
