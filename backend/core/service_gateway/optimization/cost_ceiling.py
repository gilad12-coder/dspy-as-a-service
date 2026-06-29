"""Per-job cost ceiling: hard-stop a run once its credit spend exceeds the cap.

A DSPy optimizer's token use is not linear — bootstrapping, compile steps, and
validation loops make a tight pre-run estimate dishonest. The wizard therefore
shows a projected bracket and lets the user set a Max Cost Ceiling in credits;
:class:`CostCeilingCallback` enforces that ceiling at runtime. Registered as a
``dspy`` callback alongside the timing callbacks, it re-prices the run's
accumulated per-model usage after every LM call (across the generation and
reflection LMs, on whichever worker thread the call lands) and raises
:class:`CostCeilingExceededError` the moment the run's full per-model credit cost
crosses the cap. Pricing per-model — not a flat token budget — keeps the cap
honest now that a credit means real provider cost: the same N-credit cap stops a
frontier run far sooner than a mini one. The raise unwinds out of ``service.run``
and the subprocess reports it as the job's terminal error, so the run fails (and
is never billed — debiting only happens on success) instead of silently burning
past the user's cap.
"""

from __future__ import annotations

import threading
from typing import Any

from dspy.utils.callback import BaseCallback

from ...billing.pricing import credits_for_usage, usages_from_breakdown
from ..language_models import usage_by_model_from_history


class CostCeilingExceededError(RuntimeError):
    """Raised inside a run when accumulated token usage exceeds the cost ceiling.

    Subclasses ``RuntimeError`` so the worker's generic failure handler marks the
    job ``failed`` with this message, while tests can assert on the specific type.
    """


class CostCeilingCallback(BaseCallback):
    """Hard-stop a run once its full per-model credit cost exceeds a credit cap.

    Holds the LMs whose ``history`` carries the run's token usage and a
    ``max_credits`` cap (the user's Max Cost Ceiling, in full-cost credits). After
    each LM call completes, it re-prices accumulated per-model usage and raises
    :class:`CostCeilingExceededError` once the full per-model cost exceeds the cap
    — at the next ``on_lm_end`` boundary rather than mid-call, so the just-finished
    call's usage is already counted. The check is cheap (a per-model price lookup
    over history that is read anyway for billing) and runs under a lock so DSPy's
    ``Evaluate`` worker threads can't race the tripped flag.
    """

    def __init__(self, max_credits: int, *language_models: Any) -> None:
        """Bind the ceiling to the run's LMs and full-cost credit cap.

        Args:
            max_credits: The user's Max Cost Ceiling in full-cost credits; the run
                is stopped once its per-model cost exceeds it. A non-positive cap
                makes the callback inert (it never trips).
            *language_models: The LMs whose ``history`` usage counts toward the
                ceiling — typically the generation LM and, when present, the
                reflection LM. ``None`` entries are tolerated.
        """
        self._max_credits = max_credits
        self._language_models = [lm for lm in language_models if lm is not None]
        self._lock = threading.Lock()
        self._tripped = False

    def _check(self) -> None:
        """Raise once the per-model cost exceeds the cap; latch so it raises once.

        Raises:
            CostCeilingExceededError: When the full per-model credit cost across the
                bound LMs exceeds ``max_credits``.
        """
        if self._max_credits <= 0:
            return
        breakdown = usage_by_model_from_history(*self._language_models)
        if breakdown is None:
            return
        used = credits_for_usage(usages_from_breakdown(breakdown))
        if used <= self._max_credits:
            return
        with self._lock:
            if self._tripped:
                return
            self._tripped = True
        raise CostCeilingExceededError(
            f"Run stopped at the cost ceiling: {used} credits used exceeds the "
            f"{self._max_credits}-credit Max Cost Ceiling."
        )

    def on_lm_end(
        self,
        call_id: str,
        outputs: dict[str, Any] | None,
        exception: Exception | None = None,
    ) -> None:
        """Re-check the ceiling after each completed LM call.

        Args:
            call_id: DSPy's id for the call (unused — usage is read from history).
            outputs: The LM's response payload, or ``None`` on error.
            exception: Exception raised by the LM, if any. A failed call still
                triggers the check so a run can't slip past the cap on errors.
        """
        self._check()
