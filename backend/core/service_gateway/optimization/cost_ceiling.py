"""Per-job cost ceiling: hard-stop a run once its token spend exceeds the cap.

A DSPy optimizer's token use is not linear — bootstrapping, compile steps, and
validation loops make a tight pre-run estimate dishonest. The wizard therefore
shows a projected bracket and lets the user set a Max Cost Ceiling in credits;
:class:`CostCeilingCallback` enforces that ceiling at runtime. Registered as a
``dspy`` callback alongside the timing callbacks, it re-reads the run's
accumulated token usage after every LM call (across the generation and
reflection LMs, on whichever worker thread the call lands) and raises
:class:`CostCeilingExceededError` the moment usage crosses the budget the cap
buys. The raise unwinds out of ``service.run`` and the subprocess reports it as
the job's terminal error, so the run fails (and is never billed — debiting only
happens on success) instead of silently burning past the user's cap.
"""

from __future__ import annotations

import threading
from typing import Any

from dspy.utils.callback import BaseCallback

from ..language_models import total_tokens_from_history


class CostCeilingExceededError(RuntimeError):
    """Raised inside a run when accumulated token usage exceeds the cost ceiling.

    Subclasses ``RuntimeError`` so the worker's generic failure handler marks the
    job ``failed`` with this message, while tests can assert on the specific type.
    """


class CostCeilingCallback(BaseCallback):
    """Hard-stop a run once its summed token usage exceeds a credit-derived budget.

    Holds the LMs whose ``history`` carries the run's token usage and a
    ``max_tokens`` budget (the user's credit cap translated through
    ``TOKENS_PER_CREDIT``). After each LM call completes, it totals usage across
    those LMs and raises :class:`CostCeilingExceededError` once the total exceeds
    the budget — at the next ``on_lm_end`` boundary rather than mid-call, so the
    just-finished call's usage is already counted. The check is cheap (a sum over
    the history that is read anyway for billing) and runs under a lock so DSPy's
    ``Evaluate`` worker threads can't race the tripped flag.
    """

    def __init__(self, max_tokens: int, *language_models: Any) -> None:
        """Bind the ceiling to the run's LMs and token budget.

        Args:
            max_tokens: The token budget the user's credit cap buys; the run is
                stopped once summed usage exceeds it. A non-positive budget makes
                the callback inert (it never trips).
            *language_models: The LMs whose ``history`` usage counts toward the
                ceiling — typically the generation LM and, when present, the
                reflection LM. ``None`` entries are tolerated.
        """
        self._max_tokens = max_tokens
        self._language_models = [lm for lm in language_models if lm is not None]
        self._lock = threading.Lock()
        self._tripped = False

    def _check(self) -> None:
        """Raise once accumulated usage exceeds the budget; latch so it raises once.

        Raises:
            CostCeilingExceededError: When summed token usage across the bound LMs
                exceeds ``max_tokens``.
        """
        if self._max_tokens <= 0:
            return
        used = total_tokens_from_history(*self._language_models)
        if used is None or used <= self._max_tokens:
            return
        with self._lock:
            if self._tripped:
                return
            self._tripped = True
        raise CostCeilingExceededError(
            f"Run stopped at the cost ceiling: {used} tokens used exceeds the "
            f"{self._max_tokens}-token budget set by the Max Cost Ceiling."
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
