"""Tests for the per-job cost ceiling callback.

The callback re-prices accumulated per-model LM usage after each call and
hard-stops the run once its full per-model credit cost exceeds the cap. These
exercise the trip boundary, the no-ceiling inert case, multi-LM totalling, and
the latch — with budgets set relative to the pricing engine so the assertions
don't hinge on exact per-token arithmetic.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.billing.pricing import credits_for_usage, usages_from_breakdown
from core.service_gateway.language_models import usage_by_model_from_history
from core.service_gateway.optimization.cost_ceiling import (
    CostCeilingCallback,
    CostCeilingExceededError,
)
from core.service_gateway.optimization.timing import GenLMTimingCallback, track_stage


class _FakeLM:
    """Minimal stand-in exposing a growing ``history`` and a ``model`` like ``dspy.LM``."""

    def __init__(self, model: str = "test/ceiling") -> None:
        """Start with an empty call history on an unpriced model (default costs).

        Args:
            model: Model id the LM reports; the default is unpriced so per-token
                costs fall back to the engine defaults, keeping pricing stable.
        """
        self.history: list[dict[str, Any]] = []
        self.model = model

    def record(self, input_tokens: int) -> None:
        """Append one history entry reporting ``input_tokens`` of prompt usage.

        Args:
            input_tokens: Prompt-token usage to stamp on the new history row.
        """
        self.history.append({"usage": {"prompt_tokens": input_tokens, "completion_tokens": 0}})


def _credits(*lms: _FakeLM) -> int:
    """Return the per-model credit cost of the LMs' accumulated usage."""
    breakdown = usage_by_model_from_history(*lms)
    return credits_for_usage(usages_from_breakdown(breakdown)) if breakdown else 0


def test_does_not_trip_while_usage_within_budget() -> None:
    """Usage that prices below the credit cap never raises."""
    lm = _FakeLM()
    cb = CostCeilingCallback(1000, lm)
    lm.record(50_000)
    cb.on_lm_end("c1", outputs={})
    lm.record(50_000)
    cb.on_lm_end("c2", outputs={})  # ~15 credits, well under the 1000-credit cap


def test_trips_once_cost_exceeds_budget() -> None:
    """The first call that prices the run past the credit cap raises."""
    first_chunk = _FakeLM()
    first_chunk.record(50_000)
    budget = _credits(first_chunk)  # cap == the cost of the first chunk alone
    assert budget > 0

    lm = _FakeLM()
    cb = CostCeilingCallback(budget, lm)
    lm.record(50_000)
    cb.on_lm_end("c1", outputs={})  # used == cap → no trip
    lm.record(50_000)
    with pytest.raises(CostCeilingExceededError):
        cb.on_lm_end("c2", outputs={})  # now over the cap


def test_latches_after_tripping() -> None:
    """Once tripped the callback stays tripped but raises only on the first cross."""
    lm = _FakeLM()
    cb = CostCeilingCallback(5, lm)
    lm.record(100_000)  # ~15 credits > 5
    with pytest.raises(CostCeilingExceededError):
        cb.on_lm_end("c1", outputs={})
    # A later boundary does not re-raise: the run is already unwinding.
    cb.on_lm_end("c2", outputs={})


def test_totals_cost_across_generation_and_reflection() -> None:
    """The ceiling prices usage across every bound LM, not just the generation LM."""
    gen = _FakeLM()
    refl = _FakeLM()
    gen.record(50_000)
    refl.record(50_000)
    combined = _credits(gen, refl)
    # Each LM alone prices below the cap; only their summed cost trips it.
    assert _credits(gen) < combined
    cb = CostCeilingCallback(combined - 1, gen, refl)
    with pytest.raises(CostCeilingExceededError):
        cb.on_lm_end("c1", outputs={})


def test_none_lm_is_tolerated() -> None:
    """A ``None`` LM (no reflection model) is skipped, not an error."""
    gen = _FakeLM()
    cb = CostCeilingCallback(5, gen, None)
    gen.record(100_000)  # ~15 credits > 5
    with pytest.raises(CostCeilingExceededError):
        cb.on_lm_end("c1", outputs={})


def test_non_positive_budget_is_inert() -> None:
    """A zero/negative cap disables the ceiling entirely."""
    lm = _FakeLM()
    cb = CostCeilingCallback(0, lm)
    lm.record(10_000_000)
    cb.on_lm_end("c1", outputs={})


def test_no_usage_does_not_trip() -> None:
    """Untracked usage (no history rows) reads as 'not tracked' and never trips."""
    lm = _FakeLM()
    cb = CostCeilingCallback(100, lm)
    cb.on_lm_end("c1", outputs={})


def test_cost_ceiling_callback_stays_out_of_stage_tracking() -> None:
    """Regression: the ceiling callback must not be splatted into ``track_stage``.

    It is a plain ``on_lm_end`` listener with no per-stage state, so it rides the
    dspy callbacks list (which it shares with the timing callbacks) but must be kept
    out of ``track_stage`` — which reads ``_current_stage``/``set_stage`` on each
    callback it is given. Passing the full list crashed with
    ``'CostCeilingCallback' object has no attribute '_current_stage'``; core.py now
    keeps a separate ``timing_callbacks`` list for stage tracking. This mirrors that
    split and pins both halves of the contract.
    """
    gen_lm = _FakeLM()
    gen_timing = GenLMTimingCallback(gen_lm)
    timing_callbacks = [gen_timing]
    callbacks = [*timing_callbacks, CostCeilingCallback(100, gen_lm)]

    # The ceiling belongs in the dspy callbacks list, but only timing callbacks may
    # drive stage tracking — which sets and restores the stage on them.
    assert any(isinstance(cb, CostCeilingCallback) for cb in callbacks)
    with track_stage("baseline", *timing_callbacks):
        assert gen_timing._current_stage == "baseline"
    assert gen_timing._current_stage is None

    # Splatting the full list (the original bug) is what raised AttributeError.
    with pytest.raises(AttributeError), track_stage("baseline", *callbacks):
        pass
