"""Tests for the per-job cost ceiling callback.

The callback re-reads accumulated LM-history token usage after each call and
hard-stops the run once it exceeds the credit-derived budget. These exercise the
trip boundary, the no-ceiling inert case, multi-LM totalling, and the
credit-to-token budget mapping the ceiling is built from.
"""

from __future__ import annotations

from typing import Any

import pytest

from core.billing import tokens_for_credits
from core.service_gateway.optimization.cost_ceiling import (
    CostCeilingCallback,
    CostCeilingExceededError,
)


class _FakeLM:
    """Minimal stand-in exposing a growing ``history`` list like ``dspy.LM``."""

    def __init__(self) -> None:
        """Start with an empty call history."""
        self.history: list[dict[str, Any]] = []

    def record(self, total_tokens: int) -> None:
        """Append one history entry reporting ``total_tokens`` of usage.

        Args:
            total_tokens: Token usage to stamp on the new history row.
        """
        self.history.append({"usage": {"total_tokens": total_tokens}})


def test_does_not_trip_while_usage_within_budget() -> None:
    """Usage at or below the budget never raises."""
    lm = _FakeLM()
    cb = CostCeilingCallback(100, lm)
    lm.record(60)
    cb.on_lm_end("c1", outputs={})
    lm.record(40)
    cb.on_lm_end("c2", outputs={})


def test_trips_once_usage_exceeds_budget() -> None:
    """The first call that pushes usage past the budget raises."""
    lm = _FakeLM()
    cb = CostCeilingCallback(100, lm)
    lm.record(80)
    cb.on_lm_end("c1", outputs={})
    lm.record(40)
    with pytest.raises(CostCeilingExceededError):
        cb.on_lm_end("c2", outputs={})


def test_latches_after_tripping() -> None:
    """Once tripped the callback stays tripped but raises only on the first cross."""
    lm = _FakeLM()
    cb = CostCeilingCallback(50, lm)
    lm.record(120)
    with pytest.raises(CostCeilingExceededError):
        cb.on_lm_end("c1", outputs={})
    # A later boundary does not re-raise: the run is already unwinding.
    cb.on_lm_end("c2", outputs={})


def test_totals_usage_across_generation_and_reflection() -> None:
    """The ceiling sums usage across every bound LM, not just the generation LM."""
    gen = _FakeLM()
    refl = _FakeLM()
    cb = CostCeilingCallback(100, gen, refl)
    gen.record(60)
    refl.record(60)
    with pytest.raises(CostCeilingExceededError):
        cb.on_lm_end("c1", outputs={})


def test_none_lm_is_tolerated() -> None:
    """A ``None`` LM (no reflection model) is skipped, not an error."""
    gen = _FakeLM()
    cb = CostCeilingCallback(100, gen, None)
    gen.record(150)
    with pytest.raises(CostCeilingExceededError):
        cb.on_lm_end("c1", outputs={})


def test_non_positive_budget_is_inert() -> None:
    """A zero/negative budget disables the ceiling entirely."""
    lm = _FakeLM()
    cb = CostCeilingCallback(0, lm)
    lm.record(10_000)
    cb.on_lm_end("c1", outputs={})


def test_no_usage_does_not_trip() -> None:
    """Untracked usage (no history rows) reads as 'not tracked' and never trips."""
    lm = _FakeLM()
    cb = CostCeilingCallback(100, lm)
    cb.on_lm_end("c1", outputs={})


def test_tokens_for_credits_maps_cap_to_budget() -> None:
    """The credit cap the ceiling is built from buys ``cap * TOKENS_PER_CREDIT``."""
    assert tokens_for_credits(54) == 54_000
    assert tokens_for_credits(0) == 0
    assert tokens_for_credits(-5) == 0
