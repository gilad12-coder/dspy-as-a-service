"""Tests for ``core.billing.pricing`` — per-model, per-token run pricing."""

from __future__ import annotations

import math
from unittest.mock import patch

import litellm

from core.billing.pricing import (
    CREDIT_USD_VALUE,
    DEFAULT_INPUT_COST_PER_TOKEN,
    DEFAULT_OUTPUT_COST_PER_TOKEN,
    MARKUP,
    ModelUsage,
    credits_for_usage,
    model_token_costs,
    raw_cost_usd,
    usages_from_breakdown,
)

# Controlled price table so assertions don't drift when LiteLLM updates its real
# numbers. ``test/bare`` is keyed without a provider prefix to exercise the
# prefix-stripping fallback in ``model_token_costs``.
_FAKE_COSTS = {
    "test/cheap": {"input_cost_per_token": 1e-7, "output_cost_per_token": 4e-7, "mode": "chat"},
    "test/frontier": {"input_cost_per_token": 5e-6, "output_cost_per_token": 3e-5, "mode": "chat"},
    "bare": {"input_cost_per_token": 2e-7, "output_cost_per_token": 6e-7, "mode": "chat"},
    "test/zerofield": {"input_cost_per_token": 0, "output_cost_per_token": 6e-7, "mode": "chat"},
}


def _expected_credits(raw_usd: float) -> int:
    """Mirror ``credits_for_usage`` arithmetic so tests track a tuned MARKUP."""
    cost = raw_usd * MARKUP
    return 0 if cost <= 0 else max(1, math.ceil(cost / CREDIT_USD_VALUE))


def test_model_token_costs_reads_known_model() -> None:
    """A priced model returns its LiteLLM input/output per-token costs."""
    with patch.dict(litellm.model_cost, _FAKE_COSTS, clear=False):
        assert model_token_costs("test/cheap") == (1e-7, 4e-7)


def test_model_token_costs_strips_prefix_fallback() -> None:
    """A prefixed id resolves against a bare-keyed table entry."""
    with patch.dict(litellm.model_cost, _FAKE_COSTS, clear=False):
        assert model_token_costs("someprovider/bare") == (2e-7, 6e-7)


def test_model_token_costs_unknown_model_uses_defaults() -> None:
    """An unpriced model falls back to the module default per-token costs."""
    with patch.dict(litellm.model_cost, _FAKE_COSTS, clear=False):
        assert model_token_costs("nope/not-real") == (
            DEFAULT_INPUT_COST_PER_TOKEN,
            DEFAULT_OUTPUT_COST_PER_TOKEN,
        )


def test_model_token_costs_missing_field_defaults_that_side_only() -> None:
    """A zero/absent cost field falls back for that side, keeping the other."""
    with patch.dict(litellm.model_cost, _FAKE_COSTS, clear=False):
        assert model_token_costs("test/zerofield") == (DEFAULT_INPUT_COST_PER_TOKEN, 6e-7)


def test_credits_for_usage_prices_input_and_output_separately() -> None:
    """A single-model run is priced from its split token volume, marked up."""
    with patch.dict(litellm.model_cost, _FAKE_COSTS, clear=False):
        usage = [ModelUsage("test/cheap", input_tokens=1_000_000, output_tokens=1_000_000)]
        raw = 1_000_000 * 1e-7 + 1_000_000 * 4e-7  # 0.5 USD
        assert raw_cost_usd(usage) == raw
        assert credits_for_usage(usage) == _expected_credits(raw)


def test_credits_for_usage_aggregates_multiple_models() -> None:
    """A run spanning two models sums each model's marked-up cost."""
    with patch.dict(litellm.model_cost, _FAKE_COSTS, clear=False):
        usage = [
            ModelUsage("test/cheap", input_tokens=500_000, output_tokens=200_000),
            ModelUsage("test/frontier", input_tokens=100_000, output_tokens=50_000),
        ]
        raw = (500_000 * 1e-7 + 200_000 * 4e-7) + (100_000 * 5e-6 + 50_000 * 3e-5)
        assert credits_for_usage(usage) == _expected_credits(raw)


def test_credits_for_usage_any_usage_costs_at_least_one_credit() -> None:
    """A sliver of usage rounds up to one credit, never billed zero."""
    with patch.dict(litellm.model_cost, _FAKE_COSTS, clear=False):
        assert credits_for_usage([ModelUsage("test/cheap", input_tokens=1, output_tokens=0)]) == 1


def test_credits_for_usage_zero_usage_is_zero() -> None:
    """No tokens cost no credits."""
    assert credits_for_usage([]) == 0
    with patch.dict(litellm.model_cost, _FAKE_COSTS, clear=False):
        assert credits_for_usage([ModelUsage("test/cheap", input_tokens=0, output_tokens=0)]) == 0


def test_frontier_costs_more_than_mini_for_same_volume() -> None:
    """The whole point: identical token volume prices higher on a frontier model."""
    with patch.dict(litellm.model_cost, _FAKE_COSTS, clear=False):
        mini = credits_for_usage([ModelUsage("test/cheap", 200_000, 200_000)])
        frontier = credits_for_usage([ModelUsage("test/frontier", 200_000, 200_000)])
        assert frontier > mini


def test_usages_from_breakdown_builds_model_usage_rows() -> None:
    """A ``model → (input, output)`` mapping becomes ModelUsage rows."""
    rows = usages_from_breakdown({"a/m": (10, 20), "b/n": (30, 40)})
    assert ModelUsage("a/m", 10, 20) in rows
    assert ModelUsage("b/n", 30, 40) in rows
    assert len(rows) == 2
