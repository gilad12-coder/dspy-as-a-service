"""Per-model, per-token run pricing — the shared basis for estimate and charge.

A run's credit cost is the real provider cost of its tokens (LiteLLM's
``model_cost`` registry, input and output priced separately) times the platform
:data:`MARKUP`, converted to credits at :data:`CREDIT_USD_VALUE`. The *same*
function prices a projected token volume (the pre-run estimate) and a measured
token volume (the post-run charge), so the two reconcile by construction — only
their inputs differ.

Model choice moves a run's price, and :data:`MARKUP` is the single re-priceable
margin lever (mirrored by the Stripe per-unit price) — currently 1.09, i.e.
break-even pass-through: payment-processing fees are covered, no profit margin. The module is
a leaf — it depends only on LiteLLM's static price table — so both the billing
service and any estimator can import it without cycles.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import litellm

# One credit is worth one US cent; the markup that protects margin lives in
# MARKUP (and the Stripe per-unit price), so credit counts stay legible.
CREDIT_USD_VALUE = 0.01

# Margin multiplier applied to raw provider cost before converting to credits.
# This is the single re-priceable margin lever for runs (it replaces the markup
# that used to be implicit in the flat token→credit rate). Tune here; keep it in
# step with the Stripe metered per-unit price so ledger and invoice agree.
# 1.09 = break-even pricing: covers the percentage costs of moving the money —
# OpenRouter's ~5.5% deposit fee on managed tokens and Stripe's ~2.9% cut of
# what users pay (1.055 / 0.971 ≈ 1.09) — with zero profit margin on top.
# Stripe's fixed $0.30 per purchase is deliberately not covered (it amortizes
# to noise on normal pack sizes).
MARKUP = 1.09

# Fallback per-token cost (USD) for a model LiteLLM does not price — a mid-tier
# standard rate so an unknown model estimates and charges sanely rather than at
# zero (which would give it away) or at a frontier rate (which would scare).
DEFAULT_INPUT_COST_PER_TOKEN = 1e-6
DEFAULT_OUTPUT_COST_PER_TOKEN = 3e-6


@dataclass(frozen=True)
class ModelUsage:
    """Token usage attributed to one model in a run — measured or projected.

    ``input_tokens`` and ``output_tokens`` are priced separately because output
    typically costs several times input. For a projected (estimate) usage these
    are forecasts; for a charge they are the measured prompt/completion totals.
    """

    model: str
    input_tokens: int
    output_tokens: int


def model_token_costs(model_id: str) -> tuple[float, float]:
    """Return a model's ``(input, output)`` per-token cost in USD.

    Looks the id up in LiteLLM's ``model_cost`` table, retrying without the
    provider prefix (``openai/gpt-4o-mini`` → ``gpt-4o-mini``) since the table
    keys both shapes inconsistently. A model the table doesn't price — or prices
    with a missing/zero field — falls back to the module defaults so it is never
    silently free.

    Args:
        model_id: Fully-qualified or bare model id.

    Returns:
        The ``(input_cost_per_token, output_cost_per_token)`` pair in USD.
    """
    meta = litellm.model_cost.get(model_id)
    if meta is None and "/" in model_id:
        meta = litellm.model_cost.get(model_id.split("/", 1)[1])
    if not isinstance(meta, Mapping):
        return DEFAULT_INPUT_COST_PER_TOKEN, DEFAULT_OUTPUT_COST_PER_TOKEN
    in_cost = meta.get("input_cost_per_token")
    out_cost = meta.get("output_cost_per_token")
    return (
        float(in_cost) if isinstance(in_cost, (int, float)) and in_cost > 0 else DEFAULT_INPUT_COST_PER_TOKEN,
        float(out_cost) if isinstance(out_cost, (int, float)) and out_cost > 0 else DEFAULT_OUTPUT_COST_PER_TOKEN,
    )


def raw_cost_usd(usages: Iterable[ModelUsage]) -> float:
    """Sum the raw provider cost (USD, pre-markup) of per-model token usage.

    Args:
        usages: Per-model input/output token counts.

    Returns:
        The total un-marked-up provider cost in USD.
    """
    total = 0.0
    for usage in usages:
        in_cost, out_cost = model_token_costs(usage.model)
        total += usage.input_tokens * in_cost + usage.output_tokens * out_cost
    return total


def credits_for_usage(usages: Iterable[ModelUsage]) -> int:
    """Convert per-model token usage to the credits it costs, rounding up.

    Applies :data:`MARKUP` to the raw provider cost and divides by
    :data:`CREDIT_USD_VALUE`. Any non-zero usage costs at least one credit (a
    partial credit rounds up), so a run that consumed tokens is never billed
    zero; zero usage costs zero.

    Args:
        usages: Per-model input/output token counts (measured or projected).

    Returns:
        The non-negative credit cost.
    """
    cost = raw_cost_usd(usages) * MARKUP
    if cost <= 0:
        return 0
    return max(1, math.ceil(cost / CREDIT_USD_VALUE))


def usages_from_breakdown(breakdown: Mapping[str, tuple[int, int]]) -> list[ModelUsage]:
    """Build :class:`ModelUsage` rows from a ``model → (input, output)`` mapping.

    Adapts the billing-agnostic breakdown that
    :func:`core.service_gateway.language_models.usage_by_model_from_history`
    returns into the priced unit this module consumes.

    Args:
        breakdown: Per-model ``(input_tokens, output_tokens)`` pairs.

    Returns:
        One :class:`ModelUsage` per model.
    """
    return [
        ModelUsage(model=model, input_tokens=in_out[0], output_tokens=in_out[1]) for model, in_out in breakdown.items()
    ]
