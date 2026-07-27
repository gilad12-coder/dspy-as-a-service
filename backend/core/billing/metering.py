"""Per-run LLM usage metering for the interactive surfaces.

Optimization jobs already debit through the worker at completion; this module
gives the interactive surfaces — agent turns, interview turns, tagging
predictions — the same seam. A surface hands over the ``MeteredLM`` objects a
run used; the helper harvests their per-model token usage, prices it, and
writes the debit (with measured token counts stamped on the ledger row), all
best-effort: a metering failure logs and never breaks the user-facing turn.
"""

from __future__ import annotations

import logging

from ..service_gateway.language_models import served_model_from, usage_by_model_from_history
from .pricing import usages_from_breakdown
from .service import StripeBillingService

logger = logging.getLogger("skynet.billing.metering")

_PROXY_PREFIX = "litellm_proxy/"


def _normalize_model_key(model: str) -> str:
    """Strip the managed-gateway transport prefix off a usage-breakdown key.

    Args:
        model: Model id as the LM reports it (``litellm_proxy/openrouter/…``
            behind the proxy, ``openrouter/…`` on the direct path).

    Returns:
        The catalog-shaped id the ledger and pricing table understand.
    """
    return model.removeprefix(_PROXY_PREFIX)


def meter_llm_run(
    engine,
    username: str,
    language_models,
    *,
    description: str,
    model: str | None = None,
) -> int:
    """Debit the tokens a finished interactive run consumed.

    Harvests per-model usage from ``language_models``, keys an auto-routed
    run's usage by the concrete model the router served (so pricing hits the
    real price row instead of the router's unpriced group id), and debits the
    account. A run with no tracked usage (mocked LM, zero calls) writes
    nothing. Never raises — the turn already succeeded for the user, so
    billing failures are logged and swallowed rather than surfaced as an
    error on a delivered reply.

    Args:
        engine: SQLAlchemy engine backing the billing tables; ``None`` skips
            metering entirely (legacy/in-memory stores).
        username: Account the run is billed to.
        language_models: The run's LM objects (a single LM or an iterable).
        description: Human label for the ledger row (e.g. ``"Agent chat"``).
        model: Model id to stamp on the ledger row; ``None`` derives it from
            the served/requested model of the harvest.

    Returns:
        The credits charged, or ``0`` when nothing was billed.
    """
    if engine is None or not username:
        return 0
    lms = list(language_models) if isinstance(language_models, (list, tuple)) else [language_models]
    lms = [lm for lm in lms if lm is not None]
    if not lms:
        return 0
    try:
        breakdown = usage_by_model_from_history(*lms)
        if not breakdown:
            return 0
        served = served_model_from(lms[-1]) if len(lms) == 1 else None
        rekeyed: dict[str, tuple[int, int]] = {}
        for key, in_out in breakdown.items():
            normalized = _normalize_model_key(key)
            # An auto-routed turn's usage is keyed by the router's own id;
            # the concrete pick is what the price table knows.
            if served and len(breakdown) == 1:
                normalized = served
            prior = rekeyed.get(normalized, (0, 0))
            rekeyed[normalized] = (prior[0] + in_out[0], prior[1] + in_out[1])
        ledger_model = model or next(iter(rekeyed))
        usages = usages_from_breakdown(rekeyed)
        service = StripeBillingService(engine=engine)
        credits = service.debit_run(username, usages, model=ledger_model, description=description)
    except Exception:
        logger.exception("failed to debit LLM usage for %s (%s)", username, description)
        return 0
    return credits
