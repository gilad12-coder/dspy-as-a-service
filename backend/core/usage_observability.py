"""Collect interactive LLM token usage for local on-premise observability."""

from __future__ import annotations

import logging
from typing import Any

from .service_gateway.language_models import served_model_from, usage_by_model_from_history
from .telemetry import record_server_event

logger = logging.getLogger("skynet.usage")

_PROXY_PREFIX = "litellm_proxy/"


def _coerce_lms(language_models: Any) -> list[Any]:
    """Return non-null language models from a single object or sequence.

    Args:
        language_models: One LM object or a list or tuple of LM objects.

    Returns:
        Non-null LM objects in input order.
    """
    values = (
        list(language_models)
        if isinstance(language_models, (list, tuple))
        else [language_models]
    )
    return [value for value in values if value is not None]


def collect_llm_usage(language_models: Any) -> dict[str, tuple[int, int]]:
    """Collect model-level input and output token counts.

    Args:
        language_models: One LM object or a collection used by the operation.

    Returns:
        Catalog-shaped model identifiers mapped to input and output tokens.
    """
    lms = _coerce_lms(language_models)
    if not lms:
        return {}
    breakdown = usage_by_model_from_history(*lms)
    served = served_model_from(lms[-1]) if len(lms) == 1 else None
    normalized: dict[str, tuple[int, int]] = {}
    for model, counts in breakdown.items():
        key = served if served and len(breakdown) == 1 else model.removeprefix(_PROXY_PREFIX)
        prior = normalized.get(key, (0, 0))
        normalized[key] = (prior[0] + counts[0], prior[1] + counts[1])
    return normalized


def record_llm_usage(
    engine: Any,
    username: str,
    language_models: Any,
    *,
    description: str,
) -> None:
    """Write token use to structured logs and local product telemetry.

    Args:
        engine: SQLAlchemy engine backing the local telemetry table.
        username: Authenticated identity that initiated the operation.
        language_models: LM objects whose histories carry token usage.
        description: Stable human-readable operation label.
    """
    try:
        breakdown = collect_llm_usage(language_models)
    except Exception:
        logger.exception("failed to collect LLM usage for user=%s surface=%s", username, description)
        return
    if not breakdown:
        return
    input_tokens = sum(counts[0] for counts in breakdown.values())
    output_tokens = sum(counts[1] for counts in breakdown.values())
    models = [
        {"model": model, "input_tokens": counts[0], "output_tokens": counts[1]}
        for model, counts in sorted(breakdown.items())
    ]
    logger.info(
        "llm_usage user=%s surface=%s input_tokens=%d output_tokens=%d models=%s",
        username,
        description,
        input_tokens,
        output_tokens,
        models,
    )
    record_server_event(
        engine,
        username=username or None,
        name="llm_usage_recorded",
        properties={
            "surface": description,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "models": models,
        },
    )
