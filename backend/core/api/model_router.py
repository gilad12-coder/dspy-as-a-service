"""Auto-mode model routing for the generalist agent.

The composer's default "Auto" runs the pinned default model
(``BALANCED_PINNED_MODEL_ID``); the "intelligent" tier delegates per-turn
model choice to OpenRouter's Auto Router Beta (model id
``openrouter/auto-beta``, docs:
openrouter.ai/docs/guides/routing/routers/auto-router) with its
``cost_quality_tradeoff`` dial pinned to pure quality, mirroring Cursor
Router's Intelligence mode.

Deployments without OpenRouter connectivity (air-gapped gateways) degrade
to the configured server default for both tiers.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from ..config import settings
from ..models import ModelConfig
from .model_catalog import get_catalog_cached, require_known_model

logger = logging.getLogger(__name__)

AutoTier = Literal["balanced", "intelligent"]

# Sentinel the composer sends instead of a catalog id; never billed directly.
AUTO_INTELLIGENT_ID = "auto:intelligent"

# LiteLLM id of OpenRouter's Auto Router Beta: the ``openrouter/`` provider
# prefix plus their ``openrouter/auto-beta`` model id.
OPENROUTER_AUTO_ID = "openrouter/openrouter/auto-beta"

# The default ("balanced") model, pinned per the 2026-08 nine-model eval on
# sanitized production cases (two rounds). gpt-5.6-luna scored higher
# (16/20 vs Terra's 14/20 judged passes) but Terra was kept deliberately:
# fastest (3.1s vs 7.1s avg) and cheapest of the top scorers, and the only
# candidate whose routing variants were also validated (bare beat every
# variant). Re-measure before changing — the eval is re-runnable.
BALANCED_PINNED_MODEL_ID = "openrouter/openai/gpt-5.6-terra"

# OpenRouter's dial: 0 = pure quality, 10 = cheapest wins (their default is
# 9). Only the intelligent tier rides the router now; balanced runs the
# pinned default above.
_INTELLIGENT_DIAL = 0


def resolve_auto_tier(model: str | None) -> tuple[str | None, AutoTier | None]:
    """Split the composer's model field into a catalog id or an auto tier.

    Args:
        model: Raw ``model`` value from the request — a catalog id, the
            ``AUTO_INTELLIGENT_ID`` sentinel, empty, or ``None``.

    Returns:
        ``(model_id, tier)`` — exactly one side is set: a non-empty catalog
        id with ``tier=None``, or ``model_id=None`` with the auto tier to
        route with.
    """
    name = str(model or "").strip()
    if not name:
        return None, "balanced"
    if name == AUTO_INTELLIGENT_ID:
        return None, "intelligent"
    return name, None


def _openrouter_reachable() -> bool:
    """Report whether the live catalog proves OpenRouter connectivity.

    ``openrouter/auto-beta`` is a router, not a chat model, so it never
    appears in the probed catalog itself — any ``openrouter/`` entry proves
    the credentials and connectivity it needs.

    Returns:
        True when at least one catalog model is OpenRouter-hosted.
    """
    try:
        models = get_catalog_cached().models
    except Exception:
        logger.warning("Model catalog unavailable for auto-routing; using server default")
        return False
    return any(entry.value.startswith("openrouter/") for entry in models)


def route_auto_model(tier: AutoTier, conversation_id: str | None = None) -> ModelConfig:
    """Build the model config an Auto turn should run on.

    Args:
        tier: ``"balanced"`` runs the pinned default model;
            ``"intelligent"`` rides the Auto Router at pure quality.
        conversation_id: Persisted conversation id, when known. On the
            router path it is forwarded as the ``session_id`` so model
            selection sticks across the turns of one conversation instead
            of flip-flopping; the pinned path doesn't need it.

    Returns:
        A :class:`ModelConfig` running the pinned default (balanced) or
        OpenRouter's Auto Router (intelligent), or the configured server
        default when OpenRouter isn't reachable.
    """
    if not _openrouter_reachable():
        return ModelConfig(name=settings.generalist_agent_model)
    if tier == "balanced":
        return ModelConfig(name=BALANCED_PINNED_MODEL_ID)
    body: dict[str, Any] = {
        "plugins": [{"id": "auto-router", "cost_quality_tradeoff": _INTELLIGENT_DIAL}],
    }
    if conversation_id:
        body["session_id"] = conversation_id
    return ModelConfig(name=OPENROUTER_AUTO_ID, extra={"extra_body": body})


def route_menu_model(
    model: str | None, session_id: str | None = None
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve the composer menu's model field for engines that take a bare id.

    The generalist agent threads a full :class:`ModelConfig`; the interview
    engines (code interview, tagger interview) instead take a plain model id
    plus optional LiteLLM extras. This translates the menu's three shapes —
    catalog id, absent (Auto balanced), the ``AUTO_INTELLIGENT_ID`` sentinel —
    into that calling convention with the same auto-router behaviour as the
    agent.

    Args:
        model: Raw ``model`` value from the request.
        session_id: Stable id forwarded as the router's ``session_id`` so an
            auto-routed flow sticks to one model across its turns.

    Returns:
        ``(model_id, lm_extra_body)`` — a validated catalog id with no extras,
        the pinned default with no extras, the auto router's id with its
        plugin dial, or ``(None, None)`` when OpenRouter is unreachable (the
        engine's configured default runs).

    Raises:
        DomainError: 422 when an explicit id is not a catalog model.
    """
    requested, tier = resolve_auto_tier(model)
    if requested:
        require_known_model(requested)
        return requested, None
    routed = route_auto_model(tier or "balanced", session_id)
    body = routed.extra.get("extra_body") if routed.extra else None
    if body is not None:
        return routed.name, dict(body)
    if routed.name == BALANCED_PINNED_MODEL_ID:
        return routed.name, None
    # Degraded (no OpenRouter): plain name but not the pin — let the
    # engine's own configured default run, matching pre-pin behaviour.
    return None, None
