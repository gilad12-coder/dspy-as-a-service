"""Unit tests for the generalist agent's auto-mode model routing."""

from __future__ import annotations

import pytest

from ...config import settings
from .. import model_catalog
from .. import model_router as mr
from ..errors import DomainError
from ..model_catalog import CatalogModel, ModelCatalogResponse
from ..model_router import (
    AUTO_INTELLIGENT_ID,
    BALANCED_PINNED_MODEL_ID,
    OPENROUTER_AUTO_ID,
    resolve_auto_tier,
    route_auto_model,
    route_menu_model,
)


def _catalog_with(*model_ids: str) -> ModelCatalogResponse:
    """Build a minimal catalog response listing the given model ids."""
    return ModelCatalogResponse(
        providers=[],
        models=[
            CatalogModel(value=mid, label=mid, provider="test", available=True)
            for mid in model_ids
        ],
    )


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        (None, (None, "balanced")),
        ("", (None, "balanced")),
        ("  ", (None, "balanced")),
        (AUTO_INTELLIGENT_ID, (None, "intelligent")),
        ("openai/gpt-5.5", ("openai/gpt-5.5", None)),
    ],
    ids=["none", "empty", "whitespace", "intelligent_sentinel", "explicit_id"],
)
def test_resolve_auto_tier(model: str | None, expected: tuple[str | None, str | None]) -> None:
    """Auto tiers come from absent/sentinel values; real ids pass through."""
    assert resolve_auto_tier(model) == expected


def test_balanced_runs_pinned_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Balanced Auto runs the eval-pinned default model with no extras."""
    monkeypatch.setattr(
        mr,
        "get_catalog_cached",
        lambda: _catalog_with("openrouter/anthropic/claude-sonnet-5", "openai/gpt-5.5"),
    )
    config = route_auto_model("balanced")
    assert config.name == BALANCED_PINNED_MODEL_ID
    assert config.extra == {}


def test_intelligent_runs_auto_router_pure_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    """The intelligent tier pins the router's dial to pure quality (0)."""
    monkeypatch.setattr(
        mr,
        "get_catalog_cached",
        lambda: _catalog_with("openrouter/anthropic/claude-sonnet-5"),
    )
    config = route_auto_model("intelligent")
    assert config.name == OPENROUTER_AUTO_ID
    assert config.extra == {
        "extra_body": {"plugins": [{"id": "auto-router", "cost_quality_tradeoff": 0}]}
    }


def test_conversation_id_becomes_sticky_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """A known conversation id rides along as the router's session_id."""
    monkeypatch.setattr(
        mr,
        "get_catalog_cached",
        lambda: _catalog_with("openrouter/anthropic/claude-sonnet-5"),
    )
    config = route_auto_model("intelligent", "conv-123")
    assert config.extra["extra_body"]["session_id"] == "conv-123"


def test_without_openrouter_uses_server_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """An air-gapped catalog (no openrouter/ ids) degrades to the default."""
    monkeypatch.setattr(mr, "get_catalog_cached", lambda: _catalog_with("openai/gpt-5.5"))
    for tier in ("balanced", "intelligent"):
        config = route_auto_model(tier)
        assert config.name == settings.generalist_agent_model
        assert config.extra == {}


def test_route_survives_catalog_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gateway outage routes to the server default instead of raising."""

    def _boom() -> ModelCatalogResponse:
        raise RuntimeError("gateway down")

    monkeypatch.setattr(mr, "get_catalog_cached", _boom)
    assert route_auto_model("balanced").name == settings.generalist_agent_model
    assert route_auto_model("intelligent").name == settings.generalist_agent_model


def test_route_menu_model_passes_catalog_id_through(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit catalog id is validated and returned with no extras."""
    monkeypatch.setattr(
        model_catalog, "get_catalog_cached", lambda: _catalog_with("openai/gpt-5.5")
    )
    assert route_menu_model("openai/gpt-5.5") == ("openai/gpt-5.5", None)


def test_route_menu_model_rejects_unknown_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-catalog id is refused before any LLM spend."""
    monkeypatch.setattr(
        model_catalog, "get_catalog_cached", lambda: _catalog_with("openai/gpt-5.5")
    )
    with pytest.raises(DomainError) as err:
        route_menu_model("openai/not-a-model")
    assert err.value.status_code == 422


def test_route_menu_model_routes_auto_tiers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent values run the pinned default; the sentinel rides the router."""
    monkeypatch.setattr(
        mr,
        "get_catalog_cached",
        lambda: _catalog_with("openrouter/anthropic/claude-sonnet-5"),
    )
    assert route_menu_model(None) == (BALANCED_PINNED_MODEL_ID, None)
    model, body = route_menu_model(AUTO_INTELLIGENT_ID, session_id="sess-1")
    assert model == OPENROUTER_AUTO_ID
    assert body == {
        "plugins": [{"id": "auto-router", "cost_quality_tradeoff": 0}],
        "session_id": "sess-1",
    }


def test_route_menu_model_degrades_without_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    """No OpenRouter connectivity → the engine's configured default runs."""
    monkeypatch.setattr(mr, "get_catalog_cached", lambda: _catalog_with("openai/gpt-5.5"))
    assert route_menu_model(None) == (None, None)
