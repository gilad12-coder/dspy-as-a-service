"""Unit tests for the model catalog DTOs and ``get_catalog`` helper."""

from __future__ import annotations

import litellm
import pytest

from ...config import settings
from .. import model_catalog as mc
from ..model_catalog import (  # type: ignore[attr-defined]
    CatalogModel,
    CatalogProvider,
    ModelCatalogResponse,
    _make_label,
    _provider_data_centers,
    get_catalog,
)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("openai/gpt-4o-mini", "gpt-4o-mini"),
        ("together_ai/mistral-7b-instruct-v0.1", "mistral-7b-instruct-v0.1"),
        ("anthropic/claude-3-5-haiku-20241022", "claude-3-5-haiku-20241022"),
    ],
    ids=["no_prefix", "openai_prefix", "together_ai_prefix", "anthropic_prefix"],
)
def test_make_label_strips_provider_prefix(model_id: str, expected: str) -> None:
    """``_make_label`` strips known provider prefixes from a model id."""
    assert _make_label(model_id) == expected


def test_make_label_leaves_id_without_slash_untouched() -> None:
    """An id with no provider prefix is returned unchanged."""
    assert _make_label("gpt-4o") == "gpt-4o"


def test_catalog_model_defaults() -> None:
    """Optional fields default to ``False`` / ``None`` on ``CatalogModel``."""
    m = CatalogModel(value="openai/gpt-4o", label="gpt-4o", provider="openai")
    assert m.supports_thinking is False
    assert m.supports_vision is False
    assert m.available is False
    assert m.max_input_tokens is None
    assert m.data_center is None
    assert m.input_cost_per_token is None
    assert m.output_cost_per_token is None


def test_catalog_model_stores_all_fields() -> None:
    """Explicitly-supplied fields are preserved on ``CatalogModel``."""
    m = CatalogModel(
        value="openai/gpt-4o",
        label="gpt-4o",
        provider="openai",
        supports_thinking=True,
        available=True,
        max_input_tokens=128000,
    )
    assert m.supports_thinking is True
    assert m.available is True
    assert m.max_input_tokens == 128000


def test_catalog_provider_has_env_key_defaults_false() -> None:
    """``has_env_key`` defaults to ``False`` and ``data_center`` to ``None``."""
    p = CatalogProvider(slug="openai", label="OpenAI")
    assert p.has_env_key is False
    assert p.data_center is None


def test_catalog_provider_stores_env_var_and_url() -> None:
    """Provider env vars, base URLs and data center round-trip."""
    p = CatalogProvider(
        slug="openai",
        label="OpenAI",
        data_center="On-prem gateway",
        env_var="OPENAI_API_KEY",
        default_base_url="https://api.openai.com/v1",
        has_env_key=True,
    )
    assert p.env_var == "OPENAI_API_KEY"
    assert p.default_base_url == "https://api.openai.com/v1"
    assert p.has_env_key is True
    assert p.data_center == "On-prem gateway"


def test_model_catalog_response_empty_lists() -> None:
    """An empty response carries empty ``providers`` and ``models`` lists."""
    resp = ModelCatalogResponse(providers=[], models=[])
    assert resp.providers == []
    assert resp.models == []


def test_model_catalog_response_preserves_order() -> None:
    """Model insertion order is preserved on the response."""
    models = [
        CatalogModel(value="openai/gpt-4o-mini", label="gpt-4o-mini", provider="openai", available=True),
        CatalogModel(
            value="anthropic/claude-3-5-haiku-20241022",
            label="claude-3-5-haiku-20241022",
            provider="anthropic",
            available=True,
        ),
    ]
    resp = ModelCatalogResponse(providers=[], models=models)
    assert resp.models[0].value == "openai/gpt-4o-mini"
    assert resp.models[1].provider == "anthropic"


def test_get_catalog_returns_correct_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_catalog`` returns a structured response of providers and models.

    Monkeypatches ``litellm.get_valid_models`` to avoid API keys and to make
    results deterministic. Asserts only structural invariants -- specific
    models change as LiteLLM updates its registry.
    """
    fake_cost: dict = dict(litellm.model_cost)
    fake_cost["fakeprovider-model-a"] = {
        "mode": "chat",
        "litellm_provider": "openai",
        "supports_reasoning": False,
        "max_input_tokens": 4096,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    }

    monkeypatch.setattr(litellm, "model_cost", fake_cost)
    monkeypatch.setattr(litellm, "get_valid_models", lambda: ["fakeprovider-model-a"])
    monkeypatch.setattr(mc, "_probe_all_providers", dict)

    result = get_catalog()

    assert isinstance(result, ModelCatalogResponse)
    assert isinstance(result.providers, list)
    assert isinstance(result.models, list)
    # The injected model should be available and visible
    values = [m.value for m in result.models]
    assert any("fakeprovider-model-a" in v for v in values)


def test_get_catalog_surfaces_per_token_costs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Positive LiteLLM per-token costs ride onto the catalog; zero/absent → None."""
    fake_cost: dict = {
        "priced-model": {
            "mode": "chat",
            "litellm_provider": "openai",
            "max_input_tokens": 4096,
            "input_cost_per_token": 1.5e-7,
            "output_cost_per_token": 6e-7,
        },
        "freebie-model": {
            "mode": "chat",
            "litellm_provider": "openai",
            "max_input_tokens": 4096,
            "input_cost_per_token": 0,  # zero/absent must surface as None, not free
        },
    }
    monkeypatch.setattr(litellm, "model_cost", fake_cost)
    monkeypatch.setattr(litellm, "get_valid_models", lambda: ["priced-model", "freebie-model"])
    monkeypatch.setattr(mc, "_probe_all_providers", dict)

    by_value = {m.value: m for m in get_catalog().models}
    priced = by_value["openai/priced-model"]
    assert priced.input_cost_per_token == 1.5e-7
    assert priced.output_cost_per_token == 6e-7
    freebie = by_value["openai/freebie-model"]
    assert freebie.input_cost_per_token is None
    assert freebie.output_cost_per_token is None


def test_get_catalog_only_returns_available_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Models that aren't reported by ``get_valid_models`` are filtered out."""
    fake_cost: dict = {
        "only-in-registry": {
            "mode": "chat",
            "litellm_provider": "openai",
            "supports_reasoning": False,
            "max_input_tokens": 4096,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }
    }
    monkeypatch.setattr(litellm, "model_cost", fake_cost)
    monkeypatch.setattr(litellm, "get_valid_models", list)
    monkeypatch.setattr(mc, "_probe_all_providers", dict)

    result = get_catalog()

    assert result.models == []


def test_get_catalog_deduplicates_dated_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dated variants are dropped when the base model id is also present.

    A dated variant (``gpt-4o-2024-08-06``) is excluded when the base name
    (``gpt-4o``) exists in the cost table.
    """
    fake_cost: dict = {
        "gpt-4o": {
            "mode": "chat",
            "litellm_provider": "openai",
            "supports_reasoning": False,
            "max_input_tokens": 128000,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
        "gpt-4o-2024-08-06": {
            "mode": "chat",
            "litellm_provider": "openai",
            "supports_reasoning": False,
            "max_input_tokens": 128000,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    }
    monkeypatch.setattr(litellm, "model_cost", fake_cost)
    monkeypatch.setattr(litellm, "get_valid_models", lambda: ["gpt-4o", "gpt-4o-2024-08-06"])
    monkeypatch.setattr(mc, "_probe_all_providers", dict)

    result = get_catalog()

    values = [m.value for m in result.models]
    # Dated variant must not appear
    assert not any("2024-08-06" in v for v in values)
    # Base must appear
    assert any(v.endswith("gpt-4o") for v in values)


def test_get_catalog_filters_out_non_chat_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only chat-mode entries are relevant for DSPy LMs; others are filtered."""
    fake_cost: dict = {
        "text-embedding-ada-002": {
            "mode": "embedding",
            "litellm_provider": "openai",
            "max_input_tokens": 8192,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    }
    monkeypatch.setattr(litellm, "model_cost", fake_cost)
    monkeypatch.setattr(litellm, "get_valid_models", lambda: ["text-embedding-ada-002"])
    monkeypatch.setattr(mc, "_probe_all_providers", dict)

    result = get_catalog()

    assert result.models == []


def test_get_catalog_propagates_supports_vision_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """``supports_vision`` is propagated from the cost table to the catalog."""
    fake_cost: dict = {
        "vision-model": {
            "mode": "chat",
            "litellm_provider": "openai",
            "supports_reasoning": False,
            "supports_vision": True,
            "max_input_tokens": 128000,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
        "text-only-model": {
            "mode": "chat",
            "litellm_provider": "openai",
            "supports_reasoning": False,
            "max_input_tokens": 8192,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    }
    monkeypatch.setattr(litellm, "model_cost", fake_cost)
    monkeypatch.setattr(litellm, "get_valid_models", lambda: ["vision-model", "text-only-model"])
    monkeypatch.setattr(mc, "_probe_all_providers", dict)

    result = get_catalog()

    by_value = {m.value: m for m in result.models}
    assert by_value["openai/vision-model"].supports_vision is True
    assert by_value["openai/text-only-model"].supports_vision is False


def test_get_catalog_handles_valid_models_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure inside ``get_valid_models`` is swallowed into an empty list.

    If ``get_valid_models`` raises, the catalog returns an empty models list
    rather than propagating the exception.
    """
    fake_cost: dict = {
        "gpt-4o": {
            "mode": "chat",
            "litellm_provider": "openai",
            "supports_reasoning": False,
            "max_input_tokens": 128000,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        },
    }
    monkeypatch.setattr(litellm, "model_cost", fake_cost)
    monkeypatch.setattr(litellm, "get_valid_models", lambda: (_ for _ in ()).throw(RuntimeError("API error")))
    monkeypatch.setattr(mc, "_probe_all_providers", dict)

    # Should not raise
    result = get_catalog()
    assert isinstance(result, ModelCatalogResponse)
    # model is not in valid_set so it's unavailable → filtered out
    assert result.models == []


def test_single_endpoint_provider_has_none_data_center(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider with one endpoint emits exactly one entry, ``data_center=None``.

    Preserves the historical single-DC wire shape so existing clients keep
    working when no on-prem gateway is configured.
    """
    monkeypatch.setattr(settings, "code_agent_base_url", "")
    monkeypatch.setattr(settings, "embeddings_base_url", "")
    centers = _provider_data_centers("openai")
    assert len(centers) == 1
    assert centers[0].label is None

    fake_cost: dict = {
        "gpt-4o": {
            "mode": "chat",
            "litellm_provider": "openai",
            "supports_reasoning": False,
            "max_input_tokens": 128000,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }
    }
    monkeypatch.setattr(litellm, "model_cost", fake_cost)
    monkeypatch.setattr(litellm, "get_valid_models", lambda: ["gpt-4o"])
    monkeypatch.setattr(mc, "_probe_all_providers", dict)

    result = get_catalog()

    openai_models = [m for m in result.models if m.provider == "openai"]
    assert len(openai_models) == 1
    assert openai_models[0].data_center is None
    openai_providers = [p for p in result.providers if p.slug == "openai"]
    assert len(openai_providers) == 1
    assert openai_providers[0].data_center is None


def test_on_prem_gateway_surfaces_as_extra_data_center(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured on-prem gateway adds a second OpenAI data center.

    The same model fans out to two catalog entries — the public OpenAI
    endpoint (``data_center=None``) and the internal gateway
    (``data_center="On-prem gateway"``) — each carrying its own base URL.
    """
    monkeypatch.setattr(settings, "code_agent_base_url", "https://llm.internal/v1")
    monkeypatch.setattr(settings, "embeddings_base_url", "")

    centers = _provider_data_centers("openai")
    assert len(centers) == 2
    on_prem = next(c for c in centers if c.label == "On-prem gateway")
    assert on_prem.base_url == "https://llm.internal/v1"
    assert on_prem.models_url == "https://llm.internal/v1/models"

    fake_cost: dict = {
        "gpt-4o": {
            "mode": "chat",
            "litellm_provider": "openai",
            "supports_reasoning": False,
            "max_input_tokens": 128000,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }
    }
    monkeypatch.setattr(litellm, "model_cost", fake_cost)
    monkeypatch.setattr(litellm, "get_valid_models", lambda: ["gpt-4o"])
    monkeypatch.setattr(mc, "_probe_all_providers", dict)

    result = get_catalog()

    openai_models = [m for m in result.models if m.value == "openai/gpt-4o"]
    dcs = sorted((m.data_center or "") for m in openai_models)
    assert dcs == ["", "On-prem gateway"]

    on_prem_provider = next(p for p in result.providers if p.slug == "openai" and p.data_center == "On-prem gateway")
    assert on_prem_provider.default_base_url == "https://llm.internal/v1"
    native_provider = next(p for p in result.providers if p.slug == "openai" and p.data_center is None)
    assert native_provider.default_base_url == "https://api.openai.com/v1"


def test_on_prem_falls_back_to_embeddings_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """``EMBEDDINGS_BASE_URL`` is used as the on-prem DC when no agent URL is set."""
    monkeypatch.setattr(settings, "code_agent_base_url", "")
    monkeypatch.setattr(settings, "embeddings_base_url", "https://embed.internal/v1")

    centers = _provider_data_centers("openai")
    on_prem = next(c for c in centers if c.label == "On-prem gateway")
    assert on_prem.base_url == "https://embed.internal/v1"


def test_get_catalog_adds_probe_only_models_with_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Probe-reported models absent from LiteLLM's registry are added with metadata.

    Simulates an OpenRouter ``/models`` probe that returns two MiniMax models
    LiteLLM never listed: a vision/reasoning one and a text-only one. Both must
    appear in the catalog, provider-prefixed, with vision/thinking flags and the
    context window derived defensively from the raw probe item.
    """
    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(litellm, "get_valid_models", list)

    probe = {
        ("openrouter", None): {
            "minimax/minimax-m3": {
                "id": "minimax/minimax-m3",
                "architecture": {
                    "input_modalities": ["text", "image", "video"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["reasoning", "temperature"],
                "context_length": 200000,
            },
            "minimax/minimax-m2.7": {
                "id": "minimax/minimax-m2.7",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["temperature"],
                "context_length": 100000,
            },
        }
    }
    monkeypatch.setattr(mc, "_probe_all_providers", lambda: probe)

    result = get_catalog()

    by_value = {m.value: m for m in result.models}
    assert "openrouter/minimax/minimax-m3" in by_value
    assert "openrouter/minimax/minimax-m2.7" in by_value

    m3 = by_value["openrouter/minimax/minimax-m3"]
    assert m3.supports_vision is True
    assert m3.supports_thinking is True
    assert m3.available is True
    assert m3.max_input_tokens == 200000
    assert m3.provider == "openrouter"
    assert m3.label == "minimax-m3"

    m27 = by_value["openrouter/minimax/minimax-m2.7"]
    assert m27.supports_vision is False
    assert m27.supports_thinking is False

    assert any(p.slug == "openrouter" for p in result.providers)


def test_get_catalog_skips_non_text_probe_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe item whose output modalities lack ``"text"`` is skipped.

    An embedding/image model (output_modalities without ``"text"``) must not
    leak into the chat catalog, while an item with no modality info defaults
    to being included.
    """
    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(litellm, "get_valid_models", list)

    probe = {
        ("openrouter", None): {
            "vendor/image-gen": {
                "id": "vendor/image-gen",
                "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
            },
            "vendor/bare-chat": {"id": "vendor/bare-chat"},
        }
    }
    monkeypatch.setattr(mc, "_probe_all_providers", lambda: probe)

    result = get_catalog()

    values = {m.value for m in result.models}
    assert "openrouter/vendor/image-gen" not in values
    assert "openrouter/vendor/bare-chat" in values


def test_get_catalog_does_not_duplicate_registry_models_from_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model present in both the registry and the probe is emitted only once."""
    fake_cost: dict = {
        "openrouter/vendor/known": {
            "mode": "chat",
            "litellm_provider": "openrouter",
            "supports_reasoning": False,
            "max_input_tokens": 8192,
            "input_cost_per_token": 0,
            "output_cost_per_token": 0,
        }
    }
    monkeypatch.setattr(litellm, "model_cost", fake_cost)
    monkeypatch.setattr(litellm, "get_valid_models", lambda: ["openrouter/vendor/known"])

    probe = {
        ("openrouter", None): {
            "vendor/known": {"id": "vendor/known", "context_length": 999999},
        }
    }
    monkeypatch.setattr(mc, "_probe_all_providers", lambda: probe)

    result = get_catalog()

    matches = [m for m in result.models if m.value == "openrouter/vendor/known"]
    assert len(matches) == 1
