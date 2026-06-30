"""Tests for the BYOK run-path bridge: provider resolution + key injection.

Covers that ``inject_byok_connections`` stamps the user's vault key onto each
ModelConfig in a run or grid payload (with a custom ``api_base`` / ``params``),
that a missing connection fails the run with a clear message, and that the
submit-time gate rejects a BYOK run the account has no key for. An in-memory
SQLite engine backs the vault and ``httpx.get`` is patched so no network call is
made.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr
from sqlalchemy import create_engine

from core.api.errors import DomainError
from core.api.routers.submissions import _enforce_byok_connections
from core.billing.byok_bridge import inject_byok_connections, provider_slug_for_model
from core.billing.byok_vault import ProviderKeyVault
from core.config import settings
from core.storage.models import Base


@pytest.fixture
def engine() -> Iterator[object]:
    """Yield an in-memory SQLite engine with the billing tables created."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture
def vault(engine: object, monkeypatch: pytest.MonkeyPatch) -> ProviderKeyVault:
    """A vault bound to the test engine with a real Fernet key configured."""
    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setattr(settings, "byok_vault_key", SecretStr(key))
    return ProviderKeyVault(engine=engine)


def _ok_response() -> object:
    """Return a stand-in 200 probe response so a saved key verifies."""
    return SimpleNamespace(status_code=200, is_success=True)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("openai/gpt-4o", "openai"),
        ("anthropic/claude-3-5-sonnet", "anthropic"),
        ("openrouter/minimax/minimax-01", "openrouter"),
        ("gpt-4o", None),
        ("", None),
        ("/openai/gpt-4o", "openai"),
    ],
)
def test_provider_slug_for_model(name: str, expected: str | None) -> None:
    """The provider slug is the first path segment of a LiteLLM model string."""
    assert provider_slug_for_model(name) == expected


def test_inject_stamps_run_model_configs(vault: ProviderKeyVault) -> None:
    """A run payload's student + reflection configs get the user's keys, per provider."""
    with patch("core.billing.byok_vault.httpx.get", return_value=_ok_response()):
        vault.save_key("u@x.com", "openai", "sk-openai-1111")
        vault.save_key("u@x.com", "anthropic", "sk-ant-2222")
    payload = {
        "model_config": {"name": "openai/gpt-4o", "extra": {}},
        "reflection_model_config": {"name": "anthropic/claude-3-5-sonnet", "extra": {}},
    }
    inject_byok_connections(payload, username="u@x.com", vault=vault)
    assert payload["model_config"]["extra"]["api_key"] == "sk-openai-1111"
    assert payload["reflection_model_config"]["extra"]["api_key"] == "sk-ant-2222"


def test_inject_applies_custom_api_base_and_params(vault: ProviderKeyVault) -> None:
    """A custom-endpoint connection injects its api_base and extra params too."""
    with patch("core.billing.byok_vault.httpx.get", return_value=_ok_response()):
        vault.save_key(
            "u@x.com",
            "custom",
            "sk-custom-3333",
            api_base="https://host.example/v1",
            params={"organization": "org-1"},
        )
    payload = {"model_config": {"name": "custom/my-model", "extra": {}}}
    inject_byok_connections(payload, username="u@x.com", vault=vault)
    cfg = payload["model_config"]
    assert cfg["extra"]["api_key"] == "sk-custom-3333"
    assert cfg["base_url"] == "https://host.example/v1"
    assert cfg["extra"]["organization"] == "org-1"


def test_inject_stamps_grid_model_lists(vault: ProviderKeyVault) -> None:
    """A grid payload's generation + reflection lists each get the user's key."""
    with patch("core.billing.byok_vault.httpx.get", return_value=_ok_response()):
        vault.save_key("u@x.com", "openai", "sk-openai-4444")
    payload = {
        "generation_models": [{"name": "openai/gpt-4o", "extra": {}}],
        "reflection_models": [{"name": "openai/gpt-4o-mini", "extra": {}}],
    }
    inject_byok_connections(payload, username="u@x.com", vault=vault)
    assert payload["generation_models"][0]["extra"]["api_key"] == "sk-openai-4444"
    assert payload["reflection_models"][0]["extra"]["api_key"] == "sk-openai-4444"


def test_inject_missing_connection_raises(vault: ProviderKeyVault) -> None:
    """A model whose provider has no saved connection fails the run with a clear error."""
    payload = {"model_config": {"name": "openai/gpt-4o", "extra": {}}}
    with pytest.raises(ValueError, match="openai"):
        inject_byok_connections(payload, username="u@x.com", vault=vault)


def test_inject_skips_providerless_model(vault: ProviderKeyVault) -> None:
    """A bare model name with no provider prefix is left untouched (nothing to resolve)."""
    payload = {"model_config": {"name": "gpt-4o", "extra": {}}}
    inject_byok_connections(payload, username="u@x.com", vault=vault)
    assert "api_key" not in payload["model_config"]["extra"]


def test_enforce_byok_connections_blocks_missing(vault: ProviderKeyVault, engine: object) -> None:
    """The submit gate rejects a BYOK run the account has no key for."""
    job_store = SimpleNamespace(engine=engine)
    with pytest.raises(DomainError) as exc:
        _enforce_byok_connections(job_store, "u@x.com", "byok", ["openai/gpt-4o"])
    assert exc.value.status_code == 400


def test_enforce_byok_connections_passes_when_present(vault: ProviderKeyVault, engine: object) -> None:
    """The submit gate lets a BYOK run through once the key is saved."""
    with patch("core.billing.byok_vault.httpx.get", return_value=_ok_response()):
        vault.save_key("u@x.com", "openai", "sk-openai-5555")
    job_store = SimpleNamespace(engine=engine)
    _enforce_byok_connections(job_store, "u@x.com", "byok", ["openai/gpt-4o"])


def test_enforce_byok_connections_bridges_litellm_prefix_to_vault_slug(
    vault: ProviderKeyVault, engine: object
) -> None:
    """A key saved under the vault slug satisfies a model carrying its LiteLLM prefix.

    Regression: ``gemini/...`` models resolve a key saved under ``google``. Before
    the gate bridged the prefix it looked up ``has_connection(.., "gemini")`` —
    always false — and wrongly blocked the run despite a valid saved key. Covers
    every provider whose LiteLLM prefix differs from its vault slug.
    """
    with patch("core.billing.byok_vault.httpx.get", return_value=_ok_response()):
        vault.save_key("u@x.com", "google", "AIza-google-6666")
        vault.save_key("u@x.com", "together", "sk-together-7777")
    job_store = SimpleNamespace(engine=engine)
    _enforce_byok_connections(
        job_store,
        "u@x.com",
        "byok",
        ["gemini/gemini-1.5-pro", "together_ai/mistral-7b"],
    )


def test_enforce_byok_connections_reports_bridged_slug(
    vault: ProviderKeyVault, engine: object
) -> None:
    """A missing connection is reported under the savable vault slug, not the prefix."""
    job_store = SimpleNamespace(engine=engine)
    with pytest.raises(DomainError) as exc:
        _enforce_byok_connections(job_store, "u@x.com", "byok", ["gemini/gemini-1.5-pro"])
    assert exc.value.params["provider"] == "google"


def test_enforce_byok_connections_managed_is_noop(engine: object) -> None:
    """Managed runs skip the BYOK gate entirely — no vault access, no raise."""
    job_store = SimpleNamespace(engine=engine)
    _enforce_byok_connections(job_store, "u@x.com", "managed", ["openai/gpt-4o"])
