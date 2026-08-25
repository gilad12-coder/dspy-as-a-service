"""Tests for the account-scoped BYOK model catalog."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr
from sqlalchemy import create_engine

from core.api.model_catalog import ModelCatalogResponse
from core.api.routers.billing import _byok_catalog_for_user
from core.api.routers.models import DiscoverModelsResponse
from core.billing.byok_vault import ProviderKeyVault
from core.config import settings
from core.storage.models import Base


@pytest.fixture
def vault(monkeypatch: pytest.MonkeyPatch) -> Iterator[ProviderKeyVault]:
    """Yield a configured vault backed by an isolated in-memory database.

    Args:
        monkeypatch: Pytest helper used to configure the encryption key.

    Yields:
        A provider-key vault whose tables exist for the test.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(
        settings,
        "byok_vault_key",
        SecretStr(Fernet.generate_key().decode("utf-8")),
    )
    yield ProviderKeyVault(engine=engine)
    Base.metadata.drop_all(engine)


def test_custom_catalog_fetches_models_with_stored_secret(vault: ProviderKeyVault) -> None:
    """A verified custom connection discovers models without exposing its key."""
    response = SimpleNamespace(status_code=200, is_success=True)
    with patch("core.billing.byok_vault.httpx.get", return_value=response):
        vault.save_key(
            "alice@example.com",
            "custom",
            "private-secret",
            label="Private inference",
            api_base="https://inference.example/v1",
        )

    discovered = DiscoverModelsResponse(
        models=["private-chat", "openai/second-chat"],
        base_url="https://inference.example/v1",
    )
    with (
        patch(
            "core.api.routers.billing.get_byok_catalog_cached",
            return_value=ModelCatalogResponse(providers=[], models=[]),
        ),
        patch(
            "core.api.routers.billing.discover_models_at_endpoint",
            return_value=discovered,
        ) as discover,
    ):
        catalog = _byok_catalog_for_user(vault, "alice@example.com")

    discover.assert_called_once_with("https://inference.example/v1", "private-secret")
    assert [provider.slug for provider in catalog.providers] == ["custom"]
    assert {(model.value, model.byok_provider) for model in catalog.models} == {
        ("openai/private-chat", "custom"),
        ("openai/second-chat", "custom"),
    }
    assert "private-secret" not in catalog.model_dump_json()


def test_custom_catalog_is_scoped_to_the_authenticated_account(
    vault: ProviderKeyVault,
) -> None:
    """Another account cannot see models discovered through someone else's connection."""
    response = SimpleNamespace(status_code=200, is_success=True)
    with patch("core.billing.byok_vault.httpx.get", return_value=response):
        vault.save_key(
            "alice@example.com",
            "custom",
            "private-secret",
            api_base="https://inference.example/v1",
        )
    with patch(
        "core.api.routers.billing.get_byok_catalog_cached",
        return_value=ModelCatalogResponse(providers=[], models=[]),
    ):
        catalog = _byok_catalog_for_user(vault, "bob@example.com")

    assert catalog.providers == []
    assert catalog.models == []
