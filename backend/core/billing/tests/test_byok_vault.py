"""Tests for ``ProviderKeyVault``: encrypt-at-rest, the verify probe, and routes.

Covers that secrets are stored only as ciphertext (never plaintext), that the
verify probe classifies a provider response into verified / invalid /
unverified, that decryption round-trips for the run path, and that the
mutations refuse to run without a configured vault key. Each test stands up an
in-memory SQLite engine with the billing tables and patches ``httpx.get`` so no
network call is made.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.api.errors import DomainError
from core.billing.byok_vault import (
    STATUS_INVALID,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED,
    ProviderKeyVault,
)
from core.config import settings
from core.storage.models import Base, BillingProviderKeyModel


@pytest.fixture
def engine() -> Iterator[object]:
    """Yield an in-memory SQLite engine with the billing tables created."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)


@pytest.fixture
def vault_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set a real Fernet vault key so encryption/decryption works, and return it."""
    key = Fernet.generate_key().decode("utf-8")
    monkeypatch.setattr(settings, "byok_vault_key", SecretStr(key))
    return key


def _probe_response(status_code: int) -> object:
    """Build a stand-in httpx response with the given status code.

    Args:
        status_code: The HTTP status the fake probe response reports.

    Returns:
        An object exposing the ``status_code`` / ``is_success`` httpx reads.
    """
    return SimpleNamespace(status_code=status_code, is_success=200 <= status_code < 300)


def test_save_key_stores_ciphertext_not_plaintext(engine: object, vault_key: str) -> None:
    """The stored row holds Fernet ciphertext — the plaintext never appears in the DB."""
    vault = ProviderKeyVault(engine=engine)
    secret = "sk-supersecret-abcd"
    with patch("core.billing.byok_vault.httpx.get", return_value=_probe_response(200)):
        view = vault.save_key("u@x.com", "openai", secret)
    assert view.last4 == "abcd"
    assert view.status == STATUS_VERIFIED
    with Session(engine) as session:
        row = session.get(BillingProviderKeyModel, ("u@x.com", "openai"))
        assert row is not None
        assert secret.encode("utf-8") not in row.secret_ciphertext
        # The ciphertext round-trips back to the original secret under the key.
        assert Fernet(vault_key.encode("utf-8")).decrypt(row.secret_ciphertext).decode() == secret


def test_save_key_verified_on_2xx(engine: object, vault_key: str) -> None:
    """A 2xx probe response marks the key verified on entry."""
    vault = ProviderKeyVault(engine=engine)
    with patch("core.billing.byok_vault.httpx.get", return_value=_probe_response(200)):
        view = vault.save_key("u@x.com", "anthropic", "sk-ant-1234")
    assert view.status == STATUS_VERIFIED


def test_save_key_invalid_on_auth_rejection(engine: object, vault_key: str) -> None:
    """A 401 probe response marks the key invalid."""
    vault = ProviderKeyVault(engine=engine)
    with patch("core.billing.byok_vault.httpx.get", return_value=_probe_response(401)):
        view = vault.save_key("u@x.com", "openai", "sk-bad-9999")
    assert view.status == STATUS_INVALID


def test_save_key_unverified_on_network_error(engine: object, vault_key: str) -> None:
    """A transient/network error leaves the key unverified — not condemned as invalid."""
    vault = ProviderKeyVault(engine=engine)
    with patch("core.billing.byok_vault.httpx.get", side_effect=httpx.ConnectError("down")):
        view = vault.save_key("u@x.com", "openai", "sk-maybe-0000")
    assert view.status == STATUS_UNVERIFIED


def test_save_key_unverified_on_unexpected_status(engine: object, vault_key: str) -> None:
    """A 500 from the provider is inconclusive, so the key stays unverified."""
    vault = ProviderKeyVault(engine=engine)
    with patch("core.billing.byok_vault.httpx.get", return_value=_probe_response(500)):
        view = vault.save_key("u@x.com", "openai", "sk-shrug-1111")
    assert view.status == STATUS_UNVERIFIED


def test_save_key_rotates_in_place(engine: object, vault_key: str) -> None:
    """Saving a second key for the same provider replaces the first (rotation)."""
    vault = ProviderKeyVault(engine=engine)
    with patch("core.billing.byok_vault.httpx.get", return_value=_probe_response(200)):
        vault.save_key("u@x.com", "openai", "sk-first-1111")
        vault.save_key("u@x.com", "openai", "sk-second-2222")
    snapshot = vault.list_keys("u@x.com")
    assert len(snapshot.keys) == 1
    assert snapshot.keys[0].last4 == "2222"


def test_save_key_requires_vault_key(engine: object, monkeypatch: pytest.MonkeyPatch) -> None:
    """Saving a key without a configured vault key raises 503, never a 500."""
    monkeypatch.setattr(settings, "byok_vault_key", None)
    vault = ProviderKeyVault(engine=engine)
    with pytest.raises(DomainError) as exc:
        vault.save_key("u@x.com", "openai", "sk-1234")
    assert exc.value.status_code == 503


def test_save_key_rejects_unknown_provider(engine: object, vault_key: str) -> None:
    """An unknown provider slug is rejected with a 400."""
    vault = ProviderKeyVault(engine=engine)
    with pytest.raises(DomainError) as exc:
        vault.save_key("u@x.com", "nope", "sk-1234")
    assert exc.value.status_code == 400


def test_save_key_rejects_empty_secret(engine: object, vault_key: str) -> None:
    """A blank secret is rejected with a 400 before any probe runs."""
    vault = ProviderKeyVault(engine=engine)
    with pytest.raises(DomainError) as exc:
        vault.save_key("u@x.com", "openai", "   ")
    assert exc.value.status_code == 400


def test_list_keys_is_secret_free_and_works_without_vault_key(
    engine: object, vault_key: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Listing keys returns masked views and serves even when the vault key is gone."""
    vault = ProviderKeyVault(engine=engine)
    with patch("core.billing.byok_vault.httpx.get", return_value=_probe_response(200)):
        vault.save_key("u@x.com", "openai", "sk-keep-7777")
    monkeypatch.setattr(settings, "byok_vault_key", None)
    snapshot = vault.list_keys("u@x.com")
    assert len(snapshot.keys) == 1
    view = snapshot.keys[0]
    assert view.provider == "openai"
    assert view.last4 == "7777"
    assert not hasattr(view, "secret")


def test_verify_key_reprobe_updates_status(engine: object, vault_key: str) -> None:
    """Re-verifying a key saved while the provider was down flips it to verified."""
    vault = ProviderKeyVault(engine=engine)
    with patch("core.billing.byok_vault.httpx.get", side_effect=httpx.ConnectError("down")):
        view = vault.save_key("u@x.com", "openai", "sk-later-8888")
    assert view.status == STATUS_UNVERIFIED
    with patch("core.billing.byok_vault.httpx.get", return_value=_probe_response(200)):
        reverified = vault.verify_key("u@x.com", "openai")
    assert reverified.status == STATUS_VERIFIED


def test_verify_key_missing_raises_404(engine: object, vault_key: str) -> None:
    """Verifying a provider with no stored key raises 404."""
    vault = ProviderKeyVault(engine=engine)
    with pytest.raises(DomainError) as exc:
        vault.verify_key("u@x.com", "openai")
    assert exc.value.status_code == 404


def test_remove_key_is_idempotent(engine: object, vault_key: str) -> None:
    """Removing a key forgets it; removing again is a harmless no-op."""
    vault = ProviderKeyVault(engine=engine)
    with patch("core.billing.byok_vault.httpx.get", return_value=_probe_response(200)):
        vault.save_key("u@x.com", "openai", "sk-gone-3333")
    vault.remove_key("u@x.com", "openai")
    vault.remove_key("u@x.com", "openai")
    assert vault.list_keys("u@x.com").keys == []


def test_reveal_secret_round_trips(engine: object, vault_key: str) -> None:
    """The run path can decrypt the stored secret back to its plaintext."""
    vault = ProviderKeyVault(engine=engine)
    secret = "sk-reveal-4444"
    with patch("core.billing.byok_vault.httpx.get", return_value=_probe_response(200)):
        vault.save_key("u@x.com", "openai", secret)
    assert vault.reveal_secret("u@x.com", "openai") == secret


def test_reveal_secret_missing_returns_none(engine: object, vault_key: str) -> None:
    """Revealing a provider with no stored key returns None, not an error."""
    vault = ProviderKeyVault(engine=engine)
    assert vault.reveal_secret("u@x.com", "openai") is None
