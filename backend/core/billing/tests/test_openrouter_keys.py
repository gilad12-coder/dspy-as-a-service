"""Tests for the per-user OpenRouter runtime-key provisioner and its injection.

Covers the gating (both secrets required), the mint path (limit denominated in
dollars, secret persisted only as ciphertext), the pre-dispatch limit sync
(``usage + spendable``, PATCH skipped when already in sync), the fail-open
paths (API failure, undecryptable ciphertext → ``None`` so the dispatch keeps
the shared gateway key), and the payload injection across the run and grid
shapes. Each test stands up an in-memory SQLite engine with the billing tables
and patches ``httpx.request`` so no network call is made.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from cryptography.fernet import Fernet
from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core.billing.openrouter_keys import (
    OPENROUTER_KEYS_URL,
    OpenRouterKeyProvisioner,
    inject_provisioned_openrouter_key,
)
from core.config import settings
from core.storage.models import Base, BillingOpenRouterKeyModel

_USER = "u@x.com"
_RUNTIME_SECRET = "sk-or-runtime-abcd"
_KEY_HASH = "hash-1234"


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


@pytest.fixture
def provisioning_key(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set the key-management secret so provisioning is enabled, and return it."""
    monkeypatch.setattr(settings, "openrouter_provisioning_key", SecretStr("prov-key-123"))
    return "prov-key-123"


def _api_response(body: Any, status_code: int = 200) -> object:
    """Build a stand-in httpx response carrying the given JSON body.

    Args:
        body: What ``response.json()`` returns.
        status_code: The HTTP status the fake response reports.

    Returns:
        An object exposing the ``status_code`` / ``is_success`` / ``json``
        httpx reads.
    """
    return SimpleNamespace(
        status_code=status_code,
        is_success=200 <= status_code < 300,
        json=lambda: body,
    )


class _FakeApi:
    """Record key-management calls and replay canned responses in order."""

    def __init__(self, responses: list[object]) -> None:
        """Seed the replay queue.

        Args:
            responses: Responses (or exceptions to raise) served per call.
        """
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> object:
        """Record one call and pop the next canned response.

        Args:
            method: HTTP method the module sent.
            url: Endpoint URL.
            headers: Request headers (carrying the auth shape).
            json: JSON body, when the call had one.
            timeout: Request timeout (unused).

        Returns:
            The next canned response.

        Raises:
            Exception: When the next canned entry is an exception instance.
        """
        self.calls.append({"method": method, "url": url, "headers": headers or {}, "json": json})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _stored_row(engine: object) -> BillingOpenRouterKeyModel | None:
    """Fetch the test user's persisted key row, if any."""
    with Session(engine) as session:
        return session.get(BillingOpenRouterKeyModel, _USER)


def test_disabled_without_provisioning_key(engine: object, vault_key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the key-management secret the provisioner is off and makes no calls."""
    monkeypatch.setattr(settings, "openrouter_provisioning_key", None)
    provisioner = OpenRouterKeyProvisioner(engine=engine)
    fake = _FakeApi([])
    with patch("core.billing.openrouter_keys.httpx.request", side_effect=fake):
        assert not provisioner.enabled
        assert provisioner.ensure_runtime_key(_USER, 500) is None
    assert fake.calls == []


def test_disabled_without_vault_key(engine: object, provisioning_key: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the vault key there is nowhere safe to store the secret, so it's off."""
    monkeypatch.setattr(settings, "byok_vault_key", None)
    provisioner = OpenRouterKeyProvisioner(engine=engine)
    fake = _FakeApi([])
    with patch("core.billing.openrouter_keys.httpx.request", side_effect=fake):
        assert not provisioner.enabled
        assert provisioner.ensure_runtime_key(_USER, 500) is None
    assert fake.calls == []


def test_first_use_mints_key_limited_to_balance(engine: object, vault_key: str, provisioning_key: str) -> None:
    """First dispatch POSTs a create with the balance as a dollar limit and stores ciphertext."""
    provisioner = OpenRouterKeyProvisioner(engine=engine)
    fake = _FakeApi([_api_response({"key": _RUNTIME_SECRET, "data": {"hash": _KEY_HASH}})])
    with patch("core.billing.openrouter_keys.httpx.request", side_effect=fake):
        secret = provisioner.ensure_runtime_key(_USER, 500)
    assert secret == _RUNTIME_SECRET
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == OPENROUTER_KEYS_URL
    assert call["headers"]["Authorization"] == f"Bearer {provisioning_key}"
    assert call["json"] == {"name": f"skynet-user-{_USER}", "limit": 5.0}
    row = _stored_row(engine)
    assert row is not None
    assert row.key_hash == _KEY_HASH
    assert _RUNTIME_SECRET.encode("utf-8") not in row.secret_ciphertext
    assert Fernet(vault_key.encode("utf-8")).decrypt(row.secret_ciphertext).decode() == _RUNTIME_SECRET


def test_later_dispatch_syncs_limit_to_usage_plus_balance(
    engine: object, vault_key: str, provisioning_key: str
) -> None:
    """With a stored key, dispatch reads usage and PATCHes limit = usage + spendable."""
    provisioner = OpenRouterKeyProvisioner(engine=engine)
    create = _api_response({"key": _RUNTIME_SECRET, "data": {"hash": _KEY_HASH}})
    detail = _api_response({"data": {"usage": 1.5, "limit": 5.0}})
    patched = _api_response({"data": {"limit": 6.5}})
    fake = _FakeApi([create, detail, patched])
    with patch("core.billing.openrouter_keys.httpx.request", side_effect=fake):
        provisioner.ensure_runtime_key(_USER, 500)
        secret = provisioner.ensure_runtime_key(_USER, 500)
    assert secret == _RUNTIME_SECRET
    assert [c["method"] for c in fake.calls] == ["POST", "GET", "PATCH"]
    assert fake.calls[1]["url"] == f"{OPENROUTER_KEYS_URL}/{_KEY_HASH}"
    assert fake.calls[2]["json"] == {"limit": 6.5}


def test_sync_skips_patch_when_limit_already_matches(engine: object, vault_key: str, provisioning_key: str) -> None:
    """No PATCH goes out when the key's limit already equals usage + spendable."""
    provisioner = OpenRouterKeyProvisioner(engine=engine)
    create = _api_response({"key": _RUNTIME_SECRET, "data": {"hash": _KEY_HASH}})
    detail = _api_response({"data": {"usage": 1.5, "limit": 6.5}})
    fake = _FakeApi([create, detail])
    with patch("core.billing.openrouter_keys.httpx.request", side_effect=fake):
        provisioner.ensure_runtime_key(_USER, 500)
        secret = provisioner.ensure_runtime_key(_USER, 500)
    assert secret == _RUNTIME_SECRET
    assert [c["method"] for c in fake.calls] == ["POST", "GET"]


def test_create_failure_falls_back_to_none(engine: object, vault_key: str, provisioning_key: str) -> None:
    """A failed mint returns None and persists nothing — dispatch keeps the shared key."""
    provisioner = OpenRouterKeyProvisioner(engine=engine)
    fake = _FakeApi([_api_response({"error": "nope"}, status_code=500)])
    with patch("core.billing.openrouter_keys.httpx.request", side_effect=fake):
        assert provisioner.ensure_runtime_key(_USER, 500) is None
    assert _stored_row(engine) is None


def test_create_unexpected_shape_falls_back_to_none(engine: object, vault_key: str, provisioning_key: str) -> None:
    """A create response missing the secret or hash is rejected, not half-stored."""
    provisioner = OpenRouterKeyProvisioner(engine=engine)
    fake = _FakeApi([_api_response({"data": {"hash": _KEY_HASH}})])
    with patch("core.billing.openrouter_keys.httpx.request", side_effect=fake):
        assert provisioner.ensure_runtime_key(_USER, 500) is None
    assert _stored_row(engine) is None


def test_network_error_falls_back_to_none(engine: object, vault_key: str, provisioning_key: str) -> None:
    """An unreachable key-management API yields None instead of raising."""
    provisioner = OpenRouterKeyProvisioner(engine=engine)
    fake = _FakeApi([httpx.ConnectError("down")])
    with patch("core.billing.openrouter_keys.httpx.request", side_effect=fake):
        assert provisioner.ensure_runtime_key(_USER, 500) is None


def test_sync_failure_falls_back_to_none(engine: object, vault_key: str, provisioning_key: str) -> None:
    """A stored key whose limit cannot be synced is not handed to the dispatch."""
    provisioner = OpenRouterKeyProvisioner(engine=engine)
    create = _api_response({"key": _RUNTIME_SECRET, "data": {"hash": _KEY_HASH}})
    fake = _FakeApi([create, httpx.ConnectError("down")])
    with patch("core.billing.openrouter_keys.httpx.request", side_effect=fake):
        provisioner.ensure_runtime_key(_USER, 500)
        assert provisioner.ensure_runtime_key(_USER, 500) is None


def test_undecryptable_ciphertext_falls_back_to_none(engine: object, vault_key: str, provisioning_key: str) -> None:
    """A row that no longer decrypts (rotated vault key) yields None without any API call."""
    with Session(engine) as session:
        session.merge(BillingOpenRouterKeyModel(username=_USER, key_hash=_KEY_HASH, secret_ciphertext=b"not-fernet"))
        session.commit()
    provisioner = OpenRouterKeyProvisioner(engine=engine)
    fake = _FakeApi([])
    with patch("core.billing.openrouter_keys.httpx.request", side_effect=fake):
        assert provisioner.ensure_runtime_key(_USER, 500) is None
    assert fake.calls == []


def test_inject_run_shape_pins_openrouter_and_keys() -> None:
    """Run-shape configs get the openrouter/ pin and the runtime key in extra."""
    payload = {
        "model_config": {"name": "openai/gpt-4o-mini", "extra": {"temperature": 0.2}},
        "reflection_model_config": {"name": "qwen/qwen3-8b"},
    }
    inject_provisioned_openrouter_key(payload, api_key="sk-or-x")
    assert payload["model_config"]["name"] == "openrouter/openai/gpt-4o-mini"
    assert payload["model_config"]["extra"]["api_key"] == "sk-or-x"
    assert payload["model_config"]["extra"]["temperature"] == 0.2
    assert payload["reflection_model_config"]["name"] == "openrouter/qwen/qwen3-8b"
    assert payload["reflection_model_config"]["extra"]["api_key"] == "sk-or-x"


def test_inject_grid_shape_covers_every_list_entry() -> None:
    """Grid-shape model lists all get pinned and keyed."""
    payload = {
        "generation_models": [{"name": "openai/gpt-4o"}, {"name": "anthropic/claude-3-5-haiku"}],
        "reflection_models": [{"name": "openai/gpt-4o-mini"}],
    }
    inject_provisioned_openrouter_key(payload, api_key="sk-or-x")
    for cfg in payload["generation_models"] + payload["reflection_models"]:
        assert cfg["name"].startswith("openrouter/")
        assert cfg["extra"]["api_key"] == "sk-or-x"


def test_inject_skips_byok_configs_in_mixed_payload() -> None:
    """A managed runtime key never overwrites the BYOK side of a mixed payload."""
    payload = {
        "model_config": {"name": "openrouter/openai/gpt-4o", "token_source": "byok"},
        "reflection_model_config": {
            "name": "openrouter/anthropic/claude-3-5-haiku",
            "token_source": "managed",
        },
    }

    inject_provisioned_openrouter_key(
        payload,
        api_key="sk-or-managed",
        default_token_source="managed",
    )

    assert "extra" not in payload["model_config"]
    assert payload["reflection_model_config"]["extra"]["api_key"] == "sk-or-managed"


def test_inject_does_not_double_prefix() -> None:
    """A model already pinned to openrouter/ keeps a single prefix."""
    payload = {"model_config": {"name": "openrouter/openai/gpt-4o"}}
    inject_provisioned_openrouter_key(payload, api_key="sk-or-x")
    assert payload["model_config"]["name"] == "openrouter/openai/gpt-4o"


def test_inject_leaves_already_keyed_config_untouched() -> None:
    """A config carrying its own api_key is not repinned or rekeyed."""
    payload = {"model_config": {"name": "openai/gpt-4o", "extra": {"api_key": "sk-user"}}}
    inject_provisioned_openrouter_key(payload, api_key="sk-or-x")
    assert payload["model_config"]["name"] == "openai/gpt-4o"
    assert payload["model_config"]["extra"]["api_key"] == "sk-user"


def test_inject_skips_config_without_model_name() -> None:
    """A config with no model name is left alone — there is nothing to pin."""
    payload = {"model_config": {"name": "   "}}
    inject_provisioned_openrouter_key(payload, api_key="sk-or-x")
    assert payload["model_config"] == {"name": "   "}
