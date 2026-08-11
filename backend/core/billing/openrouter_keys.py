"""Per-user OpenRouter runtime keys, provisioned and capped at the provider.

Managed runs normally authenticate with one shared gateway key, so every spend
control is backend-side (the submit gate, the cost-ceiling cap, the clamped
debit). When ``OPENROUTER_PROVISIONING_KEY`` is configured, this module mints
one OpenRouter runtime key per account through OpenRouter's key-management API
and, before each managed dispatch, syncs that key's spend limit to
``usage + spendable balance`` — so the provider itself refuses requests once
the account's prepaid credits are gone, even if every backend gate fails. The
minted secret is Fernet-encrypted at rest under the same vault key as BYOK
connections and injected into the run payload in memory only, exactly like the
BYOK bridge. Any provisioning failure falls back to the shared gateway key:
the backend-side controls remain the guarantee, this is the provider-side
backstop.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session

from ..config import settings
from ..storage.models import BillingOpenRouterKeyModel
from .byok_bridge import _model_config_dicts

logger = logging.getLogger(__name__)

OPENROUTER_KEYS_URL = "https://openrouter.ai/api/v1/keys"
_REQUEST_TIMEOUT_SECONDS = 15.0
# OpenRouter denominates key limits in account credits (1 credit = 1 USD);
# Skynet credits are cents, hence the /100 on every limit push.
_CREDITS_PER_DOLLAR = 100
# Float-compare slack for "is the limit already right": well under the
# half-cent granularity any limit we push can differ by.
_LIMIT_EPSILON_DOLLARS = 0.005


def provisioning_enabled() -> bool:
    """Report whether per-user OpenRouter keys can be minted and stored.

    Provisioning needs both the key-management secret (to call OpenRouter) and
    the vault key (to encrypt minted secrets at rest) — without either, managed
    runs keep using the shared gateway key.

    Returns:
        ``True`` when both ``OPENROUTER_PROVISIONING_KEY`` and
        ``BYOK_VAULT_KEY`` are configured.
    """
    return settings.openrouter_provisioning_key is not None and settings.byok_vault_key is not None


def inject_provisioned_openrouter_key(payload_dict: dict[str, Any], *, api_key: str) -> None:
    """Stamp the account's provisioned OpenRouter key onto a managed payload, in place.

    The managed-run mirror of :func:`.byok_bridge.inject_byok_connections`:
    each ModelConfig is pinned to litellm's native ``openrouter/`` provider
    (the same rewrite the LiteLLM proxy's wildcard performs) and authenticated
    with the per-user runtime key, so the run reaches OpenRouter directly under
    the provider-side cap instead of through the shared gateway key. A config
    already carrying an ``api_key`` is left untouched, as is one with no model
    name to pin.

    Args:
        payload_dict: The raw run/grid payload, mutated in place.
        api_key: The account's provisioned OpenRouter runtime key.
    """
    for cfg in _model_config_dicts(payload_dict):
        name = (cfg.get("name") or "").strip().strip("/")
        if not name:
            continue
        extra = cfg.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            cfg["extra"] = extra
        if "api_key" in extra:
            continue
        cfg["name"] = f"openrouter/{name.removeprefix('openrouter/')}"
        extra["api_key"] = api_key


class OpenRouterKeyProvisioner:
    """Mints one OpenRouter runtime key per account and syncs its spend limit."""

    def __init__(self, *, engine: Any) -> None:
        """Bind the provisioner to the ORM engine backing the billing tables.

        Args:
            engine: SQLAlchemy engine (``job_store.engine``) holding the
                ``billing_openrouter_keys`` table.
        """
        self._engine = engine

    @property
    def enabled(self) -> bool:
        """Report whether provisioning is configured (see :func:`provisioning_enabled`)."""
        return provisioning_enabled()

    def ensure_runtime_key(self, username: str, spendable_credits: int) -> str | None:
        """Return the account's runtime key with its headroom synced to the balance.

        Mints a key through the key-management API on first use; afterwards
        reads the key's accumulated usage and pushes ``limit = usage +
        spendable`` so the remaining provider-side headroom always equals the
        account's spendable credits — shrinking as well as growing. Every
        failure path (API unreachable, unexpected response shape, undecryptable
        stored secret) returns ``None`` so the dispatch falls back to the
        shared gateway key rather than blocking the run; deleting the account's
        ``billing_openrouter_keys`` row forces a fresh mint on the next
        dispatch.

        Args:
            username: Account the managed run bills to.
            spendable_credits: The account's spendable balance in Skynet
                credits (cents).

        Returns:
            The plaintext runtime key, or ``None`` when provisioning is
            disabled or any step failed.
        """
        if not self.enabled:
            return None
        with Session(self._engine) as session:
            row = session.get(BillingOpenRouterKeyModel, username)
            ciphertext = row.secret_ciphertext if row is not None else None
            key_hash = row.key_hash if row is not None else None
        if ciphertext is None or key_hash is None:
            return self._create_key(username, spendable_credits)
        try:
            secret = self._cipher().decrypt(ciphertext).decode("utf-8")
        except InvalidToken:
            logger.warning(
                "Stored OpenRouter key for %s does not decrypt under the current vault key; "
                "falling back to the shared gateway key",
                username,
            )
            return None
        if not self._sync_limit(username, key_hash, spendable_credits):
            return None
        return secret

    def _create_key(self, username: str, spendable_credits: int) -> str | None:
        """Mint a runtime key limited to the balance and persist it encrypted.

        Args:
            username: Account to mint for.
            spendable_credits: Initial spend limit, in Skynet credits (cents).

        Returns:
            The plaintext runtime key, or ``None`` when the mint failed.
        """
        body = self._request(
            "POST",
            OPENROUTER_KEYS_URL,
            {"name": f"skynet-user-{username}", "limit": self._dollars(spendable_credits)},
        )
        if body is None:
            return None
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        secret = body.get("key")
        key_hash = data.get("hash")
        if not isinstance(secret, str) or not secret or not isinstance(key_hash, str) or not key_hash:
            logger.warning("OpenRouter key create returned an unexpected shape for %s", username)
            return None
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            session.merge(
                BillingOpenRouterKeyModel(
                    username=username,
                    key_hash=key_hash,
                    secret_ciphertext=self._cipher().encrypt(secret.encode("utf-8")),
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()
        return secret

    def _sync_limit(self, username: str, key_hash: str, spendable_credits: int) -> bool:
        """Push ``limit = usage + spendable`` onto the account's runtime key.

        Args:
            username: Account the key belongs to (for log context only).
            key_hash: The key's identifier in the key-management API.
            spendable_credits: Desired remaining headroom, in Skynet credits.

        Returns:
            ``True`` when the key's limit now matches the balance (already in
            sync, or the update landed); ``False`` on any API failure.
        """
        detail = self._request("GET", f"{OPENROUTER_KEYS_URL}/{key_hash}")
        if detail is None:
            logger.warning("Could not read OpenRouter key state for %s; using the shared gateway key", username)
            return False
        data = detail.get("data") if isinstance(detail.get("data"), dict) else {}
        usage = data.get("usage")
        usage_dollars = float(usage) if isinstance(usage, (int, float)) else 0.0
        target = round(usage_dollars + self._dollars(spendable_credits), 2)
        current = data.get("limit")
        if isinstance(current, (int, float)) and abs(float(current) - target) < _LIMIT_EPSILON_DOLLARS:
            return True
        return self._request("PATCH", f"{OPENROUTER_KEYS_URL}/{key_hash}", {"limit": target}) is not None

    def _request(self, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Make one authenticated key-management call and return its JSON body.

        Args:
            method: HTTP method.
            url: Full endpoint URL.
            payload: Optional JSON body.

        Returns:
            The parsed JSON object, or ``None`` on any transport error,
            non-success status, or non-object body — callers treat ``None``
            uniformly as "fall back to the shared gateway key".
        """
        provisioning_key = settings.openrouter_provisioning_key
        if provisioning_key is None:
            return None
        headers = {"Authorization": f"Bearer {provisioning_key.get_secret_value()}"}
        try:
            response = httpx.request(
                method, url, headers=headers, json=payload, timeout=_REQUEST_TIMEOUT_SECONDS
            )
        except httpx.HTTPError:
            logger.warning("OpenRouter key API unreachable (%s %s)", method, url)
            return None
        if not response.is_success:
            logger.warning("OpenRouter key API %s %s failed: HTTP %s", method, url, response.status_code)
            return None
        try:
            body = response.json()
        except ValueError:
            return None
        return body if isinstance(body, dict) else None

    def _cipher(self) -> Fernet:
        """Return a Fernet cipher built from the configured vault key.

        Returns:
            The cipher encrypting/decrypting stored runtime-key secrets.

        Raises:
            RuntimeError: When no vault key is configured — unreachable behind
                the ``enabled`` gate, kept for a clear failure if that drifts.
        """
        if settings.byok_vault_key is None:
            raise RuntimeError("BYOK_VAULT_KEY is not configured")
        return Fernet(settings.byok_vault_key.get_secret_value().encode("utf-8"))

    @staticmethod
    def _dollars(credits: int) -> float:
        """Convert Skynet credits (cents) to OpenRouter's dollar denomination.

        Args:
            credits: Amount in Skynet credits; negative clamps to zero.

        Returns:
            The non-negative dollar amount.
        """
        return max(0, credits) / _CREDITS_PER_DOLLAR
