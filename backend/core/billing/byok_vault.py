"""Encrypt-at-rest vault for bring-your-own-key (BYOK) provider secrets.

When an account runs in ``byok`` token source, jobs bill the user's own provider
key instead of Skynet credits. This module is the only place that holds those
secrets: it encrypts them with Fernet (symmetric AES) under
``settings.byok_vault_key`` before they touch the database, decrypts them only
to run a verify probe or hand them to a run, and never returns plaintext to the
API surface. The stored row keeps only the ciphertext, a masked tail (``last4``)
for display, and a verification ``status`` — so a database dump never leaks a
usable key.

The verify probe makes one lightweight authenticated request to the provider on
key entry, so a typo'd or revoked key is caught before a job ever runs rather
than failing mid-optimization. A successful probe flips the stored status to
``verified``; an auth rejection flips it to ``invalid``; a transient/network
error leaves it ``unverified`` (the probe couldn't reach a verdict, not a bad
key).

Reads of the masked metadata work whether or not the vault is configured;
saving a key requires ``settings.is_byok_vault_configured`` and raises
``DomainError("billing.byok_not_configured", 503)`` otherwise — never a 500.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.orm import Session

from ..api.errors import DomainError
from ..config import settings
from ..storage.models import BillingProviderKeyModel

# Verification states a stored key can carry. ``unverified`` is the default and
# also the resting state after a probe that couldn't reach a verdict (network /
# transient error); ``verified`` and ``invalid`` are set only by a probe that
# got a definitive answer from the provider.
STATUS_UNVERIFIED = "unverified"
STATUS_VERIFIED = "verified"
STATUS_INVALID = "invalid"

# Providers a user may bring a key for. Mirrors the frontend ``BYOK_PROVIDERS``
# catalog; the value is how the verify probe reaches each provider — the models
# (or equivalent) endpoint plus the auth header shape that lists it. A bare
# authenticated GET is enough to tell a working key from a rejected one without
# spending tokens. ``header`` is templated with the secret at probe time.
_PROVIDER_PROBES: dict[str, dict[str, str]] = {
    "openai": {
        "url": "https://api.openai.com/v1/models",
        "header_name": "Authorization",
        "header_value": "Bearer {secret}",
    },
    "anthropic": {
        "url": "https://api.anthropic.com/v1/models",
        "header_name": "x-api-key",
        "header_value": "{secret}",
        "extra_header_name": "anthropic-version",
        "extra_header_value": "2023-06-01",
    },
    "google": {
        "url": "https://generativelanguage.googleapis.com/v1beta/models",
        "header_name": "x-goog-api-key",
        "header_value": "{secret}",
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/models",
        "header_name": "Authorization",
        "header_value": "Bearer {secret}",
    },
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/models",
        "header_name": "Authorization",
        "header_value": "Bearer {secret}",
    },
}

# How long a verify probe waits for the provider before giving up. A timeout is
# treated as "couldn't reach a verdict" (status stays ``unverified``), never as
# an invalid key.
_PROBE_TIMEOUT_SECONDS = 8.0


@dataclass(frozen=True)
class ProviderKeyView:
    """One stored BYOK key as the API surface sees it — never the secret.

    ``last4`` is the recognizable tail for masked display, ``status`` is the
    verification state, and ``added_at`` is the ISO-8601 instant the key was
    saved. The plaintext secret never appears here.
    """

    provider: str
    last4: str
    status: str
    added_at: str


@dataclass(frozen=True)
class VaultSnapshot:
    """The account's stored BYOK keys as a single read for the settings surface."""

    keys: list[ProviderKeyView] = field(default_factory=list)


def key_last4(secret: str) -> str:
    """Return the recognizable tail of a secret for masked display.

    Args:
        secret: The plaintext provider key.

    Returns:
        The last four characters, or a dotted placeholder when the secret is
        shorter than that.
    """
    tail = secret[-4:]
    return tail if len(tail) == 4 else "····"


class ProviderKeyVault:
    """Stores, verifies, and retrieves BYOK provider secrets, encrypted at rest."""

    def __init__(self, *, engine: Any) -> None:
        """Bind the vault to the ORM engine backing the provider-key table.

        Args:
            engine: SQLAlchemy engine (``job_store.engine``) used for every
                vault session. The Fernet cipher is built lazily per mutation,
                so constructing the vault never requires the vault key.
        """
        self._engine = engine

    def _cipher(self) -> Fernet:
        """Return a Fernet cipher built from the configured vault key.

        Returns:
            The Fernet cipher used to encrypt/decrypt stored secrets.

        Raises:
            DomainError: 503 when no BYOK vault key is configured.
        """
        if settings.byok_vault_key is None:
            raise DomainError("billing.byok_not_configured", status=503)
        return Fernet(settings.byok_vault_key.get_secret_value().encode("utf-8"))

    def list_keys(self, username: str) -> VaultSnapshot:
        """Return the account's stored keys as masked, secret-free views.

        A pure DB read — no decryption, no provider call — so it serves even when
        the vault key is unconfigured (the ciphertext is never touched). Results
        are ordered by provider for a stable settings list.

        Args:
            username: Account whose keys are listed.

        Returns:
            A :class:`VaultSnapshot` of masked key views.
        """
        with Session(self._engine) as session:
            rows = (
                session.query(BillingProviderKeyModel)
                .filter(BillingProviderKeyModel.username == username)
                .order_by(BillingProviderKeyModel.provider)
                .all()
            )
            keys = [
                ProviderKeyView(
                    provider=row.provider,
                    last4=row.last4,
                    status=row.status,
                    added_at=row.created_at.isoformat(),
                )
                for row in rows
            ]
        return VaultSnapshot(keys=keys)

    def save_key(self, username: str, provider: str, secret: str) -> ProviderKeyView:
        """Encrypt and store a provider secret, then verify it on entry.

        The secret is Fernet-encrypted before it touches the database and the
        plaintext is dropped immediately; only the ciphertext, the masked tail,
        and a verification status are persisted. Saving replaces any existing key
        for the same ``(username, provider)`` (rotation). The verify probe runs
        synchronously so the returned view already carries the entry-time verdict
        — a typo'd or revoked key is caught before a job ever runs.

        Args:
            username: Account the key belongs to.
            provider: Provider slug (must be a known BYOK provider).
            secret: The plaintext provider key; never persisted in the clear.

        Returns:
            The masked, verified view of the stored key.

        Raises:
            DomainError: 400 when ``provider`` is unknown or ``secret`` is empty;
                503 when the vault key is unconfigured.
        """
        secret = secret.strip()
        if provider not in _PROVIDER_PROBES:
            raise DomainError("billing.byok_unknown_provider", status=400, provider=provider)
        if not secret:
            raise DomainError("billing.byok_empty_secret", status=400, provider=provider)
        cipher = self._cipher()
        ciphertext = cipher.encrypt(secret.encode("utf-8"))
        status = self._probe(provider, secret)
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            row = session.get(BillingProviderKeyModel, (username, provider))
            if row is None:
                row = BillingProviderKeyModel(
                    username=username,
                    provider=provider,
                    secret_ciphertext=ciphertext,
                    last4=key_last4(secret),
                    status=status,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)
            else:
                row.secret_ciphertext = ciphertext
                row.last4 = key_last4(secret)
                row.status = status
                row.updated_at = now
            added_at = row.created_at.isoformat()
            session.commit()
        return ProviderKeyView(provider=provider, last4=key_last4(secret), status=status, added_at=added_at)

    def verify_key(self, username: str, provider: str) -> ProviderKeyView:
        """Re-run the verify probe against a stored key and persist the verdict.

        Decrypts the stored secret in memory only for the duration of the probe,
        updates the stored status to the probe's verdict, and returns the masked
        view. Used to re-check a key that was saved while the provider was
        unreachable (status stuck at ``unverified``).

        Args:
            username: Account the key belongs to.
            provider: Provider slug whose stored key is re-verified.

        Returns:
            The masked view carrying the fresh verification status.

        Raises:
            DomainError: 404 when no key is stored for the provider; 503 when the
                vault key is unconfigured.
        """
        cipher = self._cipher()
        with Session(self._engine) as session:
            row = session.get(BillingProviderKeyModel, (username, provider))
            if row is None:
                raise DomainError("billing.byok_key_not_found", status=404, provider=provider)
            try:
                secret = cipher.decrypt(row.secret_ciphertext).decode("utf-8")
            except InvalidToken as exc:
                # The stored ciphertext can't be decrypted with the current vault
                # key (the key was rotated out from under the row). Mark it invalid
                # so the UI prompts a re-entry rather than wedging on a dead probe.
                row.status = STATUS_INVALID
                row.updated_at = datetime.now(UTC)
                view = ProviderKeyView(
                    provider=provider, last4=row.last4, status=STATUS_INVALID, added_at=row.created_at.isoformat()
                )
                session.commit()
                raise DomainError("billing.byok_key_undecryptable", status=409, provider=provider) from exc
            status = self._probe(provider, secret)
            row.status = status
            row.updated_at = datetime.now(UTC)
            view = ProviderKeyView(
                provider=provider, last4=row.last4, status=status, added_at=row.created_at.isoformat()
            )
            session.commit()
        return view

    def remove_key(self, username: str, provider: str) -> None:
        """Forget a stored provider key.

        A no-op when no key is stored for the provider, so the call is idempotent.
        Requires no vault key (it never decrypts), so a key can be removed even on
        a deploy whose vault key was lost.

        Args:
            username: Account the key belongs to.
            provider: Provider slug whose key is removed.
        """
        with Session(self._engine) as session:
            row = session.get(BillingProviderKeyModel, (username, provider))
            if row is not None:
                session.delete(row)
                session.commit()

    def reveal_secret(self, username: str, provider: str) -> str | None:
        """Decrypt and return a stored secret for a run that bills the user's key.

        The only path that hands plaintext back, used by the run pipeline to bill
        a BYOK job against the user's own provider key — never exposed through the
        HTTP surface. Returns ``None`` when no key is stored.

        Args:
            username: Account the key belongs to.
            provider: Provider slug whose secret is needed.

        Returns:
            The plaintext provider key, or ``None`` when none is stored.

        Raises:
            DomainError: 503 when the vault key is unconfigured; 409 when the
                stored ciphertext can't be decrypted with the current vault key.
        """
        cipher = self._cipher()
        with Session(self._engine) as session:
            row = session.get(BillingProviderKeyModel, (username, provider))
            if row is None:
                return None
            ciphertext = row.secret_ciphertext
        try:
            return cipher.decrypt(ciphertext).decode("utf-8")
        except InvalidToken as exc:
            raise DomainError("billing.byok_key_undecryptable", status=409, provider=provider) from exc

    def _probe(self, provider: str, secret: str) -> str:
        """Make one authenticated request to a provider and classify the verdict.

        A ``2xx`` response means the key works (:data:`STATUS_VERIFIED`); a
        ``401``/``403`` means the provider rejected it (:data:`STATUS_INVALID`).
        Any other status, a timeout, or a network error means the probe couldn't
        reach a verdict, so the status stays :data:`STATUS_UNVERIFIED` — the key
        is not condemned for an outage on the provider's side.

        Args:
            provider: Provider slug (assumed present in :data:`_PROVIDER_PROBES`).
            secret: The plaintext key to authenticate the probe with.

        Returns:
            One of :data:`STATUS_VERIFIED`, :data:`STATUS_INVALID`, or
            :data:`STATUS_UNVERIFIED`.
        """
        probe = _PROVIDER_PROBES[provider]
        headers = {probe["header_name"]: probe["header_value"].format(secret=secret)}
        extra_name = probe.get("extra_header_name")
        if extra_name:
            headers[extra_name] = probe["extra_header_value"]
        try:
            response = httpx.get(probe["url"], headers=headers, timeout=_PROBE_TIMEOUT_SECONDS)
        except httpx.HTTPError:
            return STATUS_UNVERIFIED
        if response.is_success:
            return STATUS_VERIFIED
        if response.status_code in (401, 403):
            return STATUS_INVALID
        return STATUS_UNVERIFIED
