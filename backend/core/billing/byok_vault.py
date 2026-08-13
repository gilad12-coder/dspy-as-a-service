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
from ..provider_registry import LITELLM_TO_BYOK_PROVIDER
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
_RESERVED_CONNECTION_PARAMS = frozenset({"api_key", "api_base", "base_url", "model"})


def safe_connection_params(params: dict[str, Any] | None) -> dict[str, Any]:
    """Remove fields that could override the encrypted connection or selected model.

    Args:
        params: Optional extra LiteLLM keyword arguments from the caller.

    Returns:
        A copy containing only non-reserved runtime parameters.
    """
    return {key: value for key, value in (params or {}).items() if key not in _RESERVED_CONNECTION_PARAMS}


def byok_provider_for_litellm(prefix: str) -> str:
    """Return the BYOK vault slug a LiteLLM provider prefix resolves a key for.

    The run path extracts a model's leading ``provider/`` segment (a LiteLLM
    prefix such as ``gemini`` or ``together_ai``); this maps it back to the vault
    slug the user saved their key under (``google`` / ``together``). Identity for
    prefixes whose vault slug already matches.

    Args:
        prefix: The LiteLLM provider prefix from a model id.

    Returns:
        The vault provider slug to resolve a connection for.
    """
    return LITELLM_TO_BYOK_PROVIDER.get(prefix, prefix)


@dataclass(frozen=True)
class ProviderKeyView:
    """One stored BYOK connection as the API surface sees it — never the secret.

    ``id`` is the connection's stable handle, ``label`` an optional user-facing
    name, ``last4`` the recognizable tail for masked display, ``api_base`` the
    optional custom endpoint, ``status`` the verification state, and ``added_at``
    the ISO-8601 instant the connection was saved. The plaintext secret never
    appears here.
    """

    id: str
    provider: str
    label: str | None
    last4: str
    api_base: str | None
    status: str
    added_at: str


@dataclass(frozen=True)
class ResolvedConnection:
    """A decrypted BYOK connection for the run path — secret plus its endpoint.

    Handed only to the in-process run pipeline (never the HTTP surface) so a BYOK
    job can authenticate against the user's own provider key, custom ``api_base``,
    and extra LiteLLM ``params``.
    """

    provider: str
    secret: str
    api_base: str | None
    params: dict[str, Any]


@dataclass(frozen=True)
class VaultSnapshot:
    """The account's stored BYOK connections as a single read for the settings surface."""

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


def _row_view(row: BillingProviderKeyModel) -> ProviderKeyView:
    """Project a stored connection row onto its masked, secret-free view.

    Args:
        row: The persisted connection row (already flushed so ``id`` is set).

    Returns:
        The masked :class:`ProviderKeyView` the API surface returns.
    """
    return ProviderKeyView(
        id=row.id,
        provider=row.provider,
        label=row.label,
        last4=row.last4,
        api_base=row.api_base,
        status=row.status,
        added_at=row.created_at.isoformat(),
    )


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

    def _provider_row(self, session: Session, username: str, provider: str) -> BillingProviderKeyModel | None:
        """Return the account's primary connection for a provider, or ``None``.

        When several connections exist for one provider the oldest wins, so the
        provider-addressed helpers (rotate, verify, reveal) operate on a stable
        choice. ``id``-addressed callers bypass this.

        Args:
            session: Open ORM session bound to the vault engine.
            username: Account whose connection is sought.
            provider: Provider slug to match.

        Returns:
            The matching row, or ``None`` when the account has no connection for
            the provider.
        """
        return (
            session.query(BillingProviderKeyModel)
            .filter(
                BillingProviderKeyModel.username == username,
                BillingProviderKeyModel.provider == provider,
            )
            .order_by(BillingProviderKeyModel.created_at)
            .first()
        )

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
                .order_by(BillingProviderKeyModel.provider, BillingProviderKeyModel.created_at)
                .all()
            )
            keys = [_row_view(row) for row in rows]
        return VaultSnapshot(keys=keys)

    def has_connection(self, username: str, provider: str) -> bool:
        """Return whether the account has any stored connection for a provider.

        A pure existence query — no decryption — so it answers even when the
        vault key is unconfigured. Used by the submit-time BYOK gate to reject a
        run the user has no key for, before the job is ever queued.

        Args:
            username: Account to check.
            provider: Provider slug to look for.

        Returns:
            True when at least one connection is stored for the provider.
        """
        with Session(self._engine) as session:
            return (
                session.query(BillingProviderKeyModel.id)
                .filter(
                    BillingProviderKeyModel.username == username,
                    BillingProviderKeyModel.provider == provider,
                )
                .first()
                is not None
            )

    def has_verified_connection(self, username: str, provider: str) -> bool:
        """Return whether the account has a verified connection for a provider.

        Args:
            username: Account to check.
            provider: Provider slug to look for.

        Returns:
            True when at least one verified connection is stored.
        """
        with Session(self._engine) as session:
            return (
                session.query(BillingProviderKeyModel.id)
                .filter(
                    BillingProviderKeyModel.username == username,
                    BillingProviderKeyModel.provider == provider,
                    BillingProviderKeyModel.status == STATUS_VERIFIED,
                )
                .first()
                is not None
            )

    def save_key(
        self,
        username: str,
        provider: str,
        secret: str,
        *,
        label: str | None = None,
        api_base: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> ProviderKeyView:
        """Encrypt and store a provider connection, then verify it on entry.

        The secret is Fernet-encrypted before it touches the database and the
        plaintext is dropped immediately; only the ciphertext, the masked tail,
        the optional endpoint metadata, and a verification status are persisted.
        Saving rotates the account's existing connection for the same provider in
        place (the simple one-per-provider path); the verify probe runs
        synchronously — against ``api_base`` when supplied, so a custom endpoint
        is checked too — so the returned view already carries the entry-time
        verdict.

        Args:
            username: Account the connection belongs to.
            provider: Provider slug — a known BYOK provider, or any slug when a
                custom ``api_base`` is supplied.
            secret: The plaintext provider key; never persisted in the clear.
            label: Optional user-facing name for the connection.
            api_base: Optional custom endpoint; required for an unknown provider.
            params: Optional extra LiteLLM kwargs to carry with the connection.

        Returns:
            The masked, verified view of the stored connection.

        Raises:
            DomainError: 400 when ``provider`` is unknown and no ``api_base`` is
                given, or ``secret`` is empty; 503 when the vault key is
                unconfigured.
        """
        secret = secret.strip()
        api_base = (api_base or "").strip() or None
        if provider not in _PROVIDER_PROBES and api_base is None:
            raise DomainError("billing.byok_unknown_provider", status=400, provider=provider)
        if not secret:
            raise DomainError("billing.byok_empty_secret", status=400, provider=provider)
        cipher = self._cipher()
        ciphertext = cipher.encrypt(secret.encode("utf-8"))
        status = self._probe(provider, secret, api_base)
        safe_params = safe_connection_params(params)
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            row = self._provider_row(session, username, provider)
            if row is None:
                row = BillingProviderKeyModel(
                    username=username,
                    provider=provider,
                    label=label,
                    api_base=api_base,
                    params=safe_params,
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
                row.label = label
                row.api_base = api_base
                row.params = safe_params
                row.updated_at = now
            session.flush()
            view = _row_view(row)
            session.commit()
        return view

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
            row = self._provider_row(session, username, provider)
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
                session.commit()
                raise DomainError("billing.byok_key_undecryptable", status=409, provider=provider) from exc
            status = self._probe(provider, secret, row.api_base)
            row.status = status
            row.updated_at = datetime.now(UTC)
            view = _row_view(row)
            session.commit()
        return view

    def remove_key(self, username: str, provider: str) -> None:
        """Forget a stored provider key.

        A no-op when no key is stored for the provider, so the call is idempotent.
        Requires no vault key (it never decrypts), so a key can be removed even on
        a deploy whose vault key was lost.

        Args:
            username: Account the connection belongs to.
            provider: Provider slug whose connection(s) are removed.
        """
        with Session(self._engine) as session:
            rows = (
                session.query(BillingProviderKeyModel)
                .filter(
                    BillingProviderKeyModel.username == username,
                    BillingProviderKeyModel.provider == provider,
                )
                .all()
            )
            for row in rows:
                session.delete(row)
            if rows:
                session.commit()

    def resolve_connection(self, username: str, provider: str) -> ResolvedConnection | None:
        """Decrypt the account's best connection for a provider, for the run path.

        Picks the connection most likely to work — a ``verified`` one first, then
        the most recently updated — and decrypts it in memory only. Used by the
        run pipeline (BYOK bridge) to authenticate a job against the user's own
        key and endpoint; never exposed through the HTTP surface.

        Args:
            username: Account the connection belongs to.
            provider: Provider slug whose connection is resolved.

        Returns:
            The decrypted :class:`ResolvedConnection`, or ``None`` when the
            account has no connection for the provider.

        Raises:
            DomainError: 503 when the vault key is unconfigured; 409 when the
                stored ciphertext can't be decrypted with the current vault key.
        """
        cipher = self._cipher()
        with Session(self._engine) as session:
            rows = (
                session.query(BillingProviderKeyModel)
                .filter(
                    BillingProviderKeyModel.username == username,
                    BillingProviderKeyModel.provider == provider,
                )
                .all()
            )
            if not rows:
                return None
            rows.sort(key=lambda r: (r.status != STATUS_VERIFIED, -r.updated_at.timestamp()))
            row = rows[0]
            ciphertext = row.secret_ciphertext
            api_base = row.api_base
            params = dict(row.params or {})
        try:
            secret = cipher.decrypt(ciphertext).decode("utf-8")
        except InvalidToken as exc:
            raise DomainError("billing.byok_key_undecryptable", status=409, provider=provider) from exc
        return ResolvedConnection(provider=provider, secret=secret, api_base=api_base, params=params)

    def reveal_secret(self, username: str, provider: str) -> str | None:
        """Decrypt and return a stored secret for a run that bills the user's key.

        Thin wrapper over :meth:`resolve_connection` for callers that need only
        the plaintext key. Returns ``None`` when no connection is stored.

        Args:
            username: Account the connection belongs to.
            provider: Provider slug whose secret is needed.

        Returns:
            The plaintext provider key, or ``None`` when none is stored.

        Raises:
            DomainError: 503 when the vault key is unconfigured; 409 when the
                stored ciphertext can't be decrypted with the current vault key.
        """
        resolved = self.resolve_connection(username, provider)
        return resolved.secret if resolved is not None else None

    def _probe(self, provider: str, secret: str, api_base: str | None = None) -> str:
        """Make one authenticated request to a provider and classify the verdict.

        A ``2xx`` response means the key works (:data:`STATUS_VERIFIED`); a
        ``401``/``403`` means the provider rejected it (:data:`STATUS_INVALID`).
        Any other status, a timeout, or a network error means the probe couldn't
        reach a verdict, so the status stays :data:`STATUS_UNVERIFIED` — the key
        is not condemned for an outage on the provider's side. A custom
        ``api_base`` is probed at ``{api_base}/models``: with the provider's own
        auth header shape when the provider is known, or OpenAI-compatible Bearer
        auth for an unknown provider.

        Args:
            provider: Provider slug.
            secret: The plaintext key to authenticate the probe with.
            api_base: Optional custom endpoint to probe instead of the provider's
                default; required when ``provider`` is unknown.

        Returns:
            One of :data:`STATUS_VERIFIED`, :data:`STATUS_INVALID`, or
            :data:`STATUS_UNVERIFIED`.
        """
        probe = _PROVIDER_PROBES.get(provider)
        if probe is not None:
            urls = self._model_probe_urls(api_base) if api_base else [probe["url"]]
            headers = {probe["header_name"]: probe["header_value"].format(secret=secret)}
            extra_name = probe.get("extra_header_name")
            if extra_name:
                headers[extra_name] = probe["extra_header_value"]
        elif api_base:
            urls = self._model_probe_urls(api_base)
            headers = {"Authorization": f"Bearer {secret}"}
        else:
            return STATUS_UNVERIFIED
        for url in urls:
            try:
                response = httpx.get(url, headers=headers, timeout=_PROBE_TIMEOUT_SECONDS)
            except httpx.HTTPError:
                return STATUS_UNVERIFIED
            if response.is_success:
                return STATUS_VERIFIED
            if response.status_code in (401, 403):
                return STATUS_INVALID
        return STATUS_UNVERIFIED

    @staticmethod
    def _model_probe_urls(api_base: str) -> list[str]:
        """Return model-list probe URLs for an OpenAI-compatible API base.

        Args:
            api_base: User-supplied endpoint root or versioned API base.

        Returns:
            One or two candidate ``/models`` URLs in preferred order.
        """
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            return [f"{base}/models"]
        return [f"{base}/v1/models", f"{base}/models"]
