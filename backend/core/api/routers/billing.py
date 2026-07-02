"""Managed-credit billing routes: wallet, checkout, subscription, portal, webhook.

Backs the in-app wallet, the upgrade/paywall page, and the Premium subscription.
Authenticated routes resolve the caller via the shared session-JWT/PAT
dependency and key every operation on ``user.username`` (the lowercased email).
The webhook route is deliberately unauthenticated — Stripe can't present a
bearer token — and instead verifies the event's signature in the service.

Wallet reads succeed even when Stripe is unconfigured (free-tier zeros);
mutations raise ``DomainError("billing.not_configured", 503)``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from ...billing import ProviderKeyVault, StripeBillingService
from ..auth import AuthenticatedUser, get_authenticated_user

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]

# Default usage window when the dashboard omits an explicit range.
USAGE_DEFAULT_WINDOW_DAYS = 30


def _parse_instant(value: str | None, default: datetime) -> datetime:
    """Parse an ISO-8601 query value into a UTC-aware instant, or fall back.

    Tolerates a trailing ``Z`` and a naive value (assumed UTC). A malformed or
    missing value yields ``default`` rather than a 422, so a stray query string
    degrades to the default window instead of failing the dashboard read.

    Args:
        value: Raw query-string value, or ``None`` when the param was omitted.
        default: Instant returned when ``value`` is missing or unparseable.

    Returns:
        A timezone-aware UTC datetime.
    """
    if not value:
        return default
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return default
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class FreeGrantResponse(BaseModel):
    """The account's credit grant: one-time for free, monthly for Premium."""

    credits_remaining: int = Field(description="Credits left in the grant.")
    credits_total: int = Field(description="Full grant size (500 free, or the Premium allotment).")
    resets_at: str | None = Field(
        default=None,
        description="ISO-8601 instant a Premium allotment next renews; null for the one-time free grant.",
    )


class UsageEntryResponse(BaseModel):
    """One row of the credit usage ledger."""

    id: str = Field(description="Stable row id.")
    at: str = Field(description="ISO-8601 instant the entry was recorded.")
    label: str = Field(description="Human label for the row (a run name, 'Top-up', etc.).")
    model: str | None = Field(default=None, description="Model id for a run row, else null.")
    credits: int = Field(description="Signed credit delta: negative for a spend.")
    kind: str = Field(description="Entry kind: 'run', 'topup', or 'grant'.")


class WalletResponse(BaseModel):
    """The caller's wallet: purchased balance, free grant, subscription, recent ledger."""

    paid_balance_credits: int = Field(description="Purchased credit balance, on top of the free grant.")
    free_grant: FreeGrantResponse
    premium_active: bool = Field(description="Whether an active Premium subscription is in effect.")
    subscription_status: str | None = Field(
        default=None, description="Raw Stripe subscription status, or null when none."
    )
    subscription_current_period_end: str | None = Field(
        default=None, description="ISO-8601 end of the current paid period, or null."
    )
    usage: list[UsageEntryResponse] = Field(
        default_factory=list, description="Most-recent-first ledger rows."
    )


class UsageDayResponse(BaseModel):
    """One day's run spend, split into billed and refunded credits."""

    date: str = Field(description="Calendar day (YYYY-MM-DD, UTC).")
    billed_credits: int = Field(description="Gross run credits billed that day.")
    refunded_credits: int = Field(description="Run credits refunded that day by the guarantee.")


class UsageModelResponse(BaseModel):
    """One model's share of run spend over the window."""

    model: str | None = Field(default=None, description="Model id, or null for runs without one.")
    credits: int = Field(description="Gross run credits billed to this model.")
    runs: int = Field(description="Billed runs attributed to this model.")


class UsageResponse(BaseModel):
    """Date-ranged usage rollup for the billing Usage dashboard."""

    start: str = Field(description="ISO-8601 inclusive window start.")
    end: str = Field(description="ISO-8601 inclusive window end.")
    billed_credits: int = Field(description="Gross run credits billed across the window.")
    refunded_credits: int = Field(description="Run credits refunded across the window.")
    runs: int = Field(description="Billed runs across the window.")
    by_day: list[UsageDayResponse] = Field(
        default_factory=list, description="Per-day spend series, ascending by date."
    )
    by_model: list[UsageModelResponse] = Field(
        default_factory=list, description="Per-model spend series, descending by credits."
    )
    entries: list[UsageEntryResponse] = Field(
        default_factory=list, description="Most-recent-first raw ledger rows in the window."
    )


# Request to start a credit-pack checkout. pack_id is one of starter/plus/pro.
class CheckoutRequest(BaseModel):
    pack_id: str = Field(description="Credit pack to buy: 'starter', 'plus', or 'pro'.")


class CheckoutSessionResponse(BaseModel):
    """A hosted Stripe URL the client should redirect the browser to."""

    url: str = Field(description="Absolute Stripe-hosted Checkout or portal URL.")


class FoundersRateResponse(BaseModel):
    """The Founder's Rate availability: the deadline gate and the price-lock window."""

    open: bool = Field(description="Whether new Founder's Rate subscriptions are still accepted.")
    closes_at: str = Field(description="ISO-8601 instant the offer stops accepting subscribers.")
    price_locked_until: str = Field(
        description="ISO-8601 instant through which a subscriber locking in now keeps the rate."
    )


class WebhookAck(BaseModel):
    """Acknowledgement that a webhook event was received and applied."""

    received: bool = Field(description="Always true once the event was verified and applied.")


class ProviderKeyResponse(BaseModel):
    """One stored BYOK provider connection as the settings UI sees it — never the secret."""

    id: str = Field(description="Stable handle for the connection.")
    provider: str = Field(description="Provider slug the connection belongs to (e.g. 'openai').")
    label: str | None = Field(default=None, description="Optional user-facing name for the connection.")
    last4: str = Field(description="Masked tail of the secret, for recognition without revealing it.")
    api_base: str | None = Field(default=None, description="Optional custom endpoint the connection targets.")
    status: str = Field(description="Verification state: 'unverified', 'verified', or 'invalid'.")
    added_at: str = Field(description="ISO-8601 instant the connection was saved.")


class ProviderKeysResponse(BaseModel):
    """The caller's stored BYOK provider keys, masked."""

    keys: list[ProviderKeyResponse] = Field(
        default_factory=list, description="Stored keys, one per provider, ordered by provider."
    )


# Request to save (or rotate) a provider's BYOK connection. The secret is
# encrypted at rest the instant it lands and is never returned. ``api_base`` and
# ``params`` let a connection target any OpenAI-compatible host; ``api_base`` is
# required when the provider slug is not a known BYOK provider.
class SaveProviderKeyRequest(BaseModel):
    provider: str = Field(description="Provider slug to save the connection for (e.g. 'anthropic').")
    secret: str = Field(description="The plaintext provider key; stored encrypted, never echoed back.")
    label: str | None = Field(default=None, description="Optional user-facing name for the connection.")
    api_base: str | None = Field(default=None, description="Optional custom endpoint; required for an unknown provider.")
    params: dict[str, Any] = Field(default_factory=dict, description="Optional extra LiteLLM kwargs for the connection.")


def create_billing_router(*, job_store) -> APIRouter:
    """Build the managed-credit billing router.

    Args:
        job_store: Job-store instance whose ORM engine backs the billing tables.

    Returns:
        A FastAPI ``APIRouter`` exposing wallet, checkout, subscribe, portal, and
        the Stripe webhook receiver.
    """
    router = APIRouter()
    service = StripeBillingService(engine=job_store.engine)
    vault = ProviderKeyVault(engine=job_store.engine)

    @router.get(
        "/billing/wallet",
        response_model=WalletResponse,
        summary="Return the caller's credit wallet and subscription state",
    )
    def get_wallet(user: AuthenticatedUserDep) -> WalletResponse:
        """Return the caller's purchased balance, free grant, subscription, and ledger.

        Args:
            user: Authenticated caller whose wallet is read.

        Returns:
            The wallet snapshot; free-tier zeros when the caller has no billing row.
        """
        snapshot = service.get_wallet(user.username)
        return WalletResponse(
            paid_balance_credits=snapshot.paid_balance_credits,
            free_grant=FreeGrantResponse(
                credits_remaining=snapshot.free_grant_remaining,
                credits_total=snapshot.free_grant_total,
                resets_at=snapshot.free_grant_resets_at,
            ),
            premium_active=snapshot.premium_active,
            subscription_status=snapshot.subscription_status,
            subscription_current_period_end=snapshot.subscription_current_period_end,
            usage=[
                UsageEntryResponse(
                    id=row.id,
                    at=row.at,
                    label=row.label,
                    model=row.model,
                    credits=row.credits,
                    kind=row.kind,
                )
                for row in snapshot.usage
            ],
        )

    @router.get(
        "/billing/usage",
        response_model=UsageResponse,
        summary="Return a date-ranged usage rollup for the caller",
    )
    def get_usage(
        user: AuthenticatedUserDep,
        start: str | None = None,
        end: str | None = None,
    ) -> UsageResponse:
        """Return the caller's usage rollup over an optional date window.

        ``start``/``end`` are ISO-8601 instants; an omitted or unparseable bound
        defaults the window to the last :data:`USAGE_DEFAULT_WINDOW_DAYS` days
        ending now. A pure ledger read — serves on a key-less deploy (empty
        rollups for an account with no runs).

        Args:
            user: Authenticated caller whose ledger is summarized.
            start: ISO-8601 window start, or null for ``end`` minus the default window.
            end: ISO-8601 window end, or null for now.

        Returns:
            The totals, per-day and per-model series, and recent ledger rows.
        """
        now = datetime.now(UTC)
        end_dt = _parse_instant(end, now)
        start_dt = _parse_instant(start, end_dt - timedelta(days=USAGE_DEFAULT_WINDOW_DAYS))
        snapshot = service.get_usage(user.username, start_dt, end_dt)
        return UsageResponse(
            start=snapshot.start,
            end=snapshot.end,
            billed_credits=snapshot.billed_credits,
            refunded_credits=snapshot.refunded_credits,
            runs=snapshot.runs,
            by_day=[
                UsageDayResponse(
                    date=day.date,
                    billed_credits=day.billed_credits,
                    refunded_credits=day.refunded_credits,
                )
                for day in snapshot.by_day
            ],
            by_model=[
                UsageModelResponse(model=row.model, credits=row.credits, runs=row.runs)
                for row in snapshot.by_model
            ],
            entries=[
                UsageEntryResponse(
                    id=row.id,
                    at=row.at,
                    label=row.label,
                    model=row.model,
                    credits=row.credits,
                    kind=row.kind,
                )
                for row in snapshot.entries
            ],
        )

    @router.post(
        "/billing/checkout",
        response_model=CheckoutSessionResponse,
        summary="Start a Stripe Checkout session for a credit pack",
    )
    def create_checkout(body: CheckoutRequest, user: AuthenticatedUserDep) -> CheckoutSessionResponse:
        """Create a one-time Checkout session for a credit pack.

        Args:
            body: The pack to buy.
            user: Authenticated buyer; credits land on their account via webhook.

        Returns:
            The hosted Checkout URL to redirect the buyer to.
        """
        url = service.create_pack_checkout(user.username, body.pack_id)
        return CheckoutSessionResponse(url=url)

    @router.post(
        "/billing/subscribe",
        response_model=CheckoutSessionResponse,
        summary="Start a Stripe Checkout session for the Premium subscription",
    )
    def create_subscription(user: AuthenticatedUserDep) -> CheckoutSessionResponse:
        """Create a subscription Checkout session for Premium.

        Args:
            user: Authenticated subscriber.

        Returns:
            The hosted Checkout URL for the recurring subscription.
        """
        url = service.create_subscription_checkout(user.username)
        return CheckoutSessionResponse(url=url)

    @router.get(
        "/billing/founders",
        response_model=FoundersRateResponse,
        summary="Return the Founder's Rate availability and price-lock window",
    )
    def get_founders_rate(user: AuthenticatedUserDep) -> FoundersRateResponse:
        """Return whether the Founder's Rate is still open and its 12-month lock window.

        A pure config read (the deadline gate) — no Stripe call — so it serves on a
        key-less deploy. The caller is resolved only to keep the route consistent
        with the rest of the wallet surface.

        Args:
            user: Authenticated caller (unused beyond auth).

        Returns:
            The offer's open/closed state, close date, and price-lock-until date.
        """
        status = service.founders_rate_status()
        return FoundersRateResponse(
            open=status.open,
            closes_at=status.closes_at,
            price_locked_until=status.price_locked_until,
        )

    @router.post(
        "/billing/founders/subscribe",
        response_model=CheckoutSessionResponse,
        summary="Start a Stripe Checkout session for the Founder's Rate",
    )
    def create_founders_subscription(user: AuthenticatedUserDep) -> CheckoutSessionResponse:
        """Create a Founder's Rate subscription Checkout session.

        Args:
            user: Authenticated subscriber; the 12-month price-lock is stamped onto
                the subscription metadata at checkout.

        Returns:
            The hosted Checkout URL for the recurring Founder's Rate.
        """
        url = service.create_founders_checkout(user.username)
        return CheckoutSessionResponse(url=url)

    @router.post(
        "/billing/portal",
        response_model=CheckoutSessionResponse,
        summary="Open the Stripe Billing Portal for the caller",
    )
    def open_portal(user: AuthenticatedUserDep) -> CheckoutSessionResponse:
        """Create a Stripe Billing Portal session (manage card / invoices / cancel).

        Args:
            user: Authenticated account whose portal session is created.

        Returns:
            The hosted Billing Portal URL.
        """
        url = service.create_billing_portal(user.username)
        return CheckoutSessionResponse(url=url)

    @router.post(
        "/billing/webhook",
        response_model=WebhookAck,
        include_in_schema=False,
        summary="Receive and apply a Stripe webhook event",
    )
    async def stripe_webhook(request: Request) -> WebhookAck:
        """Verify a Stripe webhook signature and apply the event idempotently.

        Unauthenticated by design: authenticity comes from the Stripe signature
        over the raw request body, not a bearer token.

        Args:
            request: Raw request; its body and ``Stripe-Signature`` header are
                read directly (the signature is over the unparsed bytes).

        Returns:
            An acknowledgement once the event is verified and applied.
        """
        payload = await request.body()
        signature = request.headers.get("stripe-signature")
        service.handle_webhook(payload, signature)
        return WebhookAck(received=True)

    @router.get(
        "/billing/byok/keys",
        response_model=ProviderKeysResponse,
        summary="List the caller's stored BYOK provider keys (masked)",
    )
    def list_provider_keys(user: AuthenticatedUserDep) -> ProviderKeysResponse:
        """Return the caller's stored BYOK keys as masked, secret-free views.

        A pure DB read — no decryption, no provider call — so it serves even when
        the vault key is unconfigured.

        Args:
            user: Authenticated caller whose keys are listed.

        Returns:
            The masked keys, one per provider.
        """
        snapshot = vault.list_keys(user.username)
        return ProviderKeysResponse(
            keys=[
                ProviderKeyResponse(
                    id=k.id,
                    provider=k.provider,
                    label=k.label,
                    last4=k.last4,
                    api_base=k.api_base,
                    status=k.status,
                    added_at=k.added_at,
                )
                for k in snapshot.keys
            ]
        )

    @router.put(
        "/billing/byok/keys",
        response_model=ProviderKeyResponse,
        summary="Save (or rotate) a BYOK provider key; encrypt at rest and verify on entry",
    )
    def save_provider_key(
        body: SaveProviderKeyRequest, user: AuthenticatedUserDep
    ) -> ProviderKeyResponse:
        """Encrypt and store a provider secret, verifying it on entry.

        The secret is encrypted before it touches the database and never echoed
        back; the response carries only the masked tail and the entry-time verify
        verdict.

        Args:
            body: The provider slug and its plaintext secret.
            user: Authenticated owner of the key.

        Returns:
            The masked, verified view of the stored key.
        """
        view = vault.save_key(
            user.username,
            body.provider,
            body.secret,
            label=body.label,
            api_base=body.api_base,
            params=body.params,
        )
        return ProviderKeyResponse(
            id=view.id,
            provider=view.provider,
            label=view.label,
            last4=view.last4,
            api_base=view.api_base,
            status=view.status,
            added_at=view.added_at,
        )

    @router.post(
        "/billing/byok/keys/{provider}/verify",
        response_model=ProviderKeyResponse,
        summary="Re-run the verify probe against a stored BYOK key",
    )
    def verify_provider_key(provider: str, user: AuthenticatedUserDep) -> ProviderKeyResponse:
        """Re-verify a stored provider key and persist the fresh verdict.

        Used to re-check a key saved while the provider was unreachable (status
        stuck at ``unverified``). Decrypts the secret only for the probe.

        Args:
            provider: Provider slug whose stored key is re-verified.
            user: Authenticated owner of the key.

        Returns:
            The masked view carrying the fresh verification status.
        """
        view = vault.verify_key(user.username, provider)
        return ProviderKeyResponse(
            id=view.id,
            provider=view.provider,
            label=view.label,
            last4=view.last4,
            api_base=view.api_base,
            status=view.status,
            added_at=view.added_at,
        )

    @router.delete(
        "/billing/byok/keys/{provider}",
        response_model=ProviderKeysResponse,
        summary="Forget a stored BYOK provider key",
    )
    def remove_provider_key(provider: str, user: AuthenticatedUserDep) -> ProviderKeysResponse:
        """Forget a stored provider key and return the remaining keys.

        Idempotent: removing a provider with no stored key is a no-op. Returns the
        post-removal list so the client can re-render without a second round-trip.

        Args:
            provider: Provider slug whose key is removed.
            user: Authenticated owner of the key.

        Returns:
            The caller's remaining masked keys.
        """
        vault.remove_key(user.username, provider)
        snapshot = vault.list_keys(user.username)
        return ProviderKeysResponse(
            keys=[
                ProviderKeyResponse(
                    id=k.id,
                    provider=k.provider,
                    label=k.label,
                    last4=k.last4,
                    api_base=k.api_base,
                    status=k.status,
                    added_at=k.added_at,
                )
                for k in snapshot.keys
            ]
        )

    return router
