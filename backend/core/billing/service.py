"""Stripe-backed billing service: customers, checkout, subscriptions, webhooks.

This module is the only place that talks to Stripe. The web app calls the
billing router, which delegates here. Stripe is the source of truth for money
(charges and subscription state); the local ``billing_customers`` /
``credit_ledger`` tables are a synced cache plus an audit trail, reconciled by
:meth:`StripeBillingService.handle_webhook` on every event Stripe delivers.

Reads (the wallet) work whether or not Stripe is configured, so a deploy
without keys degrades to a read-only free tier. Mutations (checkout, subscribe,
portal) require ``settings.is_stripe_configured`` and raise
``DomainError("billing.not_configured", 503)`` otherwise — never a 500.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import stripe
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..api.errors import DomainError
from ..config import settings
from ..constants import (
    GUARANTEE_BASIS_TEST,
    GUARANTEE_BASIS_VAL,
    TOKEN_SOURCE_BYOK,
    TOKEN_SOURCE_MANAGED,
)
from ..storage.models import (
    BillingCustomerModel,
    BillingWebhookEventModel,
    CreditLedgerModel,
    GuaranteeRunModel,
)
from .pricing import ModelUsage, credits_for_usage

# Credits granted per one-time pack. Mirrors the frontend CREDIT_PACKS catalog;
# the dollar price lives in Stripe (the price id), the credits granted live here.
PACK_CREDITS: dict[str, int] = {"starter": 500, "plus": 2200, "pro": 6500}

# Renewing allowance that keeps the free tier usable on mini models. Under
# per-model pricing a credit is real marked-up provider cost, so 150 credits is
# ~$1 of mini inference (~2-3 light runs) — trial-tight, never a frontier loss.
FREE_GRANT_CREDITS = 150

# Renewing allowance for an active Premium subscriber — the monthly credit
# allotment the subscription buys, replacing (not stacking on) the free grant.
# Re-priceable here: the margin lever is the token→credit markup, not this count.
PREMIUM_GRANT_CREDITS = 2500

# The free grant rolls on a per-user 30-day window rather than a calendar month,
# so resets scatter across the month instead of all landing on the 1st. The
# window is non-cumulative: a reset tops back up to a flat FREE_GRANT_CREDITS and
# any leftover expires (no banking). Evaluated lazily on wallet read / debit.
GRANT_WINDOW_DAYS = 30

# Prefix on the placeholder ``stripe_customer_id`` of a billing row created by a
# local debit for an account that never reached Stripe. ``get_or_create_customer``
# treats such a row as having no real Stripe customer yet and provisions one.
LOCAL_CUSTOMER_PREFIX = "local:"

# Credits charged per this many run tokens. One credit is $0.01; the markup that
# protects margin lives in this mapping (and the Stripe per-unit price), not in
# the catalog, so it is re-priceable without touching credit counts elsewhere.
TOKENS_PER_CREDIT = 1000

# Share of a run's credit cost that is Skynet's platform fee (vs. pass-through
# compute). On a no-lift BYOK run the provider tokens are already spent on the
# user's own key, so only this fee is refundable; a managed no-lift run refunds
# the whole cost. Lives here so the fee is re-priceable without touching the
# guarantee logic.
PLATFORM_FEE_FRACTION = 0.20

# Stripe subscription statuses that count an account as having active Premium.
_ACTIVE_SUBSCRIPTION_STATUSES = frozenset({"active", "trialing", "past_due"})

# How long a Founder's Rate subscriber's price is held. The offer promises the
# rate is locked for 12 months; the lock instant is stamped into the
# subscription metadata at checkout so the held-through date is auditable and
# surfaced back to the subscriber.
FOUNDERS_LOCK_DAYS = 365


@dataclass(frozen=True)
class LedgerRow:
    """One usage-ledger entry as the wallet UI consumes it.

    ``credits`` is signed (negative for a spend), ``model`` is the model id for a
    run row or ``None`` for a top-up/grant, and ``at`` is an ISO-8601 instant.
    """

    id: str
    at: str
    label: str
    model: str | None
    credits: int
    kind: str


@dataclass(frozen=True)
class FoundersRateStatus:
    """The Founder's Rate availability as the upgrade page reads it.

    ``open`` is whether new subscriptions are still accepted (the deadline gate),
    ``closes_at`` is the ISO-8601 close instant, and ``price_locked_until`` is the
    ISO-8601 instant through which a subscriber locking in *now* keeps the rate.
    """

    open: bool
    closes_at: str
    price_locked_until: str


@dataclass(frozen=True)
class WalletSnapshot:
    """The account's billing state as a single read for the wallet surfaces."""

    paid_balance_credits: int
    free_grant_remaining: int
    free_grant_total: int
    free_grant_resets_at: str
    premium_active: bool
    subscription_status: str | None
    subscription_current_period_end: str | None
    usage: list[LedgerRow] = field(default_factory=list)


def credits_for_tokens(total_tokens: int) -> int:
    """Convert a run's token total to the credits it costs, rounding up.

    A partial credit's worth of tokens still costs a whole credit, so a run that
    consumed any tokens at all is never billed zero. The rate
    (:data:`TOKENS_PER_CREDIT`) carries the markup and is re-priceable here.

    Args:
        total_tokens: Tokens the run consumed; non-positive yields ``0``.

    Returns:
        The non-negative credit cost of the run.
    """
    if total_tokens <= 0:
        return 0
    return -(-total_tokens // TOKENS_PER_CREDIT)


def platform_fee_credits(total_tokens: int) -> int:
    """Return the platform-fee portion of a run's credit cost, rounding up.

    The refundable amount on a no-lift **BYOK** run: the provider tokens were
    spent on the user's own key, so only Skynet's fee
    (:data:`PLATFORM_FEE_FRACTION` of the full cost) can be returned. Always at
    least one credit when the run cost anything, so a covered run is never
    refunded zero.

    Args:
        total_tokens: Tokens the run consumed.

    Returns:
        The non-negative platform-fee credits (``0`` when the run cost nothing).
    """
    cost = credits_for_tokens(total_tokens)
    if cost <= 0:
        return 0
    return max(1, math.ceil(cost * PLATFORM_FEE_FRACTION))


def platform_fee_credits_for_usage(usages: Iterable[ModelUsage]) -> int:
    """Return the platform-fee portion of a run's per-model credit cost, rounding up.

    The per-model analogue of :func:`platform_fee_credits`: the
    :data:`PLATFORM_FEE_FRACTION` share of the run's full per-model cost
    (:func:`core.billing.pricing.credits_for_usage`). The only amount a **BYOK**
    run is charged or refunded, since the provider tokens were paid on the user's
    own key. At least one credit when the run cost anything.

    Args:
        usages: Per-model token usage for the run.

    Returns:
        The non-negative platform-fee credits (``0`` when the run cost nothing).
    """
    full = credits_for_usage(usages)
    if full <= 0:
        return 0
    return max(1, math.ceil(full * PLATFORM_FEE_FRACTION))


def run_cost_credits(usages: Iterable[ModelUsage], token_source: str) -> int:
    """Return the credits a run costs: full per-model cost, or the BYOK platform fee.

    A managed run is charged its full per-model token cost
    (:func:`core.billing.pricing.credits_for_usage`); a BYOK run is charged only
    Skynet's platform fee (:func:`platform_fee_credits_for_usage`), since the
    provider tokens were paid on the user's own key. The shared basis for both
    the live debit and the guarantee refund, so the refund always equals the
    charge.

    Args:
        usages: Per-model token usage for the run.
        token_source: ``"managed"`` (full cost) or ``"byok"`` (platform fee only).

    Returns:
        The non-negative credit cost.
    """
    if token_source == TOKEN_SOURCE_BYOK:
        return platform_fee_credits_for_usage(usages)
    return credits_for_usage(usages)


def cost_ceiling_budget(spendable: int, token_source: str) -> int:
    """Return the max per-run cost ceiling (full-cost credits) a balance can back.

    A managed run spends its full per-token credit cost, so the balance backs a
    ceiling of exactly ``spendable``. A BYOK run spends only Skynet's platform fee
    (:data:`PLATFORM_FEE_FRACTION` of the full cost — the provider tokens are paid
    on the user's own key), so the same balance backs a proportionally larger
    ceiling: the largest full-cost budget whose platform fee still fits within
    ``spendable``. Clamping ``max_cost_credits`` to this keeps a runaway BYOK run's
    fee from ever exceeding the balance, mirroring the managed clamp. Computed by
    over-estimating then stepping down against the real fee function so float
    imprecision in the fraction can only err conservative (toward the balance).

    Args:
        spendable: The account's spendable credits; non-positive yields ``0``.
        token_source: ``"managed"`` or ``"byok"`` — sets the conversion.

    Returns:
        The non-negative ceiling, in full-cost credits, to clamp the run to.
    """
    if spendable <= 0:
        return 0
    if token_source != TOKEN_SOURCE_BYOK:
        return spendable
    budget = math.ceil(spendable / PLATFORM_FEE_FRACTION)
    while budget > 1 and max(1, math.ceil(budget * PLATFORM_FEE_FRACTION)) > spendable:
        budget -= 1
    return budget


def _grant_allotment(customer: BillingCustomerModel | None) -> int:
    """Return the account's monthly grant size — larger for an active Premium sub.

    The renewing allowance the grant tops up to: :data:`PREMIUM_GRANT_CREDITS` for
    an account whose subscription is in an active state, else the free-tier
    :data:`FREE_GRANT_CREDITS`. A missing row (``None``) is a free account.

    Args:
        customer: The account's billing row, or ``None`` when it has none yet.

    Returns:
        The credit allotment the account's grant renews to.
    """
    if customer is None:
        return FREE_GRANT_CREDITS
    if customer.subscription_status in _ACTIVE_SUBSCRIPTION_STATUSES:
        return PREMIUM_GRANT_CREDITS
    return FREE_GRANT_CREDITS


def _grant_window_end(now: datetime) -> datetime:
    """Return the instant the free grant next tops up, a fixed window past ``now``.

    Args:
        now: Reference instant the window is anchored to.

    Returns:
        ``now`` advanced by :data:`GRANT_WINDOW_DAYS`, in UTC.
    """
    return now + timedelta(days=GRANT_WINDOW_DAYS)


class StripeBillingService:
    """Mediates between the billing API, the billing tables, and Stripe."""

    def __init__(self, *, engine: Any) -> None:
        """Bind the service to the ORM engine backing the billing tables.

        Args:
            engine: SQLAlchemy engine (``job_store.engine``) used for every
                billing-table session. Stripe itself is configured lazily, so
                constructing the service never requires Stripe credentials.
        """
        self._engine = engine

    def _resolve_grant(self, customer: BillingCustomerModel | None, now: datetime) -> int:
        """Apply the lazy rolling free-grant reset and return the current remaining.

        Seeds an unseeded row (NULL window — a new account or one created before
        these columns existed) to a full grant anchored ``GRANT_WINDOW_DAYS`` out.
        When the window has elapsed, tops the grant back up to a flat
        :data:`FREE_GRANT_CREDITS` (leftover expires — non-cumulative) and advances
        the anchor. Mutates ``customer`` in place; the caller commits. A missing
        row (``None``) is treated as a full grant without persisting anything — the
        row is created on the first real mutation (debit / top-up).

        Args:
            customer: The account's billing row, or ``None`` when it has none yet.
            now: Reference instant for the window comparison.

        Returns:
            Credits remaining in the account's current free grant.
        """
        if customer is None:
            return FREE_GRANT_CREDITS
        allotment = _grant_allotment(customer)
        if customer.grant_reset_at is None or customer.grant_remaining is None:
            customer.grant_remaining = allotment
            customer.grant_reset_at = _grant_window_end(now)
            customer.updated_at = now
            return allotment
        reset_at = customer.grant_reset_at
        if reset_at.tzinfo is None:
            reset_at = reset_at.replace(tzinfo=UTC)
        if now > reset_at:
            customer.grant_remaining = allotment
            customer.grant_reset_at = _grant_window_end(now)
            customer.updated_at = now
        return int(customer.grant_remaining)

    def _stripe(self) -> Any:
        """Return the ``stripe`` module configured with the secret key.

        Returns:
            The configured ``stripe`` module.

        Raises:
            DomainError: 503 when no Stripe secret key is configured.
        """
        if settings.stripe_secret_key is None:
            raise DomainError("billing.not_configured", status=503)
        stripe.api_key = settings.stripe_secret_key.get_secret_value()
        return stripe

    def _return_url(self, status: str) -> str:
        """Build a Checkout/portal return URL on the public app origin.

        Args:
            status: Query value appended as ``?status=`` (``success``/``cancel``).

        Returns:
            An absolute URL back to the in-app upgrade surface.
        """
        return f"{settings.app_public_url.rstrip('/')}/upgrade?status={status}"

    def get_or_create_customer(self, username: str) -> str:
        """Return the account's Stripe customer id, creating it on first use.

        Persists the ``username -> stripe_customer_id`` mapping so a returning
        buyer reuses one Stripe customer (and one saved card / portal history).

        Args:
            username: Lowercased-email identity the customer is billed under.

        Returns:
            The Stripe customer id (``cus_…``).
        """
        with Session(self._engine) as session:
            row = session.get(BillingCustomerModel, username)
            if row is not None and not row.stripe_customer_id.startswith(LOCAL_CUSTOMER_PREFIX):
                return row.stripe_customer_id
        stripe_mod = self._stripe()
        customer = stripe_mod.Customer.create(email=username, metadata={"username": username})
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            existing = session.get(BillingCustomerModel, username)
            if existing is not None:
                # A local debit may have created a placeholder row before the
                # account reached Stripe; upgrade it in place to the real id so the
                # accrued grant/balance carry over.
                if existing.stripe_customer_id.startswith(LOCAL_CUSTOMER_PREFIX):
                    existing.stripe_customer_id = customer.id
                    existing.updated_at = now
                    session.commit()
                    return customer.id
                return existing.stripe_customer_id
            session.add(
                BillingCustomerModel(
                    username=username,
                    stripe_customer_id=customer.id,
                    credit_balance=0,
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()
        return customer.id

    def create_pack_checkout(self, username: str, pack_id: str) -> str:
        """Create a one-time Checkout Session for a credit pack and return its URL.

        Args:
            username: Buyer identity; stamped into session metadata so the
                webhook can credit the right account.
            pack_id: One of :data:`PACK_CREDITS` (``starter``/``plus``/``pro``).

        Returns:
            The hosted Stripe Checkout URL to redirect the buyer to.

        Raises:
            DomainError: 400 when ``pack_id`` is unknown or its price id is
                unconfigured; 503 when Stripe is not configured.
        """
        price_id = settings.stripe_pack_price_ids.get(pack_id, "")
        credits = PACK_CREDITS.get(pack_id, 0)
        if not price_id or credits <= 0:
            raise DomainError("billing.unknown_pack", status=400, pack_id=pack_id)
        stripe_mod = self._stripe()
        customer_id = self.get_or_create_customer(username)
        metadata = {"username": username, "pack_id": pack_id, "credits": str(credits)}
        checkout = stripe_mod.checkout.Session.create(
            customer=customer_id,
            mode="payment",
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=self._return_url("success"),
            cancel_url=self._return_url("cancel"),
            client_reference_id=username,
            metadata=metadata,
            payment_intent_data={"metadata": metadata},
        )
        return str(checkout.url)

    def create_subscription_checkout(self, username: str) -> str:
        """Create a Checkout Session for the Premium subscription and return its URL.

        When a metered overage price is configured it is added as a second,
        quantity-less line item so the subscription can be billed for token usage.

        Args:
            username: Subscriber identity, stamped into session metadata.

        Returns:
            The hosted Stripe Checkout URL for the recurring subscription.

        Raises:
            DomainError: 503 when Stripe / the Premium price id is unconfigured.
        """
        price_id = settings.stripe_price_premium
        if not price_id:
            raise DomainError("billing.not_configured", status=503)
        stripe_mod = self._stripe()
        customer_id = self.get_or_create_customer(username)
        line_items: list[dict[str, Any]] = [{"price": price_id, "quantity": 1}]
        # The metered overage price rides on the same subscription so per-run token
        # usage (reported via report_run_usage) actually bills the subscriber. A
        # metered price is quantity-less — passing a quantity makes Checkout reject it.
        metered_price = settings.stripe_price_metered
        if metered_price:
            line_items.append({"price": metered_price})
        checkout = stripe_mod.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=line_items,
            success_url=self._return_url("success"),
            cancel_url=self._return_url("cancel"),
            client_reference_id=username,
            metadata={"username": username},
        )
        return str(checkout.url)

    def founders_rate_status(self, now: datetime | None = None) -> FoundersRateStatus:
        """Return whether the Founder's Rate is still open and its lock window.

        The deadline gate is config-driven (:attr:`settings.founders_rate_closes_at`)
        — a placeholder far-future default until the real close date is set — so no
        date is hardcoded. ``open`` is ``now <= closes_at``; once the deadline
        passes the offer is unavailable to new subscribers. ``price_locked_until``
        is a subscriber-locking-in-now date (``now + FOUNDERS_LOCK_DAYS``), the
        12-month price hold the offer promises.

        Args:
            now: Reference instant; defaults to the current UTC time.

        Returns:
            A :class:`FoundersRateStatus` for the upgrade-page deadline line and gate.
        """
        now = now or datetime.now(UTC)
        closes_at = settings.founders_rate_closes_at_dt
        return FoundersRateStatus(
            open=now <= closes_at,
            closes_at=closes_at.isoformat(),
            price_locked_until=(now + timedelta(days=FOUNDERS_LOCK_DAYS)).isoformat(),
        )

    def create_founders_checkout(self, username: str) -> str:
        """Create a Checkout Session for the Founder's Rate subscription.

        Gated by the config-driven deadline: once the close date has passed, the
        offer is unavailable and this raises rather than minting a checkout. The
        12-month price-lock is recorded as subscription metadata (``founders_rate``
        and ``price_locked_until``) so the held-through date travels with the
        subscription in Stripe and is auditable. The Founder's Rate has its own
        Stripe price; when unconfigured it falls back to the Premium price so the
        offer still works on a partly-provisioned deploy.

        Args:
            username: Subscriber identity, stamped into session metadata.

        Returns:
            The hosted Stripe Checkout URL for the recurring Founder's Rate.

        Raises:
            DomainError: 410 when the offer's deadline has passed; 503 when Stripe
                (and no fallback Premium price) is unconfigured.
        """
        now = datetime.now(UTC)
        if now > settings.founders_rate_closes_at_dt:
            raise DomainError("billing.founders_closed", status=410)
        price_id = settings.stripe_price_founders or settings.stripe_price_premium
        if not price_id:
            raise DomainError("billing.not_configured", status=503)
        stripe_mod = self._stripe()
        customer_id = self.get_or_create_customer(username)
        locked_until = (now + timedelta(days=FOUNDERS_LOCK_DAYS)).isoformat()
        metadata = {
            "username": username,
            "founders_rate": "true",
            "price_locked_until": locked_until,
        }
        line_items: list[dict[str, Any]] = [{"price": price_id, "quantity": 1}]
        # The metered overage price rides on the same subscription so per-run token
        # usage bills the subscriber. A metered price is quantity-less.
        metered_price = settings.stripe_price_metered
        if metered_price:
            line_items.append({"price": metered_price})
        checkout = stripe_mod.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=line_items,
            success_url=self._return_url("success"),
            cancel_url=self._return_url("cancel"),
            client_reference_id=username,
            metadata=metadata,
            subscription_data={"metadata": metadata},
        )
        return str(checkout.url)

    def create_billing_portal(self, username: str) -> str:
        """Create a Stripe Billing Portal session and return its URL.

        The portal is where a subscriber updates their card, views invoices, or
        cancels Premium — Stripe hosts it, so no billing-management UI is built.

        Args:
            username: Account whose portal session is created.

        Returns:
            The hosted Billing Portal URL.

        Raises:
            DomainError: 503 when Stripe is not configured.
        """
        stripe_mod = self._stripe()
        customer_id = self.get_or_create_customer(username)
        portal = stripe_mod.billing_portal.Session.create(customer=customer_id, return_url=self._return_url("portal"))
        return str(portal.url)

    def get_wallet(self, username: str) -> WalletSnapshot:
        """Return the account's wallet snapshot from the billing tables.

        A pure DB read — no Stripe call — so it serves even when Stripe is
        unconfigured (balance 0, no subscription). The free grant reflects real
        spend: run completions debit it via :meth:`debit_run`, and the rolling,
        non-cumulative reset is lazy-evaluated here — if the window has elapsed
        the grant tops back up to a flat allowance and the anchor advances (any
        leftover expires). Paid (Premium) accounts anchor the reset to the Stripe
        subscription period instead of the rolling window. A read that mutates an
        existing row (seed or reset) is persisted; a brand-new account reads a
        full grant without creating a row until its first real mutation.

        Args:
            username: Account to summarize.

        Returns:
            A :class:`WalletSnapshot` of paid balance, free grant, subscription
            state, and the most recent ledger rows.
        """
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            customer = session.get(BillingCustomerModel, username)
            grant_remaining = self._resolve_grant(customer, now)
            # Capture the seed/reset mutation before the ledger query autoflushes
            # it — once flushed, ``is_modified`` reads clean and the commit below
            # would be skipped, dropping the reset on session close.
            grant_dirty = customer is not None and session.is_modified(customer)
            rows = (
                session.query(CreditLedgerModel)
                .filter(CreditLedgerModel.username == username)
                .order_by(CreditLedgerModel.created_at.desc())
                .limit(15)
                .all()
            )
            usage = [
                LedgerRow(
                    id=str(row.id),
                    at=row.created_at.isoformat(),
                    label=row.description or row.kind,
                    model=row.model,
                    credits=row.delta_credits,
                    kind=row.kind,
                )
                for row in rows
            ]
            status = customer.subscription_status if customer else None
            period_end = (
                customer.subscription_current_period_end
                if customer and customer.subscription_current_period_end
                else None
            )
            premium_active = status in _ACTIVE_SUBSCRIPTION_STATUSES if status else False
            # Paid accounts anchor the grant top-up to their subscription period;
            # everyone else to the rolling per-user window (``grant_reset_at``).
            if premium_active and period_end is not None:
                resets_at = period_end
            elif customer is not None and customer.grant_reset_at is not None:
                resets_at = customer.grant_reset_at
            else:
                resets_at = _grant_window_end(now)
            if grant_dirty:
                session.commit()
            return WalletSnapshot(
                paid_balance_credits=customer.credit_balance if customer else 0,
                free_grant_remaining=grant_remaining,
                free_grant_total=_grant_allotment(customer),
                free_grant_resets_at=resets_at.isoformat(),
                premium_active=premium_active,
                subscription_status=status,
                subscription_current_period_end=period_end.isoformat() if period_end else None,
                usage=usage,
            )

    def spendable_credits(self, username: str) -> int:
        """Return the account's total spendable credits (free grant + paid balance).

        Applies the lazy rolling grant reset first, so an account whose window
        elapsed reads its topped-up grant rather than a stale figure. A mutated
        row (seed or reset) is persisted. This is the figure the submit gate
        checks: ``> 0`` means a managed run may start. A brand-new account reads a
        full grant without a row being created.

        Args:
            username: Account to read.

        Returns:
            Free-grant remaining plus purchased balance, never negative.
        """
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            customer = session.get(BillingCustomerModel, username)
            grant_remaining = self._resolve_grant(customer, now)
            paid = int(customer.credit_balance) if customer is not None else 0
            if customer is not None and session.is_modified(customer):
                session.commit()
        return max(grant_remaining + paid, 0)

    def debit_run(
        self,
        username: str,
        usages: Iterable[ModelUsage],
        *,
        model: str | None,
        description: str,
        token_source: str = TOKEN_SOURCE_MANAGED,
    ) -> int:
        """Charge a finished run's per-model credit cost to the account, grant first.

        Writes one signed negative ``run`` row to ``credit_ledger`` and draws the
        cost from the free grant before the purchased balance, mirroring how
        :meth:`get_wallet` reports spendable credits (grant then paid). A **managed**
        run is charged its full per-model token cost; a **BYOK** run is charged only
        Skynet's platform fee (:func:`run_cost_credits`), since the provider tokens
        were already paid on the user's own key — so credits still meter the
        platform on a BYOK run without double-charging for inference. The rolling
        grant reset is applied first so a run that completes after the window
        elapsed bills against the topped-up grant. Idempotency is the caller's
        responsibility: the worker debits inside its once-only completion claim, so
        a redelivered/re-run job never double-charges. A run costing zero credits
        writes nothing.

        Args:
            username: Account the run is billed to.
            usages: Per-model token usage for the run; priced per-model into the
                credit cost.
            model: Model id stamped on the ledger row, or ``None``.
            description: Human label for the ledger row (typically the run name).
            token_source: ``"managed"`` (full cost) or ``"byok"`` (platform fee
                only); defaults to managed.

        Returns:
            The credit cost charged (``0`` when nothing was billed).
        """
        cost = run_cost_credits(usages, token_source)
        if cost <= 0:
            return 0
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            customer = session.get(BillingCustomerModel, username)
            if customer is None:
                # A run can finish for an account that never touched Stripe; seed a
                # customer-less billing row so the debit lands and the grant tracks.
                # ``stripe_customer_id`` is uniquely indexed, so the placeholder is
                # keyed on the username to avoid colliding across free accounts; the
                # real id replaces it the first time the account reaches Stripe.
                customer = BillingCustomerModel(
                    username=username,
                    stripe_customer_id=f"{LOCAL_CUSTOMER_PREFIX}{username}",
                    credit_balance=0,
                    created_at=now,
                    updated_at=now,
                )
                session.add(customer)
            self._resolve_grant(customer, now)
            from_grant = min(int(customer.grant_remaining or 0), cost)
            customer.grant_remaining = int(customer.grant_remaining or 0) - from_grant
            customer.credit_balance = int(customer.credit_balance) - (cost - from_grant)
            customer.updated_at = now
            session.add(
                CreditLedgerModel(
                    username=username,
                    delta_credits=-cost,
                    kind="run",
                    description=description or "Run",
                    model=model,
                )
            )
            session.commit()
        return cost

    def _refund_run(
        self, session: Session, username: str, credits: int, *, model: str | None, description: str
    ) -> None:
        """Write an offsetting positive ``run`` row and restore the balance.

        The credit side of an auto-refund: returns ``credits`` to the account by
        replenishing the free grant first (mirroring how :meth:`debit_run` drew
        it down — grant before paid balance) and any remainder to the purchased
        balance, then appends a positive ``run`` ledger row so the wallet shows
        the refund as a legible line. Caller commits.

        Args:
            session: Open session the refund is written into (caller commits).
            username: Account being refunded.
            credits: Positive credit amount to restore.
            model: Model id stamped on the refund row, or ``None``.
            description: Human label for the ledger row.
        """
        now = datetime.now(UTC)
        customer = session.get(BillingCustomerModel, username)
        if customer is None:
            customer = BillingCustomerModel(
                username=username,
                stripe_customer_id=f"{LOCAL_CUSTOMER_PREFIX}{username}",
                credit_balance=0,
                created_at=now,
                updated_at=now,
            )
            session.add(customer)
        self._resolve_grant(customer, now)
        # Restore the grant up to its full allowance first (it was drawn first on
        # debit), then send any overflow to the purchased balance.
        allotment = _grant_allotment(customer)
        grant = int(customer.grant_remaining or 0)
        to_grant = min(allotment - grant, credits) if grant < allotment else 0
        to_grant = max(to_grant, 0)
        customer.grant_remaining = grant + to_grant
        customer.credit_balance = int(customer.credit_balance) + (credits - to_grant)
        customer.updated_at = now
        session.add(
            CreditLedgerModel(
                username=username,
                delta_credits=credits,
                kind="run",
                description=description,
                model=model,
            )
        )

    def adjudicate_guarantee(
        self,
        username: str,
        task_fingerprint: str,
        optimization_id: str,
        guarantee: dict[str, Any] | None,
        *,
        token_source: str,
        usages: Iterable[ModelUsage],
        model: str | None,
        description: str,
    ) -> int:
        """Apply the "No lift, no charge" guarantee to a finished run.

        Only the **first** run per ``(username, task_fingerprint)`` is covered:
        an atomic insert claims that one-time slot, so a redelivered or re-run
        job — and every later run on the same task — bills normally and returns
        ``0`` here. On the covered run, lift is read from ``guarantee`` (the
        baseline-vs-optimized scores on the test split, or the valset fallback
        when the dataset was too small): any improvement counts as lift and the
        run stays billed. No lift triggers an auto-refund — a managed run gets
        the **whole** run cost back, a BYOK run only Skynet's platform fee (the
        provider tokens were spent on the user's own key) — written as an
        offsetting ``run`` row that restores the balance, and the slot is flagged
        ``refunded``. A missing ``guarantee`` (no comparable baseline/optimized
        pair) leaves the run billed but still consumes the slot, so the account's
        one covered run is spent honestly rather than retried for free.

        Args:
            username: Account the run was billed to.
            task_fingerprint: Content hash identifying the task (same value the
                submit path computes from signature + metric + dataset).
            optimization_id: The run claiming or skipping the guarantee slot.
            guarantee: ``{"basis", "baseline", "optimized"}`` from the result, or
                ``None`` when the run produced no comparable pair.
            token_source: ``"managed"`` or ``"byok"`` — sets the refund scope.
            usages: Per-model token usage; sizes the refund (equal to the charge).
            model: Model id stamped on the refund row, or ``None``.
            description: Human label for the refund ledger row.

        Returns:
            The credits refunded (``0`` when the run was billed: it had lift,
            wasn't the first run on the task, or had no comparable scores).
        """
        if not username or not task_fingerprint:
            return 0
        with Session(self._engine) as session:
            session.add(
                GuaranteeRunModel(
                    username=username,
                    task_fingerprint=task_fingerprint,
                    optimization_id=optimization_id,
                )
            )
            try:
                # Flush the claim alone so a re-run's PK collision surfaces here
                # (this account already spent its covered run on this task) and we
                # bail before touching the ledger.
                session.flush()
            except IntegrityError:
                session.rollback()
                return 0

            # Refund only on a definitive no-lift signal. A run with lift — or one
            # with no comparable baseline/optimized pair to judge — bills, but
            # still consumes the account's one covered slot for this task.
            if not self._is_no_lift(guarantee):
                session.commit()
                return 0

            refund = run_cost_credits(usages, token_source)
            if refund <= 0:
                session.commit()
                return 0

            self._refund_run(session, username, refund, model=model, description=description)
            claim = session.get(GuaranteeRunModel, (username, task_fingerprint))
            if claim is not None:
                claim.refunded = True
            session.commit()
        return refund

    @staticmethod
    def _is_no_lift(guarantee: dict[str, Any] | None) -> bool:
        """Return whether the guarantee scores definitively show no lift.

        Lift is ``optimized > baseline`` on the run's adjudication basis (test
        split, or valset fallback); any real improvement counts — the test-split
        basis is what keeps "any improvement" honest rather than a noise gotcha.
        Returns ``True`` only when a valid basis carries both scores and the
        optimized score did not exceed the baseline. A missing or malformed
        guarantee block (no comparable pair) is *not* treated as no-lift — the
        run can't be proven a failure, so it bills.

        Args:
            guarantee: ``{"basis", "baseline", "optimized"}`` from the result, or
                ``None``.

        Returns:
            ``True`` only when both scores are present and optimized did not beat
            baseline.
        """
        if not isinstance(guarantee, dict):
            return False
        if guarantee.get("basis") not in (GUARANTEE_BASIS_TEST, GUARANTEE_BASIS_VAL):
            return False
        baseline = guarantee.get("baseline")
        optimized = guarantee.get("optimized")
        if not isinstance(baseline, (int, float)) or not isinstance(optimized, (int, float)):
            return False
        return optimized <= baseline

    def handle_webhook(self, payload: bytes, sig_header: str | None) -> None:
        """Verify and apply a Stripe webhook event, exactly once.

        Records the event id before applying its effect in the same transaction,
        so a redelivered event (Stripe guarantees at-least-once) is a no-op
        instead of a double-credit.

        Args:
            payload: Raw request body bytes (signature is over the raw bytes).
            sig_header: The ``Stripe-Signature`` header value.

        Raises:
            DomainError: 503 when the webhook secret is unconfigured; 400 when
                the signature does not verify.
        """
        if settings.stripe_webhook_secret is None:
            raise DomainError("billing.not_configured", status=503)
        stripe_mod = self._stripe()
        try:
            event = stripe_mod.Webhook.construct_event(
                payload, sig_header or "", settings.stripe_webhook_secret.get_secret_value()
            )
        except (ValueError, stripe.SignatureVerificationError) as exc:
            raise DomainError("billing.webhook_invalid", status=400) from exc
        event_id = str(event["id"])
        with Session(self._engine) as session:
            if session.get(BillingWebhookEventModel, event_id) is not None:
                return
            session.add(BillingWebhookEventModel(event_id=event_id, event_type=str(event["type"])))
            self._apply_event(session, event)
            session.commit()

    def _apply_event(self, session: Session, event: Any) -> None:
        """Dispatch a verified event to its handler; unknown types are no-ops.

        Args:
            session: Open session; the caller commits (event row + effect land
                atomically).
            event: The verified Stripe event object.
        """
        event_type = str(event["type"])
        obj = event["data"]["object"]
        if event_type == "checkout.session.completed":
            self._on_checkout_completed(session, str(event["id"]), obj)
        elif event_type in (
            "customer.subscription.created",
            "customer.subscription.updated",
            "customer.subscription.deleted",
        ):
            self._on_subscription_change(session, obj)

    def _on_checkout_completed(self, session: Session, event_id: str, obj: Any) -> None:
        """Credit a completed one-time pack purchase to the buyer's balance.

        Subscription-mode sessions are ignored here — the subscription's own
        ``customer.subscription.created`` event carries the authoritative state.

        Args:
            session: Open session (caller commits).
            event_id: Stripe event id, recorded on the ledger row for traceability.
            obj: The Checkout Session object from the event.
        """
        if obj.get("mode") != "payment" or obj.get("payment_status") != "paid":
            return
        metadata = obj.get("metadata") or {}
        username = str(metadata.get("username") or obj.get("client_reference_id") or "").lower()
        credits = int(metadata.get("credits") or 0)
        pack_id = str(metadata.get("pack_id") or "")
        if not username or credits <= 0:
            return
        customer = session.get(BillingCustomerModel, username)
        if customer is None:
            customer = BillingCustomerModel(
                username=username,
                stripe_customer_id=str(obj.get("customer") or ""),
                credit_balance=0,
            )
            session.add(customer)
        customer.credit_balance = int(customer.credit_balance) + credits
        customer.updated_at = datetime.now(UTC)
        session.add(
            CreditLedgerModel(
                username=username,
                delta_credits=credits,
                kind="topup",
                description=f"Top-up · {pack_id}" if pack_id else "Top-up",
                stripe_event_id=event_id,
            )
        )

    def _on_subscription_change(self, session: Session, obj: Any) -> None:
        """Sync a subscription's status onto the matching customer row.

        Args:
            session: Open session (caller commits).
            obj: The Subscription object from the event.
        """
        customer_id = str(obj.get("customer") or "")
        if not customer_id:
            return
        row = (
            session.query(BillingCustomerModel)
            .filter(BillingCustomerModel.stripe_customer_id == customer_id)
            .one_or_none()
        )
        if row is None:
            return
        items = (obj.get("items") or {}).get("data") or []
        price_id = items[0]["price"]["id"] if items else None
        # current_period_end sits on the subscription in older API versions and on
        # the subscription item in newer ones (2025-03+); read whichever is present.
        period_end_ts = obj.get("current_period_end") or (items[0].get("current_period_end") if items else None)
        was_active = row.subscription_status in _ACTIVE_SUBSCRIPTION_STATUSES
        row.subscription_status = str(obj.get("status") or "")
        row.subscription_price_id = price_id
        row.subscription_current_period_end = (
            datetime.fromtimestamp(int(period_end_ts), tz=UTC) if period_end_ts else None
        )
        # A fresh activation delivers the Premium monthly allotment immediately
        # (top up to it, never clawing back a larger balance), so a new subscriber
        # doesn't wait for the next rolling-window reset — every later renewal is
        # handled lazily by the premium-aware reset in ``_resolve_grant``. Anchored
        # to the billing period so the first premium window aligns to the cycle.
        if not was_active and row.subscription_status in _ACTIVE_SUBSCRIPTION_STATUSES:
            row.grant_remaining = max(int(row.grant_remaining or 0), PREMIUM_GRANT_CREDITS)
            if row.subscription_current_period_end is not None:
                row.grant_reset_at = row.subscription_current_period_end
        row.updated_at = datetime.now(UTC)

    def report_run_usage(self, username: str, credits: int) -> None:
        """Meter a finished run's credit cost to Stripe Billing Meters.

        Called once per successful optimization by the worker (see
        :class:`core.worker.engine.BackgroundWorker`). Pushes the run's per-model
        credit cost — one meter unit per credit — as an event Stripe aggregates
        against the account's metered price (the usage-based overage path).
        Metering credits, not raw tokens, keeps the meter in step with the
        per-model ledger so overage and credit burn agree; the Stripe per-unit
        price is set to one credit ($0.01). Only Premium subscribers whose
        subscription carries the metered price are actually charged; for everyone
        else the event is recorded as analytics and never billed.

        Lookup-only by design: usage is reported solely for accounts that already
        have a billing customer (bought a pack or subscribed). A user who never
        touched billing gets no Stripe customer created here, avoiding customer
        sprawl for the free tier that would never be billed anyway.

        Args:
            username: Account the usage is billed to.
            credits: The run's credit cost to meter; ignored when non-positive or
                when Stripe is unconfigured.
        """
        if credits <= 0 or settings.stripe_secret_key is None:
            return
        with Session(self._engine) as session:
            customer = session.get(BillingCustomerModel, username)
            customer_id = customer.stripe_customer_id if customer else None
        if not customer_id:
            return
        stripe_mod = self._stripe()
        stripe_mod.billing.MeterEvent.create(
            event_name=settings.stripe_meter_event_name,
            payload={"stripe_customer_id": customer_id, "value": str(credits)},
        )
