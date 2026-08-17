"""Stripe-backed billing service: customers, checkout, portal, and webhooks.

This module is the only place that talks to Stripe. The web app calls the
billing router, which delegates here. Stripe is the source of truth for money
(pack charges); the local ``billing_customers`` / ``credit_ledger`` tables are
a synced cache plus an audit trail, reconciled by
:meth:`StripeBillingService.handle_webhook` on every event Stripe delivers.

Ledger reads work whether or not Stripe is configured, so a deploy without
keys degrades to a read-only free tier. Stripe profile/history reads report
the provider as unavailable, while mutations (checkout and portal) require
``settings.is_stripe_configured`` and raise
``DomainError("billing.not_configured", 503)`` otherwise — never a 500.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import stripe
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..api.errors import DomainError
from ..config import settings
from ..constants import (
    TOKEN_SOURCE_BYOK,
    TOKEN_SOURCE_MANAGED,
)
from ..storage.models import (
    BillingCustomerModel,
    BillingWebhookEventModel,
    CreditLedgerModel,
)
from .openrouter_float import check_float
from .pricing import ModelUsage, credits_for_usage

logger = logging.getLogger("skynet.billing.service")

# Credits granted per one-time pack. Mirrors the frontend CREDIT_PACKS catalog;
# the dollar price lives in Stripe (the price id), the credits granted live here.
# Packs are at par — one credit per cent paid, exactly the Stripe unit_amount —
# so no pack sells spendable value below what it costs (no bonus-credit subsidy).
PACK_CREDITS: dict[str, int] = {"starter": 500, "plus": 2000, "pro": 5000}

# Bounds for a custom (user-chosen) top-up. One credit is worth exactly one
# cent, so a credit count doubles as a Stripe ``unit_amount``; the floor keeps
# the charge above Stripe's $0.50 minimum and the ceiling keeps a typo'd
# amount from becoming a four-figure charge. Mirrored by the frontend's
# CUSTOM_CREDITS_MIN/MAX.
CUSTOM_CREDITS_MIN = 50
CUSTOM_CREDITS_MAX = 100_000

# One-time allowance a new account gets. 0 = no free credits: every credit
# spent was paid for, so the platform never subsidizes tokens or compute.
# Accounts whose grant was seeded while this was non-zero keep their remaining
# credits (the seed logic only fills a NULL column, never tops up).
FREE_GRANT_CREDITS = 0

# Prefix on the placeholder ``stripe_customer_id`` of a billing row created by a
# local debit for an account that never reached Stripe. ``get_or_create_customer``
# treats such a row as having no real Stripe customer yet and provisions one.
LOCAL_CUSTOMER_PREFIX = "local:"

# Share of a run's full credit cost charged to a BYOK run — the provider tokens
# are paid on the user's own key, but the run still consumes Skynet's CPU and
# storage. Derived from the managed MARKUP's decomposition (~1.09 money-movement
# × ~1.10 infra × 1.25 profit ≈ 1.50): everything but the money-movement slice
# is (1.50 − 1.09)×raw = 0.41/1.50 ≈ 0.273 of the full cost, grossed up for
# Stripe's ~2.9% cut of the money behind the fee credits (÷0.971) ≈ 0.28. No
# OpenRouter deposit fee applies (no managed tokens) — the fee covers compute +
# storage plus the same margin managed runs carry.
PLATFORM_FEE_FRACTION = 0.28

# Ceiling handed to fee-less BYOK runs: far above any real run's full cost,
# small enough to stay a safe int everywhere credits are summed.
_BYOK_UNCAPPED_CEILING = 10**9

# Most recent ledger rows the usage dashboard carries back per window. The
# per-day/per-model rollups span every row in range; only the raw activity list
# (and the per-run breakdown derived from it) is bounded, to cap payload size.
USAGE_ENTRY_LIMIT = 200

# Stripe-backed billing surfaces stay intentionally bounded. The UI only needs
# a concise payment-method list and a recent, date-filtered purchase history.
PAYMENT_METHOD_LIMIT = 20
TRANSACTION_LIMIT = 100


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
class WalletSnapshot:
    """The account's billing state as a single read for the wallet surfaces."""

    paid_balance_credits: int
    free_grant_remaining: int
    free_grant_total: int
    usage: list[LedgerRow] = field(default_factory=list)


@dataclass(frozen=True)
class UsageDayRow:
    """One calendar day's billed run spend, in credits."""

    date: str
    billed_credits: int


@dataclass(frozen=True)
class UsageModelRow:
    """One model's share of run spend over the window: gross billed credits and run count.

    ``input_tokens``/``output_tokens`` sum the measured usage stamped on the
    model's spend rows; rows written before token metering carry no counts and
    contribute zero.
    """

    model: str | None
    credits: int
    runs: int
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class UsageSnapshot:
    """A date-ranged usage rollup for the billing Usage dashboard.

    Aggregates the credit ledger over ``[start, end]`` into the totals, per-day,
    and per-model series the dashboard charts, plus the most recent raw ledger
    rows for the activity list. ``billed_credits`` is gross run spend (the
    absolute value of negative run rows). Top-ups and grants are excluded from
    the spend rollups but still surface in ``entries``.
    """

    start: str
    end: str
    billed_credits: int
    runs: int
    by_day: list[UsageDayRow] = field(default_factory=list)
    by_model: list[UsageModelRow] = field(default_factory=list)
    entries: list[LedgerRow] = field(default_factory=list)


@dataclass(frozen=True)
class BillingAddressSnapshot:
    """Represent display-safe billing address fields stored by Stripe."""

    line1: str | None = None
    line2: str | None = None
    city: str | None = None
    state: str | None = None
    postal_code: str | None = None
    country: str | None = None


@dataclass(frozen=True)
class PaymentMethodSnapshot:
    """Represent a saved payment method without exposing sensitive details."""

    id: str
    type: str
    brand: str | None
    last4: str | None
    exp_month: int | None
    exp_year: int | None
    is_default: bool


@dataclass(frozen=True)
class BillingProfileSnapshot:
    """Represent the Stripe-backed billing profile shown in Settings."""

    available: bool
    has_customer: bool
    email: str | None = None
    name: str | None = None
    phone: str | None = None
    address: BillingAddressSnapshot = field(default_factory=BillingAddressSnapshot)
    payment_methods: list[PaymentMethodSnapshot] = field(default_factory=list)


@dataclass(frozen=True)
class BillingTransactionSnapshot:
    """Represent one completed Stripe Checkout transaction."""

    id: str
    at: str
    amount: int
    currency: str
    status: str
    credits: int | None
    pack_id: str | None
    document_url: str | None


@dataclass(frozen=True)
class BillingTransactionsSnapshot:
    """Represent a bounded Stripe purchase history for the current account."""

    available: bool
    entries: list[BillingTransactionSnapshot] = field(default_factory=list)


def _stripe_value(obj: Any, key: str, default: Any = None) -> Any:
    """Read one field from a Stripe object or a test mapping.

    Args:
        obj: Stripe resource, mapping, or ``None``.
        key: Field name to read.
        default: Value returned when the field is absent.

    Returns:
        The field value, or ``default``.
    """
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _stripe_id(value: Any) -> str | None:
    """Return an expandable Stripe field's id without assuming expansion state.

    Args:
        value: Resource object, mapping, string id, or ``None``.

    Returns:
        Stripe id when present, otherwise ``None``.
    """
    if isinstance(value, str):
        return value
    resource_id = _stripe_value(value, "id")
    return str(resource_id) if resource_id else None


def platform_fee_credits_for_usage(usages: Iterable[ModelUsage]) -> int:
    """Return the platform-fee portion of a run's per-model credit cost, rounding up.

    The :data:`PLATFORM_FEE_FRACTION` share of the run's full per-model cost
    (:func:`core.billing.pricing.credits_for_usage`). The only amount a **BYOK**
    run is charged, since the provider tokens were paid on the user's own key.
    At least one credit when the run cost anything.

    Args:
        usages: Per-model token usage for the run.

    Returns:
        The non-negative platform-fee credits (``0`` when the run cost nothing).
    """
    full = credits_for_usage(usages)
    fee = full * PLATFORM_FEE_FRACTION
    if fee <= 0:
        return 0
    return max(1, math.ceil(fee))


def run_cost_credits(
    usages: Iterable[ModelUsage],
    token_source: str,
    token_sources_by_model: Mapping[str, str] | None = None,
) -> int:
    """Return the credits a run costs: full per-model cost, or the BYOK platform fee.

    A managed run is charged its full per-model token cost
    (:func:`core.billing.pricing.credits_for_usage`); a BYOK run is charged only
    Skynet's platform fee (:func:`platform_fee_credits_for_usage`), since the
    provider tokens were paid on the user's own key.

    Args:
        usages: Per-model token usage for the run.
        token_source: Fallback source for legacy or unrecognized model rows.
        token_sources_by_model: Optional per-model sources for a mixed job.

    Returns:
        The non-negative credit cost.
    """
    usage_rows = list(usages)
    if token_sources_by_model is None:
        if token_source == TOKEN_SOURCE_BYOK:
            return platform_fee_credits_for_usage(usage_rows)
        return credits_for_usage(usage_rows)
    managed = [
        usage for usage in usage_rows if token_sources_by_model.get(usage.model, token_source) != TOKEN_SOURCE_BYOK
    ]
    byok = [usage for usage in usage_rows if token_sources_by_model.get(usage.model, token_source) == TOKEN_SOURCE_BYOK]
    return credits_for_usage(managed) + platform_fee_credits_for_usage(byok)


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
    if PLATFORM_FEE_FRACTION <= 0:
        # A fee-less BYOK run can never touch the balance, so any positive
        # balance backs an effectively unlimited ceiling.
        return _BYOK_UNCAPPED_CEILING
    budget = math.ceil(spendable / PLATFORM_FEE_FRACTION)
    while budget > 1 and max(1, math.ceil(budget * PLATFORM_FEE_FRACTION)) > spendable:
        budget -= 1
    return budget


def committed_spend_credits(budget: int, token_source: str) -> int:
    """Return the most balance credits a run's cost-ceiling budget can consume.

    The inverse of :func:`cost_ceiling_budget`: given a run's ``max_cost_credits``
    (a full-cost ceiling), how many credits of the account's balance it holds a
    claim on. A managed run can debit up to the full budget; a BYOK run debits
    only the platform fee of it (rounded up to at least one credit, matching
    :func:`platform_fee_credits_for_usage`). Used at submit time to subtract the
    commitments of still-active runs from the balance, so concurrent submissions
    cannot collectively promise more than the account holds.

    Args:
        budget: The run's cost-ceiling budget in full-cost credits; non-positive
            yields ``0``.
        token_source: ``"managed"`` or ``"byok"`` — sets the conversion.

    Returns:
        The non-negative credits of balance the run can still debit.
    """
    if budget <= 0:
        return 0
    if token_source != TOKEN_SOURCE_BYOK:
        return budget
    fee = budget * PLATFORM_FEE_FRACTION
    if fee <= 0:
        return 0
    return max(1, math.ceil(fee))


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
        """Seed the one-time free grant if unseeded; return the remaining credits.

        Seeds an unseeded row (NULL ``grant_remaining`` — a new account) to the
        one-time :data:`FREE_GRANT_CREDITS`. The grant is lifetime — once seeded
        it only ever draws down; there is no renewal. Mutates ``customer`` in
        place; the caller commits. A missing row (``None``) reads a full free
        grant without persisting — the row is created on the first real mutation
        (debit / top-up).

        Args:
            customer: The account's billing row, or ``None`` when it has none yet.
            now: Instant stamped as ``updated_at`` when the seed mutates the row.

        Returns:
            Credits remaining in the account's grant.
        """
        if customer is None:
            return FREE_GRANT_CREDITS
        if customer.grant_remaining is None:
            customer.grant_remaining = FREE_GRANT_CREDITS
            customer.updated_at = now
            return FREE_GRANT_CREDITS
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
        """Build a Checkout return URL on the public app origin.

        Args:
            status: Query value appended as ``?billing=`` (``success``/``cancel``).

        Returns:
            An absolute URL back to the app root, where the credit provider
            picks up the ``billing`` param and syncs the wallet.
        """
        return f"{settings.app_public_url.rstrip('/')}/?billing={status}"

    def _billing_settings_url(self) -> str:
        """Build the public return URL that reopens Billing settings.

        Returns:
            An absolute app URL with the billing settings deep link.
        """
        return f"{settings.app_public_url.rstrip('/')}/?settings=billing"

    def _existing_customer_id(self, username: str) -> str | None:
        """Return an account's real Stripe customer id without provisioning one.

        Args:
            username: Account identity whose billing link is read.

        Returns:
            A ``cus_...`` id, or ``None`` for a customerless/local-only account.
        """
        with Session(self._engine) as session:
            row = session.get(BillingCustomerModel, username)
            if row is None or row.stripe_customer_id.startswith(LOCAL_CUSTOMER_PREFIX):
                return None
            return row.stripe_customer_id

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

    def get_billing_profile(self, username: str) -> BillingProfileSnapshot:
        """Return display-safe billing details and saved payment methods from Stripe.

        Opening Settings never provisions a Stripe customer. Accounts that have
        not checked out yet receive an available, empty profile; configured
        accounts read Stripe directly so stale billing details are never cached
        in the application database.

        Args:
            username: Authenticated account whose Stripe profile is read.

        Returns:
            Billing details and masked payment-method metadata.

        Raises:
            DomainError: 502 when Stripe cannot serve the profile read.
        """
        customer_id = self._existing_customer_id(username)
        if customer_id is None:
            return BillingProfileSnapshot(available=settings.is_stripe_configured, has_customer=False)
        if not settings.is_stripe_configured:
            return BillingProfileSnapshot(available=False, has_customer=True)
        stripe_mod = self._stripe()
        try:
            customer = stripe_mod.Customer.retrieve(customer_id)
            methods = stripe_mod.Customer.list_payment_methods(customer_id, limit=PAYMENT_METHOD_LIMIT)
        except stripe.StripeError as exc:
            raise DomainError("billing.provider_unavailable", status=502) from exc

        invoice_settings = _stripe_value(customer, "invoice_settings", {})
        default_method_id = _stripe_id(_stripe_value(invoice_settings, "default_payment_method"))
        payment_methods: list[PaymentMethodSnapshot] = []
        for method in _stripe_value(methods, "data", []) or []:
            method_type = str(_stripe_value(method, "type", "unknown"))
            details = _stripe_value(method, method_type, {})
            method_id = _stripe_id(method)
            if method_id is None:
                continue
            payment_methods.append(
                PaymentMethodSnapshot(
                    id=method_id,
                    type=method_type,
                    brand=_stripe_value(details, "brand"),
                    last4=_stripe_value(details, "last4"),
                    exp_month=_stripe_value(details, "exp_month"),
                    exp_year=_stripe_value(details, "exp_year"),
                    is_default=method_id == default_method_id,
                )
            )

        address = _stripe_value(customer, "address", {})
        return BillingProfileSnapshot(
            available=True,
            has_customer=True,
            email=_stripe_value(customer, "email"),
            name=_stripe_value(customer, "name"),
            phone=_stripe_value(customer, "phone"),
            address=BillingAddressSnapshot(
                line1=_stripe_value(address, "line1"),
                line2=_stripe_value(address, "line2"),
                city=_stripe_value(address, "city"),
                state=_stripe_value(address, "state"),
                postal_code=_stripe_value(address, "postal_code"),
                country=_stripe_value(address, "country"),
            ),
            payment_methods=payment_methods,
        )

    def get_transactions(self, username: str, start: datetime, end: datetime) -> BillingTransactionsSnapshot:
        """Return completed Checkout purchases for a date window from Stripe.

        Args:
            username: Authenticated account whose purchases are read.
            start: Inclusive lower bound on Checkout Session creation time.
            end: Inclusive upper bound on Checkout Session creation time.

        Returns:
            A bounded, most-recent-first purchase history.

        Raises:
            DomainError: 502 when Stripe cannot serve the history read.
        """
        customer_id = self._existing_customer_id(username)
        if customer_id is None:
            return BillingTransactionsSnapshot(available=settings.is_stripe_configured)
        if not settings.is_stripe_configured:
            return BillingTransactionsSnapshot(available=False)
        stripe_mod = self._stripe()
        try:
            sessions = stripe_mod.checkout.Session.list(
                customer=customer_id,
                status="complete",
                created={"gte": int(start.timestamp()), "lte": int(end.timestamp())},
                limit=TRANSACTION_LIMIT,
                expand=["data.payment_intent.latest_charge", "data.invoice"],
            )
        except stripe.StripeError as exc:
            raise DomainError("billing.provider_unavailable", status=502) from exc

        entries: list[BillingTransactionSnapshot] = []
        for checkout in _stripe_value(sessions, "data", []) or []:
            payment_intent = _stripe_value(checkout, "payment_intent", {})
            charge = _stripe_value(payment_intent, "latest_charge", {})
            invoice = _stripe_value(checkout, "invoice", {})
            amount = int(_stripe_value(checkout, "amount_total", 0) or 0)
            refunded = int(_stripe_value(charge, "amount_refunded", 0) or 0)
            if _stripe_value(charge, "disputed", False):
                status = "disputed"
            elif refunded >= amount > 0:
                status = "refunded"
            elif refunded > 0:
                status = "partially_refunded"
            elif _stripe_value(checkout, "payment_status") == "paid":
                status = "paid"
            else:
                status = "processing"
            metadata = _stripe_value(checkout, "metadata", {})
            credits_raw = _stripe_value(metadata, "credits")
            try:
                credits = int(credits_raw) if credits_raw is not None else None
            except (TypeError, ValueError):
                credits = None
            created = int(_stripe_value(checkout, "created", 0) or 0)
            entries.append(
                BillingTransactionSnapshot(
                    id=str(_stripe_value(checkout, "id", "")),
                    at=datetime.fromtimestamp(created, UTC).isoformat(),
                    amount=amount,
                    currency=str(_stripe_value(checkout, "currency", "usd") or "usd").upper(),
                    status=status,
                    credits=credits,
                    pack_id=_stripe_value(metadata, "pack_id"),
                    document_url=_stripe_value(invoice, "hosted_invoice_url") or _stripe_value(charge, "receipt_url"),
                )
            )
        return BillingTransactionsSnapshot(available=True, entries=entries)

    def create_portal_session(self, username: str, *, payment_method_update: bool) -> str:
        """Create a Stripe-hosted billing-management session for the account.

        Args:
            username: Authenticated account whose customer portal is opened.
            payment_method_update: Deep-link directly into adding/updating a
                payment method when true; otherwise open the portal home.

        Returns:
            The hosted Customer Portal URL.

        Raises:
            DomainError: 502 when Stripe cannot create the portal session; 503
                when Stripe is not configured.
        """
        stripe_mod = self._stripe()
        customer_id = self.get_or_create_customer(username)
        return_url = self._billing_settings_url()
        kwargs: dict[str, Any] = {"customer": customer_id, "return_url": return_url}
        if payment_method_update:
            kwargs["flow_data"] = {
                "type": "payment_method_update",
                "after_completion": {
                    "type": "redirect",
                    "redirect": {"return_url": return_url},
                },
            }
        try:
            portal = stripe_mod.billing_portal.Session.create(**kwargs)
        except stripe.StripeError as exc:
            raise DomainError("billing.provider_unavailable", status=502) from exc
        return str(portal.url)

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
        return self._create_checkout(username, {"price": price_id, "quantity": 1}, pack_id, credits)

    def create_custom_checkout(self, username: str, credits: int) -> str:
        """Create a one-time Checkout Session for a user-chosen credit amount.

        One credit is one cent, so ``credits`` is passed to Stripe verbatim as
        the ad-hoc ``unit_amount``; the webhook credits the account from the
        session metadata exactly as for a fixed pack.

        Args:
            username: Buyer identity; stamped into session metadata so the
                webhook can credit the right account.
            credits: Credit amount to buy, within
                :data:`CUSTOM_CREDITS_MIN`..:data:`CUSTOM_CREDITS_MAX`.

        Returns:
            The hosted Stripe Checkout URL to redirect the buyer to.

        Raises:
            DomainError: 400 when ``credits`` is out of bounds; 503 when
                Stripe is not configured.
        """
        if not CUSTOM_CREDITS_MIN <= credits <= CUSTOM_CREDITS_MAX:
            raise DomainError("billing.invalid_amount", status=400, credits=credits)
        line_item = {
            "price_data": {
                "currency": "usd",
                "unit_amount": credits,
                "product_data": {"name": f"Skynet credits · {credits}"},
            },
            "quantity": 1,
        }
        return self._create_checkout(username, line_item, "custom", credits)

    def _create_checkout(self, username: str, line_item: dict[str, Any], pack_id: str, credits: int) -> str:
        """Create the Stripe Checkout Session shared by pack and custom top-ups.

        Args:
            username: Buyer identity; stamped into session metadata so the
                webhook can credit the right account.
            line_item: The single Stripe line item to charge (a fixed price id
                or ad-hoc ``price_data``).
            pack_id: Pack id (or ``"custom"``) recorded in metadata for the
                ledger description.
            credits: Credits the webhook grants once the session completes.

        Returns:
            The hosted Stripe Checkout URL.
        """
        stripe_mod = self._stripe()
        customer_id = self.get_or_create_customer(username)
        metadata = {"username": username, "pack_id": pack_id, "credits": str(credits)}
        checkout = stripe_mod.checkout.Session.create(
            customer=customer_id,
            mode="payment",
            line_items=[line_item],
            billing_address_collection="required",
            customer_update={"address": "auto", "name": "auto"},
            invoice_creation={"enabled": True},
            saved_payment_method_options={
                "payment_method_save": "enabled",
                "payment_method_remove": "enabled",
            },
            success_url=self._return_url("success"),
            cancel_url=self._return_url("cancel"),
            client_reference_id=username,
            metadata=metadata,
            payment_intent_data={"metadata": metadata},
        )
        return str(checkout.url)

    def get_wallet(self, username: str) -> WalletSnapshot:
        """Return the account's wallet snapshot from the billing tables.

        A pure DB read — no Stripe call — so it serves even when Stripe is
        unconfigured (balance 0). The free grant reflects real spend: run
        completions debit it via :meth:`debit_run`. The grant is one-time
        (seeded once, never renewed). A read that seeds an existing row is
        persisted; a brand-new account reads a full grant without creating a
        row until its first real mutation.

        Args:
            username: Account to summarize.

        Returns:
            A :class:`WalletSnapshot` of paid balance, free grant, and the most
            recent ledger rows.
        """
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            customer = session.get(BillingCustomerModel, username)
            grant_remaining = self._resolve_grant(customer, now)
            # Capture the seed mutation before the ledger query autoflushes
            # it — once flushed, ``is_modified`` reads clean and the commit below
            # would be skipped, dropping the seed on session close.
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
            if grant_dirty:
                session.commit()
            return WalletSnapshot(
                paid_balance_credits=customer.credit_balance if customer else 0,
                free_grant_remaining=grant_remaining,
                free_grant_total=FREE_GRANT_CREDITS,
                usage=usage,
            )

    def get_usage(self, username: str, start: datetime, end: datetime) -> UsageSnapshot:
        """Aggregate the account's credit ledger over a date window for the dashboard.

        A pure DB read — no Stripe call. Sums run rows in ``[start, end]`` into
        gross billed spend, a run count, a per-day billed series (ascending by
        date), and a per-model spend series (descending by credits); top-ups
        and grants are excluded from those rollups. The most recent :data:`USAGE_ENTRY_LIMIT` raw ledger rows in the
        window ride along as ``entries`` for the activity list and the per-run
        breakdown, while the rollups span every row in range regardless of that
        cap.

        Args:
            username: Account to summarize.
            start: Inclusive lower bound on ``created_at``.
            end: Inclusive upper bound on ``created_at``.

        Returns:
            A :class:`UsageSnapshot` of totals, per-day and per-model series, and
            the most recent ledger rows.
        """
        with Session(self._engine) as session:
            # Project only the columns the fold below reads — the full entity
            # (with description text on every row) is materialized for
            # thousands of rows on a wide window, only to be reduced to sums.
            rows = (
                session.query(
                    CreditLedgerModel.id,
                    CreditLedgerModel.created_at,
                    CreditLedgerModel.kind,
                    CreditLedgerModel.model,
                    CreditLedgerModel.delta_credits,
                    CreditLedgerModel.description,
                    CreditLedgerModel.input_tokens,
                    CreditLedgerModel.output_tokens,
                )
                .filter(
                    CreditLedgerModel.username == username,
                    CreditLedgerModel.created_at >= start,
                    CreditLedgerModel.created_at <= end,
                )
                .order_by(CreditLedgerModel.created_at.desc())
                .all()
            )
        billed = 0
        runs = 0
        per_day: dict[str, int] = {}
        per_model: dict[str | None, list[int]] = {}
        # Only debits count as spend; a positive ``run`` row (a legacy
        # correction) is ignored by the rollups but still rides in ``entries``.
        for row in rows:
            if row.kind != "run" or row.delta_credits >= 0:
                continue
            spent = -row.delta_credits
            billed += spent
            runs += 1
            day = row.created_at.date().isoformat()
            per_day[day] = per_day.get(day, 0) + spent
            model = per_model.setdefault(row.model, [0, 0, 0, 0])
            model[0] += spent
            model[1] += 1
            model[2] += row.input_tokens or 0
            model[3] += row.output_tokens or 0
        by_day = [UsageDayRow(date=date, billed_credits=credits) for date, credits in sorted(per_day.items())]
        by_model = [
            UsageModelRow(
                model=model,
                credits=vals[0],
                runs=vals[1],
                input_tokens=vals[2],
                output_tokens=vals[3],
            )
            for model, vals in sorted(per_model.items(), key=lambda kv: kv[1][0], reverse=True)
        ]
        entries = [
            LedgerRow(
                id=str(row.id),
                at=row.created_at.isoformat(),
                label=row.description or row.kind,
                model=row.model,
                credits=row.delta_credits,
                kind=row.kind,
            )
            for row in rows[:USAGE_ENTRY_LIMIT]
        ]
        return UsageSnapshot(
            start=start.isoformat(),
            end=end.isoformat(),
            billed_credits=billed,
            runs=runs,
            by_day=by_day,
            by_model=by_model,
            entries=entries,
        )

    def spendable_credits(self, username: str) -> int:
        """Return the account's total spendable credits (free grant + paid balance).

        Resolves the grant first (seeding a new account's one-time free grant),
        so the figure is never stale. A seeded row is persisted. This is the
        figure the submit gate checks: ``> 0`` means a managed run may start. A
        brand-new account reads a full grant without a row being created.

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

    def total_outstanding_credits(self) -> int:
        """Return the total unspent credit liability across every account.

        Sums each account's purchased ``credit_balance`` plus its remaining free
        grant. This is the pool the shared OpenRouter float ultimately backs:
        credits users have paid for (or been granted) but not yet spent on runs.
        A pure DB read — no grant seeding, no Stripe call — so it reflects only
        liability already recorded.

        Returns:
            Sum of paid balances and remaining grants across all customers,
            never negative.
        """
        with Session(self._engine) as session:
            paid, granted = session.query(
                func.coalesce(func.sum(BillingCustomerModel.credit_balance), 0),
                func.coalesce(func.sum(BillingCustomerModel.grant_remaining), 0),
            ).one()
        return max(int(paid) + int(granted), 0)

    def credits_spent_since(self, since: datetime) -> int:
        """Return platform-wide credits spent on runs since a timestamp.

        Sums the magnitude of the negative ledger deltas (run charges written by
        :meth:`debit_run`) posted at or after ``since``; positive rows (top-ups,
        grants, refunds) are excluded. Backs the global daily spend kill-switch,
        which refuses new submissions once a trailing-window total is reached.

        Args:
            since: Lower bound (inclusive) on ``created_at``; pass a timezone-aware
                UTC datetime such as ``now - 24h``.

        Returns:
            Total credits spent in the window, never negative.
        """
        with Session(self._engine) as session:
            spent = (
                session.query(func.coalesce(func.sum(-CreditLedgerModel.delta_credits), 0))
                .filter(
                    CreditLedgerModel.delta_credits < 0,
                    CreditLedgerModel.created_at >= since,
                )
                .scalar()
            )
        return max(int(spent or 0), 0)

    def debit_run(
        self,
        username: str,
        usages: Iterable[ModelUsage],
        *,
        model: str | None,
        description: str,
        token_source: str = TOKEN_SOURCE_MANAGED,
        token_sources_by_model: Mapping[str, str] | None = None,
    ) -> int:
        """Charge a finished run's per-model credit cost to the account, grant first.

        Writes one signed negative ``run`` row to ``credit_ledger`` and draws the
        cost from the free grant before the purchased balance, mirroring how
        :meth:`get_wallet` reports spendable credits (grant then paid). A **managed**
        run is charged its full per-model token cost; a **BYOK** run is charged only
        Skynet's platform fee (:func:`run_cost_credits`), since the provider tokens
        were already paid on the user's own key — so credits still meter the
        platform on a BYOK run without double-charging for inference. The grant is
        resolved first (seeding a new account's one-time free grant) so the
        debit lands against a current grant. Idempotency is the caller's
        responsibility: the worker debits inside its once-only completion claim, so
        a redelivered/re-run job never double-charges. A run costing zero credits
        writes nothing.

        The debit can never drive the account negative: the customer row is read
        under ``FOR UPDATE`` so concurrent debits serialize, and the charge is
        clamped to what the account actually holds — a run that cost more than
        the remaining balance drains it to exactly zero, the shortfall is logged
        as absorbed, and the ledger row records the clamped (actually charged)
        amount. The DB backs this up with ``CHECK`` constraints on the balance
        columns, so a bug here fails the transaction instead of persisting a
        negative.

        Args:
            username: Account the run is billed to.
            usages: Per-model token usage for the run; priced per-model into the
                credit cost.
            model: Model id stamped on the ledger row, or ``None``.
            description: Human label for the ledger row (typically the run name).
            token_source: ``"managed"`` (full cost) or ``"byok"`` (platform fee
                only); defaults to managed.
            token_sources_by_model: Optional per-model source map for a mixed job.

        Returns:
            The credit cost actually charged (``0`` when nothing was billed) —
            at most the account's spendable balance, so it can undershoot the
            run's full cost on a depleted account.
        """
        usage_rows = list(usages)
        cost = run_cost_credits(usage_rows, token_source, token_sources_by_model)
        if cost <= 0:
            return 0
        # The ledger row records the measured tokens behind the charge — the
        # per-model Usage tab reads these back, so the invoice-side credit
        # figure and the token figure come from the same measurement.
        input_tokens = sum(usage.input_tokens for usage in usage_rows)
        output_tokens = sum(usage.output_tokens for usage in usage_rows)
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            customer = session.get(BillingCustomerModel, username, with_for_update=True)
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
            grant = max(int(customer.grant_remaining or 0), 0)
            paid = max(int(customer.credit_balance), 0)
            charged = min(cost, grant + paid)
            if charged < cost:
                logger.warning(
                    "debit for %s clamped to balance: cost=%d charged=%d absorbed=%d (%s)",
                    username,
                    cost,
                    charged,
                    cost - charged,
                    description or "Run",
                )
            from_grant = min(grant, charged)
            customer.grant_remaining = grant - from_grant
            customer.credit_balance = paid - (charged - from_grant)
            customer.updated_at = now
            if charged > 0:
                session.add(
                    CreditLedgerModel(
                        username=username,
                        delta_credits=-charged,
                        kind="run",
                        description=description or "Run",
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                    )
                )
            session.commit()
        return charged

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
        # After the top-up commits — so a fresh read sees the new liability — check
        # that the shared OpenRouter float still covers what users are owed. Runs
        # only on the freshly-applied path (a redelivery returns above), and never
        # on the money path itself: the purchase is already committed.
        if str(event["type"]) == "checkout.session.completed":
            self._monitor_float()

    def _monitor_float(self) -> None:
        """Check the OpenRouter float after a purchase, swallowing every failure.

        A best-effort tripwire: reads the master-account balance and warns when
        it has fallen below the configured floor. Wrapped so a monitor failure
        (HTTP timeout, DB hiccup on the liability sum) can never propagate into
        the webhook handler, which has already committed the credit.
        """
        try:
            check_float(self.total_outstanding_credits())
        except Exception:
            logger.exception("OpenRouter float monitor failed")

    def _apply_event(self, session: Session, event: Any) -> None:
        """Dispatch a verified event to its handler; unknown types are no-ops.

        Args:
            session: Open session; the caller commits (event row + effect land
                atomically).
            event: The verified Stripe event object.
        """
        event_type = str(event["type"])
        obj = event["data"]["object"]
        event_id = str(event["id"])
        if event_type == "checkout.session.completed":
            self._on_checkout_completed(session, event_id, obj)
        elif event_type == "charge.refunded":
            self._on_charge_refunded(session, event_id, obj)
        elif event_type == "charge.dispute.created":
            self._on_dispute_created(session, event_id, obj)

    def _on_checkout_completed(self, session: Session, event_id: str, obj: Any) -> None:
        """Credit a completed one-time pack purchase to the buyer's balance.

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
        payment_intent = str(obj.get("payment_intent") or "") or None
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
                stripe_payment_intent_id=payment_intent,
            )
        )

    def _pi_clawback_context(self, session: Session, payment_intent: str) -> tuple[str | None, int, int]:
        """Sum a PaymentIntent's ledger into (account, credits granted, credits already reversed).

        Reads every ledger row stamped with ``payment_intent`` — the top-up that
        granted the credits plus any earlier refund/dispute clawbacks against it —
        so a handler can cap a new clawback at what the top-up granted and net out
        what was already reversed. Resolving the account from the top-up row (not
        the charge's ``customer`` field) is what lets a dispute event, which carries
        no customer, still find its account.

        Args:
            session: Open session to read the ledger under.
            payment_intent: The Stripe PaymentIntent id (``pi_…``) to sum over.

        Returns:
            ``(username, granted, clawed)`` — the account the PaymentIntent's
            top-up credited (``None`` when no top-up row matches), the credits that
            top-up granted, and the credits already reversed by refunds/disputes.
        """
        rows = (
            session.query(
                CreditLedgerModel.username,
                CreditLedgerModel.kind,
                CreditLedgerModel.delta_credits,
            )
            .filter(CreditLedgerModel.stripe_payment_intent_id == payment_intent)
            .all()
        )
        username: str | None = None
        granted = 0
        clawed = 0
        for row in rows:
            if row.kind == "topup":
                username = row.username
                granted += int(row.delta_credits)
            elif row.kind in ("refund", "dispute"):
                username = username or row.username
                clawed += -int(row.delta_credits)
        return username, granted, clawed

    def _on_charge_refunded(self, session: Session, event_id: str, obj: Any) -> None:
        """Claw back credits for a refunded charge, netting out earlier partial refunds.

        Fires on every ``charge.refunded`` event. ``amount_refunded`` is the
        charge's cumulative refunded cents (one credit per cent), so the credits to
        remove now are that cumulative figure — capped at what the top-up granted —
        minus what earlier refunds already reversed. A charge Skynet never credited
        (no matching top-up) is logged and skipped.

        Args:
            session: Open session (caller commits).
            event_id: Stripe event id, recorded on the clawback ledger row.
            obj: The Charge object from the event.
        """
        payment_intent = str(obj.get("payment_intent") or "")
        refunded = int(obj.get("amount_refunded") or 0)
        if not payment_intent or refunded <= 0:
            return
        username, granted, clawed = self._pi_clawback_context(session, payment_intent)
        if username is None:
            logger.warning(
                "refund for unrecognized payment_intent %s (event %s); nothing to claw back",
                payment_intent,
                event_id,
            )
            return
        delta = max(0, min(refunded, granted) - clawed)
        if delta <= 0:
            return
        self._write_clawback(
            session,
            event_id=event_id,
            payment_intent=payment_intent,
            username=username,
            credits=delta,
            kind="refund",
            description="Refund",
        )

    def _on_dispute_created(self, session: Session, event_id: str, obj: Any) -> None:
        """Claw back credits when a charge is disputed — the chargeback pulls the funds back.

        Fires on ``charge.dispute.created``. ``amount`` is the disputed cents (one
        credit per cent); the funds have left Skynet's Stripe balance, so the
        matching credits are removed, capped at the top-up's still-unreversed credits
        so a dispute after a partial refund never double-counts. A dispute on a charge
        Skynet never credited is logged and skipped.

        Args:
            session: Open session (caller commits).
            event_id: Stripe event id, recorded on the clawback ledger row.
            obj: The Dispute object from the event.
        """
        payment_intent = str(obj.get("payment_intent") or "")
        disputed = int(obj.get("amount") or 0)
        if not payment_intent or disputed <= 0:
            return
        username, granted, clawed = self._pi_clawback_context(session, payment_intent)
        if username is None:
            logger.warning(
                "dispute for unrecognized payment_intent %s (event %s); nothing to claw back",
                payment_intent,
                event_id,
            )
            return
        delta = max(0, min(disputed, granted - clawed))
        if delta <= 0:
            return
        self._write_clawback(
            session,
            event_id=event_id,
            payment_intent=payment_intent,
            username=username,
            credits=delta,
            kind="dispute",
            description="Chargeback",
        )

    def _write_clawback(
        self,
        session: Session,
        *,
        event_id: str,
        payment_intent: str,
        username: str,
        credits: int,
        kind: str,
        description: str,
    ) -> None:
        """Remove up to ``credits`` from the account's purchased balance, flooring at zero.

        A refund or dispute returns money that only ever bought the purchased
        balance, so the clawback draws from ``credit_balance`` alone — never the free
        grant — under ``FOR UPDATE`` so it serializes with concurrent debits. The
        balance may already be spent below what is owed; the charge is clamped to what
        remains (the DB ``CHECK`` forbids going negative) and the uncollectable
        shortfall is logged rather than carried as user debt. A clawback that can
        collect nothing writes no ledger row, mirroring a zero-credit debit.

        Args:
            session: Open session (caller commits).
            event_id: Stripe event id, recorded on the ledger row.
            payment_intent: The PaymentIntent id, stamped on the ledger row so a
                later clawback nets against it.
            username: Account to draw the credits back from.
            credits: Credits owed back to Stripe (positive).
            kind: Ledger kind (``"refund"`` or ``"dispute"``).
            description: Human label for the ledger row.
        """
        now = datetime.now(UTC)
        customer = session.get(BillingCustomerModel, username, with_for_update=True)
        paid = max(int(customer.credit_balance), 0) if customer is not None else 0
        charged = min(credits, paid)
        if charged < credits:
            logger.warning(
                "%s clawback for %s clamped to balance: owed=%d collected=%d uncollectable=%d (pi %s)",
                kind,
                username,
                credits,
                charged,
                credits - charged,
                payment_intent,
            )
        if charged <= 0:
            return
        customer.credit_balance = paid - charged
        customer.updated_at = now
        session.add(
            CreditLedgerModel(
                username=username,
                delta_credits=-charged,
                kind=kind,
                description=description,
                stripe_event_id=event_id,
                stripe_payment_intent_id=payment_intent,
            )
        )
