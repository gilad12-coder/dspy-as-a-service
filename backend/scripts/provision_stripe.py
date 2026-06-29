"""Provision the Stripe products, prices, and meter that the billing code expects.

Run once (per Stripe account / mode) after setting ``STRIPE_SECRET_KEY`` in
``backend/.env``. Idempotent: prices are keyed by a stable ``lookup_key`` and the
meter by its ``event_name``, so re-running reuses what already exists instead of
duplicating. On success it prints the ``STRIPE_PRICE_*`` env lines to paste back
into ``backend/.env``.

    cd backend && python scripts/provision_stripe.py

The metered-overage price is optional and best-effort: if the Billing Meter API
is unavailable it is skipped with a warning, and packs + Premium still provision.
See docs/stripe-setup.md for the full walkthrough.
"""

from __future__ import annotations

import sys

import stripe

from core.config import settings

# (lookup_key, display name, unit amount in cents). The credits each pack grants
# live in core.billing.service.PACK_CREDITS — Stripe only holds the dollar price.
_PACKS: list[tuple[str, str, int]] = [
    ("skynet_pack_starter", "Skynet Credits — Starter", 500),
    ("skynet_pack_plus", "Skynet Credits — Plus", 2000),
    ("skynet_pack_pro", "Skynet Credits — Pro", 5000),
]
_PREMIUM = ("skynet_premium", "Skynet Premium", 2000)
_FOUNDERS = ("skynet_founders", "Skynet Founder's Rate", 2000)
# One meter unit = one credit, priced at 1 cent ($0.01 = one credit's value), so
# metered overage matches the per-model credit ledger one-to-one.
_METERED = ("skynet_metered", "Skynet Credit Overage", 1)


def _find_price(lookup_key: str) -> str | None:
    """Return an existing active price id for ``lookup_key``, or None.

    Args:
        lookup_key: Stable lookup key the price was created with.

    Returns:
        The matching price id, or ``None`` when none exists yet.
    """
    existing = stripe.Price.list(lookup_keys=[lookup_key], active=True, limit=1)
    return existing.data[0].id if existing.data else None


def _ensure_price(lookup_key: str, name: str, unit_amount: int, recurring: dict | None = None) -> str:
    """Return the price id for a pack/subscription, creating it if absent.

    Args:
        lookup_key: Stable idempotency key; a re-run reuses the matching price.
        name: Product display name (an inline product is created with it).
        unit_amount: Price in the smallest currency unit (cents).
        recurring: Optional Stripe ``recurring`` block for subscriptions/metered
            prices; ``None`` makes a one-time price.

    Returns:
        The Stripe price id.
    """
    found = _find_price(lookup_key)
    if found:
        print(f"  reuse  {lookup_key} -> {found}")
        return found
    params: dict = {
        "currency": "usd",
        "unit_amount": unit_amount,
        "lookup_key": lookup_key,
        "product_data": {"name": name},
    }
    if recurring:
        params["recurring"] = recurring
    price = stripe.Price.create(**params)
    print(f"  create {lookup_key} -> {price.id}")
    return price.id


def _ensure_meter(event_name: str) -> str | None:
    """Return a Billing Meter id for ``event_name``, creating it if absent.

    Best-effort: returns ``None`` (with a warning) if the Meter API is
    unavailable, so the rest of provisioning still completes.

    Args:
        event_name: Meter event name the metered price aggregates.

    Returns:
        The meter id, or ``None`` when meters could not be provisioned.
    """
    try:
        for meter in stripe.billing.Meter.list(status="active", limit=100).auto_paging_iter():
            if meter.event_name == event_name:
                print(f"  reuse  meter {event_name} -> {meter.id}")
                return meter.id
        meter = stripe.billing.Meter.create(
            display_name="Skynet credit usage",
            event_name=event_name,
            default_aggregation={"formula": "sum"},
            customer_mapping={"event_payload_key": "stripe_customer_id", "type": "by_id"},
        )
        print(f"  create meter {event_name} -> {meter.id}")
        return meter.id
    except Exception as exc:  # meter is optional; must never block packs/Premium
        print(f"  WARN   metered overage skipped ({exc})", file=sys.stderr)
        return None


def _ensure_metered_price(lookup_key: str, name: str, unit_amount: int, meter_id: str) -> str:
    """Return a metered price bound to ``meter_id``, rebinding on a meter cutover.

    A metered price is permanently tied to one meter, so changing
    ``STRIPE_METER_EVENT_NAME`` (a new meter) requires a fresh price — reusing the
    existing one by ``lookup_key`` would leave overage billing pointed at the old
    meter. When the price found under ``lookup_key`` aggregates a different meter,
    the new price is created with ``transfer_lookup_key`` so Stripe atomically
    moves the key onto it; the old price is left active and bound to the old meter
    so in-flight subscriptions keep billing until they are migrated off it (it is
    also the default price of its product, which Stripe refuses to archive). An
    already-correct price is reused unchanged.

    Args:
        lookup_key: Stable idempotency key for the metered price.
        name: Product display name used when a new price is created.
        unit_amount: Price per metered unit in cents.
        meter_id: Billing Meter id the price must aggregate.

    Returns:
        The Stripe price id bound to ``meter_id``.
    """
    params: dict = {
        "currency": "usd",
        "unit_amount": unit_amount,
        "lookup_key": lookup_key,
        "product_data": {"name": name},
        "recurring": {"interval": "month", "usage_type": "metered", "meter": meter_id},
    }
    existing = stripe.Price.list(lookup_keys=[lookup_key], active=True, limit=1)
    if existing.data:
        price = existing.data[0]
        bound = price.recurring.get("meter") if price.recurring else None
        if bound == meter_id:
            print(f"  reuse  {lookup_key} -> {price.id}")
            return price.id
        params["transfer_lookup_key"] = True
        print(f"  rebind {lookup_key}: moving key off {price.id} (was meter {bound})")
    price = stripe.Price.create(**params)
    print(f"  create {lookup_key} -> {price.id}")
    return price.id


def main() -> int:
    """Provision every Stripe price + meter and print the env lines to set.

    Returns:
        Process exit code (0 on success, 1 when Stripe is unconfigured).
    """
    if settings.stripe_secret_key is None:
        print("STRIPE_SECRET_KEY is not set in backend/.env — nothing to do.", file=sys.stderr)
        return 1
    stripe.api_key = settings.stripe_secret_key.get_secret_value()

    print("Provisioning credit packs ...")
    pack_ids = {key: _ensure_price(key, name, amount) for key, name, amount in _PACKS}

    print("Provisioning Premium subscription ...")
    premium_id = _ensure_price(_PREMIUM[0], _PREMIUM[1], _PREMIUM[2], recurring={"interval": "month"})

    print("Provisioning Founder's Rate subscription ...")
    founders_id = _ensure_price(_FOUNDERS[0], _FOUNDERS[1], _FOUNDERS[2], recurring={"interval": "month"})

    print("Provisioning metered overage (optional) ...")
    meter_id = _ensure_meter(settings.stripe_meter_event_name)
    metered_id = None
    if meter_id:
        metered_id = _ensure_metered_price(_METERED[0], _METERED[1], _METERED[2], meter_id)

    print("\nDone. Paste these into backend/.env:\n")
    print(f"STRIPE_PRICE_PACK_STARTER={pack_ids['skynet_pack_starter']}")
    print(f"STRIPE_PRICE_PACK_PLUS={pack_ids['skynet_pack_plus']}")
    print(f"STRIPE_PRICE_PACK_PRO={pack_ids['skynet_pack_pro']}")
    print(f"STRIPE_PRICE_PREMIUM={premium_id}")
    print(f"STRIPE_PRICE_FOUNDERS={founders_id}")
    if metered_id:
        print(f"STRIPE_PRICE_METERED={metered_id}")
        print(f"STRIPE_METER_EVENT_NAME={settings.stripe_meter_event_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
