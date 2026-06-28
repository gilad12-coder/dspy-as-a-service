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
_METERED = ("skynet_metered", "Skynet Token Overage", 1)


def _find_price(lookup_key: str) -> str | None:
    """Return an existing active price id for ``lookup_key``, or None.

    Args:
        lookup_key: Stable lookup key the price was created with.

    Returns:
        The matching price id, or ``None`` when none exists yet.
    """
    existing = stripe.Price.list(lookup_keys=[lookup_key], active=True, limit=1)
    return existing.data[0].id if existing.data else None


def _ensure_price(
    lookup_key: str, name: str, unit_amount: int, recurring: dict | None = None
) -> str:
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
            display_name="Skynet token usage",
            event_name=event_name,
            default_aggregation={"formula": "sum"},
            customer_mapping={"event_payload_key": "stripe_customer_id", "type": "by_id"},
        )
        print(f"  create meter {event_name} -> {meter.id}")
        return meter.id
    except Exception as exc:  # meter is optional; must never block packs/Premium
        print(f"  WARN   metered overage skipped ({exc})", file=sys.stderr)
        return None


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
    premium_id = _ensure_price(
        _PREMIUM[0], _PREMIUM[1], _PREMIUM[2], recurring={"interval": "month"}
    )

    print("Provisioning Founder's Rate subscription ...")
    founders_id = _ensure_price(
        _FOUNDERS[0], _FOUNDERS[1], _FOUNDERS[2], recurring={"interval": "month"}
    )

    print("Provisioning metered overage (optional) ...")
    meter_id = _ensure_meter(settings.stripe_meter_event_name)
    metered_id = None
    if meter_id:
        metered_id = _ensure_price(
            _METERED[0],
            _METERED[1],
            _METERED[2],
            recurring={"interval": "month", "usage_type": "metered", "meter": meter_id},
        )

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
