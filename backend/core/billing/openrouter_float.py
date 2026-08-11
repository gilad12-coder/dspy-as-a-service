"""Monitor the OpenRouter master-account float against outstanding credit liability.

Native Auto Top-Up (configured in the OpenRouter dashboard) is what actually
refills the shared prepaid balance from a saved card. This module does not move
money; it is a tripwire. It reads the master-account balance (GET
``/api/v1/credits``) and warns when that balance has fallen below a configured
floor — the signal that Auto Top-Up has failed (a declined card) or that demand
is outrunning refills. Outstanding credit liability (credits users have bought
but not yet spent) rides along in the warning for context.

Everything here fails open: an unset key, an unreachable endpoint, a non-2xx
response, or an unexpected body yields no status and no alert, never an
exception into the caller. The Stripe webhook that triggers a check must never
break because a monitor read timed out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from ..config import settings

logger = logging.getLogger("skynet.billing.openrouter_float")

OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
_REQUEST_TIMEOUT_SECONDS = 6.0
_CREDITS_PER_DOLLAR = 100


@dataclass(frozen=True)
class FloatStatus:
    """Snapshot of the OpenRouter float against its floor and outstanding liability.

    All three figures are in credits (1 credit = 1 cent), so they compare
    directly. ``balance_credits`` is the master account's prepaid remainder,
    ``floor_credits`` the low-water mark below which the float is considered
    thin, and ``liability_credits`` the credits users have bought but not spent.
    """

    balance_credits: int
    floor_credits: int
    liability_credits: int

    @property
    def covered(self) -> bool:
        """Return whether the balance is at or above the floor.

        Returns:
            True when the float still clears its low-water mark.
        """
        return self.balance_credits >= self.floor_credits


def read_account_balance_credits() -> int | None:
    """Read the OpenRouter master-account prepaid balance, in credits.

    Calls ``GET /api/v1/credits`` with the master inference key and returns the
    remaining balance (``total_credits - total_usage``) converted from dollars to
    credits. Fails open.

    Returns:
        Remaining balance in credits, or ``None`` when the key is unset or the
        read fails (unreachable, non-2xx, or an unexpected body shape).
    """
    key = settings.openrouter_api_key
    if key is None:
        return None
    try:
        response = httpx.get(
            OPENROUTER_CREDITS_URL,
            headers={"Authorization": f"Bearer {key.get_secret_value()}"},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        logger.warning("OpenRouter balance read failed: %s", exc)
        return None
    if response.status_code >= 400:
        logger.warning("OpenRouter balance read returned HTTP %s", response.status_code)
        return None
    try:
        data = response.json().get("data") or {}
        remaining_dollars = float(data["total_credits"]) - float(data["total_usage"])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning("OpenRouter balance read had an unexpected shape: %s", exc)
        return None
    return round(remaining_dollars * _CREDITS_PER_DOLLAR)


def check_float(outstanding_credits: int) -> FloatStatus | None:
    """Compare the OpenRouter float to its floor and warn when it runs thin.

    Reads the master-account balance and builds a :class:`FloatStatus` against
    the configured floor (``settings.openrouter_balance_floor_credits``). Logs a
    WARNING carrying balance, floor, and outstanding liability when the balance
    is below the floor; stays quiet otherwise. Fails open — a disabled monitor
    or an unreadable balance simply yields ``None``.

    Args:
        outstanding_credits: Credits users have bought (or been granted) but not
            yet spent — the liability the shared float ultimately backs. Logged
            for context; does not affect whether the warning fires.

    Returns:
        The float status, or ``None`` when the monitor is disabled (floor ``<=
        0`` or the key is unset) or the balance could not be read.
    """
    floor = settings.openrouter_balance_floor_credits
    if floor <= 0:
        return None
    balance = read_account_balance_credits()
    if balance is None:
        return None
    status = FloatStatus(
        balance_credits=balance,
        floor_credits=floor,
        liability_credits=outstanding_credits,
    )
    if not status.covered:
        logger.warning(
            "OpenRouter float low: balance $%.2f below floor $%.2f "
            "(outstanding liability $%.2f). Check Auto Top-Up and the saved card.",
            balance / _CREDITS_PER_DOLLAR,
            floor / _CREDITS_PER_DOLLAR,
            outstanding_credits / _CREDITS_PER_DOLLAR,
        )
    return status
