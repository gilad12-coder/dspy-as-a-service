"""Monitor the OpenRouter master-account float against outstanding credit liability.

Native Auto Top-Up (configured in the OpenRouter dashboard) is what actually
refills the shared prepaid balance from a saved card. This module does not move
money; it is a tripwire. It reads the master-account balance (GET
``/api/v1/credits``) and warns when that balance has fallen below a configured
floor — the signal that Auto Top-Up has failed (a declined card) or that demand
is outrunning refills. Outstanding credit liability (credits users have bought
but not yet spent) rides along in the warning for context.

Two things trigger a check: the Stripe webhook after every credit purchase
(:meth:`~core.billing.service.StripeBillingService._monitor_float`) and the
periodic :class:`OpenRouterFloatSweeper` started by the API lifespan, so a
declined card on a quiet day is still caught. A breach fans out to the WARNING
log, the alert webhook and an operator email, rate-limited by a cooldown that
is shared across replicas through Redis when one is configured.

Everything here fails open: an unset key, an unreachable endpoint, a non-2xx
response, or an unexpected body yields no status and no alert, never an
exception into the caller. The Stripe webhook that triggers a check must never
break because a monitor read timed out.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
from redis.exceptions import RedisError
from sqlalchemy import text

from ..api.alerts import send_alert
from ..api.email_sender import email_configured, send_email
from ..api.rate_limit import shared_redis_client
from ..config import settings

logger = logging.getLogger("skynet.billing.openrouter_float")

OPENROUTER_CREDITS_URL = "https://openrouter.ai/api/v1/credits"
_REQUEST_TIMEOUT_SECONDS = 6.0
_CREDITS_PER_DOLLAR = 100
# Continues the 7421370000xx advisory-lock series in core/api/observability.py;
# defined here rather than imported so billing does not pull in the router
# graph that observability imports.
OPENROUTER_FLOAT_SWEEP_LOCK_KEY = 742137000005
_ALERT_COOLDOWN_REDIS_KEY = "skynet:openrouter-float:alert-cooldown"
_MIN_SWEEP_INTERVAL_SECONDS = 60.0

_local_cooldown_lock = threading.Lock()
_local_cooldown_until = 0.0


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
    the configured floor (``settings.openrouter_balance_floor_credits``). Below
    the floor it logs a WARNING carrying balance, floor, and outstanding
    liability and notifies the operator (alert webhook + email, cooldown-gated);
    stays quiet otherwise. Fails open — a disabled monitor or an unreadable
    balance simply yields ``None``.

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
        notify_low_float(status)
    return status


def _cooldown_allows(now: float | None = None) -> bool:
    """Claim the low-float notification slot, honouring the cooldown.

    Tries a Redis ``SET NX EX`` first so replicas share one cooldown window;
    when Redis is unset or unreachable it falls back to a per-process window.
    A cooldown of ``0`` always allows.

    Args:
        now: Monotonic clock override for the per-process window (tests).

    Returns:
        True when a notification may be sent now.
    """
    global _local_cooldown_until
    cooldown = settings.openrouter_float_alert_cooldown_seconds
    if cooldown <= 0:
        return True
    client = shared_redis_client()
    if client is not None:
        try:
            return bool(client.set(_ALERT_COOLDOWN_REDIS_KEY, "1", nx=True, ex=int(cooldown)))
        except RedisError as exc:
            logger.debug("Float alert cooldown fell back to process-local: %s", exc)
    stamp = now if now is not None else time.monotonic()
    with _local_cooldown_lock:
        if stamp < _local_cooldown_until:
            return False
        _local_cooldown_until = stamp + cooldown
        return True


def _format_low_float(status: FloatStatus) -> tuple[str, str]:
    """Render the operator-facing subject and body for a low-float breach.

    Args:
        status: The breaching float snapshot.

    Returns:
        ``(subject, body)`` in plain text.
    """
    subject = (
        f"OpenRouter float low: ${status.balance_credits / _CREDITS_PER_DOLLAR:.2f} "
        f"below floor ${status.floor_credits / _CREDITS_PER_DOLLAR:.2f}"
    )
    body = (
        f"OpenRouter master-account balance: ${status.balance_credits / _CREDITS_PER_DOLLAR:.2f}\n"
        f"Configured floor: ${status.floor_credits / _CREDITS_PER_DOLLAR:.2f}\n"
        f"Outstanding credit liability: ${status.liability_credits / _CREDITS_PER_DOLLAR:.2f}\n\n"
        "Auto Top-Up has likely failed (declined or expired card) or demand is "
        "outrunning refills. Check the saved card and Auto Top-Up settings at "
        "https://openrouter.ai/settings/credits — managed runs return 402 once "
        "the balance hits zero."
    )
    return subject, body


def notify_low_float(status: FloatStatus, *, now: float | None = None) -> bool:
    """Fan a low-float breach out to the alert webhook and the operator email.

    Both channels degrade to a no-op when unconfigured. The email is sent on a
    daemon thread so an SMTP stall never blocks the caller (which may be the
    Stripe webhook handler). Swallows every failure.

    Args:
        status: The breaching float snapshot.
        now: Monotonic clock override for the cooldown (tests).

    Returns:
        True when the cooldown allowed a notification to be dispatched.
    """
    try:
        if not _cooldown_allows(now):
            return False
        subject, body = _format_low_float(status)
        send_alert(subject, body=body, level="WARNING")
        recipient = settings.openrouter_float_alert_email.strip()
        if recipient and email_configured():
            threading.Thread(
                target=_deliver_email,
                args=(recipient, subject, body),
                name="openrouter-float-alert-email",
                daemon=True,
            ).start()
        return True
    except Exception:
        logger.exception("OpenRouter float notification failed")
        return False


def _deliver_email(recipient: str, subject: str, body: str) -> None:
    """Send the low-float email, logging (never raising) on failure.

    Args:
        recipient: Operator address.
        subject: Message subject.
        body: Plain-text body.
    """
    try:
        send_email(recipient, subject, body)
    except Exception as exc:
        logger.warning("OpenRouter float alert email to %s failed: %s", recipient, exc)


class OpenRouterFloatSweeper:
    """Periodically read the OpenRouter float so a failed Auto Top-Up is caught.

    The post-purchase check only runs when a customer buys credits; on a quiet
    day a declined card would surface as a 402 on the next managed run. This
    loop checks every ``settings.openrouter_float_check_interval_seconds`` on
    the API pods. Leader election uses a Postgres transaction-scoped advisory
    lock so only one replica per tick spends the read; on non-PostgreSQL
    dialects (tests / SQLite) the check runs unconditionally.
    """

    def __init__(
        self,
        engine: Any,
        outstanding_credits: Callable[[], int],
        interval_seconds: float | None = None,
    ) -> None:
        """Initialize the sweeper.

        Args:
            engine: SQLAlchemy engine the advisory lock is taken on.
            outstanding_credits: Callable returning the platform-wide unspent
                credit liability (``StripeBillingService.total_outstanding_credits``);
                injected so this module never imports the service that imports it.
            interval_seconds: Override for the polling interval; defaults to
                ``settings.openrouter_float_check_interval_seconds``.
        """
        self._engine = engine
        self._outstanding_credits = outstanding_credits
        resolved = (
            interval_seconds if interval_seconds is not None else settings.openrouter_float_check_interval_seconds
        )
        self._interval_seconds = max(_MIN_SWEEP_INTERVAL_SECONDS, float(resolved))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background check loop."""
        self._thread = threading.Thread(target=self._run, name="openrouter-float-sweeper", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background check loop."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def sweep_once(self) -> FloatStatus | None:
        """Win the leader lock and run one float check.

        Returns:
            The float status, or ``None`` when a peer replica holds the lock,
            the monitor is disabled, or the check failed.
        """
        try:
            if self._engine is not None and self._engine.dialect.name == "postgresql":
                with self._engine.begin() as conn:
                    acquired = conn.execute(
                        text("SELECT pg_try_advisory_xact_lock(:k)"),
                        {"k": OPENROUTER_FLOAT_SWEEP_LOCK_KEY},
                    ).scalar()
                    if not acquired:
                        return None
                    return check_float(self._outstanding_credits())
            return check_float(self._outstanding_credits())
        except Exception:
            logger.warning("OpenRouter float sweep failed", exc_info=True)
            return None

    def _run(self) -> None:
        """Run the check loop until stopped."""
        self.sweep_once()
        while not self._stop_event.wait(self._interval_seconds):
            self.sweep_once()


def start_openrouter_float_sweeper(
    engine: Any, outstanding_credits: Callable[[], int]
) -> OpenRouterFloatSweeper | None:
    """Start the periodic float sweeper when the monitor is configured.

    Args:
        engine: SQLAlchemy engine the advisory lock is taken on.
        outstanding_credits: Callable returning the platform-wide unspent
            credit liability, forwarded to :class:`OpenRouterFloatSweeper`.

    Returns:
        The started sweeper, or ``None`` when the periodic check is disabled
        (interval ``0``), the floor is non-positive, or no master key is set —
        callers should invoke ``stop()`` on a returned sweeper during shutdown.
    """
    if (
        settings.openrouter_float_check_interval_seconds <= 0
        or settings.openrouter_balance_floor_credits <= 0
        or settings.openrouter_api_key is None
    ):
        return None
    sweeper = OpenRouterFloatSweeper(engine, outstanding_credits)
    sweeper.start()
    return sweeper
