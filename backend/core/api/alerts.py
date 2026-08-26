"""Forward operational alerts to a chat webhook.

The backend already logs the things an operator needs to know about — an
unhandled 500, a dead worker, a balance below its floor — but on a hosted
deployment those log lines scroll past unwatched. This module forwards the
important ones to an incoming chat webhook so a human actually sees them.

Two entry points:

* :class:`AlertLogHandler` — a logging handler that forwards every record at or
  above ``ALERT_MIN_LEVEL`` to the webhook, so existing ``logger.error(...)``
  call sites become alerts with no change to them.
* :func:`send_alert` — an explicit call for events worth alerting on that log
  below the handler's threshold (e.g. a ``WARNING`` balance-floor breach), which
  fires regardless of ``ALERT_MIN_LEVEL``.

Everything degrades to a no-op when ``ALERT_WEBHOOK_URL`` is unset: alerting is
opt-in, records still reach the logs, and a webhook outage never propagates into
the request path. Delivery runs on a daemon thread and swallows its own
failures, and identical alerts are throttled so an error loop can't flood the
channel.

The payload is Slack's ``{"text": …}`` shape, which Mattermost and Google Chat
also accept. Discord's webhook wants ``{"content": …}`` and needs a small
translating proxy in front.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import traceback
import urllib.request

from ..config import settings

logger = logging.getLogger(__name__)

_DELIVERY_TIMEOUT_SECONDS = 5.0
_MAX_TITLE_CHARS = 300
_MAX_BODY_CHARS = 3000

# Duplicated from observability._resolve_pod_name rather than imported:
# observability imports this module to install the handler, so importing it
# back would form a cycle. One line is cheaper than the seam.
_POD_NAME = os.environ.get("POD_NAME") or socket.gethostname()


class _Throttle:
    """Collapse identical alert payloads within a cooldown window.

    Keyed on the alert's ``(level, title, body)``; an entry older than the
    window is forgotten so the map stays bounded under a stream of distinct
    alerts. Guarded by a lock because delivery is dispatched from arbitrary
    logging threads.
    """

    def __init__(self) -> None:
        """Initialise an empty throttle map with its own lock."""
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float, window: float) -> bool:
        """Return whether an alert may be sent, recording the send when it may.

        Args:
            key: Stable identity of the alert (level, title and body).
            now: Current monotonic clock reading.
            window: Cooldown length in seconds; ``0`` disables throttling.

        Returns:
            True when no identical alert was sent within ``window`` seconds.
        """
        if window <= 0:
            return True
        with self._lock:
            self._prune(now, window)
            last = self._last_sent.get(key)
            if last is not None and now - last < window:
                return False
            self._last_sent[key] = now
            return True

    def _prune(self, now: float, window: float) -> None:
        """Drop entries whose cooldown has fully elapsed.

        Args:
            now: Current monotonic clock reading.
            window: Cooldown length in seconds.
        """
        expired = [key for key, stamp in self._last_sent.items() if now - stamp >= window]
        for key in expired:
            del self._last_sent[key]


_throttle = _Throttle()


def alerts_configured() -> bool:
    """Return whether an outbound alert webhook is configured."""
    return bool(settings.alert_webhook_url)


def _truncate(text_value: str, limit: int) -> str:
    """Return ``text_value`` shortened to ``limit`` characters with an elision marker.

    Args:
        text_value: The text to bound.
        limit: Maximum number of characters to keep.

    Returns:
        The original text when within the limit, otherwise a truncated copy
        with a trailing note of how many characters were dropped.
    """
    if len(text_value) <= limit:
        return text_value
    dropped = len(text_value) - limit
    return f"{text_value[:limit]}\n… (+{dropped} more chars)"


def _render(title: str, body: str, level: str) -> str:
    """Build the webhook message text for an alert.

    Args:
        title: One-line summary of the alert.
        body: Optional detail (e.g. a traceback); truncated when long.
        level: Severity name shown in the header.

    Returns:
        The plain-text message posted to the webhook.
    """
    env = settings.alert_environment.strip()
    prefix = f"[{env}] " if env else ""
    lines = [f"{prefix}{level.upper()}: {_truncate(title, _MAX_TITLE_CHARS)}", f"pod={_POD_NAME}"]
    detail = body.strip()
    if detail:
        lines.append(_truncate(detail, _MAX_BODY_CHARS))
    return "\n".join(lines)


def _post(url: str, text_value: str, timeout: float) -> None:
    """POST a Slack-compatible ``{"text": …}`` payload to the webhook.

    Args:
        url: The webhook URL.
        text_value: The message body.
        timeout: Socket timeout in seconds.

    Raises:
        urllib.error.URLError, OSError: On any network or protocol failure.
    """
    payload = json.dumps({"text": text_value}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def _deliver(url: str, text_value: str) -> None:
    """Deliver one alert, swallowing failures so the sender thread never raises.

    Args:
        url: The webhook URL.
        text_value: The rendered message text.
    """
    try:
        _post(url, text_value, _DELIVERY_TIMEOUT_SECONDS)
    except Exception as exc:
        # DEBUG (not WARNING) on purpose: a webhook outage must not recurse back
        # through AlertLogHandler and try to alert about failing to alert.
        logger.debug("Alert delivery to webhook failed: %s", exc)


def send_alert(
    title: str,
    *,
    body: str = "",
    level: str = "ERROR",
    now: float | None = None,
) -> threading.Thread | None:
    """Forward an alert to the configured webhook on a background thread.

    A no-op when no webhook is configured or when an identical alert was sent
    within the throttle window. Delivery is fire-and-forget: the returned thread
    is a daemon and its failures are swallowed.

    Args:
        title: One-line summary; becomes the alert header.
        body: Optional detail such as a traceback.
        level: Severity name (e.g. ``"ERROR"``, ``"WARNING"``).
        now: Monotonic clock override for the throttle; tests pass an explicit
            value, production leaves it ``None`` to read the real clock.

    Returns:
        The daemon delivery thread when an alert was dispatched, else ``None``.
    """
    url = settings.alert_webhook_url
    if not url:
        return None
    stamp = now if now is not None else time.monotonic()
    key = f"{level}\x00{title}\x00{body}"
    if not _throttle.allow(key, stamp, settings.alert_throttle_seconds):
        return None
    text_value = _render(title, body, level)
    thread = threading.Thread(target=_deliver, args=(url, text_value), name="alert-webhook", daemon=True)
    thread.start()
    return thread


class AlertLogHandler(logging.Handler):
    """Forward log records at or above the handler level to the alert webhook."""

    def emit(self, record: logging.LogRecord) -> None:
        """Forward one record via :func:`send_alert`.

        Records emitted by this module are skipped so a delivery-failure log can
        never recurse into another alert.

        Args:
            record: The log record to forward.
        """
        if record.name == __name__:
            return
        try:
            body = ""
            if record.exc_info:
                body = "".join(traceback.format_exception(*record.exc_info))
            send_alert(record.getMessage(), body=body, level=record.levelname)
        except Exception:
            self.handleError(record)


def install_alert_log_handler(root: logging.Logger | None = None) -> AlertLogHandler | None:
    """Attach an :class:`AlertLogHandler` to the root logger when configured.

    The handler level is set from ``ALERT_MIN_LEVEL``. It only sees records the
    root logger already admits, so a threshold below ``LOG_LEVEL`` is capped by
    ``LOG_LEVEL`` in practice.

    Args:
        root: Logger to attach to; defaults to the root logger.

    Returns:
        The installed handler, or ``None`` when no webhook is configured.
    """
    if not settings.alert_webhook_url:
        return None
    target = root if root is not None else logging.getLogger()
    handler = AlertLogHandler()
    handler.setLevel(getattr(logging, settings.alert_min_level, logging.ERROR))
    target.addHandler(handler)
    return handler
