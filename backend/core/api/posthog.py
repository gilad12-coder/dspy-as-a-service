"""Privacy-preserving PostHog export for accepted first-party telemetry."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import requests

from ..config import settings

logger = logging.getLogger(__name__)

_EVENT_NAMES = {
    "page_view",
    "element_click",
    "login_succeeded",
    "login_failed",
    "signup_started",
    "signup_succeeded",
    "run_submitted",
    "grid_search_submitted",
    "settings_opened",
    "settings_tab_changed",
}
_PROPERTY_KEYS = {
    "generation_models",
    "has_reflection",
    "href",
    "id",
    "label",
    "method",
    "name",
    "path",
    "react",
    "reflection_models",
    "role",
    "tab",
    "tag",
    "testid",
    "type",
}
_STRUCTURAL_STRING = re.compile(r"^[A-Za-z0-9_./:-]{1,128}$")


def _distinct_id(
    username: str | None,
    anonymous_id: str | None,
    session_id: str | None,
) -> str | None:
    """Build a stable non-PII identity for PostHog event grouping.

    Args:
        username: Server-trusted account identity, when authenticated.
        anonymous_id: Opaque browser id, when supplied by the client.
        session_id: Opaque visit id used only as a final fallback.

    Returns:
        Hashed user id or opaque client id, never the account email itself.
    """
    if username:
        digest = hashlib.sha256(username.encode("utf-8")).hexdigest()
        return f"user-{digest}"
    if anonymous_id:
        return f"anon-{anonymous_id}"
    if session_id:
        return f"session-{session_id}"
    return None


def _safe_properties(raw: Any) -> dict[str, str | int | float | bool]:
    """Keep only bounded structural properties from the browser contract.

    Args:
        raw: Untrusted event properties accepted by the public ingest endpoint.

    Returns:
        Allowlisted primitive properties that cannot contain email addresses or
        free-form user content.
    """
    if not isinstance(raw, dict):
        return {}
    safe: dict[str, str | int | float | bool] = {}
    for property_key, value in raw.items():
        if property_key not in _PROPERTY_KEYS:
            continue
        if isinstance(value, (bool, int, float)) or (isinstance(value, str) and _STRUCTURAL_STRING.fullmatch(value)):
            safe[property_key] = value
    return safe


def export_telemetry_events(
    *,
    username: str | None,
    anonymous_id: str | None,
    session_id: str | None,
    events: list[dict[str, Any]],
) -> None:
    """Send an accepted telemetry batch to PostHog without blocking ingest.

    The caller runs this as a FastAPI background task. Provider failures are
    logged and swallowed because Postgres remains the source of truth and
    analytics must never affect the product request.

    Args:
        username: Authenticated identity used only to derive a one-way hash.
        anonymous_id: Opaque browser identifier.
        session_id: Opaque visit identifier.
        events: Sanitized accepted event dictionaries.
    """
    key = settings.posthog_project_api_key
    distinct_id = _distinct_id(username, anonymous_id, session_id)
    if key is None or distinct_id is None or not events:
        return
    batch = []
    for event in events:
        if event.get("name") not in _EVENT_NAMES:
            continue
        properties = {
            **_safe_properties(event.get("properties")),
            "distinct_id": distinct_id,
            "$process_person_profile": False,
            "$geoip_disable": True,
            "$lib": "skynet-first-party",
            "$session_id": session_id,
            "$pathname": event.get("path"),
            "locale": event.get("locale"),
            "app_version": event.get("app_version"),
        }
        batch.append(
            {
                "event": event["name"],
                "properties": {property_key: value for property_key, value in properties.items() if value is not None},
                "timestamp": event.get("timestamp"),
            }
        )
    if not batch:
        return
    try:
        response = requests.post(
            f"{settings.posthog_host.rstrip('/')}/batch/",
            json={"api_key": key.get_secret_value(), "batch": batch},
            timeout=3,
        )
        response.raise_for_status()
    except requests.RequestException:
        logger.warning("PostHog telemetry export failed", exc_info=True)
