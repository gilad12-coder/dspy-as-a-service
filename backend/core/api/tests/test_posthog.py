"""Tests for privacy-preserving PostHog telemetry export."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import requests
from pydantic import SecretStr

from ...config import settings
from .. import posthog


def _events() -> list[dict[str, object]]:
    """Build a representative accepted telemetry batch."""
    return [
        {
            "name": "run_submitted",
            "timestamp": "2026-08-13T12:00:00+00:00",
            "path": "/submit",
            "locale": "en",
            "app_version": "abc123",
            "properties": {
                "react": True,
                "label": "submit-run",
                "email": "leak@example.com",
                "name": "leak@example.com",
            },
            "context": {"secret": "must-not-export"},
        }
    ]


def test_export_hashes_identity_and_filters_untrusted_properties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PostHog receives a non-person profile with no raw identity or content."""
    monkeypatch.setattr(settings, "posthog_project_api_key", SecretStr("phc_test"))
    monkeypatch.setattr(settings, "posthog_host", "https://eu.i.posthog.com")
    captured: dict[str, object] = {}

    def fake_post(url: str, *, json: dict[str, object], timeout: int) -> SimpleNamespace:
        """Capture the outbound request and mimic a successful response."""
        captured.update(url=url, json=json, timeout=timeout)
        return SimpleNamespace(raise_for_status=lambda: None)

    monkeypatch.setattr(posthog.requests, "post", fake_post)

    posthog.export_telemetry_events(
        username="alice@example.com",
        anonymous_id="anon-1",
        session_id="session-1",
        events=_events(),
    )

    payload = captured["json"]
    assert isinstance(payload, dict)
    assert payload["api_key"] == "phc_test"
    batch = payload["batch"]
    assert isinstance(batch, list)
    properties = batch[0]["properties"]
    expected_hash = hashlib.sha256(b"alice@example.com").hexdigest()
    assert properties["distinct_id"] == f"user-{expected_hash}"
    assert properties["$process_person_profile"] is False
    assert properties["react"] is True
    assert properties["label"] == "submit-run"
    serialized = repr(payload)
    assert "alice@example.com" not in serialized
    assert "leak@example.com" not in serialized
    assert "must-not-export" not in serialized


def test_export_ignores_unknown_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown public-ingest names never consume third-party analytics quota."""
    monkeypatch.setattr(settings, "posthog_project_api_key", SecretStr("phc_test"))
    called = False

    def fake_post(*args: object, **kwargs: object) -> None:
        """Record an unexpected provider request."""
        nonlocal called
        called = True

    monkeypatch.setattr(posthog.requests, "post", fake_post)

    posthog.export_telemetry_events(
        username=None,
        anonymous_id="anon-1",
        session_id=None,
        events=[{"name": "attacker_event"}],
    )

    assert called is False


def test_export_swallows_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PostHog outage cannot fail the completed product request."""
    monkeypatch.setattr(settings, "posthog_project_api_key", SecretStr("phc_test"))

    def fail_post(*args: object, **kwargs: object) -> None:
        """Simulate a provider timeout."""
        raise requests.Timeout("provider unavailable")

    monkeypatch.setattr(posthog.requests, "post", fail_post)

    posthog.export_telemetry_events(
        username=None,
        anonymous_id="anon-1",
        session_id=None,
        events=_events(),
    )
