"""Tests for ``core.notifications.notifier`` HTML email orchestration."""

from __future__ import annotations

import pytest

import core.notifications.notifier as notifier_module
from core.i18n import t
from core.notifications.notifier import (
    notify_job_completed,
    notify_job_started,
    notify_ownership_transfer,
    notify_role_change,
    notify_share_invite,
)

from .conftest import FakeMail


def _patch_delivery(monkeypatch, fake_mail, resolver=lambda username: username):
    """Route the notifier through ``fake_mail`` with a stub email resolver.

    Args:
        monkeypatch: pytest's ``MonkeyPatch`` fixture.
        fake_mail: The ``FakeMail`` recording deliveries.
        resolver: ``resolve_email`` stand-in; defaults to the identity so the
            recorded ``to`` equals the username. Return ``None`` to simulate
            a legacy non-email identity.
    """
    monkeypatch.setattr(notifier_module, "send_mail", fake_mail.send_mail)
    monkeypatch.setattr(notifier_module, "resolve_email", resolver)


def test_job_started_delivers_once_to_owner(monkeypatch: pytest.MonkeyPatch, fake_mail: FakeMail) -> None:
    """A submitted job emails its owner exactly once."""
    _patch_delivery(monkeypatch, fake_mail)

    notify_job_started(
        optimization_id="abc123",
        username="alice",
        optimization_type="run",
        optimizer_name="GEPA",
        module_name="MyModule",
    )

    assert fake_mail.call_count == 1
    assert fake_mail.last()["to"] == "alice"


def test_job_started_is_minimal_title_and_link(monkeypatch: pytest.MonkeyPatch, fake_mail: FakeMail) -> None:
    """The submission email shows the title + run link, with no config metadata."""
    _patch_delivery(monkeypatch, fake_mail)

    notify_job_started(
        optimization_id="abc123",
        username="alice",
        optimization_type="run",
        optimizer_name="GEPA",
        module_name="MyModule",
        model_name="gpt-4o",
    )

    call = fake_mail.last()
    assert call["subject"] == t("notifier.title.new")
    assert "abc123" in call["html"]  # run link + id reference
    # Optimization metadata is intentionally not rendered.
    assert "GEPA" not in call["html"]
    assert "MyModule" not in call["html"]
    assert "gpt-4o" not in call["html"]
    assert t("notifier.label.type") not in call["html"]


def test_job_started_skips_when_email_unresolved(monkeypatch: pytest.MonkeyPatch, fake_mail: FakeMail) -> None:
    """A legacy non-email account identity is skipped."""
    _patch_delivery(monkeypatch, fake_mail, resolver=lambda username: None)

    notify_job_started(
        optimization_id="abc123",
        username="alice",
        optimization_type="run",
        optimizer_name="GEPA",
        module_name="MyModule",
    )

    assert fake_mail.call_count == 0


def test_job_completed_success_renders_score_line(monkeypatch: pytest.MonkeyPatch, fake_mail: FakeMail) -> None:
    """A successful job emails the owner with the score improvement line."""
    _patch_delivery(monkeypatch, fake_mail)

    notify_job_completed(
        optimization_id="abc123",
        username="alice",
        status="success",
        baseline_score=40.0,
        optimized_score=55.0,
    )

    call = fake_mail.last()
    assert call["subject"] == t("notifier.title.completed")
    assert "40.0%" in call["html"]
    assert "55.0%" in call["html"]
    assert "+15.0%" in call["html"]


def test_job_completed_cancelled_subject(monkeypatch: pytest.MonkeyPatch, fake_mail: FakeMail) -> None:
    """A cancelled job uses the cancelled subject."""
    _patch_delivery(monkeypatch, fake_mail)

    notify_job_completed(optimization_id="abc123", username="alice", status="cancelled")

    assert fake_mail.last()["subject"] == t("notifier.title.cancelled")


def test_job_completed_failed_includes_error(monkeypatch: pytest.MonkeyPatch, fake_mail: FakeMail) -> None:
    """A failed job uses the failed subject and renders the error message."""
    _patch_delivery(monkeypatch, fake_mail)

    notify_job_completed(
        optimization_id="abc123",
        username="alice",
        status="failed",
        message="boom happened",
    )

    call = fake_mail.last()
    assert call["subject"] == t("notifier.title.failed")
    assert "boom happened" in call["html"]


def test_job_completed_unknown_status_skips(
    monkeypatch: pytest.MonkeyPatch,
    fake_mail: FakeMail,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unknown status logs a warning and sends nothing."""
    _patch_delivery(monkeypatch, fake_mail)

    with caplog.at_level("WARNING", logger="core.notifications.notifier"):
        notify_job_completed(optimization_id="abc123", username="alice", status="weird")

    assert fake_mail.call_count == 0
    assert "Skipping notification for unknown job status: weird" in caplog.text


def test_share_invite_emails_grantee_with_role_and_link(monkeypatch: pytest.MonkeyPatch, fake_mail: FakeMail) -> None:
    """An explicit invite emails the grantee with the inviter, tier, and link."""
    _patch_delivery(monkeypatch, fake_mail)

    notify_share_invite(optimization_id="abc123", grantee="bob", inviter="alice", role="editor")

    call = fake_mail.last()
    assert call["to"] == "bob"
    assert call["subject"] == t("notifier.share.invite.subject")
    assert "alice" in call["html"]
    assert t("notifier.share.role.editor") in call["html"]
    assert "abc123" in call["html"]


def test_share_invite_skips_when_email_unresolved(monkeypatch: pytest.MonkeyPatch, fake_mail: FakeMail) -> None:
    """A legacy non-email account identity receives no invite email."""
    _patch_delivery(monkeypatch, fake_mail, resolver=lambda username: None)

    notify_share_invite(optimization_id="abc123", grantee="bob", inviter="alice", role="viewer")

    assert fake_mail.call_count == 0


def test_role_change_emails_member(monkeypatch: pytest.MonkeyPatch, fake_mail: FakeMail) -> None:
    """A role change emails the member with the actor and the new tier."""
    _patch_delivery(monkeypatch, fake_mail)

    notify_role_change(optimization_id="abc123", grantee="bob", actor="alice", role="viewer")

    call = fake_mail.last()
    assert call["to"] == "bob"
    assert call["subject"] == t("notifier.share.role_change.subject")
    assert "alice" in call["html"]
    assert t("notifier.share.role.viewer") in call["html"]


def test_ownership_transfer_emails_new_owner(monkeypatch: pytest.MonkeyPatch, fake_mail: FakeMail) -> None:
    """A transfer emails the new owner naming the previous owner."""
    _patch_delivery(monkeypatch, fake_mail)

    notify_ownership_transfer(optimization_id="abc123", new_owner="bob", actor="alice")

    call = fake_mail.last()
    assert call["to"] == "bob"
    assert call["subject"] == t("notifier.share.transfer.subject")
    assert "alice" in call["html"]
