"""Notification orchestration.

Builds the localized HTML email for each Skynet event (job lifecycle + sharing)
and hands it to the shared SMTP transport. Recipient resolution is centralized
in :func:`core.notifications.comms.resolve_email`; non-email identities are
skipped and logged, so every helper here is safe to call unconditionally from
the API and worker layers.
"""

from __future__ import annotations

import html
import logging
import os

from ..i18n import t
from ._wordmark import WORDMARK_CSS, wordmark_svg
from .comms import resolve_email, send_mail
from .preferences import NotificationCategory, notification_category_enabled

logger = logging.getLogger(__name__)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3001")

# Email palette mirrors the app design tokens (frontend globals.css) so a
# notification reads like a page of the product: warm off-white canvas, near-black
# warm ink, the brown primary, taupe muted text, and the shared 8px radius. Email
# clients can't read CSS vars, so the concrete hex values are inlined here.
_BG = "#faf8f5"  # --background
_CARD = "#ffffff"  # --card
_FG = "#1c1612"  # --foreground
_MUTED = "#8c7a6b"  # --muted-foreground
_PRIMARY = "#3d2e22"  # --primary
_PRIMARY_FG = "#faf8f5"  # --primary-foreground
_BORDER = "#ddd6cc"  # --border
# Heebo is the app's UI typeface; emails rely on the recipient's locally-available
# Heebo, falling back to Segoe UI/Arial — no external font is fetched (air-gap safe).
_FONT = "'Heebo','Heebo Variable','Segoe UI',Arial,sans-serif"
# JetBrains Mono is the app's monospace face; used for the optimization id.
_MONO = "'JetBrains Mono','JetBrains Mono Variable','SFMono-Regular',Consolas,'Courier New',monospace"
_PRIMARY_HOVER = "#473528"  # primary lightened, echoing the app's hover:bg-primary/90

# CTA hover feedback (the button keeps its resting shape): lighten on hover with a
# soft lift shadow, press in on active — matching the app's primary button. Needs
# a class since inline styles can't carry pseudo-states; Outlook strips it and
# keeps the static button.
_BUTTON_CSS = (
    ".btn{transition:background-color .15s ease,box-shadow .15s ease,transform .08s ease}"
    f".btn:hover{{background:{_PRIMARY_HOVER}!important;box-shadow:0 6px 16px rgba(61,46,34,0.20)}}"
    ".btn:active{transform:scale(0.97)}"
)
# Unicode bidi isolate (LRI .. PDI): wraps an LTR token inside RTL text so its
# parentheses/sign keep their order. Built via chr() to keep raw control
# characters out of the source (ruff PLE2502).
_LRI = chr(0x2066)
_PDI = chr(0x2069)


def _job_url(optimization_id: str) -> str:
    """Return the full URL for the optimization detail page."""
    return f"{FRONTEND_URL}/optimizations/{optimization_id}"


def _render_body(body_lines: list[str]) -> str:
    """Render body lines as centered paragraphs.

    A line containing ``": "`` renders as a centered detail line (muted label,
    ink value); a line without one renders as a centered lead sentence. The same
    input shape powers both the job (key/value) and sharing (sentence) emails.

    Args:
        body_lines: Pre-localized text lines for the message body.

    Returns:
        The inner HTML for the body paragraphs.
    """
    # Spacing lives on each block's TOP margin (heading has none) so the gap
    # above the CTA is a constant 32px whether or not a body line is present —
    # the button sits identically across every email variant.
    parts = []
    for line in body_lines:
        label, sep, value = line.partition(": ")
        if sep:
            parts.append(
                f'<p style="margin:20px 0 0;font-size:14px;line-height:1.6;text-align:center;">'
                f'<span style="color:{_MUTED};">{html.escape(label)}:</span>&nbsp;'
                f'<span style="color:{_FG};font-weight:600;">{html.escape(value)}</span></p>'
            )
        else:
            parts.append(
                f'<p style="margin:20px 0 0;font-size:16px;line-height:1.7;text-align:center;color:{_FG};">'
                f"{html.escape(line)}</p>"
            )
    return "".join(parts)


def _html_email(heading: str, body_lines: list[str], cta_label: str, cta_url: str, ref_id: str) -> str:
    """Render a full, app-themed, RTL HTML email document.

    Full-width single column: a branded header band, a heading, a body
    (key/value rows or a lead sentence), a primary CTA pill, the optimization id
    for reference, and a footer — all styled from the app's design tokens with
    inline styles for email-client robustness.

    Args:
        heading: Title shown under the brand header.
        body_lines: Pre-localized message lines (see :func:`_render_body`).
        cta_label: Visible text of the call-to-action button.
        cta_url: Destination the CTA button links to.
        ref_id: Optimization id shown (monospace) beneath the CTA.

    Returns:
        A self-contained HTML document suitable for ``mail.HTMLBody``.
    """
    safe_url = html.escape(cta_url, quote=True)
    home_url = html.escape(FRONTEND_URL, quote=True)
    return (
        "<!doctype html>"
        '<html lang="he" dir="rtl"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<style>"
        "@media (max-width:600px){.px{padding-left:24px!important;padding-right:24px!important}}"
        f"{WORDMARK_CSS}{_BUTTON_CSS}"
        "</style>"
        f"<title>{html.escape(heading)}</title></head>"
        f'<body style="margin:0;padding:0;background:{_BG};">'
        f'<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" '
        f'style="background:{_BG};font-family:{_FONT};border-collapse:collapse;">'
        f'<tr><td class="px" align="center" style="padding:26px 44px;background:{_CARD};'
        f'border-bottom:1px solid {_BORDER};text-align:center;">'
        f'<a class="wm" href="{home_url}" aria-label="SKYNET" '
        'style="text-decoration:none;display:inline-block;">'
        '<!--[if mso]><span style="font-family:Arial,sans-serif;font-size:22px;font-weight:800;'
        f'letter-spacing:-1px;color:{_FG};">SKYNET</span><![endif]-->'
        f"<!--[if !mso]><!-->{wordmark_svg()}<!--<![endif]-->"
        "</a></td></tr>"
        f'<tr><td class="px" align="center" style="padding:40px 44px;background:{_CARD};text-align:center;">'
        f'<h1 style="margin:0;font-size:23px;font-weight:700;line-height:1.4;'
        f'letter-spacing:-0.02em;color:{_FG};text-align:center;">{html.escape(heading)}</h1>'
        f"{_render_body(body_lines)}"
        '<table role="presentation" align="center" cellpadding="0" cellspacing="0" border="0" '
        'style="margin:32px auto 0;">'
        f'<tr><td align="center" style="border-radius:8px;background:{_PRIMARY};">'
        f'<a class="btn" href="{safe_url}" style="display:inline-block;padding:13px 32px;font-size:15px;'
        f'font-weight:600;color:{_PRIMARY_FG};text-decoration:none;border-radius:8px;font-family:{_FONT};">'
        f"{html.escape(cta_label)}</a></td></tr></table>"
        f'<p style="margin:20px 0 0;font-size:12px;color:{_MUTED};text-align:center;font-family:{_MONO};">'
        f"{html.escape(t('notifier.email.ref'))}: {html.escape(ref_id)}</p>"
        "</td></tr>"
        f'<tr><td class="px" align="center" style="padding:22px 44px;color:{_MUTED};font-size:12px;'
        f'line-height:1.6;text-align:center;">{html.escape(t("notifier.email.footer"))}</td></tr>'
        "</table></body></html>"
    )


def _deliver(
    recipient_username: str | None,
    subject: str,
    html_body: str,
    category: NotificationCategory,
) -> None:
    """Resolve the recipient's address and send, skipping (logged) when unknown.

    Args:
        recipient_username: Skynet account to notify.
        subject: Mail subject line.
        html_body: Rendered HTML body.
        category: Optional product-email category checked against preferences.
    """
    if not recipient_username:
        return
    if not notification_category_enabled(recipient_username, category):
        logger.info("Account notification preference disabled category %s; skipping", category)
        return
    address = resolve_email(recipient_username)
    if address is None:
        logger.info("Account %r has no email identity; skipping notification %r", recipient_username, subject)
        return
    send_mail(address, subject, html_body)


def notify_job_started(
    optimization_id: str,
    username: str,
    optimization_type: str,
    optimizer_name: str,
    module_name: str,
    model_name: str | None = None,
) -> None:
    """Email the owner when their job is submitted.

    The submission email is intentionally minimal — the headline plus a link to
    the run — so the optimization-metadata params are accepted (callers pass
    them) but not rendered.

    Args:
        optimization_id: Job identifier used for the detail link and id line.
        username: Owner who submitted the job (and the email recipient).
        optimization_type: Submission metadata; accepted but not rendered.
        optimizer_name: Submission metadata; accepted but not rendered.
        module_name: Submission metadata; accepted but not rendered.
        model_name: Submission metadata; accepted but not rendered.
    """
    subject = t("notifier.title.new")
    html_body = _html_email(subject, [], t("notifier.link.follow"), _job_url(optimization_id), optimization_id)
    _deliver(username, subject, html_body, "job_updates")


def notify_job_completed(
    optimization_id: str,
    username: str,
    status: str,
    message: str | None = None,
    baseline_score: float | None = None,
    optimized_score: float | None = None,
) -> None:
    """Email the owner when their job finishes (success, cancelled, or failed).

    Args:
        optimization_id: Job identifier used to render the detail link.
        username: Owner who submitted the job (and the email recipient).
        status: One of ``"success"``, ``"cancelled"`` or ``"failed"``;
            unknown statuses are logged and skipped.
        message: Optional error/context message; rendered (truncated to 150
            chars) only for the failed branch.
        baseline_score: Pre-optimization score; combined with
            ``optimized_score`` to render an improvement line on success.
        optimized_score: Post-optimization score; rendered with the improvement
            delta when both scores are present.
    """
    # The owner is the recipient, so a "user: <owner>" line is redundant.
    lines: list[str] = []

    if status == "success":
        subject = t("notifier.title.completed")
        if baseline_score is not None and optimized_score is not None:
            improvement = optimized_score - baseline_score
            sign = "+" if improvement >= 0 else ""
            # Arrow points left so baseline->optimized reads correctly in RTL; the
            # signed delta is wrapped in a bidi LTR isolate so its parentheses and
            # sign don't reorder to "(22.6%+)".
            lines.append(
                f"{t('notifier.label.score')}: "
                f"{baseline_score:.1f}% ← {optimized_score:.1f}% "
                f"{_LRI}({sign}{improvement:.1f}%){_PDI}"
            )
        cta_label = t("notifier.link.results")
    elif status == "cancelled":
        subject = t("notifier.title.cancelled")
        cta_label = t("notifier.link.details")
    elif status == "failed":
        subject = t("notifier.title.failed")
        truncated = f"{message[:150]}..." if message and len(message) > 150 else message
        if truncated:
            lines.append(f"{t('notifier.label.error')}: {truncated}")
        cta_label = t("notifier.link.details")
    else:
        logger.warning("Skipping notification for unknown job status: %s", status)
        return

    html_body = _html_email(subject, lines, cta_label, _job_url(optimization_id), optimization_id)
    _deliver(username, subject, html_body, "job_updates")


def notify_share_invite(optimization_id: str, grantee: str, inviter: str, role: str) -> None:
    """Email a user when they are explicitly invited to an optimization.

    Args:
        optimization_id: Optimization that was shared.
        grantee: Invited username (the email recipient).
        inviter: Username of the owner/editor who issued the invite.
        role: Tier granted (``"viewer"`` or ``"editor"``).
    """
    role_label = t(f"notifier.share.role.{role}")
    subject = t("notifier.share.invite.subject")
    body = t("notifier.share.invite.body", inviter=inviter, role=role_label)
    html_body = _html_email(subject, [body], t("notifier.link.open"), _job_url(optimization_id), optimization_id)
    _deliver(grantee, subject, html_body, "sharing_updates")


def notify_role_change(optimization_id: str, grantee: str, actor: str, role: str) -> None:
    """Email a member when their access tier on an optimization changes.

    Args:
        optimization_id: Optimization whose grant changed.
        grantee: Affected username (the email recipient).
        actor: Username of the owner/editor who changed the role.
        role: New tier (``"viewer"`` or ``"editor"``).
    """
    role_label = t(f"notifier.share.role.{role}")
    subject = t("notifier.share.role_change.subject")
    body = t("notifier.share.role_change.body", actor=actor, role=role_label)
    html_body = _html_email(subject, [body], t("notifier.link.open"), _job_url(optimization_id), optimization_id)
    _deliver(grantee, subject, html_body, "sharing_updates")


def notify_ownership_transfer(optimization_id: str, new_owner: str, actor: str) -> None:
    """Email the new owner when an optimization's ownership is transferred to them.

    Args:
        optimization_id: Optimization whose ownership moved.
        new_owner: Username receiving ownership (the email recipient).
        actor: Username of the previous owner/admin who transferred it.
    """
    subject = t("notifier.share.transfer.subject")
    body = t("notifier.share.transfer.body", actor=actor)
    html_body = _html_email(subject, [body], t("notifier.link.open"), _job_url(optimization_id), optimization_id)
    _deliver(new_owner, subject, html_body, "sharing_updates")
