"""Account security: two-factor enrollment and WebAuthn passkeys. [INTERNAL]

Two audiences share this router. The ``/auth/security/*`` routes are normal
bearer-authenticated endpoints — the signed-in user manages their own TOTP,
emailed-code, and passkey enrollment from the settings modal, with identity
taken from the session token (never the request body). The ``/auth/2fa/*``
and ``/auth/webauthn/options|verify`` routes run *before* a session exists,
so like ``/auth/register|login`` they are gated by the shared
``BACKEND_AUTH_SECRET`` and only callable by the trusted frontend server.

Passkeys are open to every identity (a Google/GitHub sign-in has no ``users``
row but can still register one — the email is the cross-provider identity),
while TOTP/email 2FA only guards password sign-ins and therefore requires a
local account.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, Header
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from webauthn import (
    base64url_to_bytes,
    generate_authentication_options,
    generate_registration_options,
    options_to_json,
    verify_authentication_response,
    verify_registration_response,
)
from webauthn.helpers import bytes_to_base64url
from webauthn.helpers.exceptions import WebAuthnException
from webauthn.helpers.structs import (
    AuthenticatorSelectionCriteria,
    PublicKeyCredentialDescriptor,
    ResidentKeyRequirement,
    UserVerificationRequirement,
)

from ...config import settings
from ...storage.models import UserModel, WebAuthnChallengeModel, WebAuthnCredentialModel
from ..auth import AuthenticatedUser, get_authenticated_user
from ..email_sender import email_configured, send_email
from ..errors import DomainError
from ..monthly_active_users import enforce_monthly_active_user_limit
from ..passwords import verify_password
from ..two_factor import (
    consume_recovery_code,
    generate_recovery_codes,
    generate_totp_secret,
    hash_recovery_codes,
    issue_email_code,
    totp_provisioning_uri,
    verify_totp,
)
from .accounts import AccountInfo, OkResponse, _normalise_email, _require_internal_auth, _role_for

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]

_RP_NAME = "Skynet"
_CHALLENGE_TTL = timedelta(minutes=5)
_NICKNAME_MAX = 64
_DEV_ORIGINS = ("http://localhost:3000", "http://localhost:3002")


def _rp_id() -> str:
    """Resolve the WebAuthn relying-party id for this deployment.

    Returns:
        ``WEBAUTHN_RP_ID`` when set, else the hostname of ``APP_PUBLIC_URL``,
        else ``localhost``.
    """
    if settings.webauthn_rp_id:
        return settings.webauthn_rp_id
    return urlsplit(settings.app_public_url).hostname or "localhost"


def _expected_origins() -> list[str]:
    """Resolve the browser origins accepted in passkey ceremonies.

    Returns:
        ``WEBAUTHN_ORIGINS`` (comma-separated) when set, else the public app
        origin plus the localhost dev origins.
    """
    if settings.webauthn_origins.strip():
        return [o.strip().rstrip("/") for o in settings.webauthn_origins.split(",") if o.strip()]
    origins = [settings.app_public_url.rstrip("/"), *_DEV_ORIGINS]
    return list(dict.fromkeys(origins))


def _store_challenge(session: Session, challenge: bytes, purpose: str, user_email: str | None) -> None:
    """Persist a single-use ceremony challenge, purging expired rows.

    Args:
        session: Open ORM session (caller commits).
        challenge: Raw challenge bytes from the generated options.
        purpose: ``"register"`` or ``"auth"``.
        user_email: Owner for registration challenges; None for sign-in.
    """
    now = datetime.now(UTC)
    session.query(WebAuthnChallengeModel).filter(WebAuthnChallengeModel.expires_at < now).delete()
    session.add(
        WebAuthnChallengeModel(
            challenge=bytes_to_base64url(challenge),
            purpose=purpose,
            user_email=user_email,
            expires_at=now + _CHALLENGE_TTL,
        )
    )


def _client_challenge(credential: dict[str, Any]) -> str:
    """Extract the base64url challenge echoed in a credential's clientDataJSON.

    Args:
        credential: The JSON credential from ``navigator.credentials``.

    Returns:
        The challenge string the browser signed over.

    Raises:
        DomainError: 422 when the credential payload is malformed.
    """
    try:
        client_data = json.loads(base64url_to_bytes(credential["response"]["clientDataJSON"]))
        challenge = client_data["challenge"]
    except (KeyError, TypeError, ValueError):
        raise DomainError("webauthn.invalid_credential", status=422) from None
    if not isinstance(challenge, str) or not challenge:
        raise DomainError("webauthn.invalid_credential", status=422)
    return challenge


def _consume_challenge(
    session: Session, credential: dict[str, Any], purpose: str
) -> tuple[bytes, str | None]:
    """Look up, validate, and burn the server-issued challenge for a ceremony.

    The row is deleted (and committed) before signature verification runs, so
    a challenge can never be replayed even when verification then fails.

    Args:
        session: Open ORM session.
        credential: The JSON credential from ``navigator.credentials``.
        purpose: Expected ceremony purpose (``"register"`` / ``"auth"``).

    Returns:
        The raw challenge bytes and the email the challenge was issued to
        (None for sign-in challenges).

    Raises:
        DomainError: 401 ``webauthn.challenge_expired`` when the challenge is
            unknown, already used, past its TTL, or for another purpose.
    """
    encoded = _client_challenge(credential)
    row = session.get(WebAuthnChallengeModel, encoded)
    if row is None or str(row.purpose) != purpose:
        raise DomainError("webauthn.challenge_expired", status=401)
    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    user_email = row.user_email
    session.delete(row)
    session.commit()
    if expires_at < datetime.now(UTC):
        raise DomainError("webauthn.challenge_expired", status=401)
    return base64url_to_bytes(encoded), str(user_email) if user_email is not None else None


def _credential_key(credential: dict[str, Any]) -> str:
    """Canonicalize a client credential's id to the stored base64url form.

    Args:
        credential: The JSON credential from ``navigator.credentials``.

    Returns:
        The unpadded-base64url credential id used as the storage key.

    Raises:
        DomainError: 422 when the payload carries no decodable id.
    """
    raw = credential.get("rawId") or credential.get("id")
    if not isinstance(raw, str) or not raw:
        raise DomainError("webauthn.invalid_credential", status=422)
    try:
        return bytes_to_base64url(base64url_to_bytes(raw))
    except ValueError:
        raise DomainError("webauthn.invalid_credential", status=422) from None


def _passkey_info(row: WebAuthnCredentialModel) -> PasskeyInfo:
    """Map a credential row to its API shape.

    Args:
        row: A stored passkey row.

    Returns:
        The client-facing passkey summary (never the public key).
    """
    return PasskeyInfo(
        credential_id=str(row.credential_id),
        nickname=str(row.nickname),
        created_at=row.created_at,
        last_used_at=row.last_used_at,
    )


# A registered passkey as shown in the settings list — metadata only.
class PasskeyInfo(BaseModel):
    credential_id: str = Field(description="Base64url credential id (the delete handle).")
    nickname: str = Field(description="User-chosen label for this passkey.")
    created_at: datetime = Field(description="When the passkey was registered.")
    last_used_at: datetime | None = Field(description="Last successful sign-in with it, if any.")


# The signed-in user's full security posture, driving the settings tab.
class SecurityStatus(BaseModel):
    has_password: bool = Field(description="Whether a local email/password account exists (2FA applies only to those).")
    totp_enabled: bool = Field(description="Whether an authenticator app is enrolled.")
    email_2fa_enabled: bool = Field(description="Whether emailed sign-in codes are enabled.")
    email_2fa_available: bool = Field(description="Whether this deployment can deliver email at all.")
    passkeys: list[PasskeyInfo] = Field(description="Registered passkeys, oldest first.")


# Fresh TOTP enrollment material for the QR step.
class TotpSetupResponse(BaseModel):
    secret: str = Field(description="Base32 secret for manual entry.")
    otpauth_url: str = Field(description="otpauth:// URI to render as a QR code.")


# First authenticator code, proving the app was enrolled.
class TotpCodeRequest(BaseModel):
    code: str = Field(description="6-digit authenticator code (or, for disable, a recovery code).")


# One-time reveal of the recovery codes minted at enablement.
class TotpEnableResponse(BaseModel):
    recovery_codes: list[str] = Field(description="Single-use recovery codes; shown exactly once.")


# Toggle for emailed sign-in codes.
class EmailCodesRequest(BaseModel):
    enabled: bool = Field(description="Desired state of email-code 2FA.")


# Browser-produced WebAuthn credential plus its display label.
class PasskeyRegisterRequest(BaseModel):
    credential: dict[str, Any] = Field(description="JSON result of navigator.credentials.create().")
    nickname: str = Field(default="", description="Label for the settings list; defaults to 'Passkey'.")


class PasskeyRenameRequest(BaseModel):
    nickname: str = Field(description="New label for the settings list.")


# Browser-produced WebAuthn assertion for passkey sign-in.
class PasskeyAssertRequest(BaseModel):
    credential: dict[str, Any] = Field(description="JSON result of navigator.credentials.get().")


# Request to email a one-time sign-in code mid-login (password re-proved).
class EmailCodeSendRequest(BaseModel):
    email: str = Field(description="Account email.")
    password: str = Field(description="Account password, re-verified before any email is sent.")


def create_account_security_router(*, job_store) -> APIRouter:
    """Build the account-security router (2FA + passkeys).

    Args:
        job_store: Job-store instance whose ORM engine backs the routes.

    Returns:
        A FastAPI ``APIRouter`` with the bearer-authenticated enrollment
        routes and the internal-secret ceremony routes.
    """
    router = APIRouter()

    def _local_account(session: Session, email: str) -> UserModel:
        """Fetch the local account row 2FA settings hang off, or 422.

        Args:
            session: Open ORM session.
            email: Normalized account email.

        Returns:
            The ``users`` row.

        Raises:
            DomainError: 422 when the identity has no local account (OAuth
                sign-ins delegate 2FA to their provider).
        """
        row = session.get(UserModel, email)
        if row is None:
            raise DomainError("accounts.two_factor_unavailable", status=422)
        return row

    @router.get(
        "/auth/security",
        response_model=SecurityStatus,
        summary="The signed-in user's 2FA and passkey posture",
    )
    def security_status(user: AuthenticatedUserDep) -> SecurityStatus:
        """Report 2FA enrollment state and registered passkeys.

        Args:
            user: The bearer-authenticated caller.

        Returns:
            The caller's security posture.
        """
        email = _normalise_email(user.username)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, email)
            passkeys = session.scalars(
                select(WebAuthnCredentialModel)
                .where(WebAuthnCredentialModel.user_email == email)
                .order_by(WebAuthnCredentialModel.created_at)
            ).all()
            return SecurityStatus(
                has_password=row is not None,
                totp_enabled=bool(row is not None and row.totp_secret),
                email_2fa_enabled=bool(row is not None and row.email_2fa_enabled),
                email_2fa_available=email_configured(),
                passkeys=[_passkey_info(p) for p in passkeys],
            )

    @router.post(
        "/auth/security/totp/setup",
        response_model=TotpSetupResponse,
        summary="Begin authenticator-app enrollment",
    )
    def totp_setup(user: AuthenticatedUserDep) -> TotpSetupResponse:
        """Mint a pending TOTP secret and its provisioning URI.

        Args:
            user: The bearer-authenticated caller.

        Returns:
            The secret (manual entry) and otpauth URI (QR code).

        Raises:
            DomainError: 422 when the identity has no local account.
        """
        email = _normalise_email(user.username)
        secret = generate_totp_secret()
        with Session(job_store.engine) as session:
            row = _local_account(session, email)
            row.totp_pending_secret = secret
            session.commit()
        return TotpSetupResponse(secret=secret, otpauth_url=totp_provisioning_uri(secret, email))

    @router.post(
        "/auth/security/totp/enable",
        response_model=TotpEnableResponse,
        summary="Verify the first authenticator code and enable TOTP",
    )
    def totp_enable(body: TotpCodeRequest, user: AuthenticatedUserDep) -> TotpEnableResponse:
        """Promote the pending secret after a correct first code.

        Args:
            body: The first 6-digit code from the authenticator app.
            user: The bearer-authenticated caller.

        Returns:
            The freshly minted recovery codes (shown exactly once).

        Raises:
            DomainError: 422 without a local account or a pending setup; 401
                when the code doesn't verify.
        """
        email = _normalise_email(user.username)
        with Session(job_store.engine) as session:
            row = _local_account(session, email)
            pending = row.totp_pending_secret
            if not pending:
                raise DomainError("accounts.totp_setup_required", status=422)
            if not verify_totp(str(pending), body.code):
                raise DomainError("accounts.invalid_second_factor", status=401)
            codes = generate_recovery_codes()
            row.totp_secret = pending
            row.totp_pending_secret = None
            row.recovery_codes = hash_recovery_codes(codes)
            session.commit()
        return TotpEnableResponse(recovery_codes=codes)

    @router.post(
        "/auth/security/totp/disable",
        response_model=OkResponse,
        summary="Disable TOTP after re-proving a current code",
    )
    def totp_disable(body: TotpCodeRequest, user: AuthenticatedUserDep) -> OkResponse:
        """Turn TOTP off; accepts a live code or a recovery code.

        Args:
            body: A current authenticator code or an unused recovery code.
            user: The bearer-authenticated caller.

        Returns:
            Acknowledgement.

        Raises:
            DomainError: 422 without a local account or with TOTP off; 401
                when the code doesn't verify.
        """
        email = _normalise_email(user.username)
        with Session(job_store.engine) as session:
            row = _local_account(session, email)
            secret = row.totp_secret
            if not secret:
                raise DomainError("accounts.totp_setup_required", status=422)
            if not verify_totp(str(secret), body.code) and not consume_recovery_code(
                row, body.code
            ):
                raise DomainError("accounts.invalid_second_factor", status=401)
            row.totp_secret = None
            row.totp_pending_secret = None
            row.recovery_codes = None
            session.commit()
        return OkResponse()

    @router.put(
        "/auth/security/email-codes",
        response_model=OkResponse,
        summary="Toggle emailed one-time sign-in codes",
    )
    def email_codes_toggle(body: EmailCodesRequest, user: AuthenticatedUserDep) -> OkResponse:
        """Enable or disable email-code 2FA for the caller's local account.

        Args:
            body: The desired state.
            user: The bearer-authenticated caller.

        Returns:
            Acknowledgement.

        Raises:
            DomainError: 422 without a local account; 422
                ``accounts.email_delivery_unavailable`` when enabling on a
                deployment with no SMTP relay.
        """
        email = _normalise_email(user.username)
        with Session(job_store.engine) as session:
            row = _local_account(session, email)
            if body.enabled and not email_configured():
                raise DomainError("accounts.email_delivery_unavailable", status=422)
            row.email_2fa_enabled = body.enabled
            session.commit()
        return OkResponse()

    @router.post(
        "/auth/security/passkeys/options",
        summary="Begin passkey registration (creation options)",
    )
    def passkey_register_options(user: AuthenticatedUserDep) -> dict[str, Any]:
        """Issue WebAuthn creation options for the signed-in identity.

        Args:
            user: The bearer-authenticated caller.

        Returns:
            The ``PublicKeyCredentialCreationOptions`` JSON for the browser.
        """
        email = _normalise_email(user.username)
        with Session(job_store.engine) as session:
            existing = session.scalars(
                select(WebAuthnCredentialModel).where(
                    WebAuthnCredentialModel.user_email == email
                )
            ).all()
            options = generate_registration_options(
                rp_id=_rp_id(),
                rp_name=_RP_NAME,
                user_id=email.encode("utf-8"),
                user_name=email,
                authenticator_selection=AuthenticatorSelectionCriteria(
                    resident_key=ResidentKeyRequirement.REQUIRED,
                    user_verification=UserVerificationRequirement.PREFERRED,
                ),
                exclude_credentials=[
                    PublicKeyCredentialDescriptor(id=base64url_to_bytes(str(c.credential_id)))
                    for c in existing
                ],
            )
            _store_challenge(session, options.challenge, "register", email)
            session.commit()
        return json.loads(options_to_json(options))

    @router.post(
        "/auth/security/passkeys",
        response_model=PasskeyInfo,
        status_code=201,
        summary="Finish passkey registration (verify + store)",
    )
    def passkey_register_verify(
        body: PasskeyRegisterRequest, user: AuthenticatedUserDep
    ) -> PasskeyInfo:
        """Verify the browser's attestation and store the new passkey.

        Args:
            body: The credential produced by ``navigator.credentials.create()``
                plus an optional nickname.
            user: The bearer-authenticated caller.

        Returns:
            The stored passkey's summary.

        Raises:
            DomainError: 401 on an expired/foreign challenge; 422 when the
                attestation doesn't verify or the credential already exists.
        """
        email = _normalise_email(user.username)
        with Session(job_store.engine) as session:
            challenge, challenge_email = _consume_challenge(session, body.credential, "register")
            if challenge_email != email:
                raise DomainError("webauthn.challenge_expired", status=401)
            try:
                verification = verify_registration_response(
                    credential=body.credential,
                    expected_challenge=challenge,
                    expected_rp_id=_rp_id(),
                    expected_origin=_expected_origins(),
                )
            except (WebAuthnException, ValueError, TypeError, KeyError) as exc:
                raise DomainError("webauthn.invalid_credential", status=422) from exc
            credential_id = bytes_to_base64url(verification.credential_id)
            if session.get(WebAuthnCredentialModel, credential_id) is not None:
                raise DomainError("webauthn.invalid_credential", status=422)
            transports = body.credential.get("response", {}).get("transports")
            row = WebAuthnCredentialModel(
                credential_id=credential_id,
                user_email=email,
                public_key=bytes_to_base64url(verification.credential_public_key),
                sign_count=verification.sign_count,
                transports=",".join(transports) if isinstance(transports, list) else None,
                nickname=(body.nickname.strip() or "Passkey")[:_NICKNAME_MAX],
            )
            session.add(row)
            session.commit()
            return _passkey_info(row)

    @router.patch(
        "/auth/security/passkeys/{credential_id}",
        response_model=PasskeyInfo,
        summary="Rename one of the caller's passkeys",
    )
    def passkey_rename(
        credential_id: str,
        body: PasskeyRenameRequest,
        user: AuthenticatedUserDep,
    ) -> PasskeyInfo:
        """Rename a passkey the caller owns.

        Args:
            credential_id: Base64url id from the passkey list.
            body: Replacement display label.
            user: The bearer-authenticated caller.

        Returns:
            The updated client-facing passkey summary.

        Raises:
            DomainError: 404 when no such passkey belongs to the caller.
        """
        email = _normalise_email(user.username)
        with Session(job_store.engine) as session:
            row = session.get(WebAuthnCredentialModel, credential_id)
            if row is None or str(row.user_email) != email:
                raise DomainError("webauthn.not_found", status=404)
            row.nickname = (body.nickname.strip() or "Passkey")[:_NICKNAME_MAX]
            session.commit()
            return _passkey_info(row)

    @router.delete(
        "/auth/security/passkeys/{credential_id}",
        response_model=OkResponse,
        summary="Remove one of the caller's passkeys",
    )
    def passkey_delete(credential_id: str, user: AuthenticatedUserDep) -> OkResponse:
        """Delete a passkey the caller owns.

        Args:
            credential_id: Base64url id from the passkey list.
            user: The bearer-authenticated caller.

        Returns:
            Acknowledgement.

        Raises:
            DomainError: 404 when no such passkey belongs to the caller.
        """
        email = _normalise_email(user.username)
        with Session(job_store.engine) as session:
            row = session.get(WebAuthnCredentialModel, credential_id)
            if row is None or str(row.user_email) != email:
                raise DomainError("webauthn.not_found", status=404)
            session.delete(row)
            session.commit()
        return OkResponse()

    @router.post(
        "/auth/2fa/email/send",
        response_model=OkResponse,
        summary="Email a one-time sign-in code (internal)",
    )
    def email_code_send(
        body: EmailCodeSendRequest,
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> OkResponse:
        """Deliver an emailed second-factor code mid-login.

        Password re-verification keeps this from becoming a mail-bomb oracle:
        only a caller who already holds valid credentials can trigger a send.

        Args:
            body: Account email and password.
            x_internal_auth: Shared-secret header proving the trusted frontend.

        Returns:
            Acknowledgement.

        Raises:
            DomainError: 403 on a bad internal secret; 401 on bad credentials;
                422 when email 2FA is off for the account or undeliverable;
                502 when the SMTP relay rejects the send.
        """
        _require_internal_auth(x_internal_auth)
        email = _normalise_email(body.email)
        with Session(job_store.engine) as session:
            row = session.get(UserModel, email)
            if row is None or not verify_password(body.password, str(row.password_hash)):
                raise DomainError("accounts.invalid_credentials", status=401)
            if not bool(row.email_2fa_enabled):
                raise DomainError("accounts.two_factor_unavailable", status=422)
            if not email_configured():
                raise DomainError("accounts.email_delivery_unavailable", status=422)
            code = issue_email_code(session, email)
            session.commit()
        try:
            send_email(
                email,
                "Your Skynet sign-in code",
                f"Your Skynet sign-in code is {code}. It expires in 10 minutes.\n\n"
                "If you didn't try to sign in, you can ignore this email.",
            )
        except (OSError, RuntimeError) as exc:
            raise DomainError("accounts.email_send_failed", status=502) from exc
        return OkResponse()

    @router.post(
        "/auth/webauthn/options",
        summary="Begin passkey sign-in (request options, internal)",
    )
    def passkey_auth_options(
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        """Issue discoverable-credential authentication options.

        Args:
            x_internal_auth: Shared-secret header proving the trusted frontend.

        Returns:
            The ``PublicKeyCredentialRequestOptions`` JSON for the browser.

        Raises:
            DomainError: 403 on a bad internal secret.
        """
        _require_internal_auth(x_internal_auth)
        options = generate_authentication_options(
            rp_id=_rp_id(),
            user_verification=UserVerificationRequirement.PREFERRED,
        )
        with Session(job_store.engine) as session:
            _store_challenge(session, options.challenge, "auth", None)
            session.commit()
        return json.loads(options_to_json(options))

    @router.post(
        "/auth/webauthn/verify",
        response_model=AccountInfo,
        summary="Finish passkey sign-in (verify assertion, internal)",
    )
    def passkey_auth_verify(
        body: PasskeyAssertRequest,
        x_internal_auth: Annotated[str | None, Header()] = None,
    ) -> AccountInfo:
        """Verify a passkey assertion and resolve the signing identity.

        Args:
            body: The credential produced by ``navigator.credentials.get()``.
            x_internal_auth: Shared-secret header proving the trusted frontend.

        Returns:
            The authenticated account, from which the frontend mints a session.

        Raises:
            DomainError: 403 on a bad internal secret; 401 on an expired
                challenge or unknown/failed credential.
        """
        _require_internal_auth(x_internal_auth)
        with Session(job_store.engine) as session:
            challenge, _ = _consume_challenge(session, body.credential, "auth")
            row = session.get(WebAuthnCredentialModel, _credential_key(body.credential))
            if row is None:
                raise DomainError("webauthn.invalid_credential", status=401)
            try:
                verification = verify_authentication_response(
                    credential=body.credential,
                    expected_challenge=challenge,
                    expected_rp_id=_rp_id(),
                    expected_origin=_expected_origins(),
                    credential_public_key=base64url_to_bytes(str(row.public_key)),
                    credential_current_sign_count=int(row.sign_count),
                )
            except (WebAuthnException, ValueError, TypeError, KeyError) as exc:
                raise DomainError("webauthn.invalid_credential", status=401) from exc
            now = datetime.now(UTC)
            row.sign_count = verification.new_sign_count
            row.last_used_at = now
            email = str(row.user_email)
            user_row = session.get(UserModel, email)
            name = email
            if user_row is not None:
                user_row.last_login_at = now
                name = str(user_row.name)
            enforce_monthly_active_user_limit(
                job_store,
                email,
                exempt=_role_for(email) == "admin",
            )
            session.commit()
        return AccountInfo(email=email, name=name, role=_role_for(email))

    return router
