"""Self-service account data export and irreversible account deletion. [INTERNAL]

Backs the GDPR/CCPA "download my data" and "delete my account" controls in
Settings. Both routes are bearer-authenticated and act only on the caller's own
identity (``user.username`` — the lowercased email every table keys on).
Deletion hard-removes the account's own content and secrets and anonymizes the
financial and audit records retention law requires be kept; it makes no live
call to Stripe or any other provider (see :mod:`core.api.account_data_service`).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...storage.models import UserModel
from ..account_data_service import delete_account, export_account
from ..auth import AuthenticatedUser, get_authenticated_user
from ..errors import DomainError
from ..passwords import verify_password

AuthenticatedUserDep = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]


# Confirmation payload for irreversible account deletion.
class DeleteAccountRequest(BaseModel):
    password: str = Field(
        default="",
        description="Current password, required to confirm deletion of a local (email/password) account.",
    )


# Row-count outcome returned to the client after a deletion completes.
class AccountDeletionResult(BaseModel):
    deleted_rows: int = Field(description="Rows hard-deleted across the account's data.")
    anonymized_rows: int = Field(
        description="Retained rows (financial, audit, telemetry) whose identity link was severed."
    )


def create_account_data_router(*, job_store) -> APIRouter:
    """Build the account data export + deletion router.

    Args:
        job_store: Job-store instance whose ORM engine backs the routes.

    Returns:
        A FastAPI ``APIRouter`` exposing export + delete for the caller's own
        account.
    """
    router = APIRouter()

    @router.get(
        "/account/export",
        summary="Export all of the caller's account data as a JSON download",
    )
    def export_my_data(user: AuthenticatedUserDep) -> JSONResponse:
        """Gather every record the caller owns into a downloadable JSON bundle.

        Secrets (password hash, 2FA secrets, encrypted provider keys, token
        hashes) are never included — only the non-sensitive fact that each
        exists. The response is served as a file attachment.

        Args:
            user: Authenticated caller; only their own data is gathered.

        Returns:
            A JSON response carrying the export bundle as a file attachment.
        """
        with Session(job_store.engine) as session:
            bundle = export_account(session, user.username)
        return JSONResponse(
            content=bundle,
            headers={
                "Content-Disposition": 'attachment; filename="skynet-account-export.json"'
            },
        )

    @router.post(
        "/account/delete",
        response_model=AccountDeletionResult,
        summary="Irreversibly delete the caller's account and all its data",
    )
    def delete_my_account(
        body: DeleteAccountRequest, user: AuthenticatedUserDep
    ) -> AccountDeletionResult:
        """Permanently delete the caller's account and everything it owns.

        Local (email/password) accounts must re-confirm with their current
        password; OAuth accounts have no password on file and are authorized by
        their bearer token alone. Financial and audit rows are anonymized rather
        than deleted, to satisfy record-retention obligations.

        Args:
            body: Confirmation payload carrying the current password.
            user: Authenticated caller whose account is deleted.

        Returns:
            Counts of hard-deleted and anonymized rows.

        Raises:
            DomainError: 401 (``accounts.invalid_credentials``) when a local
                account's confirmation password is missing or does not match.
        """
        with Session(job_store.engine) as session:
            row = session.get(UserModel, user.username)
            if row is not None and not verify_password(body.password, str(row.password_hash)):
                raise DomainError("accounts.invalid_credentials", status=401)
            summary = delete_account(session, user.username)
            session.commit()
        return AccountDeletionResult(
            deleted_rows=summary.deleted_rows, anonymized_rows=summary.anonymized_rows
        )

    return router
