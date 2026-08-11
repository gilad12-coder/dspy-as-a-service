"""Gather-and-purge logic behind self-service data export and account deletion.

Both operations key entirely on the caller's ``username`` — the lowercased
email every table owns rows by. There are no database foreign keys pointing at
the ``users`` row (OAuth accounts have no such row yet still own work), so
deletion cannot lean on a cascade from a single identity row: every owned table
is cleared explicitly here, children before parents so the Postgres foreign
keys inside each object tree stay satisfied.

Two deliberate exceptions to "hard delete everything":

* Financial records (``billing_customers``, ``credit_ledger``) and the quota
  audit trail are *anonymized*, not deleted — their retention has an accounting
  and audit basis, so the identity link is severed to a stable, PII-free
  tombstone token while the rows themselves survive.
* Product telemetry keeps its aggregate rows but drops the ``username`` that
  tied them to a person, leaving only the already-anonymous per-browser id.

Nothing here makes a live call to Stripe, OpenRouter, or any other provider;
severing the local link is the whole of the operation. Deprovisioning the
upstream Stripe customer or OpenRouter runtime key is left to a follow-up.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from ..storage.models import (
    AgentConversationModel,
    AgentMemoryModel,
    AgentMemorySettingsModel,
    AgentMemorySummaryModel,
    AgentMessageModel,
    AgentStagedDatasetModel,
    ApiTokenModel,
    BillingCustomerModel,
    BillingOpenRouterKeyModel,
    BillingProviderKeyModel,
    ConversationEmbeddingModel,
    CreditLedgerModel,
    DatasetBlobModel,
    DatasetModel,
    DatasetShareGrantModel,
    DatasetShareLinkModel,
    GepaCheckpointModel,
    GridPairResultModel,
    JobEmbeddingModel,
    JobModel,
    LogEntryModel,
    OptimizationShareGrantModel,
    OptimizationShareLinkModel,
    ProgressEventModel,
    TaggingSessionModel,
    TaggingSessionShareGrantModel,
    TaggingSessionShareLinkModel,
    TelemetryEventModel,
    TwoFactorEmailCodeModel,
    UserModel,
    UserQuotaAuditModel,
    UserQuotaOverrideModel,
    UserStorageQuotaOverrideModel,
    WebAuthnChallengeModel,
    WebAuthnCredentialModel,
)

EXPORT_FORMAT = "skynet-account-export/v1"


@dataclass
class AccountDeletionSummary:
    """Row-count outcome of an account deletion.

    Attributes:
        deleted_rows: Rows hard-deleted across every table the account owned.
        anonymized_rows: Retained rows whose identity link was severed
            (financial, audit, and telemetry records).
    """

    deleted_rows: int
    anonymized_rows: int


def _iso(value: datetime | None) -> str | None:
    """Render a datetime as an ISO-8601 string, passing ``None`` through.

    Args:
        value: A timezone-aware datetime, or ``None``.

    Returns:
        The ISO-8601 rendering, or ``None`` when the input was ``None``.
    """
    return value.isoformat() if value is not None else None


def _anonymized_username(username: str) -> str:
    """Derive the stable, PII-free tombstone that replaces a deleted identity.

    The token is a truncated SHA-256 of the username, so the same person maps
    to the same token across every retained table (keeping the financial trail
    internally joinable) while carrying no recoverable email.

    Args:
        username: The account's lowercased-email identity.

    Returns:
        A ``deleted-user-<hash>`` token with no personal data.
    """
    digest = hashlib.sha256(username.encode("utf-8")).hexdigest()[:16]
    return f"deleted-user-{digest}"


def export_account(session: Session, username: str) -> dict[str, Any]:
    """Collect every record owned by ``username`` into a JSON-safe bundle.

    Secrets are never included — the password hash, TOTP/recovery secrets,
    encrypted provider keys, and token hashes are all withheld; only the
    non-sensitive fact that each exists (a masked tail, a boolean, a count) is
    reported. Bulk payloads that are recoverable through their own download
    surface (raw run payloads, raw dataset bytes, raw tagger source rows) are
    summarized rather than inlined to keep the bundle bounded.

    Args:
        session: An open session on the store's engine.
        username: The caller's lowercased-email identity.

    Returns:
        A JSON-serializable dict describing the account and all data it owns.
    """
    user = session.get(UserModel, username)
    account: dict[str, Any] | None = None
    if user is not None:
        recovery_codes = json.loads(user.recovery_codes) if user.recovery_codes else []
        account = {
            "email": user.email,
            "name": user.name,
            "created_at": _iso(user.created_at),
            "last_login_at": _iso(user.last_login_at),
            "use_case": user.use_case,
            "experience_level": user.experience_level,
            "job_role": user.job_role,
            "two_factor": {
                "totp_enabled": user.totp_secret is not None,
                "email_codes_enabled": bool(user.email_2fa_enabled),
                "recovery_codes_remaining": len(recovery_codes),
            },
        }

    jobs = session.scalars(
        select(JobModel).where(JobModel.username == username).order_by(JobModel.created_at)
    ).all()
    optimizations = [
        {
            "optimization_id": job.optimization_id,
            "status": job.status,
            "optimization_type": job.optimization_type,
            "created_at": _iso(job.created_at),
            "started_at": _iso(job.started_at),
            "completed_at": _iso(job.completed_at),
            "latest_metrics": job.latest_metrics,
            "payload_overview": job.payload_overview,
            "result": job.result,
        }
        for job in jobs
    ]

    datasets = session.scalars(
        select(DatasetModel)
        .where(DatasetModel.owner_username == username)
        .order_by(DatasetModel.created_at)
    ).all()
    datasets_out = [
        {
            "id": dataset.id,
            "name": dataset.name,
            "source": dataset.source,
            "row_count": dataset.row_count,
            "column_count": dataset.column_count,
            "byte_size": dataset.byte_size,
            "column_schema": dataset.column_schema,
            "created_at": _iso(dataset.created_at),
        }
        for dataset in datasets
    ]

    tagging_sessions = session.scalars(
        select(TaggingSessionModel)
        .where(TaggingSessionModel.username == username)
        .order_by(TaggingSessionModel.created_at)
    ).all()
    tagging_out = [
        {
            "id": tagging.id,
            "name": tagging.name,
            "phase": tagging.phase,
            "row_count": tagging.row_count,
            "tagged_count": tagging.tagged_count,
            "config": tagging.config,
            "annotations": tagging.annotations,
            "created_at": _iso(tagging.created_at),
            "updated_at": _iso(tagging.updated_at),
        }
        for tagging in tagging_sessions
    ]

    conversations = session.scalars(
        select(AgentConversationModel)
        .where(AgentConversationModel.username == username)
        .order_by(AgentConversationModel.created_at)
    ).all()
    conversations_out = []
    for conversation in conversations:
        messages = session.scalars(
            select(AgentMessageModel)
            .where(AgentMessageModel.conversation_id == conversation.id)
            .order_by(AgentMessageModel.created_at)
        ).all()
        conversations_out.append(
            {
                "id": conversation.id,
                "title": conversation.title,
                "created_at": _iso(conversation.created_at),
                "updated_at": _iso(conversation.updated_at),
                "messages": [
                    {
                        "role": message.role,
                        "content": message.content,
                        "model": message.model,
                        "created_at": _iso(message.created_at),
                    }
                    for message in messages
                ],
            }
        )

    memories = session.scalars(
        select(AgentMemoryModel)
        .where(AgentMemoryModel.username == username)
        .order_by(AgentMemoryModel.seq)
    ).all()
    memories_out = [
        {"seq": memory.seq, "content": memory.content, "created_at": _iso(memory.created_at)}
        for memory in memories
    ]

    customer = session.get(BillingCustomerModel, username)
    ledger = session.scalars(
        select(CreditLedgerModel)
        .where(CreditLedgerModel.username == username)
        .order_by(CreditLedgerModel.created_at)
    ).all()
    billing = {
        "credit_balance": int(customer.credit_balance) if customer is not None else 0,
        "grant_remaining": (
            int(customer.grant_remaining)
            if customer is not None and customer.grant_remaining is not None
            else None
        ),
        "ledger": [
            {
                "delta_credits": int(entry.delta_credits),
                "kind": entry.kind,
                "description": entry.description,
                "model": entry.model,
                "created_at": _iso(entry.created_at),
            }
            for entry in ledger
        ],
    }

    token = session.get(ApiTokenModel, username)
    api_token = (
        {
            "last4": token.last4,
            "created_at": _iso(token.created_at),
            "last_used_at": _iso(token.last_used_at),
        }
        if token is not None
        else None
    )

    provider_keys = session.scalars(
        select(BillingProviderKeyModel).where(BillingProviderKeyModel.username == username)
    ).all()
    provider_keys_out = [
        {
            "provider": key.provider,
            "label": key.label,
            "last4": key.last4,
            "status": key.status,
            "created_at": _iso(key.created_at),
        }
        for key in provider_keys
    ]

    passkeys = session.scalars(
        select(WebAuthnCredentialModel).where(WebAuthnCredentialModel.user_email == username)
    ).all()
    passkeys_out = [
        {
            "nickname": passkey.nickname,
            "created_at": _iso(passkey.created_at),
            "last_used_at": _iso(passkey.last_used_at),
        }
        for passkey in passkeys
    ]

    return {
        "export_format": EXPORT_FORMAT,
        "exported_at": _iso(datetime.now(UTC)),
        "username": username,
        "account": account,
        "optimizations": optimizations,
        "datasets": datasets_out,
        "tagging_sessions": tagging_out,
        "agent_conversations": conversations_out,
        "agent_memories": memories_out,
        "billing": billing,
        "api_token": api_token,
        "provider_keys": provider_keys_out,
        "passkeys": passkeys_out,
    }


def delete_account(session: Session, username: str) -> AccountDeletionSummary:
    """Purge everything owned by ``username`` within the given session.

    Runs entirely inside the caller's transaction (the caller commits), so a
    failure part-way rolls the whole deletion back. Owned content and secrets
    are hard-deleted; financial, audit, and telemetry rows are anonymized in
    place. See the module docstring for why those three survive.

    Args:
        session: An open session on the store's engine; the caller commits.
        username: The caller's lowercased-email identity.

    Returns:
        Counts of hard-deleted and anonymized rows.
    """
    deleted = 0
    anonymized = 0

    def _run_delete(statement: Any) -> None:
        nonlocal deleted
        result = session.execute(statement, execution_options={"synchronize_session": False})
        deleted += result.rowcount or 0

    def _run_anonymize(statement: Any) -> None:
        nonlocal anonymized
        result = session.execute(statement, execution_options={"synchronize_session": False})
        anonymized += result.rowcount or 0

    owned_job_ids = list(
        session.scalars(select(JobModel.optimization_id).where(JobModel.username == username))
    )
    child_job_ids = (
        list(
            session.scalars(
                select(JobModel.optimization_id).where(
                    JobModel.parent_optimization_id.in_(owned_job_ids)
                )
            )
        )
        if owned_job_ids
        else []
    )
    job_ids = list({*owned_job_ids, *child_job_ids})
    if job_ids:
        _run_delete(delete(ProgressEventModel).where(ProgressEventModel.optimization_id.in_(job_ids)))
        _run_delete(delete(LogEntryModel).where(LogEntryModel.optimization_id.in_(job_ids)))
        _run_delete(delete(GepaCheckpointModel).where(GepaCheckpointModel.optimization_id.in_(job_ids)))
        _run_delete(delete(GridPairResultModel).where(GridPairResultModel.optimization_id.in_(job_ids)))
        _run_delete(
            delete(OptimizationShareGrantModel).where(
                OptimizationShareGrantModel.optimization_id.in_(job_ids)
            )
        )
        _run_delete(
            delete(OptimizationShareLinkModel).where(
                OptimizationShareLinkModel.optimization_id.in_(job_ids)
            )
        )
        _run_delete(delete(JobModel).where(JobModel.optimization_id.in_(job_ids)))
    _run_delete(delete(JobEmbeddingModel).where(JobEmbeddingModel.user_id == username))
    _run_delete(
        delete(OptimizationShareLinkModel).where(OptimizationShareLinkModel.created_by == username)
    )
    _run_delete(
        delete(OptimizationShareGrantModel).where(
            OptimizationShareGrantModel.grantee_username == username
        )
    )
    _run_delete(
        delete(OptimizationShareGrantModel).where(OptimizationShareGrantModel.created_by == username)
    )

    dataset_ids = list(
        session.scalars(select(DatasetModel.id).where(DatasetModel.owner_username == username))
    )
    if dataset_ids:
        _run_delete(delete(DatasetBlobModel).where(DatasetBlobModel.dataset_id.in_(dataset_ids)))
        _run_delete(
            delete(DatasetShareLinkModel).where(DatasetShareLinkModel.dataset_id.in_(dataset_ids))
        )
        _run_delete(
            delete(DatasetShareGrantModel).where(DatasetShareGrantModel.dataset_id.in_(dataset_ids))
        )
        _run_delete(delete(DatasetModel).where(DatasetModel.id.in_(dataset_ids)))
    _run_delete(delete(DatasetShareLinkModel).where(DatasetShareLinkModel.created_by == username))
    _run_delete(
        delete(DatasetShareGrantModel).where(DatasetShareGrantModel.grantee_username == username)
    )
    _run_delete(delete(DatasetShareGrantModel).where(DatasetShareGrantModel.created_by == username))

    session_ids = list(
        session.scalars(select(TaggingSessionModel.id).where(TaggingSessionModel.username == username))
    )
    if session_ids:
        _run_delete(
            delete(TaggingSessionShareLinkModel).where(
                TaggingSessionShareLinkModel.session_id.in_(session_ids)
            )
        )
        _run_delete(
            delete(TaggingSessionShareGrantModel).where(
                TaggingSessionShareGrantModel.session_id.in_(session_ids)
            )
        )
        _run_delete(delete(TaggingSessionModel).where(TaggingSessionModel.id.in_(session_ids)))
    _run_delete(
        delete(TaggingSessionShareLinkModel).where(
            TaggingSessionShareLinkModel.created_by == username
        )
    )
    _run_delete(
        delete(TaggingSessionShareGrantModel).where(
            TaggingSessionShareGrantModel.grantee_username == username
        )
    )
    _run_delete(
        delete(TaggingSessionShareGrantModel).where(
            TaggingSessionShareGrantModel.created_by == username
        )
    )

    conversation_ids = list(
        session.scalars(
            select(AgentConversationModel.id).where(AgentConversationModel.username == username)
        )
    )
    if conversation_ids:
        _run_delete(
            delete(AgentMessageModel).where(AgentMessageModel.conversation_id.in_(conversation_ids))
        )
    _run_delete(
        delete(ConversationEmbeddingModel).where(ConversationEmbeddingModel.username == username)
    )
    _run_delete(delete(AgentConversationModel).where(AgentConversationModel.username == username))

    _run_delete(delete(ApiTokenModel).where(ApiTokenModel.username == username))
    _run_delete(delete(TwoFactorEmailCodeModel).where(TwoFactorEmailCodeModel.email == username))
    _run_delete(delete(WebAuthnCredentialModel).where(WebAuthnCredentialModel.user_email == username))
    _run_delete(delete(WebAuthnChallengeModel).where(WebAuthnChallengeModel.user_email == username))
    _run_delete(delete(BillingProviderKeyModel).where(BillingProviderKeyModel.username == username))
    _run_delete(delete(BillingOpenRouterKeyModel).where(BillingOpenRouterKeyModel.username == username))
    _run_delete(delete(UserQuotaOverrideModel).where(UserQuotaOverrideModel.username == username))
    _run_delete(
        delete(UserStorageQuotaOverrideModel).where(
            UserStorageQuotaOverrideModel.username == username
        )
    )
    _run_delete(delete(AgentStagedDatasetModel).where(AgentStagedDatasetModel.username == username))
    _run_delete(delete(AgentMemoryModel).where(AgentMemoryModel.username == username))
    _run_delete(delete(AgentMemorySummaryModel).where(AgentMemorySummaryModel.username == username))
    _run_delete(delete(AgentMemorySettingsModel).where(AgentMemorySettingsModel.username == username))

    tombstone = _anonymized_username(username)
    _run_anonymize(
        update(TelemetryEventModel)
        .where(TelemetryEventModel.username == username)
        .values(username=None)
    )
    _run_anonymize(
        update(BillingCustomerModel)
        .where(BillingCustomerModel.username == username)
        .values(username=tombstone)
    )
    _run_anonymize(
        update(CreditLedgerModel)
        .where(CreditLedgerModel.username == username)
        .values(username=tombstone)
    )
    _run_anonymize(
        update(UserQuotaAuditModel)
        .where(UserQuotaAuditModel.actor == username)
        .values(actor=tombstone)
    )
    _run_anonymize(
        update(UserQuotaAuditModel)
        .where(UserQuotaAuditModel.target_username == username)
        .values(target_username=tombstone)
    )

    _run_delete(delete(UserModel).where(UserModel.email == username))

    return AccountDeletionSummary(deleted_rows=deleted, anonymized_rows=anonymized)
