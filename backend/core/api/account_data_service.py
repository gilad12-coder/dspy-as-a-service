"""Permanently remove an on-premise account and all user-owned data."""

from __future__ import annotations

from dataclasses import dataclass
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
    ByokProviderKeyModel,
    ConversationEmbeddingModel,
    DatasetBlobModel,
    DatasetModel,
    DatasetShareGrantModel,
    DatasetShareLinkModel,
    GepaCheckpointModel,
    GridPairResultModel,
    JobEmbeddingModel,
    JobModel,
    LogEntryModel,
    NotificationPreferenceModel,
    OptimizationShareGrantModel,
    OptimizationShareLinkModel,
    ProgressEventModel,
    TaggingSessionModel,
    TaggingSessionShareGrantModel,
    TaggingSessionShareLinkModel,
    TelemetryEventModel,
    UserModel,
    UserStorageQuotaOverrideModel,
)


@dataclass(frozen=True)
class AccountDeletionSummary:
    """Describe the outcome of an irreversible account purge."""

    deleted_rows: int
    anonymized_rows: int = 0


def _delete_rows(session: Session, statement: Any) -> int:
    """Execute one deletion and return the affected row count.

    Args:
        session: Open transaction used for the account purge.
        statement: SQLAlchemy delete statement.

    Returns:
        Number of deleted rows reported by the database.
    """
    result = session.execute(statement, execution_options={"synchronize_session": False})
    return result.rowcount or 0


def delete_account(session: Session, username: str) -> AccountDeletionSummary:
    """Purge an account and every record linked to its normalized username.

    The caller owns the transaction and commits only after all deletions
    succeed. A later valid ADFS login may recreate only the identity row; none
    of the purged data is restored.

    Args:
        session: Open database session used for the purge.
        username: Normalized account identity to remove.

    Returns:
        Count of rows permanently deleted.
    """
    deleted = 0

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
        deleted += _delete_rows(
            session,
            delete(ProgressEventModel).where(ProgressEventModel.optimization_id.in_(job_ids)),
        )
        deleted += _delete_rows(
            session,
            delete(LogEntryModel).where(LogEntryModel.optimization_id.in_(job_ids)),
        )
        deleted += _delete_rows(
            session,
            delete(GepaCheckpointModel).where(GepaCheckpointModel.optimization_id.in_(job_ids)),
        )
        deleted += _delete_rows(
            session,
            delete(GridPairResultModel).where(GridPairResultModel.optimization_id.in_(job_ids)),
        )
        deleted += _delete_rows(
            session,
            delete(OptimizationShareGrantModel).where(
                OptimizationShareGrantModel.optimization_id.in_(job_ids)
            ),
        )
        deleted += _delete_rows(
            session,
            delete(OptimizationShareLinkModel).where(
                OptimizationShareLinkModel.optimization_id.in_(job_ids)
            ),
        )
        deleted += _delete_rows(session, delete(JobModel).where(JobModel.optimization_id.in_(job_ids)))

    deleted += _delete_rows(
        session, delete(JobEmbeddingModel).where(JobEmbeddingModel.user_id == username)
    )
    deleted += _delete_rows(
        session,
        delete(OptimizationShareLinkModel).where(OptimizationShareLinkModel.created_by == username),
    )
    deleted += _delete_rows(
        session,
        delete(OptimizationShareGrantModel).where(
            (OptimizationShareGrantModel.grantee_username == username)
            | (OptimizationShareGrantModel.created_by == username)
        ),
    )

    dataset_ids = list(
        session.scalars(select(DatasetModel.id).where(DatasetModel.owner_username == username))
    )
    if dataset_ids:
        deleted += _delete_rows(
            session, delete(DatasetBlobModel).where(DatasetBlobModel.dataset_id.in_(dataset_ids))
        )
        deleted += _delete_rows(
            session,
            delete(DatasetShareGrantModel).where(DatasetShareGrantModel.dataset_id.in_(dataset_ids)),
        )
        deleted += _delete_rows(
            session,
            delete(DatasetShareLinkModel).where(DatasetShareLinkModel.dataset_id.in_(dataset_ids)),
        )
        deleted += _delete_rows(session, delete(DatasetModel).where(DatasetModel.id.in_(dataset_ids)))
    deleted += _delete_rows(
        session,
        delete(DatasetShareLinkModel).where(DatasetShareLinkModel.created_by == username),
    )
    deleted += _delete_rows(
        session,
        delete(DatasetShareGrantModel).where(
            (DatasetShareGrantModel.grantee_username == username)
            | (DatasetShareGrantModel.created_by == username)
        ),
    )

    tagging_ids = list(
        session.scalars(select(TaggingSessionModel.id).where(TaggingSessionModel.username == username))
    )
    if tagging_ids:
        deleted += _delete_rows(
            session,
            delete(TaggingSessionShareGrantModel).where(
                TaggingSessionShareGrantModel.session_id.in_(tagging_ids)
            ),
        )
        deleted += _delete_rows(
            session,
            delete(TaggingSessionShareLinkModel).where(
                TaggingSessionShareLinkModel.session_id.in_(tagging_ids)
            ),
        )
        deleted += _delete_rows(
            session, delete(TaggingSessionModel).where(TaggingSessionModel.id.in_(tagging_ids))
        )
    deleted += _delete_rows(
        session,
        delete(TaggingSessionShareLinkModel).where(TaggingSessionShareLinkModel.created_by == username),
    )
    deleted += _delete_rows(
        session,
        delete(TaggingSessionShareGrantModel).where(
            (TaggingSessionShareGrantModel.grantee_username == username)
            | (TaggingSessionShareGrantModel.created_by == username)
        ),
    )

    conversation_ids = list(
        session.scalars(
            select(AgentConversationModel.id).where(AgentConversationModel.username == username)
        )
    )
    if conversation_ids:
        deleted += _delete_rows(
            session,
            delete(AgentMessageModel).where(AgentMessageModel.conversation_id.in_(conversation_ids)),
        )
    deleted += _delete_rows(
        session,
        delete(ConversationEmbeddingModel).where(ConversationEmbeddingModel.username == username),
    )
    deleted += _delete_rows(
        session,
        delete(AgentConversationModel).where(AgentConversationModel.username == username),
    )

    for model in (
        ApiTokenModel,
        ByokProviderKeyModel,
        UserStorageQuotaOverrideModel,
        AgentStagedDatasetModel,
        AgentMemoryModel,
        AgentMemorySummaryModel,
        AgentMemorySettingsModel,
        NotificationPreferenceModel,
        TelemetryEventModel,
    ):
        deleted += _delete_rows(session, delete(model).where(model.username == username))

    session.execute(
        update(UserModel).where(UserModel.created_by == username).values(created_by=None),
        execution_options={"synchronize_session": False},
    )
    deleted += _delete_rows(session, delete(UserModel).where(UserModel.username == username))

    return AccountDeletionSummary(deleted_rows=deleted)
