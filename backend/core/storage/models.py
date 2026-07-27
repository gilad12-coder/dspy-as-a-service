"""SQLAlchemy ORM models for job storage.

Defines the shared database models used by the PostgreSQL storage backend.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 512
JSON_STORE = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all models."""


class ApiTokenModel(Base):
    """Per-user personal access token (one active token per user).

    Stores only the SHA-256 hash of the issued token; the plaintext is shown
    to the user once at creation and never persisted. ``username`` is the
    primary key, so generating a new token replaces the user's previous one
    (rotation). ``token_hash`` is uniquely indexed for the auth lookup, which
    queries by hash without ever holding the plaintext.
    """

    __tablename__ = "api_tokens"

    username: Mapped[str] = mapped_column(String(255), primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    last4: Mapped[str] = mapped_column(String(4), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UserModel(Base):
    """A Skynet-native account for email/password sign-in.

    Backs only the "create an account in Skynet" path — OAuth (Google/GitHub)
    users authenticate against their provider and never get a row here. The
    lowercased ``email`` is the primary key because it is also the cross-app
    identity (the ``username`` every other table — jobs, datasets, shares —
    keys ownership on), so a local account and the work it owns line up on a
    single value. Only the scrypt ``password_hash`` is persisted; the plaintext
    password is never stored.
    """

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Optional profile captured on the rich sign-up form. Nullable because OAuth
    # accounts (no ``users`` row) and rows created before these columns existed
    # never set them, and ``job_role`` is optional even on the sign-up form.
    use_case: Mapped[str | None] = mapped_column(String(32), nullable=True)
    experience_level: Mapped[str | None] = mapped_column(String(16), nullable=True)
    job_role: Mapped[str | None] = mapped_column(String(16), nullable=True)


class BillingCustomerModel(Base):
    """Per-user billing state synced from Stripe (one row per paying identity).

    Keyed on ``username`` (the lowercased email every other table owns rows by)
    rather than a foreign key to ``users`` so SSO accounts — which never get a
    ``users`` row — are billed too. ``stripe_customer_id`` is the durable link to
    Stripe. ``credit_balance`` is the denormalized spendable purchased-credit
    total, kept in step with ``credit_ledger`` on every mutation so a balance
    read is a single fast integer.
    """

    __tablename__ = "billing_customers"

    username: Mapped[str] = mapped_column(String(255), primary_key=True)
    stripe_customer_id: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    credit_balance: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        nullable=False,
        default=0,
        server_default="0",
    )
    # ``grant_remaining`` is what is left of the account's one-time free credit
    # grant (500 credits, seeded once and never renewed). NULL until the first
    # wallet read or run seeds it; seeding is lazy-evaluated on read, never
    # cron'd.
    grant_remaining: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class CreditLedgerModel(Base):
    """Immutable, append-only record of every credit movement for an account.

    Each row is a signed ``delta_credits`` against ``username``: positive for a
    top-up (pack purchase) or monthly grant, negative for a run charge. The
    running sum is denormalized onto ``billing_customers.credit_balance`` for
    fast gating; this table is the audit trail that explains that balance and
    backs the wallet's usage ledger. ``stripe_event_id`` ties a top-up row to the
    webhook event that created it so a redelivered event can't double-credit.
    """

    __tablename__ = "credit_ledger"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    username: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    delta_credits: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    description: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Measured usage behind a run row's charge; None on top-ups/grants and on
    # rows written before token metering landed.
    input_tokens: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"), nullable=True
    )
    output_tokens: Mapped[int | None] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"), nullable=True
    )
    stripe_event_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )


class BillingWebhookEventModel(Base):
    """Idempotency ledger of Stripe webhook events already applied.

    The webhook endpoint records each ``evt_…`` id here inside the same
    transaction that applies the event's effect, and skips any event whose id is
    already present. Stripe guarantees at-least-once delivery, so without this a
    retried ``checkout.session.completed`` would credit the same purchase twice.
    """

    __tablename__ = "billing_webhook_events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class BillingProviderKeyModel(Base):
    """One stored BYOK provider connection for an account, encrypted at rest.

    Backs BYOK mode: when an account runs in ``byok`` token source, jobs bill the
    user's own provider key instead of Skynet credits. The secret is never stored
    in plaintext — only ``secret_ciphertext`` (the Fernet-encrypted bytes) is
    persisted, so a database dump never leaks a usable key. ``last4`` is the
    recognizable tail kept for masked display, and ``status`` records whether the
    connection was checked against its provider (``unverified`` / ``verified`` /
    ``invalid``) so the UI can tell a typo'd key from a working one before a job
    ever runs. ``api_base`` and ``params`` carry an optional custom endpoint and
    extra LiteLLM kwargs so a connection can target any OpenAI-compatible host,
    and ``label`` is an optional user-facing name. The surrogate ``id`` is the
    primary key, so an account may hold several connections for one provider
    (e.g. two OpenAI-compatible endpoints); ``(username, provider)`` is indexed
    for the run-path and settings lookups.
    """

    __tablename__ = "billing_provider_keys"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid4().hex)
    username: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str | None] = mapped_column(String(120), nullable=True)
    api_base: Mapped[str | None] = mapped_column(String(255), nullable=True)
    params: Mapped[dict[str, Any]] = mapped_column(JSON_STORE, nullable=False, default=dict)
    secret_ciphertext: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    last4: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="unverified", server_default="unverified"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_billing_provider_keys_username_provider", "username", "provider"),
    )


class SearchQueryLogModel(Base):
    """Anonymous log of public-corpus search queries, powering trending searches.

    One row is written per public ``POST /dashboard/search`` that carries a
    non-empty query, first page only — later pages are pagination of the same
    search, not new intent. Owner-scoped ("mine") searches are never logged;
    only the cross-user public corpus is. No user identity, IP, session, or
    filter state is stored — just the normalized query text and a timestamp —
    so a top-N trending aggregate can be computed without profiling anyone.
    """

    __tablename__ = "search_query_log"

    # BigInteger on Postgres (BIGSERIAL); INTEGER on SQLite so its rowid
    # autoincrement kicks in for the create_all-based test stores.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    query_text: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )


class TelemetryEventModel(Base):
    """Append-only product-telemetry event — how the app is actually used.

    One row per captured interaction (a page view, a labelled click, a named
    flow event). The pipeline is first-party: the browser SDK batches events to
    ``POST /telemetry/events`` and they land here, queryable beside the rest of
    the product data — no third-party analytics service ever sees a user.

    Identity is split deliberately. ``username`` is the *server-trusted* caller
    derived from the request's auth token (never from the request body), so it
    is set only when the batch was sent authenticated; the client can't forge
    another user's identity onto an event. ``anonymous_id`` is an opaque,
    per-browser id the SDK generates (no email, no name), letting pre-login and
    logged-out activity be funnel-counted without profiling a person.
    ``session_id`` scopes a single browsing session. No IP address is stored,
    and ``properties`` is contractually PII-free (the SDK only emits structural
    descriptors), so aggregates can be computed without holding personal data.

    ``occurred_at`` is the client-reported event time (best-effort, clock-skewed);
    ``received_at`` is the authoritative server insert time the read endpoints
    order and bucket on.
    """

    __tablename__ = "telemetry_events"

    # BigInteger on Postgres (BIGSERIAL); INTEGER on SQLite so its rowid
    # autoincrement kicks in for the create_all-based test stores.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer(), "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    event_name: Mapped[str] = mapped_column(String(80), nullable=False)
    occurred_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    anonymous_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    locale: Mapped[str | None] = mapped_column(String(16), nullable=True)
    app_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    properties: Mapped[dict[str, Any]] = mapped_column(JSON_STORE, default=dict)
    context: Mapped[dict[str, Any]] = mapped_column(JSON_STORE, default=dict)

    # Top-events-over-time and per-user activity are the two read shapes; each
    # filters on its leading column and orders by recency, so a composite beats
    # two standalone indexes. The bare received_at index above still serves the
    # global time-series (all events, no name/user filter).
    __table_args__ = (
        Index("ix_telemetry_events_name_received", "event_name", "received_at"),
        Index("ix_telemetry_events_user_received", "username", "received_at"),
    )


class OptimizationShareLinkModel(Base):
    """Per-optimization sharing config keyed by a public link token.

    The ``token`` is the unguessable capability identifier embedded in the
    public ``/share/<token>`` URL. It is stored in plaintext because it IS the
    public identifier (like a ChatGPT share id), not a credential hash. The
    active (``revoked_at IS NULL``) row per optimization holds the sharing
    config. ``general_access`` selects the link policy: ``'restricted'`` (owner
    + invited members only) or ``'anyone'`` (anyone holding the link has
    access). ``general_role`` is the tier an ``'anyone'`` link grants a
    *signed-in* visitor — ``'viewer'`` or ``'editor'`` (never ``'owner'``;
    ownership is not transferred by link). An anonymous (logged-out) visitor
    never elevates past the read-only ``view`` tier regardless of
    ``general_role``, so a bare URL can never run inference on the owner's key.
    Revoking sets ``revoked_at`` so the public route returns 404 thereafter.
    Rows are removed when the optimization is deleted.
    """

    __tablename__ = "optimization_share_links"

    token: Mapped[str] = mapped_column(String(48), primary_key=True)
    optimization_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    general_access: Mapped[str] = mapped_column(
        String(16), nullable=False, default="restricted", server_default="restricted"
    )
    general_role: Mapped[str] = mapped_column(
        String(16), nullable=False, default="viewer", server_default="viewer"
    )


class OptimizationShareGrantModel(Base):
    """A single per-user access grant on a shared optimization.

    Each row invites one ``grantee_username`` to an optimization with a tier
    ``role`` (``'viewer'`` / ``'editor'`` / ``'owner'``). The pair
    ``(optimization_id, grantee_username)`` is the primary key, so re-inviting a
    user replaces their existing grant. ``general_access`` on the link and these
    per-user grants coexist: an anyone-link can be on while named members hold
    higher roles. Rows are removed when the optimization is deleted.
    """

    __tablename__ = "optimization_share_grants"

    optimization_id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
    grantee_username: Mapped[str] = mapped_column(String(255), primary_key=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # The composite PK leads with optimization_id, so "list everything shared
    # with this user" (filtering on grantee_username alone) could not use it.
    __table_args__ = (Index("ix_optimization_share_grants_grantee", "grantee_username"),)


class JobModel(Base):
    """SQLAlchemy model for the jobs table.

    The ``claimed_by`` / ``claimed_at`` / ``lease_expires_at`` triplet implements
    a DB-backed work queue safe for multi-pod horizontal scaling: each worker
    atomically claims a row via ``SELECT ... FOR UPDATE SKIP LOCKED`` and
    extends the lease while it holds the job. A pod that crashes leaves an
    expired lease which any other pod is free to re-claim on its next poll.
    """

    __tablename__ = "jobs"

    optimization_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    estimated_remaining_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    latest_metrics: Mapped[dict[str, Any]] = mapped_column(JSON_STORE, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON_STORE, nullable=True)
    payload_overview: Mapped[dict[str, Any]] = mapped_column(JSON_STORE, default=dict)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON_STORE, nullable=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    optimization_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    code_version: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Set once, by the worker that wins the CAS update inside
    # ``claim_completion_notification`` — guarantees a single Slack/Teams
    # message per job even when orphan recovery re-runs a row.
    notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Optional client-supplied dedup key; lookups are scoped per submitter so
    # two users may legitimately reuse the same key without colliding.
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Precomputed byte size of this job's JSON columns (payload + result +
    # overview), maintained at write time so the unified per-user storage total
    # is a single indexed SUM rather than a scan that re-serializes every payload.
    stored_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0, server_default="0")
    # Running sum of every completed leg's wall-clock duration. ``requeue_for_resume``
    # folds the finished leg in before clearing the timestamps, so elapsed reports net
    # active compute across resumes without ever counting the paused gap between legs.
    accumulated_runtime_seconds: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, server_default="0"
    )

    __table_args__ = (
        Index("ix_jobs_status_created_at", "status", "created_at"),
        # "my optimizations, newest first" — the dominant list/sidebar query —
        # becomes a single backward index range scan instead of filter + sort.
        Index("ix_jobs_username_created_at", "username", "created_at"),
        Index("ix_jobs_lease_expires_at", "lease_expires_at"),
        # Lookup index for idempotency dedup; the corresponding PG-only
        # uniqueness guard (partial on idempotency_key IS NOT NULL) lives in
        # the alembic migration so two concurrent submits with the same key
        # cannot both create rows.
        Index("ix_jobs_username_idempotency_key", "username", "idempotency_key"),
    )


class ProgressEventModel(Base):
    """SQLAlchemy model for the job_progress_events table."""

    __tablename__ = "job_progress_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    optimization_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    event: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON_STORE, default=dict)

    __table_args__ = (Index("ix_job_progress_events_optimization_timestamp", "optimization_id", "timestamp"),)


class LogEntryModel(Base):
    """SQLAlchemy model for the job_logs table."""

    __tablename__ = "job_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    optimization_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    level: Mapped[str] = mapped_column(String(20), nullable=False)
    logger: Mapped[str] = mapped_column(String(255), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    pair_index: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Backs the per-job ordered reads (get_logs) and the oldest-first cap
    # eviction in append_log, which otherwise scan on the single-column
    # optimization_id index and sort timestamps in memory.
    __table_args__ = (Index("ix_job_logs_optimization_timestamp", "optimization_id", "timestamp"),)


class GepaCheckpointModel(Base):
    """The latest GEPA optimizer state for one in-flight GEPA run.

    Holds the pickled ``gepa_state.bin`` the GEPA engine writes at every
    iteration, so a run that dies mid-optimization (provider timeout, stalled
    worker, pod crash, user cancel) can be resumed from its last completed
    iteration rather than restarting and re-spending the whole budget. Keyed by
    ``(optimization_id, pair_index)``: a single run uses the sentinel
    ``pair_index = -1`` (one row), while a grid search keeps one row per
    in-flight model pair (``pair_index`` 0..N-1) — each pair is its own GEPA run.
    A row is replaced in place on each iteration and deleted once that run/pair
    succeeds, so its ``stored_bytes`` (the blob length, folded into the owner's
    "optimizations" footprint) only counts while a resumable failure is pending.
    Deleted with its parent job via the cascading foreign key. Postgres stores
    ``data`` as BYTEA; the per-row whole-blob read/write is the only access
    pattern, so it lives apart from the hot ``jobs`` table.
    """

    __tablename__ = "gepa_checkpoints"

    optimization_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.optimization_id", ondelete="CASCADE"), primary_key=True
    )
    pair_index: Mapped[int] = mapped_column(Integer, primary_key=True, default=-1)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    stored_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class GridPairResultModel(Base):
    """One completed grid-search pair's result, persisted so resume can skip it.

    A grid search runs one GEPA optimization per model pair and keeps no partial
    results until the whole sweep finishes — so an interrupted grid would
    otherwise re-run every pair. Persisting each pair's :class:`PairResult` as it
    completes lets a resumed grid keep the finished pairs (``pair_index`` →
    result) and re-run only the rest. Keyed by ``(optimization_id, pair_index)``;
    ``stored_bytes`` folds into the owner's "optimizations" footprint and the
    rows are dropped once the grid succeeds (or with the parent job via the
    cascading foreign key).
    """

    __tablename__ = "grid_pair_results"

    optimization_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("jobs.optimization_id", ondelete="CASCADE"), primary_key=True
    )
    pair_index: Mapped[int] = mapped_column(Integer, primary_key=True)
    result: Mapped[dict[str, Any]] = mapped_column(JSON_STORE, nullable=False)
    stored_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class UserQuotaOverrideModel(Base):
    """SQLAlchemy model for live per-user quota overrides."""

    __tablename__ = "user_quota_overrides"

    username: Mapped[str] = mapped_column(String(255), primary_key=True)
    quota: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)


class UserQuotaAuditModel(Base):
    """SQLAlchemy model for quota administration audit events."""

    __tablename__ = "user_quota_audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    actor: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    target_username: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    old_quota: Mapped[int | None] = mapped_column(Integer, nullable=True)
    new_quota: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class UserStorageQuotaOverrideModel(Base):
    """SQLAlchemy model for per-user storage-budget overrides.

    ``quota_bytes`` is a per-user ceiling in bytes that replaces the global
    ``settings.user_storage_quota_bytes`` default for this user. Uses
    ``BigInteger`` because an admin override may grant multi-gigabyte budgets
    that exceed the signed 32-bit range (2_147_483_647).
    """

    __tablename__ = "user_storage_quota_overrides"

    username: Mapped[str] = mapped_column(String(255), primary_key=True)
    quota_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_by: Mapped[str | None] = mapped_column(String(255), nullable=True)


class JobEmbeddingModel(Base):
    """Per-job embedding row backing the recommendation service.

    One row is written after a job finishes successfully. Three named
    aspects are embedded independently so a similarity search can
    weigh them separately (``summary`` = LLM-authored task description,
    ``code`` = signature + metric source, ``schema`` = dataset schema
    digest). All use the configured embedding API model,
    MRL-truncated to ``EMBEDDING_DIM``.

    Metadata (``optimization_type``, ``winning_model``, ``winning_rank``)
    is denormalized from ``jobs`` so the search can filter and rerank
    without an extra join per-candidate.
    """

    __tablename__ = "job_embeddings"

    optimization_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    optimization_type: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    winning_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    winning_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    embedding_summary: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_code: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    embedding_schema: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    is_recommendable: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false", index=True
    )
    is_private: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false", index=True)
    baseline_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    optimized_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    signature_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    metric_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    optimizer_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    optimizer_kwargs: Mapped[dict[str, Any] | None] = mapped_column(JSON_STORE, nullable=True)
    module_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    task_name: Mapped[str | None] = mapped_column(String(255), nullable=True)


class AgentConversationModel(Base):
    """Persisted generalist-agent conversation header.

    One row per user-owned thread. ``title`` is auto-derived from the first
    user message and may be edited via PATCH. ``pinned`` and ``archived_at``
    are mutually independent — an archived row may still be pinned.
    """

    __tablename__ = "agent_conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_agent_conversations_user_updated", "username", "updated_at"),
        Index("ix_agent_conversations_user_pinned", "username", "pinned"),
    )


class AgentMessageModel(Base):
    """Single turn inside an :class:`AgentConversationModel`.

    ``tool_calls`` is the rendered ``AgentToolCall[]`` payload exactly as the
    frontend stores it in React state — kept as JSON rather than normalized
    so the renderer needs no migration when tool shapes change.
    """

    __tablename__ = "agent_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tool_calls: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON_STORE, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    # Five nullable training-metadata columns feed the training-ground harness
    # (training_ground_SPEC.md §4). Old rows predate the migration and stay
    # NULL; the optimize CLI filters them out via WHERE wizard_state_before IS NOT NULL.
    wizard_state_before: Mapped[dict[str, Any] | None] = mapped_column(JSON_STORE, nullable=True)
    wizard_state_after: Mapped[dict[str, Any] | None] = mapped_column(JSON_STORE, nullable=True)
    allowed_tools: Mapped[list[str] | None] = mapped_column(JSON_STORE, nullable=True)
    tool_schema_hashes: Mapped[dict[str, str] | None] = mapped_column(JSON_STORE, nullable=True)
    router_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON_STORE, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )

    __table_args__ = (Index("ix_agent_messages_conv_created", "conversation_id", "created_at"),)


class ConversationEmbeddingModel(Base):
    """Per-conversation embedding row backing the agent-history search.

    Mirrors :class:`JobEmbeddingModel` so the search dispatch and the
    backfill / purge plumbing can be lifted from the optimization corpus
    with a different source table. The ``summary_text`` column holds the
    exact prose that was embedded — concatenated user turns (and a slice
    of assistant replies) capped to a budget — so lexical fallback can hit
    the same text the vector was built from.
    """

    __tablename__ = "conversation_embeddings"

    conversation_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("agent_conversations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    username: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    embedding_summary: Mapped[Any] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Snapshot of ``conversation.updated_at`` at embed time. Used by the
    # backfill sweep to detect stale rows (conversation got new turns after
    # the last embed) without diffing message content.
    embedded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


# Holds rows the wizard parsed in the browser so the generalist agent can
# submit ``/run`` without re-shipping the dataset through its context. Rows
# live here only between upload and submit; the frontend stages on upload,
# the /run handler dereferences by id, and stale rows are evicted by user/age.
class AgentStagedDatasetModel(Base):
    """Server-side cache of wizard dataset rows for agent-driven submits."""

    __tablename__ = "agent_staged_datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dataset_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    rows: Mapped[list[dict[str, Any]]] = mapped_column(JSON_STORE, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC), index=True
    )

    __table_args__ = (Index("ix_agent_staged_datasets_user_created", "username", "created_at"),)


class TaggingSessionModel(Base):
    """Persisted text-labeling (tagger) session — one row per saved session.

    Mirrors the in-memory ``useTagger`` hook so a user can resume annotating
    across refreshes and devices, the way optimizations persist. ``config``,
    ``columns`` and ``data`` are captured once when annotating begins and are
    immutable thereafter; ``annotations``, ``current_index`` and ``phase`` (with
    the denormalized ``tagged_count``) advance as rows are labeled and are the
    only columns the autosave PUT rewrites, so the large ``data`` TOAST is not
    re-written on every keystroke. ``row_count``/``tagged_count`` are
    denormalized so the sidebar list never loads the heavy JSON columns.
    Ownership is by ``username`` (compared in application code, not a DB FK).
    """

    __tablename__ = "tagging_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    username: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    phase: Mapped[str] = mapped_column(String(20), nullable=False, default="annotating")
    config: Mapped[dict[str, Any]] = mapped_column(JSON_STORE, nullable=False, default=dict)
    columns: Mapped[list[str]] = mapped_column(JSON_STORE, nullable=False, default=list)
    data: Mapped[list[dict[str, Any]]] = mapped_column(JSON_STORE, nullable=False, default=list)
    annotations: Mapped[dict[str, Any]] = mapped_column(JSON_STORE, nullable=False, default=dict)
    # AI co-tagging state (assist mode, interview, rubric, predictions with
    # provenance/confidence, review rounds, auto-tag job bookkeeping). NULL for
    # plain manual sessions; ``annotations`` stays the single source of truth
    # for final labels regardless of who produced them.
    assist: Mapped[dict[str, Any] | None] = mapped_column(JSON_STORE, nullable=True, default=None)
    current_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tagged_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (
        Index("ix_tagging_sessions_user_updated", "username", "updated_at"),
        Index("ix_tagging_sessions_user_pinned", "username", "pinned"),
    )


class DatasetModel(Base):
    """One saved dataset in a user's personal library — a single file reference.

    Metadata only: the row bytes live in the one-to-one :class:`DatasetBlobModel`
    so this hot table stays narrow (a JSONB-inlined dataset would TOAST and slow
    every list query). ``byte_size`` is the uncompressed logical size shown to
    the user; ``stored_bytes`` is the compressed size that counts against the
    per-user quota. ``content_hash`` is the SHA-256 of the canonical uncompressed
    bytes — indexed with ``owner_username`` so a re-save of identical bytes can
    dedupe instead of storing a second copy. ``column_schema`` carries the saved
    column roles/kinds/order so picking the dataset in the submit wizard
    pre-fills the column step. ``source`` records the producing surface
    (``'upload'`` / ``'tagger'`` / ``'optimization'``).
    """

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_username: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="upload")
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    column_count: Mapped[int] = mapped_column(Integer, nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    stored_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    column_schema: Mapped[dict[str, Any]] = mapped_column(JSON_STORE, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    __table_args__ = (Index("ix_datasets_owner_content_hash", "owner_username", "content_hash"),)


class DatasetBlobModel(Base):
    """The row bytes for one :class:`DatasetModel`, stored compressed and apart.

    Split from the metadata table so the whole-file read pattern avoids the
    random-access penalty TOAST imposes on a large JSONB column, and so the
    bytes can later move behind an object-store seam without touching the
    metadata schema. ``data`` holds the ``compression``-encoded serialization of
    the rows in ``content_type`` (``'csv'`` / ``'json'``). Deleted with its
    parent dataset via the cascading foreign key.
    """

    __tablename__ = "dataset_blobs"

    dataset_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("datasets.id", ondelete="CASCADE"), primary_key=True
    )
    content_type: Mapped[str] = mapped_column(String(16), nullable=False)
    compression: Mapped[str] = mapped_column(String(16), nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class DatasetShareLinkModel(Base):
    """Per-dataset sharing config keyed by a public link token.

    Mirrors :class:`OptimizationShareLinkModel` for the dataset library: the
    ``token`` is the unguessable capability embedded in the public
    ``/datasets/share/<token>`` URL, stored in plaintext because it IS the
    public identifier, not a credential hash. The active (``revoked_at IS
    NULL``) row per dataset holds the config. ``general_access`` is
    ``'restricted'`` (owner + invited members only) or ``'anyone'`` (anyone
    holding the link). ``general_role`` is the tier an ``'anyone'`` link grants
    a signed-in visitor — ``'viewer'`` or ``'editor'`` (never ``'owner'``).
    Unlike the optimization variant, the ``dataset_id`` foreign key cascades, so
    deleting a dataset removes its link rows automatically.
    """

    __tablename__ = "dataset_share_links"

    token: Mapped[str] = mapped_column(String(48), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    general_access: Mapped[str] = mapped_column(
        String(16), nullable=False, default="restricted", server_default="restricted"
    )
    general_role: Mapped[str] = mapped_column(
        String(16), nullable=False, default="viewer", server_default="viewer"
    )


class DatasetShareGrantModel(Base):
    """A single per-user access grant on a shared library dataset.

    Mirrors :class:`OptimizationShareGrantModel`: each row invites one
    ``grantee_username`` to a dataset with a tier ``role`` (``'viewer'`` /
    ``'editor'`` / ``'owner'``). The pair ``(dataset_id, grantee_username)`` is
    the primary key, so re-inviting a user replaces their grant. The
    ``dataset_id`` foreign key cascades, removing grants when the dataset is
    deleted.
    """

    __tablename__ = "dataset_share_grants"

    dataset_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    grantee_username: Mapped[str] = mapped_column(String(255), primary_key=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # The composite PK leads with dataset_id, so "list everything shared with
    # this user" (filtering on grantee_username alone) could not use it.
    __table_args__ = (Index("ix_dataset_share_grants_grantee", "grantee_username"),)


class TaggingSessionShareLinkModel(Base):
    """Per-session sharing config keyed by a public link token.

    Mirrors :class:`DatasetShareLinkModel` for saved tagger sessions: the
    ``token`` is the unguessable capability embedded in the public
    ``/tagger/share/<token>`` URL, stored in plaintext because it IS the public
    identifier, not a credential hash. The active (``revoked_at IS NULL``) row
    per session holds the config; ``general_access`` and ``general_role`` carry
    the same semantics as the dataset variant. The ``session_id`` foreign key
    cascades, so deleting a session removes its link rows automatically.
    """

    __tablename__ = "tagging_session_share_links"

    token: Mapped[str] = mapped_column(String(48), primary_key=True)
    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tagging_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    general_access: Mapped[str] = mapped_column(
        String(16), nullable=False, default="restricted", server_default="restricted"
    )
    general_role: Mapped[str] = mapped_column(
        String(16), nullable=False, default="viewer", server_default="viewer"
    )


class TaggingSessionShareGrantModel(Base):
    """A single per-user access grant on a shared tagger session.

    Mirrors :class:`DatasetShareGrantModel`: each row invites one
    ``grantee_username`` to a session with a tier ``role`` (``'viewer'`` /
    ``'editor'`` / ``'owner'``). The pair ``(session_id, grantee_username)`` is
    the primary key, so re-inviting a user replaces their grant. The
    ``session_id`` foreign key cascades, removing grants when the session is
    deleted.
    """

    __tablename__ = "tagging_session_share_grants"

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("tagging_sessions.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    grantee_username: Mapped[str] = mapped_column(String(255), primary_key=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    created_by: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )

    # The composite PK leads with session_id, so "list everything shared with
    # this user" (filtering on grantee_username alone) could not use it.
    __table_args__ = (Index("ix_tagging_session_share_grants_grantee", "grantee_username"),)
