"""Public dashboard aggregator for the anonymous /explore page (PER-11 Feature B).

Builds the corpus point list that feeds the /explore list view's count,
filters, and model/optimizer options.

1. Fingerprint check — cheap ``COUNT(*) + MAX(created_at)`` query gates the
   expensive recompute. Same fingerprint = serve cached payload.
2. Bulk fetch — every public success job's lightweight metadata. Heavy
   fields (``signature_code``, ``optimizer_kwargs``, ``metric_name``,
   ``winning_rank``, ``is_recommendable``) are not used by the explore UI
   and are dropped to keep the payload under ~5 MB (gzipped) at 100k points.
3. Cache — keyed by fingerprint, 5 min TTL.

No personal information is exposed. ``signature_code`` is dropped from the
bulk response (it is not consumed by the explore page). Jobs flagged
``is_private`` are excluded from both the fingerprint and the bulk fetch,
so they never appear in the corpus payload and do not invalidate the cache
when added.
"""

from __future__ import annotations

import logging
import threading
import time
import weakref
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import DateTime, bindparam, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import settings
from ..constants import (
    PAYLOAD_OVERVIEW_DESCRIPTION,
    PAYLOAD_OVERVIEW_MODEL_NAME,
    PAYLOAD_OVERVIEW_MODULE_NAME,
    PAYLOAD_OVERVIEW_NAME,
    PAYLOAD_OVERVIEW_OPTIMIZER_NAME,
)
from .embedding_pipeline.embeddings import get_embedder

logger = logging.getLogger(__name__)


# Defensive ceiling. The /explore page is designed for up to 100k points;
# beyond that, building the payload on the request thread becomes a real
# outage risk.
MAX_POINTS = 100_000

# Free-text fields are truncated for the bulk response. Full text is
# only useful when a point is selected — and the truncated text is what
# the tooltip / detail panel header already render.
SUMMARY_TEXT_MAX = 200

_CACHE_TTL_SECONDS = 300
_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"fingerprint": None, "at": 0.0, "payload": None}

# "Shared with me" scope: restrict to optimizations the caller holds a member
# grant on AND does not own. Mirrors RemoteJobStore.list_jobs_shared_with —
# match on the lowercased grantee with no is_private gate, since the grant
# authorizes access to private runs the caller was explicitly invited to. The
# owner-exclusion makes "shared with me" mean runs *others* shared with the
# caller, never their own, even if a self-grant ever slips into the table.
# ``IS DISTINCT FROM`` is the NULL-safe inequality — it keeps rows whose owner
# column is NULL rather than dropping them the way ``<>`` would.
_SHARED_GRANT_SCOPE_SQL = (
    "j.optimization_id IN ("
    "SELECT optimization_id FROM optimization_share_grants "
    "WHERE grantee_username = :shared_with_username) "
    "AND j.username IS DISTINCT FROM :shared_with_username"
)

# Columns the explore/corpus SQL reads off a job's embedding row. When the
# job_embeddings table is absent — embeddings disabled or pgvector unavailable,
# in which case RemoteJobStore deliberately skips the Vector tables so a plain
# Postgres still boots — every success job is unembedded. Joining this empty,
# typed relation in the table's place makes each LEFT JOIN behave exactly like
# "table exists, holds no rows": the queries fall back to payload_overview via
# the COALESCEs they already apply, instead of raising UndefinedTable (which the
# browser then sees as a CORS / "can't connect" failure on /dashboard/*).
_EMPTY_JOB_EMBEDDINGS_REL = (
    "(SELECT "
    "NULL::text AS optimization_id, "
    "NULL::text AS optimization_type, "
    "NULL::text AS winning_model, "
    "NULL::text AS optimizer_name, "
    "NULL::text AS module_name, "
    "NULL::text AS task_name, "
    "NULL::text AS summary_text, "
    "NULL::double precision AS baseline_metric, "
    "NULL::double precision AS optimized_metric, "
    "NULL::boolean AS is_private, "
    "NULL::timestamptz AS created_at, "
    "NULL::text AS embedding_summary "
    "WHERE FALSE)"
)

_EMBEDDINGS_TABLE_PRESENT: "weakref.WeakKeyDictionary[Any, bool]" = weakref.WeakKeyDictionary()


def _job_embeddings_table_present(job_store: Any) -> bool:
    """Return whether the ``job_embeddings`` table exists, cached per engine.

    The schema is fixed for a process's lifetime, so the catalog lookup runs
    once per engine. ``to_regclass`` yields NULL (not an error) when the
    relation is absent, so the probe never raises on a healthy connection; any
    probe failure is treated as "absent" so the caller degrades safely.

    Args:
        job_store: A store exposing a SQLAlchemy ``engine`` attribute.

    Returns:
        True when the table is present, False otherwise.
    """
    engine = job_store.engine
    cached = _EMBEDDINGS_TABLE_PRESENT.get(engine)
    if cached is not None:
        return cached
    try:
        with Session(engine) as session:
            present = bool(
                session.execute(text("SELECT to_regclass('job_embeddings')")).scalar()
            )
    except Exception as exc:
        logger.warning("job_embeddings presence probe failed, assuming absent: %s", exc)
        present = False
    _EMBEDDINGS_TABLE_PRESENT[engine] = present
    return present


def _job_embeddings_relation(job_store: Any) -> str:
    """Return the SQL relation to stand in for ``job_embeddings`` in explore joins.

    The real table when it exists, else an empty typed relation (see
    :data:`_EMPTY_JOB_EMBEDDINGS_REL`) so the corpus / facets / lexical-search
    queries degrade to a jobs-only read on a database where the Vector tables
    were never created, rather than raising ``UndefinedTable``.

    Args:
        job_store: A store exposing a SQLAlchemy ``engine`` attribute.

    Returns:
        ``"job_embeddings"`` when the table is present, otherwise the empty
        stand-in relation.
    """
    if _job_embeddings_table_present(job_store):
        return "job_embeddings"
    return _EMPTY_JOB_EMBEDDINGS_REL


def _fetch_fingerprint(session: Session, je_rel: str) -> str:
    """Cheap content fingerprint over the searchable corpus.

    Used as the cache key. Includes both embedded and unembedded public
    success jobs so backfill progress (or new submissions while embeddings
    are off) reliably invalidates the cached corpus payload.

    Args:
        session: Active SQLAlchemy session.
        je_rel: SQL relation to use for the ``job_embeddings`` join (the real
            table, or the empty stand-in when it does not exist).

    Returns:
        ``"<embedded>|<embedded_max_ts>|<unembedded>|<unembedded_max_ts>"``
        — opaque but compact.
    """
    embedded = (
        session.execute(
            text(
                "SELECT COUNT(*) AS n, MAX(je.created_at) AS max_ts "
                f"FROM {je_rel} je "
                "INNER JOIN jobs j ON j.optimization_id = je.optimization_id "
                "WHERE j.status = 'success' "
                "AND je.embedding_summary IS NOT NULL AND je.is_private = FALSE"
            )
        )
        .mappings()
        .first()
    )
    unembedded = (
        session.execute(
            text(
                "SELECT COUNT(*) AS n, MAX(j.created_at) AS max_ts "
                "FROM jobs j "
                f"LEFT JOIN {je_rel} je ON je.optimization_id = j.optimization_id "
                "WHERE j.status = 'success' "
                "AND (je.optimization_id IS NULL OR je.embedding_summary IS NULL) "
                "AND NOT COALESCE((j.payload_overview->>'is_private')::boolean, FALSE)"
            )
        )
        .mappings()
        .first()
    )
    e_n = int(embedded["n"]) if embedded else 0
    e_ts = embedded["max_ts"] if embedded else None
    u_n = int(unembedded["n"]) if unembedded else 0
    u_ts = unembedded["max_ts"] if unembedded else None
    return (
        f"{e_n}|{e_ts.isoformat() if e_ts else 'none'}|"
        f"{u_n}|{u_ts.isoformat() if u_ts else 'none'}"
    )


def _as_float(value: Any) -> float | None:
    """Best-effort coerce ``value`` to ``float``; return ``None`` on ``None`` or parse failure.

    Args:
        value: Anything ``float()`` might accept.

    Returns:
        The parsed float, or ``None`` if conversion fails.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fetch_corpus_points(session: Session, je_rel: str) -> list[dict[str, Any]]:
    """Return every public success-state job as a corpus point.

    Drives the /explore list view's corpus count, filters, and
    model/optimizer options. Both embedded and unembedded jobs are
    included via a ``LEFT JOIN`` that prefers the embedding row's values
    and falls back to ``payload_overview`` for jobs not yet embedded.

    Args:
        session: An open SQLAlchemy session bound to the job-store engine.
        je_rel: SQL relation to use for the ``job_embeddings`` join (the real
            table, or the empty stand-in when it does not exist).

    Returns:
        A list of point dicts carrying the metadata the /explore payload
        exposes; heavy fields (signature_code, optimizer_kwargs,
        metric_name) are omitted.
    """
    rows = (
        session.execute(
            text(
                "SELECT j.optimization_id, "
                "COALESCE(je.optimization_type, j.optimization_type) AS optimization_type, "
                f"COALESCE(je.winning_model, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODEL_NAME}') "
                "AS winning_model, "
                "je.baseline_metric, je.optimized_metric, "
                "je.summary_text, "
                f"COALESCE(j.payload_overview->>'{PAYLOAD_OVERVIEW_NAME}', je.task_name) AS task_name, "
                f"COALESCE(je.module_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODULE_NAME}') "
                "AS module_name, "
                f"COALESCE(je.optimizer_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_OPTIMIZER_NAME}') "
                "AS optimizer_name, "
                "j.created_at, "
                f"j.payload_overview->>'{PAYLOAD_OVERVIEW_DESCRIPTION}' AS task_description "
                "FROM jobs j "
                f"LEFT JOIN {je_rel} je ON je.optimization_id = j.optimization_id "
                "WHERE j.status = 'success' "
                "AND NOT COALESCE(je.is_private, "
                "(j.payload_overview->>'is_private')::boolean, FALSE) "
                "ORDER BY j.created_at DESC, j.optimization_id DESC "
                f"LIMIT {MAX_POINTS}"
            )
        )
        .mappings()
        .all()
    )
    points: list[dict[str, Any]] = []
    for row in rows:
        summary = row["summary_text"] or row.get("task_description")
        if isinstance(summary, str) and len(summary) > SUMMARY_TEXT_MAX:
            summary = summary[:SUMMARY_TEXT_MAX].rstrip() + "…"
        points.append(
            {
                "optimization_id": row["optimization_id"],
                "optimization_type": row["optimization_type"],
                "winning_model": row["winning_model"],
                "baseline_metric": _as_float(row["baseline_metric"]),
                "optimized_metric": _as_float(row["optimized_metric"]),
                "summary_text": summary,
                "task_name": row["task_name"],
                "module_name": row["module_name"],
                "optimizer_name": row["optimizer_name"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "siblings": [],
                "task_fingerprint": None,
                "compare_fingerprint": None,
            }
        )
    return points


def fetch_public_dashboard(*, job_store: Any) -> dict[str, Any]:
    """Return the public corpus point list for ``GET /dashboard/public``.

    Cached by content fingerprint with a 5 min TTL so the corpus count,
    filters, and model/optimizer options the /explore list view derives
    aren't recomputed per request.

    Args:
        job_store: A store exposing a SQLAlchemy ``engine`` attribute.

    Returns:
        ``{"points": [...]}`` — one entry per public success-state job.
    """
    engine = job_store.engine
    je_rel = _job_embeddings_relation(job_store)
    with Session(engine) as session:
        fingerprint = _fetch_fingerprint(session, je_rel)
        now = time.time()
        with _LOCK:
            cached = _CACHE
            if (
                cached["fingerprint"] == fingerprint
                and cached["payload"] is not None
                and now - float(cached["at"]) < _CACHE_TTL_SECONDS
            ):
                return cached["payload"]

        payload = {"points": _fetch_corpus_points(session, je_rel)}
        with _LOCK:
            _CACHE["fingerprint"] = fingerprint
            _CACHE["at"] = now
            _CACHE["payload"] = payload
        return payload


def invalidate_public_dashboard_cache() -> None:
    """Force the next ``fetch_public_dashboard`` call to recompute."""
    with _LOCK:
        _CACHE["fingerprint"] = None
        _CACHE["at"] = 0.0
        _CACHE["payload"] = None


def fetch_corpus_facets(
    *,
    job_store: Any,
    owner_username: str | None = None,
    shared_with_username: str | None = None,
) -> dict[str, list[str]]:
    """Return the distinct model / optimizer / module values in one corpus.

    Backs ``GET /dashboard/facets`` so each /explore tab lists the filter
    options drawn from its OWN scope — the mine tab surfaces a user's private
    react runs, not just whatever appears in the public archive. The scope
    predicate and the payload-first / embedded-first ``COALESCE`` derivation
    mirror :func:`_fetch_corpus_points` and :func:`_search_semantic` exactly,
    so every value returned here lines up with a run the same scope can
    actually filter to.

    Args:
        job_store: A store exposing a SQLAlchemy ``engine`` attribute.
        owner_username: When set, scope to that user's own jobs (including
            private rows) instead of the public corpus.
        shared_with_username: When set (and ``owner_username`` is not), scope to
            jobs shared with that user via a member grant.

    Returns:
        ``{"models": [...], "optimizers": [...], "modules": [...]}`` — each a
        case-sensitively sorted list of distinct non-empty values.
    """
    params: dict[str, Any] = {}
    if owner_username is not None:
        scope_sql = "j.username = :owner_username"
        params["owner_username"] = owner_username
    elif shared_with_username is not None:
        scope_sql = _SHARED_GRANT_SCOPE_SQL
        params["shared_with_username"] = shared_with_username
    else:
        scope_sql = (
            "NOT COALESCE(je.is_private, "
            "(j.payload_overview->>'is_private')::boolean, FALSE)"
        )
    je_rel = _job_embeddings_relation(job_store)
    with Session(job_store.engine) as session:
        row = (
            session.execute(
                text(
                    "SELECT "
                    "ARRAY_AGG(DISTINCT model) FILTER (WHERE model <> '') AS models, "
                    "ARRAY_AGG(DISTINCT optimizer) FILTER (WHERE optimizer <> '') "
                    "AS optimizers, "
                    "ARRAY_AGG(DISTINCT module) FILTER (WHERE module <> '') AS modules "
                    "FROM ("
                    "SELECT "
                    "COALESCE(je.winning_model, "
                    f"j.payload_overview->>'{PAYLOAD_OVERVIEW_MODEL_NAME}') AS model, "
                    "COALESCE(je.optimizer_name, "
                    f"j.payload_overview->>'{PAYLOAD_OVERVIEW_OPTIMIZER_NAME}') AS optimizer, "
                    "COALESCE(je.module_name, "
                    f"j.payload_overview->>'{PAYLOAD_OVERVIEW_MODULE_NAME}') AS module "
                    "FROM jobs j "
                    f"LEFT JOIN {je_rel} je "
                    "ON je.optimization_id = j.optimization_id "
                    f"WHERE j.status = 'success' AND {scope_sql}"
                    ") sub"
                ),
                params,
            )
            .mappings()
            .first()
        )
    return {
        "models": sorted(row["models"] or []) if row else [],
        "optimizers": sorted(row["optimizers"] or []) if row else [],
        "modules": sorted(row["modules"] or []) if row else [],
    }


SEARCH_SORT_RELEVANCE = "relevance"
SEARCH_SORT_RECENT = "recent"
SEARCH_SORT_GAIN = "gain"
SEARCH_SORTS = (SEARCH_SORT_RELEVANCE, SEARCH_SORT_RECENT, SEARCH_SORT_GAIN)

SEARCH_PAGE_SIZE_DEFAULT = 30
SEARCH_PAGE_SIZE_MAX = 50
SEARCH_MATCHED_IDS_CAP = 5_000

POPULAR_QUERIES_LIMIT_DEFAULT = 8
POPULAR_QUERIES_WINDOW_DAYS_DEFAULT = 30
_SEARCH_QUERY_LOG_MIN_LEN = 2
_SEARCH_QUERY_LOG_MAX_LEN = 200


def _normalize_query_for_log(query: str) -> str | None:
    """Normalize a query for trending storage, or None when it isn't worth logging.

    Lowercases, collapses internal whitespace, and caps length so trivially
    different spellings of the same search coalesce into one trending bucket.

    Args:
        query: The raw (already caller-trimmed) public query string.

    Returns:
        The normalized query, or ``None`` when it is shorter than
        :data:`_SEARCH_QUERY_LOG_MIN_LEN` and thus too noisy to count.
    """
    normalized = " ".join(query.split()).lower()[:_SEARCH_QUERY_LOG_MAX_LEN]
    if len(normalized) < _SEARCH_QUERY_LOG_MIN_LEN:
        return None
    return normalized


def record_public_search_query(job_store: Any, query: str) -> None:
    """Record one public search query for trending, best-effort.

    Called only on an explicit commit (Enter or opening a result) via
    ``POST /dashboard/search/log`` — never on every debounced keystroke — so
    half-typed prefixes don't pollute the trending counts. Writes a single
    anonymous row to ``search_query_log`` and never raises: logging is a side
    effect and must not break the caller when the store has no engine (test
    stubs) or the insert fails.

    Args:
        job_store: Job store exposing a SQLAlchemy ``engine`` attribute.
        query: The public query to record (normalized internally).
    """
    normalized = _normalize_query_for_log(query)
    if normalized is None:
        return
    try:
        with Session(job_store.engine) as session:
            session.execute(
                text(
                    "INSERT INTO search_query_log (query_text, created_at) "
                    "VALUES (:q, :ts)"
                ).bindparams(bindparam("ts", type_=DateTime(timezone=True))),
                {"q": normalized, "ts": datetime.now(UTC)},
            )
            session.commit()
    except Exception as exc:
        logger.debug("search query logging skipped: %s", exc)


def fetch_popular_queries(
    job_store: Any,
    *,
    limit: int = POPULAR_QUERIES_LIMIT_DEFAULT,
    window_days: int = POPULAR_QUERIES_WINDOW_DAYS_DEFAULT,
) -> list[dict[str, Any]]:
    """Return the most frequent public search queries over a recent window.

    Aggregates ``search_query_log`` into a top-N ranking by occurrence count,
    most popular first. Best-effort: returns an empty list when the store has
    no engine or the aggregate fails, so a missing/empty log never breaks the
    /explore zero-state (the UI falls back to corpus-frequency terms).

    Args:
        job_store: Job store exposing a SQLAlchemy ``engine`` attribute.
        limit: Maximum number of queries to return (clamped to ``[1, 50]``).
        window_days: Only count queries logged within this many days.

    Returns:
        A list of ``{"query": str, "count": int}`` dicts, ranked by count
        descending then query text.
    """
    limit = max(1, min(50, limit))
    cutoff = datetime.now(UTC) - timedelta(days=max(1, window_days))
    try:
        with Session(job_store.engine) as session:
            rows = session.execute(
                text(
                    "SELECT query_text, COUNT(*) AS n "
                    "FROM search_query_log "
                    "WHERE created_at >= :cutoff "
                    "GROUP BY query_text "
                    "ORDER BY n DESC, query_text ASC "
                    "LIMIT :limit"
                ).bindparams(bindparam("cutoff", type_=DateTime(timezone=True))),
                {"cutoff": cutoff, "limit": limit},
            ).all()
    except Exception as exc:
        logger.debug("popular query fetch failed: %s", exc)
        return []
    return [{"query": row[0], "count": int(row[1])} for row in rows]


def _vector_literal(vector: list[float]) -> str:
    """Format a Python float list as a pgvector text literal.

    Args:
        vector: The query embedding as a list of floats.

    Returns:
        The pgvector ``"[v1,v2,...]"`` literal — pgvector parses this on input.
    """
    return "[" + ",".join(f"{v:.7f}" for v in vector) + "]"


def search_optimizations(
    *,
    job_store: Any,
    query: str | None,
    models: list[str] | None = None,
    optimizers: list[str] | None = None,
    optimization_types: list[str] | None = None,
    tasks: list[str] | None = None,
    modules: list[str] | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    sort: str = SEARCH_SORT_RELEVANCE,
    page: int = 1,
    size: int = SEARCH_PAGE_SIZE_DEFAULT,
    owner_username: str | None = None,
    shared_with_username: str | None = None,
) -> dict[str, Any]:
    """Search the optimization corpus, semantic when possible, lexical otherwise.

    Dispatch rules:

    * ``embeddings_enabled`` is off → lexical.
    * The corpus contains any success-state job that lacks a summary embedding
      → lexical (so partially-embedded corpora stay fully searchable instead
      of silently hiding the unembedded rows).
    * A query is supplied but the embedder can't encode it → lexical.
    * Otherwise → semantic (pgvector cosine similarity).

    Both paths support the same structured filters, paging, and sort modes.
    Lexical results have ``relevance = None`` since there's no continuous
    similarity score to surface.

    Args:
        job_store: Job store exposing the SQLAlchemy ``engine`` attribute.
        query: Free-text query (embedded server-side when semantic) or ``None``.
        models: Optional model whitelist (matches embedded ``winning_model`` or
            the payload-overview model name for unembedded jobs).
        optimizers: Optional optimizer whitelist.
        optimization_types: Optional ``optimization_type`` whitelist.
        tasks: Optional ``task_name`` whitelist (matched against the same
            payload-overview-first COALESCE the corpus options derive from).
        modules: Optional ``module_name`` whitelist (matched against the same
            embedded-first COALESCE the corpus options derive from).
        date_from: Inclusive lower bound on ``created_at`` (date precision).
        date_to: Inclusive upper bound on ``created_at`` (date precision).
        sort: One of :data:`SEARCH_SORTS`.
        page: 1-indexed page number.
        size: Page size; clamped to ``[1, SEARCH_PAGE_SIZE_MAX]``.
        owner_username: When set, scope the search to jobs owned by this user
            (including their private rows) instead of the public corpus. The
            caller is responsible for verifying the requested owner matches the
            authenticated session.
        shared_with_username: When set (and ``owner_username`` is not), scope the
            search to jobs shared with this user via a member grant — runs they
            were invited to but do not own, including private ones the grant
            authorizes. The caller verifies the requested user is the session.

    Returns:
        ``{"results": [...], "total": int, "matched_ids": [...], "search_type": str}``,
        where ``search_type`` is ``"semantic"`` or ``"lexical"`` depending on
        which dispatch branch served the query.
    """
    if sort not in SEARCH_SORTS:
        sort = SEARCH_SORT_RELEVANCE
    page = max(1, page)
    size = max(1, min(SEARCH_PAGE_SIZE_MAX, size))

    je_rel = _job_embeddings_relation(job_store)
    query_clean = (query or "").strip()

    use_lexical = not settings.embeddings_enabled
    query_vector: list[float] | None = None

    if not use_lexical:
        if _has_unembedded_success_jobs(
            job_store,
            owner_username=owner_username,
            shared_with_username=shared_with_username,
        ):
            use_lexical = True
        elif query_clean:
            query_vector = get_embedder().encode(query_clean, task="retrieval.query")
            if query_vector is None:
                logger.info("search_optimizations: query embedding unavailable, using lexical")
                use_lexical = True

    if use_lexical:
        # BM25 ranks by relevance, so it only serves the relevance sort with a
        # query present; explicit gain/recent sorts keep the ILIKE path's
        # ordering. Any pg_search failure degrades to the ILIKE search below.
        if (
            query_clean
            and sort == SEARCH_SORT_RELEVANCE
            and settings.search_bm25_enabled
            and getattr(job_store, "bm25_search_enabled", False)
        ):
            try:
                return _search_bm25(
                    job_store=job_store,
                    je_rel=je_rel,
                    query=query_clean,
                    models=models,
                    optimizers=optimizers,
                    optimization_types=optimization_types,
                    tasks=tasks,
                    modules=modules,
                    date_from=date_from,
                    date_to=date_to,
                    page=page,
                    size=size,
                    owner_username=owner_username,
                    shared_with_username=shared_with_username,
                )
            except SQLAlchemyError as exc:
                logger.warning(
                    "BM25 search failed (%s); falling back to ILIKE lexical search.", exc
                )
        return _search_lexical(
            job_store=job_store,
            je_rel=je_rel,
            query=query_clean,
            models=models,
            optimizers=optimizers,
            optimization_types=optimization_types,
            tasks=tasks,
            modules=modules,
            date_from=date_from,
            date_to=date_to,
            sort=sort,
            page=page,
            size=size,
            owner_username=owner_username,
            shared_with_username=shared_with_username,
        )

    return _search_semantic(
        job_store=job_store,
        query_vector=query_vector,
        models=models,
        optimizers=optimizers,
        optimization_types=optimization_types,
        tasks=tasks,
        modules=modules,
        date_from=date_from,
        date_to=date_to,
        sort=sort,
        page=page,
        size=size,
        owner_username=owner_username,
        shared_with_username=shared_with_username,
    )


def _has_unembedded_success_jobs(
    job_store: Any,
    *,
    owner_username: str | None = None,
    shared_with_username: str | None = None,
) -> bool:
    """Return True if any in-scope successful job lacks a summary embedding row.

    Cheap ``LIMIT 1`` probe used to decide whether the search dispatcher
    should fall back to lexical matching. The query is index-friendly
    (``jobs.status`` is indexed) and short-circuits as soon as one
    qualifying row is found.

    Args:
        job_store: Job store exposing the SQLAlchemy ``engine`` attribute.
        owner_username: When set, restrict the probe to that user's jobs so a
            mine-corpus query isn't downgraded to lexical because some other
            user has unembedded rows.
        shared_with_username: When set (and ``owner_username`` is not), restrict
            the probe to jobs shared with that user via a member grant.

    Returns:
        True when at least one in-scope success-state job has no embedding,
        False otherwise (including on transient query failure — we prefer
        the semantic path to a hard error).
    """
    params: dict[str, Any] = {}
    if owner_username is not None:
        scope_sql = "j.username = :owner_username"
        params["owner_username"] = owner_username
    elif shared_with_username is not None:
        scope_sql = _SHARED_GRANT_SCOPE_SQL
        params["shared_with_username"] = shared_with_username
    else:
        scope_sql = "NOT COALESCE((j.payload_overview->>'is_private')::boolean, FALSE)"
    je_rel = _job_embeddings_relation(job_store)
    try:
        with Session(job_store.engine) as session:
            row = session.execute(
                text(
                    "SELECT 1 FROM jobs j "
                    f"LEFT JOIN {je_rel} je ON je.optimization_id = j.optimization_id "
                    "WHERE j.status = 'success' "
                    "AND (je.optimization_id IS NULL OR je.embedding_summary IS NULL) "
                    f"AND {scope_sql} "
                    "LIMIT 1"
                ),
                params,
            ).first()
            return row is not None
    except Exception as exc:
        logger.warning("Unembedded-job probe failed, assuming all embedded: %s", exc)
        return False


def _search_semantic(
    *,
    job_store: Any,
    query_vector: list[float] | None,
    models: list[str] | None,
    optimizers: list[str] | None,
    optimization_types: list[str] | None,
    tasks: list[str] | None,
    modules: list[str] | None,
    date_from: date | None,
    date_to: date | None,
    sort: str,
    page: int,
    size: int,
    owner_username: str | None = None,
    shared_with_username: str | None = None,
) -> dict[str, Any]:
    """Rank the embedded corpus by pgvector cosine similarity (or recency / gain).

    Args:
        job_store: Job store exposing the SQLAlchemy ``engine`` attribute.
        query_vector: Encoded query vector, or ``None`` when no probe is given.
        models: Optional ``winning_model`` whitelist.
        optimizers: Optional ``optimizer_name`` whitelist.
        optimization_types: Optional ``optimization_type`` whitelist.
        tasks: Optional ``task_name`` whitelist.
        modules: Optional ``module_name`` whitelist.
        date_from: Inclusive lower bound on ``created_at``.
        date_to: Inclusive upper bound on ``created_at``.
        sort: One of :data:`SEARCH_SORTS`.
        page: 1-indexed page number.
        size: Page size (already clamped).
        owner_username: When set, scope to that user (including private rows)
            instead of the public corpus.
        shared_with_username: When set (and ``owner_username`` is not), scope to
            jobs shared with that user via a member grant.

    Returns:
        ``{"results": [...], "total": int, "matched_ids": [...], "search_type": "semantic"}``.
    """
    use_similarity = query_vector is not None and sort == SEARCH_SORT_RELEVANCE

    # INNER JOIN with jobs so deleted/orphan embedding rows can't leak through,
    # and so the status filter is enforced regardless of how the embedding row
    # was written.
    from_sql = (
        "FROM job_embeddings je "
        "INNER JOIN jobs j ON j.optimization_id = je.optimization_id"
    )
    where_parts: list[str] = [
        "j.status = 'success'",
        "je.embedding_summary IS NOT NULL",
    ]
    params: dict[str, Any] = {}
    if owner_username is not None:
        where_parts.append("j.username = :owner_username")
        params["owner_username"] = owner_username
    elif shared_with_username is not None:
        where_parts.append(_SHARED_GRANT_SCOPE_SQL)
        params["shared_with_username"] = shared_with_username
    else:
        where_parts.append("je.is_private = FALSE")
    if models:
        where_parts.append("je.winning_model = ANY(:models)")
        params["models"] = list(models)
    if optimizers:
        where_parts.append("je.optimizer_name = ANY(:optimizers)")
        params["optimizers"] = list(optimizers)
    if optimization_types:
        where_parts.append("je.optimization_type = ANY(:optimization_types)")
        params["optimization_types"] = list(optimization_types)
    # Match the same payload-first / embedded-first COALESCE the corpus options
    # derive from (``_fetch_corpus_points``) so a filter value always lines up
    # with the chip the user picked, even for jobs renamed after embedding.
    if tasks:
        where_parts.append(
            f"COALESCE(j.payload_overview->>'{PAYLOAD_OVERVIEW_NAME}', je.task_name) "
            "= ANY(:tasks)"
        )
        params["tasks"] = list(tasks)
    if modules:
        where_parts.append(
            f"COALESCE(je.module_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODULE_NAME}') "
            "= ANY(:modules)"
        )
        params["modules"] = list(modules)
    if date_from is not None:
        where_parts.append("je.created_at >= :date_from")
        params["date_from"] = date_from
    if date_to is not None:
        where_parts.append("je.created_at < :date_to_excl")
        params["date_to_excl"] = date_to + timedelta(days=1)

    where_sql = " AND ".join(where_parts)

    if use_similarity:
        params["query_vec"] = _vector_literal(query_vector)  # type: ignore[arg-type]
        order_sql = "je.embedding_summary <=> CAST(:query_vec AS vector) ASC, je.created_at DESC"
        relevance_sql = "1 - (je.embedding_summary <=> CAST(:query_vec AS vector))"
    elif sort == SEARCH_SORT_GAIN:
        # Plain ``optimized - baseline`` (not COALESCE-to-0): a row missing
        # either metric yields NULL and sinks via NULLS LAST, rather than a
        # baseline-less run posing as a gain equal to its raw optimized score.
        order_sql = (
            "(je.optimized_metric - je.baseline_metric) DESC NULLS LAST, "
            "je.created_at DESC"
        )
        relevance_sql = "NULL::float"
    else:
        order_sql = "je.created_at DESC, je.optimization_id DESC"
        relevance_sql = "NULL::float"

    engine = job_store.engine
    with Session(engine) as session:
        # Pull every match in rank order up to the id cap so total and
        # matched_ids cover the full result set, then page in Python.
        ranked_rows = (
            session.execute(
                text(
                    "SELECT je.optimization_id, j.payload_overview, "
                    f"{relevance_sql} AS relevance "
                    f"{from_sql} "
                    f"WHERE {where_sql} "
                    f"ORDER BY {order_sql} "
                    "LIMIT :ids_cap"
                ),
                {**params, "ids_cap": SEARCH_MATCHED_IDS_CAP},
            )
            .mappings()
            .all()
        )

        leaders = [row["optimization_id"] for row in ranked_rows]
        relevance_by_id = {
            row["optimization_id"]: _as_float(row.get("relevance")) for row in ranked_rows
        }
        total = len(leaders)
        offset = (page - 1) * size
        page_ids = leaders[offset : offset + size]

        page_rows: list[Mapping[str, Any]] = []
        if page_ids:
            # COALESCE the user-renamable fields against payload_overview so a
            # job that was renamed after embedding shows the current name in
            # the search results, not the stale embedded snapshot.
            page_rows = (
                session.execute(
                    text(
                        "SELECT je.optimization_id, je.optimization_type, je.winning_model, "
                        "je.baseline_metric, je.optimized_metric, je.summary_text, "
                        f"COALESCE(j.payload_overview->>'{PAYLOAD_OVERVIEW_NAME}', je.task_name) AS task_name, "
                        "je.module_name, je.optimizer_name, je.created_at "
                        f"{from_sql} "
                        "WHERE je.optimization_id = ANY(:page_ids)"
                    ),
                    {"page_ids": page_ids},
                )
                .mappings()
                .all()
            )

    by_id = {row["optimization_id"]: row for row in page_rows}
    results: list[dict[str, Any]] = []
    for opt_id in page_ids:
        row = by_id.get(opt_id)
        if row is None:
            continue
        summary = row["summary_text"]
        if isinstance(summary, str) and len(summary) > SUMMARY_TEXT_MAX:
            summary = summary[:SUMMARY_TEXT_MAX].rstrip() + "…"
        results.append(
            {
                "optimization_id": opt_id,
                "optimization_type": row["optimization_type"],
                "winning_model": row["winning_model"],
                "baseline_metric": _as_float(row["baseline_metric"]),
                "optimized_metric": _as_float(row["optimized_metric"]),
                "summary_text": summary,
                "task_name": row["task_name"],
                "module_name": row["module_name"],
                "optimizer_name": row["optimizer_name"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "relevance": relevance_by_id.get(opt_id),
            }
        )
    return {
        "results": results,
        "total": total,
        "matched_ids": leaders,
        "search_type": "semantic",
    }


# Lexical text matched against the union of these fields. The COALESCE order
# matters: embedded fields are authoritative (post-optimization winners,
# canonical summaries) and fall back to payload_overview values for jobs
# that haven't been embedded yet.
_LEXICAL_HAYSTACK_SQL = (
    "lower(coalesce("
    "  coalesce(j.payload_overview->>'name', je.task_name, '') || ' ' || "
    "  coalesce(je.summary_text, '') || ' ' || "
    "  coalesce(j.payload_overview->>'description', '') || ' ' || "
    "  coalesce(je.optimizer_name, j.payload_overview->>'optimizer_name', '') || ' ' || "
    "  coalesce(je.winning_model, j.payload_overview->>'model_name', '') || ' ' || "
    "  coalesce(je.module_name, j.payload_overview->>'module_name', ''), "
    "''))"
)


def _lexical_tokens(query: str) -> list[str]:
    """Split a free-text query into searchable lowercase tokens.

    Single-character tokens are dropped — they're either accidental
    whitespace artifacts or too noisy to be useful for ILIKE matching at
    the corpus sizes we expect.

    Args:
        query: Raw query string from the request.

    Returns:
        Lowercase tokens, deduped while preserving first-seen order.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in query.split():
        token = raw.strip().lower()
        if len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _search_lexical(
    *,
    job_store: Any,
    je_rel: str,
    query: str,
    models: list[str] | None,
    optimizers: list[str] | None,
    optimization_types: list[str] | None,
    tasks: list[str] | None,
    modules: list[str] | None,
    date_from: date | None,
    date_to: date | None,
    sort: str,
    page: int,
    size: int,
    owner_username: str | None = None,
    shared_with_username: str | None = None,
) -> dict[str, Any]:
    """Lexical ILIKE search across the corpus.

    Walks ``jobs LEFT JOIN job_embeddings`` so unembedded successful jobs
    are still returned — their text comes from ``payload_overview`` rather
    than the LLM-authored summary, and structured filters fall back to the
    payload values when the embedding row is missing.

    The relevance sort is degraded to recency, since lexical matching has
    no continuous similarity score and emitting a synthetic one would be
    misleading on the result badge.

    Args:
        job_store: Job store exposing the SQLAlchemy ``engine`` attribute.
        je_rel: SQL relation to use for the ``job_embeddings`` join (the real
            table, or the empty stand-in when it does not exist).
        query: Pre-trimmed query string (empty string allowed).
        models: Optional model whitelist.
        optimizers: Optional optimizer whitelist.
        optimization_types: Optional ``optimization_type`` whitelist.
        tasks: Optional ``task_name`` whitelist.
        modules: Optional ``module_name`` whitelist.
        date_from: Inclusive lower bound on ``created_at``.
        date_to: Inclusive upper bound on ``created_at``.
        sort: One of :data:`SEARCH_SORTS` (relevance is treated as recent).
        page: 1-indexed page number.
        size: Page size (already clamped).
        owner_username: When set, scope to that user (including private rows)
            instead of the public corpus.
        shared_with_username: When set (and ``owner_username`` is not), scope to
            jobs shared with that user via a member grant.

    Returns:
        ``{"results": [...], "total": int, "matched_ids": [...], "search_type": "lexical"}``.
    """
    where_parts: list[str] = ["j.status = 'success'"]
    params: dict[str, Any] = {}
    if owner_username is not None:
        where_parts.append("j.username = :owner_username")
        params["owner_username"] = owner_username
    elif shared_with_username is not None:
        where_parts.append(_SHARED_GRANT_SCOPE_SQL)
        params["shared_with_username"] = shared_with_username
    else:
        # Private flag may live in the embedding row OR the payload overview
        # (for jobs that haven't been embedded yet). Treat either as private.
        where_parts.append(
            "NOT COALESCE(je.is_private, "
            "(j.payload_overview->>'is_private')::boolean, FALSE)"
        )

    if models:
        where_parts.append(
            f"COALESCE(je.winning_model, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODEL_NAME}') "
            "= ANY(:models)"
        )
        params["models"] = list(models)
    if optimizers:
        where_parts.append(
            f"COALESCE(je.optimizer_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_OPTIMIZER_NAME}') "
            "= ANY(:optimizers)"
        )
        params["optimizers"] = list(optimizers)
    if optimization_types:
        where_parts.append(
            "COALESCE(je.optimization_type, j.optimization_type) = ANY(:optimization_types)"
        )
        params["optimization_types"] = list(optimization_types)
    # Same COALESCE expressions as the corpus options (``_fetch_corpus_points``)
    # so a picked chip's value always matches, including for unembedded jobs.
    if tasks:
        where_parts.append(
            f"COALESCE(j.payload_overview->>'{PAYLOAD_OVERVIEW_NAME}', je.task_name) "
            "= ANY(:tasks)"
        )
        params["tasks"] = list(tasks)
    if modules:
        where_parts.append(
            f"COALESCE(je.module_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODULE_NAME}') "
            "= ANY(:modules)"
        )
        params["modules"] = list(modules)
    if date_from is not None:
        where_parts.append("j.created_at >= :date_from")
        params["date_from"] = date_from
    if date_to is not None:
        where_parts.append("j.created_at < :date_to_excl")
        params["date_to_excl"] = date_to + timedelta(days=1)

    tokens = _lexical_tokens(query)
    for idx, token in enumerate(tokens):
        param_name = f"tok_{idx}"
        where_parts.append(f"{_LEXICAL_HAYSTACK_SQL} LIKE :{param_name}")
        params[param_name] = f"%{token}%"

    where_sql = " AND ".join(where_parts)

    if sort == SEARCH_SORT_GAIN:
        # Plain ``optimized - baseline`` (not COALESCE-to-0): a row missing
        # either metric yields NULL and sinks via NULLS LAST, rather than a
        # baseline-less run posing as a gain equal to its raw optimized score.
        order_sql = (
            "(je.optimized_metric - je.baseline_metric) DESC NULLS LAST, "
            "j.created_at DESC, j.optimization_id DESC"
        )
    else:
        order_sql = "j.created_at DESC, j.optimization_id DESC"

    select_cols = (
        "j.optimization_id, "
        "COALESCE(je.optimization_type, j.optimization_type) AS optimization_type, "
        f"COALESCE(je.winning_model, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODEL_NAME}') "
        "AS winning_model, "
        "je.baseline_metric, je.optimized_metric, "
        "je.summary_text, "
        f"COALESCE(je.task_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_NAME}') AS task_name, "
        f"COALESCE(je.module_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODULE_NAME}') "
        "AS module_name, "
        f"COALESCE(je.optimizer_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_OPTIMIZER_NAME}') "
        "AS optimizer_name, "
        "j.created_at, "
        f"j.payload_overview->>'{PAYLOAD_OVERVIEW_DESCRIPTION}' AS task_description"
    )

    engine = job_store.engine
    with Session(engine) as session:
        # Pull every match in rank order up to the id cap so total and
        # matched_ids cover the full result set, then page in Python.
        ranked_rows = (
            session.execute(
                text(
                    "SELECT j.optimization_id, j.payload_overview, "
                    "NULL::float AS relevance "
                    "FROM jobs j "
                    f"LEFT JOIN {je_rel} je ON je.optimization_id = j.optimization_id "
                    f"WHERE {where_sql} "
                    f"ORDER BY {order_sql} "
                    "LIMIT :ids_cap"
                ),
                {**params, "ids_cap": SEARCH_MATCHED_IDS_CAP},
            )
            .mappings()
            .all()
        )

        leaders = [row["optimization_id"] for row in ranked_rows]
        total = len(leaders)
        offset = (page - 1) * size
        page_ids = leaders[offset : offset + size]

        page_rows: list[Mapping[str, Any]] = []
        if page_ids:
            page_rows = (
                session.execute(
                    text(
                        f"SELECT {select_cols} FROM jobs j "
                        f"LEFT JOIN {je_rel} je ON je.optimization_id = j.optimization_id "
                        "WHERE j.optimization_id = ANY(:page_ids)"
                    ),
                    {"page_ids": page_ids},
                )
                .mappings()
                .all()
            )

    by_id = {row["optimization_id"]: row for row in page_rows}
    results: list[dict[str, Any]] = []
    for opt_id in page_ids:
        row = by_id.get(opt_id)
        if row is None:
            continue
        # Prefer the LLM-authored summary; fall back to the user-supplied
        # task description so unembedded rows still have something to render
        # in the result snippet.
        summary = row["summary_text"] or row.get("task_description")
        if isinstance(summary, str) and len(summary) > SUMMARY_TEXT_MAX:
            summary = summary[:SUMMARY_TEXT_MAX].rstrip() + "…"
        results.append(
            {
                "optimization_id": opt_id,
                "optimization_type": row["optimization_type"],
                "winning_model": row["winning_model"],
                "baseline_metric": _as_float(row["baseline_metric"]),
                "optimized_metric": _as_float(row["optimized_metric"]),
                "summary_text": summary,
                "task_name": row["task_name"],
                "module_name": row["module_name"],
                "optimizer_name": row["optimizer_name"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "relevance": None,
            }
        )
    return {
        "results": results,
        "total": total,
        "matched_ids": leaders,
        "search_type": "lexical",
    }


def _search_bm25(
    *,
    job_store: Any,
    je_rel: str,
    query: str,
    models: list[str] | None,
    optimizers: list[str] | None,
    optimization_types: list[str] | None,
    tasks: list[str] | None,
    modules: list[str] | None,
    date_from: date | None,
    date_to: date | None,
    page: int,
    size: int,
    owner_username: str | None = None,
    shared_with_username: str | None = None,
) -> dict[str, Any]:
    """BM25-ranked lexical search over the jobs payload_overview corpus.

    Mirrors :func:`_search_lexical`'s structured filters and result assembly,
    but ranks by ParadeDB ``paradedb.score`` instead of recency and surfaces a
    real ``relevance`` per row. Requires the ``pg_search`` extension and the
    ``idx_jobs_bm25`` index (created best-effort at store init); the caller only
    routes here when ``job_store.bm25_search_enabled`` is true and wraps the call
    so any failure falls back to :func:`_search_lexical`.

    Args:
        job_store: Job store exposing the SQLAlchemy ``engine`` attribute.
        je_rel: SQL relation to use for the ``job_embeddings`` join (the real
            table, or the empty stand-in when it does not exist).
        query: Pre-trimmed, non-empty query string (the BM25 match text).
        models: Optional model whitelist.
        optimizers: Optional optimizer whitelist.
        optimization_types: Optional ``optimization_type`` whitelist.
        tasks: Optional ``task_name`` whitelist.
        modules: Optional ``module_name`` whitelist.
        date_from: Inclusive lower bound on ``created_at``.
        date_to: Inclusive upper bound on ``created_at``.
        page: 1-indexed page number.
        size: Page size (already clamped).
        owner_username: When set, scope to that user (including private rows).
        shared_with_username: When set (and ``owner_username`` is not), scope to
            jobs shared with that user via a member grant.

    Returns:
        ``{"results": [...], "total": int, "matched_ids": [...], "search_type": "bm25"}``.
    """
    where_parts: list[str] = ["j.status = 'success'"]
    params: dict[str, Any] = {}
    if owner_username is not None:
        where_parts.append("j.username = :owner_username")
        params["owner_username"] = owner_username
    elif shared_with_username is not None:
        where_parts.append(_SHARED_GRANT_SCOPE_SQL)
        params["shared_with_username"] = shared_with_username
    else:
        where_parts.append(
            "NOT COALESCE(je.is_private, "
            "(j.payload_overview->>'is_private')::boolean, FALSE)"
        )

    if models:
        where_parts.append(
            f"COALESCE(je.winning_model, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODEL_NAME}') "
            "= ANY(:models)"
        )
        params["models"] = list(models)
    if optimizers:
        where_parts.append(
            f"COALESCE(je.optimizer_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_OPTIMIZER_NAME}') "
            "= ANY(:optimizers)"
        )
        params["optimizers"] = list(optimizers)
    if optimization_types:
        where_parts.append(
            "COALESCE(je.optimization_type, j.optimization_type) = ANY(:optimization_types)"
        )
        params["optimization_types"] = list(optimization_types)
    if tasks:
        where_parts.append(
            f"COALESCE(j.payload_overview->>'{PAYLOAD_OVERVIEW_NAME}', je.task_name) "
            "= ANY(:tasks)"
        )
        params["tasks"] = list(tasks)
    if modules:
        where_parts.append(
            f"COALESCE(je.module_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODULE_NAME}') "
            "= ANY(:modules)"
        )
        params["modules"] = list(modules)
    if date_from is not None:
        where_parts.append("j.created_at >= :date_from")
        params["date_from"] = date_from
    if date_to is not None:
        where_parts.append("j.created_at < :date_to_excl")
        params["date_to_excl"] = date_to + timedelta(days=1)

    # BM25 match: @@@ against the indexed payload_overview corpus. paradedb.score
    # then ranks the matched rows. Both require the pg_search bm25 index on jobs.
    where_parts.append("j.payload_overview @@@ :bm25_query")
    params["bm25_query"] = query

    where_sql = " AND ".join(where_parts)

    select_cols = (
        "j.optimization_id, "
        "COALESCE(je.optimization_type, j.optimization_type) AS optimization_type, "
        f"COALESCE(je.winning_model, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODEL_NAME}') "
        "AS winning_model, "
        "je.baseline_metric, je.optimized_metric, "
        "je.summary_text, "
        f"COALESCE(je.task_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_NAME}') AS task_name, "
        f"COALESCE(je.module_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_MODULE_NAME}') "
        "AS module_name, "
        f"COALESCE(je.optimizer_name, j.payload_overview->>'{PAYLOAD_OVERVIEW_OPTIMIZER_NAME}') "
        "AS optimizer_name, "
        "j.created_at, "
        f"j.payload_overview->>'{PAYLOAD_OVERVIEW_DESCRIPTION}' AS task_description"
    )

    engine = job_store.engine
    with Session(engine) as session:
        ranked_rows = (
            session.execute(
                text(
                    "SELECT j.optimization_id, "
                    "paradedb.score(j.optimization_id) AS relevance "
                    "FROM jobs j "
                    f"LEFT JOIN {je_rel} je ON je.optimization_id = j.optimization_id "
                    f"WHERE {where_sql} "
                    "ORDER BY relevance DESC, j.created_at DESC, j.optimization_id DESC "
                    "LIMIT :ids_cap"
                ),
                {**params, "ids_cap": SEARCH_MATCHED_IDS_CAP},
            )
            .mappings()
            .all()
        )

        leaders = [row["optimization_id"] for row in ranked_rows]
        relevance_by_id = {
            row["optimization_id"]: _as_float(row.get("relevance")) for row in ranked_rows
        }
        total = len(leaders)
        offset = (page - 1) * size
        page_ids = leaders[offset : offset + size]

        page_rows: list[Mapping[str, Any]] = []
        if page_ids:
            page_rows = (
                session.execute(
                    text(
                        f"SELECT {select_cols} FROM jobs j "
                        f"LEFT JOIN {je_rel} je ON je.optimization_id = j.optimization_id "
                        "WHERE j.optimization_id = ANY(:page_ids)"
                    ),
                    {"page_ids": page_ids},
                )
                .mappings()
                .all()
            )

    by_id = {row["optimization_id"]: row for row in page_rows}
    results: list[dict[str, Any]] = []
    for opt_id in page_ids:
        row = by_id.get(opt_id)
        if row is None:
            continue
        summary = row["summary_text"] or row.get("task_description")
        if isinstance(summary, str) and len(summary) > SUMMARY_TEXT_MAX:
            summary = summary[:SUMMARY_TEXT_MAX].rstrip() + "…"
        results.append(
            {
                "optimization_id": opt_id,
                "optimization_type": row["optimization_type"],
                "winning_model": row["winning_model"],
                "baseline_metric": _as_float(row["baseline_metric"]),
                "optimized_metric": _as_float(row["optimized_metric"]),
                "summary_text": summary,
                "task_name": row["task_name"],
                "module_name": row["module_name"],
                "optimizer_name": row["optimizer_name"],
                "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                "relevance": relevance_by_id.get(opt_id),
            }
        )
    return {
        "results": results,
        "total": total,
        "matched_ids": leaders,
        "search_type": "bm25",
    }
