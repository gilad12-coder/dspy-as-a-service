"""Tests for the public-dashboard aggregator (PER-11 Feature B)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy import create_engine, event, text
from sqlalchemy.pool import StaticPool

from core.constants import OPTIMIZATION_TYPE_TAGGING
from core.service_gateway import dashboard
from core.storage.models import SearchQueryLogModel


def test_invalidate_public_dashboard_cache_resets_state() -> None:
    """``invalidate_public_dashboard_cache`` clears the fingerprint, timestamp, and payload."""
    dashboard._CACHE["fingerprint"] = "stale"
    dashboard._CACHE["at"] = 9e18
    dashboard._CACHE["payload"] = {"points": [{"optimization_id": "stale"}], "meta": {}}
    dashboard.invalidate_public_dashboard_cache()
    assert dashboard._CACHE["fingerprint"] is None
    assert dashboard._CACHE["at"] == 0.0
    assert dashboard._CACHE["payload"] is None


def test_fingerprint_includes_embedding_and_completion_freshness() -> None:
    """An in-place embedding refresh changes the public-dashboard fingerprint."""
    session = MagicMock(name="session")
    embedded = MagicMock(name="embedded-result")
    embedded.mappings.return_value.first.return_value = {
        "n": 2,
        "updated_max_ts": datetime(2026, 8, 2, 12, 0, tzinfo=UTC),
        "created_max_ts": datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
        "completed_max_ts": datetime(2026, 8, 2, 11, 0, tzinfo=UTC),
    }
    unembedded = MagicMock(name="unembedded-result")
    unembedded.mappings.return_value.first.return_value = {
        "n": 1,
        "completed_max_ts": datetime(2026, 8, 2, 10, 0, tzinfo=UTC),
        "created_max_ts": datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
    }
    session.execute.side_effect = [embedded, unembedded]

    fingerprint = dashboard._fetch_fingerprint(session, "job_embeddings")

    assert fingerprint == (
        "2|2026-08-02T12:00:00+00:00|2026-08-01T12:00:00+00:00|"
        "2026-08-02T11:00:00+00:00|1|2026-08-02T10:00:00+00:00|"
        "2026-08-01T10:00:00+00:00"
    )


def test_corpus_fetch_has_no_artificial_point_cap() -> None:
    """The public corpus query returns every matching success row."""
    session = MagicMock(name="session")
    session.execute.return_value.mappings.return_value.all.return_value = []

    assert dashboard._fetch_corpus_points(session, "job_embeddings") == []
    statement = session.execute.call_args.args[0]
    assert "LIMIT" not in statement.text.upper()


def test_user_facing_corpus_sql_keeps_legacy_rows_and_drops_internal_rows() -> None:
    """The corpus predicate admits user-facing rows and rejects internal ones.

    Executes the real SQL against an in-memory schema shaped like the columns
    the predicate reads. Legacy rows (no ``optimization_type`` recorded
    anywhere) must stay discoverable — the open/detail paths serve them — and
    tagger jobs and distributed grid-pair child rows must never surface.
    """
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE jobs (optimization_id TEXT, optimization_type TEXT, "
                "payload_overview TEXT, parent_optimization_id TEXT)"
            )
        )
        conn.execute(
            text("CREATE TABLE job_embeddings (optimization_id TEXT, optimization_type TEXT)")
        )
        conn.execute(
            text("INSERT INTO jobs VALUES (:id, :otype, :payload, :parent)"),
            [
                {"id": "legacy", "otype": None, "payload": None, "parent": None},
                {"id": "plain-run", "otype": "run", "payload": None, "parent": None},
                {"id": "grid-parent", "otype": "grid_search", "payload": None, "parent": None},
                {
                    "id": "payload-only",
                    "otype": None,
                    "payload": '{"optimization_type": "run"}',
                    "parent": None,
                },
                {
                    "id": "tagger",
                    "otype": OPTIMIZATION_TYPE_TAGGING,
                    "payload": None,
                    "parent": None,
                },
                {
                    "id": "grid-child",
                    "otype": "grid_search",
                    "payload": None,
                    "parent": "grid-parent",
                },
            ],
        )
        kept = (
            conn.execute(
                text(
                    "SELECT j.optimization_id FROM jobs j "
                    "LEFT JOIN job_embeddings je "
                    "ON je.optimization_id = j.optimization_id "
                    f"WHERE {dashboard._USER_FACING_CORPUS_SQL} "
                    "ORDER BY j.optimization_id"
                )
            )
            .scalars()
            .all()
        )
    assert kept == ["grid-parent", "legacy", "payload-only", "plain-run"]


def _sqlite_jsonb_typeof(value: str | None) -> str | None:
    """sqlite stand-in for Postgres ``jsonb_typeof`` over ``->`` output.

    sqlite's ``->`` yields the JSON text of the addressed value (or NULL), so
    parsing it and mapping Python types onto Postgres's type names is enough
    for the corpus metric fragment, which only compares against ``'number'``.
    """
    if value is None:
        return None
    parsed = json.loads(value)
    if isinstance(parsed, bool):
        return "boolean"
    if isinstance(parsed, (int, float)):
        return "number"
    return "other"


def test_corpus_metric_sql_falls_back_to_job_scores_for_gain_ranking() -> None:
    """Unembedded rows rank by their own job scores under the gain sort.

    Executes the real metric-fallback SQL against an in-memory schema. The
    embedded pair must win when present; otherwise runs read
    ``latest_metrics`` then ``result`` and grid jobs read
    ``result.best_pair``, mirroring the embedding pipeline's
    ``_extract_scores``. Rows with no numeric pair anywhere (including a
    malformed non-numeric value, which must not error) sink below every
    scored row and fall back to recency among themselves.
    """
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def _register_jsonb_typeof(dbapi_conn, _record) -> None:
        dbapi_conn.create_function("jsonb_typeof", 1, _sqlite_jsonb_typeof)

    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE jobs (optimization_id TEXT, payload_overview TEXT, "
                "latest_metrics TEXT, result TEXT, created_at TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE job_embeddings (optimization_id TEXT, "
                "baseline_metric REAL, optimized_metric REAL)"
            )
        )
        conn.execute(
            text("INSERT INTO jobs VALUES (:id, :overview, :metrics, :result, :created)"),
            [
                {
                    "id": "papillon",
                    "overview": None,
                    "metrics": '{"optimized_test_metric": 89.93}',
                    "result": '{"baseline_test_metric": 74.29, "optimized_test_metric": 89.93}',
                    "created": "2026-07-23",
                },
                {
                    "id": "flat",
                    "overview": None,
                    "metrics": '{"optimized_test_metric": 73.33}',
                    "result": '{"baseline_test_metric": 73.33}',
                    "created": "2026-07-24",
                },
                {
                    "id": "regressed",
                    "overview": None,
                    "metrics": '{"optimized_test_metric": 93.0}',
                    "result": '{"baseline_test_metric": 98.0}',
                    "created": "2026-07-26",
                },
                {
                    "id": "grid",
                    "overview": '{"optimization_type": "grid_search"}',
                    "metrics": None,
                    "result": (
                        '{"best_pair": {"baseline_test_metric": 50.0, '
                        '"optimized_test_metric": 60.0}}'
                    ),
                    "created": "2026-07-22",
                },
                {
                    "id": "embedded",
                    "overview": None,
                    "metrics": '{"optimized_test_metric": 11.0}',
                    "result": '{"baseline_test_metric": 10.0}',
                    "created": "2026-07-21",
                },
                {
                    "id": "scoreless",
                    "overview": None,
                    "metrics": None,
                    "result": None,
                    "created": "2026-07-25",
                },
                {
                    "id": "malformed",
                    "overview": None,
                    "metrics": '{"baseline_test_metric": 5.0, "optimized_test_metric": "n/a"}',
                    "result": None,
                    "created": "2026-07-27",
                },
            ],
        )
        conn.execute(
            text("INSERT INTO job_embeddings VALUES ('embedded', 10.0, 30.0)")
        )
        ranked = (
            conn.execute(
                text(
                    "SELECT j.optimization_id FROM jobs j "
                    "LEFT JOIN job_embeddings je "
                    "ON je.optimization_id = j.optimization_id "
                    f"ORDER BY ({dashboard._CORPUS_OPTIMIZED_METRIC_SQL} - "
                    f"{dashboard._CORPUS_BASELINE_METRIC_SQL}) DESC NULLS LAST, "
                    "j.created_at DESC"
                )
            )
            .scalars()
            .all()
        )
    assert ranked == [
        "embedded",
        "papillon",
        "grid",
        "flat",
        "regressed",
        "malformed",
        "scoreless",
    ]


def test_normalize_query_for_log_collapses_and_drops_short() -> None:
    """Normalization lowercases, collapses whitespace, and drops sub-2-char noise."""
    assert dashboard._normalize_query_for_log("  GPT-4o   MINI ") == "gpt-4o mini"
    assert dashboard._normalize_query_for_log("a") is None
    assert dashboard._normalize_query_for_log("   ") is None


def test_log_and_fetch_popular_queries_ranks_by_count() -> None:
    """Recorded public queries aggregate into a count-ranked trending list."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    SearchQueryLogModel.__table__.create(engine)
    store = SimpleNamespace(engine=engine)

    dashboard.record_public_search_query(store, "GPT-4o")
    dashboard.record_public_search_query(store, "  gpt-4o ")
    dashboard.record_public_search_query(store, "dspy MIPRO")

    popular = dashboard.fetch_popular_queries(store, limit=5, window_days=30)
    assert popular[0] == {"query": "gpt-4o", "count": 2}
    assert {"query": "dspy mipro", "count": 1} in popular


def test_job_embeddings_relation_degrades_when_table_absent() -> None:
    """The explore join falls back to an empty stand-in when job_embeddings is gone.

    Mirrors a plain Postgres where RemoteJobStore skipped the Vector tables: the
    presence probe finds nothing, so the corpus / facets / search SQL reads
    ``jobs`` alone instead of raising ``UndefinedTable`` (which the browser would
    surface as a "can't connect to the server" failure). The jobs-only query
    itself is Postgres-specific (``::`` casts, ``WHERE FALSE``) and is validated
    against a real table-less Postgres out-of-band, not here on SQLite.
    """
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    store = SimpleNamespace(engine=engine)
    dashboard._EMBEDDINGS_TABLE_PRESENT.pop(engine, None)
    assert dashboard._job_embeddings_table_present(store) is False
    assert dashboard._job_embeddings_relation(store) == dashboard._EMPTY_JOB_EMBEDDINGS_REL


def test_job_embeddings_relation_uses_real_table_when_present(monkeypatch) -> None:
    """When the table exists, the join targets ``job_embeddings`` unchanged."""
    monkeypatch.setattr(dashboard, "_job_embeddings_table_present", lambda _store: True)
    assert dashboard._job_embeddings_relation(SimpleNamespace(engine=object())) == "job_embeddings"


def test_search_optimizations_forces_lexical_when_embeddings_table_absent(monkeypatch) -> None:
    """A semantic backend with no ``job_embeddings`` table must avoid ``_search_semantic``.

    Regression: with ``SEARCH_BACKEND=semantic`` but the table never created (an
    airgap deploy without pgvector), an owner/shared scope whose corpus has no
    unembedded success rows fell straight through to ``_search_semantic`` — which
    references ``job_embeddings`` by name and raised ``UndefinedTable`` -> 500,
    surfacing in the browser as "failed to load results" for the Mine / Shared
    tabs while the public scope (diverted to lexical by the unembedded probe)
    kept working. The dispatcher now forces lexical whenever the table is absent.
    """
    monkeypatch.setattr(dashboard.settings, "embeddings_enabled", True)
    monkeypatch.setattr(dashboard, "_job_embeddings_table_present", lambda _store: False)

    def _must_not_run(**_kwargs: object) -> dict[str, object]:
        raise AssertionError("semantic/bm25 search must not run without the table")

    monkeypatch.setattr(dashboard, "_search_semantic", _must_not_run)
    monkeypatch.setattr(dashboard, "_search_bm25", _must_not_run)
    sentinel = {"results": [], "total": 0, "matched_ids": [], "search_type": "lexical"}
    monkeypatch.setattr(dashboard, "_search_lexical", lambda **_kwargs: sentinel)

    out = dashboard.search_optimizations(
        job_store=SimpleNamespace(engine=object()),
        query="anything",
        owner_username="someone@example.com",
    )
    assert out is sentinel
