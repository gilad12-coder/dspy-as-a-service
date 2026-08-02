"""Tests for the public-dashboard aggregator (PER-11 Feature B)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

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
