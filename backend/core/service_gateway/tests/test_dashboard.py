"""Tests for the public-dashboard aggregator (PER-11 Feature B)."""

from __future__ import annotations

from types import SimpleNamespace

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
