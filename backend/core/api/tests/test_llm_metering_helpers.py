"""Tests for the interactive-turn billing helpers in ``routers._helpers``.

Covers the 402 credit gate (``enforce_llm_credits``) and the SSE metering
wrapper (``stream_with_llm_metering``) — including the early-teardown path,
where the client drops the stream before the terminal event and the turn must
still be billed from the sink.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from ...storage.models import Base, BillingCustomerModel, CreditLedgerModel
from ..errors import DomainError
from ..routers._helpers import enforce_llm_credits, stream_with_llm_metering


class _StubStore:
    """Job-store double exposing only the engine the billing path touches."""

    def __init__(self, engine: Engine | None) -> None:
        """Store the engine (or None to model a legacy/in-memory store)."""
        self.engine = engine


@pytest.fixture
def engine() -> Iterator[Engine]:
    """Yield an in-memory SQLite engine with the billing tables created."""
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(eng, tables=[BillingCustomerModel.__table__, CreditLedgerModel.__table__])
    yield eng
    Base.metadata.drop_all(eng)


def _deplete(engine: Engine, username: str) -> None:
    """Seed a billing row whose grant and paid balance are both exhausted."""
    with Session(engine) as session:
        session.add(
            BillingCustomerModel(
                username=username,
                stripe_customer_id=f"cus_{username}",
                credit_balance=0,
                grant_remaining=0,
            )
        )
        session.commit()


class _FakeLm:
    """History-carrying LM double matching what ``usage_by_model_from_history`` reads."""

    def __init__(self, history: list[dict[str, Any]], model: str = "openrouter/test/unpriced") -> None:
        """Store the canned history entries and reported model id."""
        self.history = history
        self.model = model


def test_enforce_llm_credits_passes_fresh_account(engine: Engine) -> None:
    """A brand-new account rides its free grant through the gate."""
    enforce_llm_credits(_StubStore(engine), "new@x.io")


def test_enforce_llm_credits_rejects_depleted_account(engine: Engine) -> None:
    """A zero-balance account is refused with the 402 insufficient-credits code."""
    _deplete(engine, "broke@x.io")
    with pytest.raises(DomainError) as err:
        enforce_llm_credits(_StubStore(engine), "broke@x.io")
    assert err.value.status_code == 402


def test_enforce_llm_credits_skips_engineless_store(engine: Engine) -> None:
    """A store without a SQL engine streams ungated, matching the submit path."""
    enforce_llm_credits(_StubStore(None), "anyone@x.io")
    enforce_llm_credits(_StubStore(engine), "")


async def test_stream_with_llm_metering_bills_on_completion(engine: Engine) -> None:
    """A fully-drained stream passes events through and writes the debit."""
    sink = [_FakeLm([{"usage": {"prompt_tokens": 100_000, "completion_tokens": 40_000}}])]

    async def source() -> AsyncIterator[dict[str, Any]]:
        yield {"event": "message_patch", "data": {"chunk": "hi"}}
        yield {"event": "done", "data": {}}

    events = [
        event
        async for event in stream_with_llm_metering(
            source(),
            job_store=_StubStore(engine),
            username="alice@x.io",
            description="Agent chat",
            usage_sink=sink,
        )
    ]
    assert [e["event"] for e in events] == ["message_patch", "done"]
    with Session(engine) as session:
        row = session.query(CreditLedgerModel).one()
    assert row.description == "Agent chat"
    assert row.delta_credits < 0
    assert row.input_tokens == 100_000


async def test_stream_with_llm_metering_bills_on_early_teardown(engine: Engine) -> None:
    """A stream dropped before its terminal event still bills the sink's usage."""
    sink = [_FakeLm([{"usage": {"prompt_tokens": 50_000, "completion_tokens": 5_000}}])]

    async def source() -> AsyncIterator[dict[str, Any]]:
        yield {"event": "message_patch", "data": {"chunk": "hi"}}
        yield {"event": "done", "data": {}}

    stream = stream_with_llm_metering(
        source(),
        job_store=_StubStore(engine),
        username="alice@x.io",
        description="Agent chat",
        usage_sink=sink,
    )
    assert (await anext(stream))["event"] == "message_patch"
    await stream.aclose()
    with Session(engine) as session:
        row = session.query(CreditLedgerModel).one()
    assert row.delta_credits < 0


async def test_stream_with_llm_metering_skips_empty_sink(engine: Engine) -> None:
    """A turn whose run function never built an LM bills nothing."""

    async def source() -> AsyncIterator[dict[str, Any]]:
        yield {"event": "error", "data": {"error": "boom"}}

    events = [
        event
        async for event in stream_with_llm_metering(
            source(),
            job_store=_StubStore(engine),
            username="alice@x.io",
            description="Agent chat",
            usage_sink=[],
        )
    ]
    assert [e["event"] for e in events] == ["error"]
    with Session(engine) as session:
        assert session.query(CreditLedgerModel).count() == 0
