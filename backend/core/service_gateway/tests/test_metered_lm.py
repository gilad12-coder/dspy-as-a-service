"""Tests for ``MeteredLM`` usage aggregation and its history-free contract.

``MeteredLM`` replaces dspy's per-call history retention (full prompts and
responses, three lists deep) with running usage totals. These tests pin the
two properties the swap relies on: nothing is retained anywhere, and the
billing/telemetry readers see exactly the numbers the old history walk
produced — including the fallback path for plain/mocked LMs.
"""

import threading
import time
from types import SimpleNamespace

import dspy
from dspy.clients.base_lm import GLOBAL_HISTORY

from core.service_gateway.language_models import (
    LmUsageTotals,
    MeteredLM,
    activate_job_lm_budget,
    lm_call_count,
    total_tokens_from_history,
    usage_by_model_from_history,
)


def _entry(prompt: int | None = None, completion: int | None = None, total: int | None = None) -> dict:
    """Build a minimal history entry with the given usage numbers.

    Args:
        prompt: ``prompt_tokens`` value, omitted when ``None``.
        completion: ``completion_tokens`` value, omitted when ``None``.
        total: ``total_tokens`` value, omitted when ``None``.

    Returns:
        A dict shaped like the entries ``dspy.LM`` hands to ``update_history``.
    """
    usage: dict = {}
    if prompt is not None:
        usage["prompt_tokens"] = prompt
    if completion is not None:
        usage["completion_tokens"] = completion
    if total is not None:
        usage["total_tokens"] = total
    return {"prompt": "q", "response": "r", "usage": usage}


def _metered() -> MeteredLM:
    """Construct a MeteredLM without any network access."""
    return MeteredLM(model="openai/gpt-4o-mini", cache=False)


def test_update_history_aggregates_and_retains_nothing():
    """Usage folds into totals; per-LM and global history stay empty."""
    lm = _metered()
    global_before = len(GLOBAL_HISTORY)

    lm.update_history(_entry(prompt=100, completion=40))
    lm.update_history(_entry(total=60))
    lm.update_history({"prompt": "q", "response": "r", "usage": {}})

    assert lm.history == []
    assert len(GLOBAL_HISTORY) == global_before
    totals = lm.usage_totals
    assert totals.calls == 3
    assert totals.total_tokens == 140 + 60
    assert totals.input_tokens == 100 + 60  # total-only entries attribute to input
    assert totals.output_tokens == 40
    assert totals.total_found
    assert totals.split_found


def test_no_usage_reads_as_untracked():
    """Calls without usage count as calls but leave token readers at None."""
    lm = _metered()
    lm.update_history({"prompt": "q", "response": "r", "usage": {}})

    assert lm_call_count(lm) == 1
    assert total_tokens_from_history(lm) is None
    assert usage_by_model_from_history(lm) is None


def test_readers_prefer_metered_totals():
    """The aggregate feeds both token readers and the call counter."""
    lm = _metered()
    lm.update_history(_entry(prompt=10, completion=5))
    lm.update_history(_entry(prompt=20, completion=15))

    assert lm_call_count(lm) == 2
    assert total_tokens_from_history(lm) == 50
    assert usage_by_model_from_history(lm) == {lm.model: (30, 20)}


def test_history_fallback_matches_old_semantics():
    """Plain/mocked LMs with a history list still total the old way."""
    stub = SimpleNamespace(model="mock/model", history=[_entry(prompt=7, completion=3), _entry(total=10)])

    assert lm_call_count(stub) == 2
    assert total_tokens_from_history(stub) == 20
    assert usage_by_model_from_history(stub) == {"mock/model": (17, 3)}
    assert lm_call_count(SimpleNamespace()) is None


def test_mixed_metered_and_history_lms_sum():
    """One metered and one history-backed LM combine in a single read."""
    lm = _metered()
    lm.update_history(_entry(prompt=100, completion=50))
    stub = SimpleNamespace(model="mock/model", history=[_entry(prompt=1, completion=2)])

    assert total_tokens_from_history(lm, stub, None) == 153
    assert usage_by_model_from_history(lm, stub) == {lm.model: (100, 50), "mock/model": (1, 2)}


def test_copy_shares_usage_totals():
    """Calls made through dspy's internal ``copy()`` clones are still counted."""
    lm = _metered()
    clone = lm.copy()
    clone.update_history(_entry(prompt=5, completion=5))

    assert isinstance(clone.usage_totals, LmUsageTotals)
    assert lm.usage_totals.calls == 1
    assert total_tokens_from_history(lm) == 10


def test_job_lm_budget_serializes_concurrent_forward_calls(monkeypatch):
    """With a budget of 1, a second ``forward`` waits for the first's permit."""
    state = {"active": 0, "max": 0}
    lock = threading.Lock()
    release = threading.Event()

    def fake_forward(self, *args, **kwargs):
        """Track overlapping entries while holding until released."""
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        release.wait(timeout=5)
        with lock:
            state["active"] -= 1
        return "ok"

    monkeypatch.setattr(dspy.LM, "forward", fake_forward)
    activate_job_lm_budget(1)
    try:
        lm = _metered()
        threads = [threading.Thread(target=lm.forward) for _ in range(2)]
        for t in threads:
            t.start()
        # Let the second thread reach the gate; it must block outside
        # fake_forward while the first holds the single permit.
        time.sleep(0.05)
        with lock:
            assert state["active"] == 1
        release.set()
        for t in threads:
            t.join(timeout=5)
        assert state["max"] == 1
    finally:
        activate_job_lm_budget(0)


def test_job_lm_budget_zero_disables_the_gate(monkeypatch):
    """A non-positive budget removes the gate entirely; calls overlap freely."""
    state = {"active": 0, "max": 0}
    lock = threading.Lock()
    barrier = threading.Barrier(2, timeout=5)

    def fake_forward(self, *args, **kwargs):
        """Rendezvous both calls inside the LM to prove they overlap."""
        with lock:
            state["active"] += 1
            state["max"] = max(state["max"], state["active"])
        barrier.wait()
        with lock:
            state["active"] -= 1
        return "ok"

    monkeypatch.setattr(dspy.LM, "forward", fake_forward)
    activate_job_lm_budget(0)
    lm = _metered()
    threads = [threading.Thread(target=lm.forward) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert state["max"] == 2
