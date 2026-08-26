"""Tests for the cgroup memory probe and the worker's admission deferral.

The probe must read whichever cgroup layout the container runs (v2 first,
v1 fallback) and, critically, report ``None`` — never a guess — when no
limit is readable, because ``None`` is what keeps the gate inert on dev
machines and unlimited pods.
"""

import threading
from pathlib import Path

import pytest

from core.config import settings
from core.worker.engine import BackgroundWorker
from core.worker.memory_guard import memory_usage_fraction


def _paths(tmp_path: Path, **files: str) -> dict[str, Path]:
    """Write the given cgroup files into ``tmp_path`` and return all four probe paths.

    Args:
        tmp_path: Test-scoped directory to hold the fake cgroup files.
        **files: ``name → content`` for the files that should exist; the four
            probe kwargs always point into ``tmp_path`` so unnamed ones read
            as missing.

    Returns:
        Kwargs for :func:`memory_usage_fraction`.
    """
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    return {
        "current_path": tmp_path / "current",
        "max_path": tmp_path / "max",
        "v1_usage_path": tmp_path / "v1_usage",
        "v1_limit_path": tmp_path / "v1_limit",
    }


def test_cgroup_v2_fraction(tmp_path):
    """v2 usage/limit produces the plain ratio."""
    kwargs = _paths(tmp_path, current="1073741824", max="4294967296")
    assert memory_usage_fraction(**kwargs) == pytest.approx(0.25)


def test_cgroup_v2_unlimited_falls_back_to_v1(tmp_path):
    """A v2 'max' token means unlimited there; v1 files still apply."""
    kwargs = _paths(tmp_path, current="1", max="max", v1_usage="500", v1_limit="1000")
    assert memory_usage_fraction(**kwargs) == pytest.approx(0.5)


def test_cgroup_v1_unlimited_sentinel_reads_as_none(tmp_path):
    """v1's huge no-limit sentinel must not be treated as a real limit."""
    kwargs = _paths(tmp_path, v1_usage="500", v1_limit=str(1 << 62))
    assert memory_usage_fraction(**kwargs) is None


def test_missing_files_read_as_none(tmp_path):
    """No cgroup files at all (dev machine) → None, keeping the gate inert."""
    assert memory_usage_fraction(**_paths(tmp_path)) is None


def test_garbage_content_reads_as_none(tmp_path):
    """Unparseable file contents must not crash or produce a fraction."""
    kwargs = _paths(tmp_path, current="not-a-number", max="4096")
    assert memory_usage_fraction(**kwargs) is None


class _StubWorker:
    """Bare object carrying just the state ``_defer_for_memory_pressure`` uses."""

    def __init__(self) -> None:
        """Initialize the admission-log state the real worker holds."""
        self._admission_last_log = 0.0
        self._activity_lock = threading.Lock()


def test_defer_for_memory_pressure_gates_on_threshold(monkeypatch):
    """Above the threshold the claim defers; below it (or unknown) it proceeds."""
    stub = _StubWorker()
    defer = BackgroundWorker._defer_for_memory_pressure

    monkeypatch.setattr(settings, "job_admission_max_memory_fraction", 0.85)
    monkeypatch.setattr("core.worker.engine.memory_usage_fraction", lambda: 0.9)
    assert defer(stub) is True

    monkeypatch.setattr("core.worker.engine.memory_usage_fraction", lambda: 0.5)
    assert defer(stub) is False

    monkeypatch.setattr("core.worker.engine.memory_usage_fraction", lambda: None)
    assert defer(stub) is False

    monkeypatch.setattr(settings, "job_admission_max_memory_fraction", 0.0)
    monkeypatch.setattr("core.worker.engine.memory_usage_fraction", lambda: 0.99)
    assert defer(stub) is False
