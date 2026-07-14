"""Container memory-usage probe for job admission control.

Reads the cgroup accounting the OOM killer actually enforces (v2 first, v1
fallback) so the worker can stop claiming new jobs when the pod is near its
memory limit — the job then waits in the Postgres queue for this pod (or a
peer) to free up, instead of the whole container being OOM-killed mid-run.
"""

from __future__ import annotations

from pathlib import Path

_CGROUP_V2_CURRENT = Path("/sys/fs/cgroup/memory.current")
_CGROUP_V2_MAX = Path("/sys/fs/cgroup/memory.max")
_CGROUP_V1_USAGE = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
_CGROUP_V1_LIMIT = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")

# cgroup v1 reports "no limit" as a huge page-aligned sentinel rather than a
# token; anything this large means the container is effectively unlimited.
_V1_UNLIMITED_FLOOR = 1 << 60


def _read_bytes(path: Path) -> int | None:
    """Parse one cgroup byte-count file; ``None`` for missing/`max`/garbage."""
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def memory_usage_fraction(
    current_path: Path = _CGROUP_V2_CURRENT,
    max_path: Path = _CGROUP_V2_MAX,
    v1_usage_path: Path = _CGROUP_V1_USAGE,
    v1_limit_path: Path = _CGROUP_V1_LIMIT,
) -> float | None:
    """Return the container's memory usage as a fraction of its limit.

    Args:
        current_path: cgroup v2 usage file (overridable for tests).
        max_path: cgroup v2 limit file.
        v1_usage_path: cgroup v1 usage file, tried when v2 is absent.
        v1_limit_path: cgroup v1 limit file.

    Returns:
        ``usage / limit`` in [0, ∞), or ``None`` when no readable limit exists
        (unlimited cgroup, or a non-Linux dev machine) — callers must treat
        ``None`` as "no admission gating possible".
    """
    current = _read_bytes(current_path)
    limit = _read_bytes(max_path)
    if current is None or limit is None:
        current = _read_bytes(v1_usage_path)
        limit = _read_bytes(v1_limit_path)
        if limit is not None and limit >= _V1_UNLIMITED_FLOOR:
            return None
    if current is None or limit is None or limit <= 0:
        return None
    return current / limit
