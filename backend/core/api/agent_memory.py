"""Permanent per-user memory for the generalist agent — an OptMem port.

Ports github.com/VictorTaelin/OptMem from fixed-width files to the shared
Postgres store, keyed by ``username``. The memory is an append-only log of
one-line entries; aligned power-of-two blocks of it compress into one-line
summaries forming a binary merge tree. The agent itself authors every
compression — the system only ever asks for the next one, one at a time, in
the output of ``note`` — and the ``wake`` document rendered into each turn
tiles the whole log with at most :data:`WAKE_LINES` lines, finest near the
present, so detail decays with age.

Leaf module (imports no routers) so the memory router and any future caller
share it without a cycle. All strings returned here are agent-facing
English: they land in tool observations and the ``memory_context`` input,
never in the UI.
"""

from __future__ import annotations

from collections import deque

import regex
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..storage.models import AgentMemoryModel, AgentMemorySummaryModel
from .errors import DomainError

MEMO_CHARS = 280
# ~64 dense lines ≈ 4–5k tokens worst case, the ceiling of what a chat turn
# can absorb as standing context (OptMem's CLI default is 96 for a whole
# session; a per-turn injection warrants a tighter budget).
WAKE_LINES = 64
# Blocks up to this many memories compress straight from the raw log; larger
# ones compress their two half-summaries instead.
RAW_MAX = 16
RECALL_CHARS = 4000
_PATTERN_CHARS = 200
# Per-row match budget. Rows are ≤280 chars, so any pattern that needs more
# than this is catastrophically backtracking — cut it off instead of letting
# one hostile pattern pin a worker thread.
_REGEX_TIMEOUT_SECONDS = 0.05
_NOTE_RETRIES = 3


def _cover(total: int, alpha: float) -> list[tuple[int, int]]:
    """Tile ``[0, total)`` with aligned power-of-two blocks, coarseness ``alpha``.

    A block stays whole iff its size is at most ``alpha`` times its age
    (``total - lo``); bigger alpha means coarser tiling with fewer lines.

    Args:
        total: Length of the log being tiled.
        alpha: Size-to-age ratio above which a block splits.

    Returns:
        The tiling as sorted ``[lo, hi)`` pairs.
    """
    root = 1
    while root < total:
        root *= 2
    out: list[tuple[int, int]] = []
    stack = [(0, root)]
    while stack:
        lo, hi = stack.pop()
        if lo >= total:
            continue
        size = hi - lo
        if size > 1 and (hi > total or size > alpha * (total - lo)):
            mid = (lo + hi) // 2
            stack.append((mid, hi))
            stack.append((lo, mid))
        else:
            out.append((lo, hi))
    out.sort()
    return out


def cover(total: int, budget: int) -> list[tuple[int, int]]:
    """The blocks the wake document prints: at most ``budget``, finest near ``total``.

    Detail decays with age, so recent memories stay verbatim and ancient ones
    collapse. If everything fits, nothing is compressed at all.

    Args:
        total: Length of the log being rendered.
        budget: Maximum number of blocks to return.

    Returns:
        Sorted ``[lo, hi)`` pairs covering ``[0, total)`` exactly.
    """
    if total <= 0:
        return []
    if total <= budget:
        return [(i, i + 1) for i in range(total)]
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if len(_cover(total, mid)) > budget:
            lo = mid
        else:
            hi = mid
    out = _cover(total, hi)
    # Block sizes jump in powers of two, so alpha alone can undershoot the
    # budget. Spend what is left on the present, where detail is worth most.
    while len(out) < budget:
        i = max((i for i, b in enumerate(out) if b[1] - b[0] > 1), default=None)
        if i is None:
            break
        lo_, hi_ = out[i]
        mid_ = (lo_ + hi_) // 2
        out[i : i + 1] = [(lo_, mid_), (mid_, hi_)]
    return out


def log_len(session: Session, username: str) -> int:
    """Return how many memories ``username`` has recorded.

    The log is dense (``seq`` runs 0..N-1 with no gaps), so a row count is
    also one past the highest ``seq``.
    """
    return int(
        session.scalar(
            select(func.count()).select_from(AgentMemoryModel).where(AgentMemoryModel.username == username)
        )
        or 0
    )


def check_text(text: str) -> str:
    """Validate one memory (or summary) line and return it stripped.

    Args:
        text: Candidate memory text.

    Returns:
        The stripped single-line text.

    Raises:
        DomainError: 422 when empty, multi-line, or over :data:`MEMO_CHARS`.
    """
    text = text.strip()
    if not text:
        raise DomainError("agent_memory.note_empty", status=422)
    if "\n" in text or "\r" in text:
        raise DomainError("agent_memory.note_multiline", status=422)
    if len(text) > MEMO_CHARS:
        raise DomainError("agent_memory.note_too_long", status=422, length=len(text), limit=MEMO_CHARS)
    return text


def parse_block(block: str) -> tuple[int, int]:
    """Parse ``<lo>-<hi>`` (inclusive, as printed) into a ``[lo, hi)`` block.

    Args:
        block: Block id string like ``"16-31"``.

    Returns:
        The half-open ``(lo, hi)`` range.

    Raises:
        DomainError: 422 when the string is not an aligned power-of-two block.
    """
    m = regex.fullmatch(r"(\d+)-(\d+)", block.strip())
    if not m:
        raise DomainError("agent_memory.invalid_block", status=422, block=block)
    lo, hi = int(m.group(1)), int(m.group(2)) + 1
    n = hi - lo
    if n < 2 or n & (n - 1) or lo % n:
        raise DomainError("agent_memory.invalid_block", status=422, block=block)
    return lo, hi


def _log_slice(session: Session, username: str, lo: int, hi: int) -> list[tuple[int, str, str]]:
    """Return memories ``[lo, hi)`` as ``(seq, date, text)`` rows."""
    rows = session.scalars(
        select(AgentMemoryModel)
        .where(AgentMemoryModel.username == username, AgentMemoryModel.seq >= lo, AgentMemoryModel.seq < hi)
        .order_by(AgentMemoryModel.seq)
    ).all()
    return [(row.seq, row.created_at.date().isoformat(), row.content) for row in rows]


def _summary(session: Session, username: str, lo: int, hi: int) -> str | None:
    """Return the cached summary of block ``[lo, hi)``, or ``None`` if unbuilt."""
    size = hi - lo
    row = session.get(AgentMemorySummaryModel, (username, size, lo // size))
    return row.content if row is not None else None


def _level_counts(session: Session, username: str) -> dict[int, int]:
    """Return ``{block_size: settled_block_count}`` for ``username``.

    Each level is a dense prefix (blocks are built strictly in order), so a
    per-level count says exactly how far that level got — one grouped query,
    never a scan.
    """
    rows = session.execute(
        select(AgentMemorySummaryModel.block_size, func.count())
        .where(AgentMemorySummaryModel.username == username)
        .group_by(AgentMemorySummaryModel.block_size)
    ).all()
    return {int(size): int(n) for size, n in rows}


def pending(session: Session, username: str, total: int, limit: int | None = None) -> list[tuple[int, int]]:
    """Blocks that can be built and have not been, smallest size first.

    Args:
        session: Open ORM session.
        username: Memory owner.
        total: Current log length.
        limit: Optional cap on how many blocks to list.

    Returns:
        Pending ``[lo, hi)`` blocks in build order.
    """
    counts = _level_counts(session, username)
    todo: list[tuple[int, int]] = []
    size = 2
    while size <= total:
        for k in range(counts.get(size, 0), total // size):
            todo.append((k * size, (k + 1) * size))
            if limit and len(todo) >= limit:
                return todo
        size *= 2
    return todo


def pending_count(session: Session, username: str, total: int) -> int:
    """How many blocks :func:`pending` would list, without listing them."""
    counts = _level_counts(session, username)
    n, size = 0, 2
    while size <= total:
        n += max(0, total // size - counts.get(size, 0))
        size *= 2
    return n


def _nap_prompt(session: Session, username: str, lo: int, hi: int, left: int) -> str:
    """Render the compression request for block ``[lo, hi)``.

    Args:
        session: Open ORM session.
        username: Memory owner.
        lo: Block start (inclusive).
        hi: Block end (exclusive).
        left: How many compressions remain after this one.

    Returns:
        An agent-facing instruction naming the block, its source lines, and
        the ``memory_nap`` call that settles it.
    """
    if hi - lo <= RAW_MAX:
        body = "\n".join(f"  #{seq} {date} {text}" for seq, date, text in _log_slice(session, username, lo, hi))
    else:
        # pending() lists a block only after both halves settled (a smaller
        # gap would be the pending head instead), so the halves always exist.
        mid = (lo + hi) // 2
        body = "\n".join(
            f"  #{a}-{b - 1} {_summary(session, username, a, b)}" for a, b in ((lo, mid), (mid, hi))
        )
    tail = ""
    if left == 1:
        tail = "\n1 compression remains after this one."
    elif left > 1:
        tail = f"\n{left} compressions remain after this one."
    return (
        f"Compress memories #{lo}-{hi - 1} into one line of at most {MEMO_CHARS} characters.\n"
        "Keep what has lasting effect, drop what does not. Invent nothing.\n\n"
        f"{body}\n{tail}\n"
        f'Call: memory_nap(block="{lo}-{hi - 1}", summary="<your line>")'
    )


def next_nap(session: Session, username: str, total: int) -> str | None:
    """Return the compression request for the next pending block, if any."""
    todo = pending(session, username, total, limit=1)
    if not todo:
        return None
    lo, hi = todo[0]
    return _nap_prompt(session, username, lo, hi, pending_count(session, username, total) - 1)


def note(session: Session, username: str, text: str) -> tuple[int, str | None]:
    """Append one memory and return its id plus the next compression request.

    Args:
        session: Open ORM session (committed here).
        username: Memory owner.
        text: The memory line (validated via :func:`check_text`).

    Returns:
        ``(seq, compression_request)`` — the id the memory saved as, and the
        next pending compression to hand the agent, or ``None``.

    Raises:
        DomainError: 422 on invalid text.
    """
    text = check_text(text)
    # Ids are assigned from the dense row count; a concurrent note by the
    # same user collides on the (username, seq) PK, so retry with a fresh
    # count instead of a lock.
    for attempt in range(_NOTE_RETRIES):
        seq = log_len(session, username)
        session.add(AgentMemoryModel(username=username, seq=seq, content=text))
        try:
            session.commit()
            break
        except IntegrityError:
            session.rollback()
            if attempt == _NOTE_RETRIES - 1:
                raise
    return seq, next_nap(session, username, seq + 1)


def save_nap(session: Session, username: str, block: str, summary: str) -> tuple[str, str | None]:
    """Settle one pending compression, enforcing build order.

    Args:
        session: Open ORM session (committed here).
        username: Memory owner.
        block: Block id string as printed by the compression request.
        summary: The agent-authored one-line compression.

    Returns:
        ``(status, compression_request)`` — what happened to this block, and
        the next pending compression, or ``None``.

    Raises:
        DomainError: 422 on a malformed block id, invalid summary text, or a
            block that is not the next pending one.
    """
    lo, hi = parse_block(block)
    total = log_len(session, username)
    todo = pending(session, username, total, limit=1)
    if not todo:
        return "Nothing left to compress.", None
    if (lo, hi) != todo[0]:
        if _summary(session, username, lo, hi) is not None:
            status = f"#{lo}-{hi - 1} is already settled."
        else:
            nxt = todo[0]
            raise DomainError(
                "agent_memory.wrong_block", status=422, block=block, next_block=f"{nxt[0]}-{nxt[1] - 1}"
            )
    else:
        size = hi - lo
        session.add(
            AgentMemorySummaryModel(
                username=username, block_size=size, block_index=lo // size, content=check_text(summary)
            )
        )
        try:
            session.commit()
            status = f"#{lo}-{hi - 1} saved."
        except IntegrityError:
            # A parallel turn settled it meanwhile — the summary that landed
            # first wins, matching OptMem's append-once tree files.
            session.rollback()
            status = f"#{lo}-{hi - 1} was settled meanwhile."
    return status, next_nap(session, username, total)


def _render_block(session: Session, username: str, lo: int, hi: int) -> list[str]:
    """Render block ``[lo, hi)`` as wake lines, splitting when unsummarized.

    OptMem's wake refuses to render until the needed summary exists; a chat
    turn cannot refuse, so an unbuilt block falls back to its two halves —
    more lines, never a hole — and the appended compression request drains
    the backlog.
    """
    if hi - lo == 1:
        seq, date, text = _log_slice(session, username, lo, hi)[0]
        return [f"#{seq} {date} {text}"]
    s = _summary(session, username, lo, hi)
    if s is not None:
        return [f"#{lo}-{hi - 1} {s}"]
    mid = (lo + hi) // 2
    return _render_block(session, username, lo, mid) + _render_block(session, username, mid, hi)


def wake_document(session: Session, username: str) -> str:
    """Render the memory context injected into every generalist turn.

    Args:
        session: Open ORM session.
        username: Memory owner.

    Returns:
        The whole log tiled into at most ~:data:`WAKE_LINES` lines (raw
        ``#i date text`` entries and ``#lo-hi summary`` nodes, oldest
        first), followed by the next pending compression request, if any.
    """
    total = log_len(session, username)
    if total == 0:
        return (
            "No memories yet. When something with lasting effect happens, record it with "
            'memory_note(text="<one line>").'
        )
    lines: list[str] = []
    for lo, hi in cover(total, WAKE_LINES):
        lines.extend(_render_block(session, username, lo, hi))
    memories = "1 memory" if total == 1 else f"{total} memories"
    doc = f"Your memory, oldest first ({memories}):\n" + "\n".join(lines)
    nap = next_nap(session, username, total)
    if nap:
        doc += "\n\n" + nap
    return doc


def recall(session: Session, username: str, pattern: str) -> str:
    """Search every memory ever recorded, newest matches kept within a cap.

    Args:
        session: Open ORM session.
        username: Memory owner.
        pattern: Case-insensitive regular expression.

    Returns:
        The newest matching ``#seq date text`` lines that fit
        :data:`RECALL_CHARS`, followed by a match count (or ``"No match."``).

    Raises:
        DomainError: 422 on an invalid, oversized, or catastrophically
            backtracking pattern.
    """
    if len(pattern) > _PATTERN_CHARS:
        raise DomainError("agent_memory.bad_pattern", status=422)
    try:
        pat = regex.compile(pattern, regex.IGNORECASE)
    except regex.error as exc:
        raise DomainError("agent_memory.bad_pattern", status=422) from exc
    hits, size = 0, 0
    out: deque[str] = deque()
    query = (
        select(AgentMemoryModel)
        .where(AgentMemoryModel.username == username)
        .order_by(AgentMemoryModel.seq)
        .execution_options(yield_per=500)
    )
    try:
        for row in session.scalars(query):
            line = f"#{row.seq} {row.created_at.date().isoformat()} {row.content}"
            if not pat.search(line, timeout=_REGEX_TIMEOUT_SECONDS):
                continue
            hits += 1
            out.append(line)
            size += len(line) + 1
            while size > RECALL_CHARS:
                size -= len(out.popleft()) + 1
    except TimeoutError as exc:
        raise DomainError("agent_memory.bad_pattern", status=422) from exc
    if not hits:
        return "No match."
    matches = "1 match" if hits == 1 else f"{hits} matches"
    body = "\n".join(out)
    if len(out) < hits:
        return f"{body}\nNewest {len(out)} of {matches}. Narrow the regex."
    return f"{body}\n{matches}."


def zoom(session: Session, username: str, block: str) -> str:
    """Open one tree node: its two halves, each a summary or a raw memory.

    Args:
        session: Open ORM session.
        username: Memory owner.
        block: Block id string as the wake document prints them.

    Returns:
        One line per half — ``#seq date text`` once a half is single,
        ``#lo-hi summary`` otherwise.

    Raises:
        DomainError: 422 on a malformed id or a block beyond the log.
    """
    lo, hi = parse_block(block)
    total = log_len(session, username)
    if lo >= total:
        raise DomainError("agent_memory.block_beyond_log", status=422, block=block, count=total)
    mid = (lo + hi) // 2
    lines: list[str] = []
    for a, b in ((lo, mid), (mid, hi)):
        if a >= total:
            continue
        if b - a == 1:
            seq, date, text = _log_slice(session, username, a, b)[0]
            lines.append(f"#{seq} {date} {text}")
        else:
            s = _summary(session, username, a, b)
            lines.append(f"#{a}-{b - 1} {s if s is not None else 'not compressed yet'}")
    return "\n".join(lines)
