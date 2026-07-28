"""Persistence seams for resuming GEPA optimizations — checkpoints and pair results.

The GEPA engine writes its full state (``gepa_state.bin``) at every iteration:
the candidate population, per-candidate validation scores, the Pareto front, the
iteration counter and the consumed metric-call budget. Persisting the latest
copy lets a run that died mid-optimization resume from its last completed
iteration with no budget double-spend, instead of restarting from scratch.

Both seams are keyed by ``(optimization_id, pair_index)``: a single run uses the
sentinel ``pair_index = -1``; a grid search runs one GEPA optimization per model
pair, so it keeps a checkpoint per in-flight pair plus a stored pair result
for each pair that already finished — a resumed grid keeps the finished pairs and
re-runs only the rest. The bytes go through these stores — mirroring the
dataset-library blob seam — so they can later move behind an object store without
touching the worker, the resume endpoint, or the storage meter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from .models import GepaCheckpointModel, GridPairResultModel

SINGLE_RUN_PAIR_INDEX = -1


@dataclass(frozen=True)
class GepaCheckpoint:
    """One run/pair's saved GEPA state plus the iteration it was taken at."""

    optimization_id: str
    pair_index: int
    iteration: int
    data: bytes
    stored_bytes: int


def _json_bytes(value: dict[str, Any]) -> int:
    """Return the compact-JSON UTF-8 byte length used for storage accounting."""
    return len(json.dumps(value, separators=(",", ":"), default=str).encode("utf-8"))


class PostgresCheckpointBlobStore:
    """GEPA state blobs in ``gepa_checkpoints``, keyed by ``(optimization_id, pair_index)``."""

    def __init__(self, engine: Engine) -> None:
        """Bind the checkpoint store to a SQLAlchemy engine.

        Args:
            engine: Engine whose schema carries ``gepa_checkpoints``.
        """
        self._engine = engine

    def put(self, optimization_id: str, *, data: bytes, iteration: int, pair_index: int = SINGLE_RUN_PAIR_INDEX) -> None:
        """Insert or replace the GEPA state bytes for one run or grid pair.

        Args:
            optimization_id: Owning job id.
            data: The raw ``gepa_state.bin`` bytes for the latest iteration.
            iteration: The iteration index the state was saved at.
            pair_index: Grid pair index, or ``-1`` for a single run.
        """
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            existing = session.get(GepaCheckpointModel, (optimization_id, pair_index))
            if existing is None:
                session.add(
                    GepaCheckpointModel(
                        optimization_id=optimization_id,
                        pair_index=pair_index,
                        iteration=iteration,
                        data=data,
                        stored_bytes=len(data),
                        updated_at=now,
                    )
                )
            else:
                existing.iteration = iteration
                existing.data = data
                existing.stored_bytes = len(data)
                existing.updated_at = now
            session.commit()

    def get(self, optimization_id: str, pair_index: int = SINGLE_RUN_PAIR_INDEX) -> GepaCheckpoint | None:
        """Return the saved checkpoint for one run/pair, or ``None``.

        Args:
            optimization_id: Job whose checkpoint is read.
            pair_index: Grid pair index, or ``-1`` for a single run.

        Returns:
            The :class:`GepaCheckpoint`, or ``None`` when no row exists.
        """
        with Session(self._engine) as session:
            row = session.get(GepaCheckpointModel, (optimization_id, pair_index))
            return self._to_checkpoint(row) if row is not None else None

    def list_for_optimization(self, optimization_id: str) -> list[GepaCheckpoint]:
        """Return every saved checkpoint for a job (all grid pairs, or the single run).

        Args:
            optimization_id: Job whose checkpoints are read.

        Returns:
            The job's :class:`GepaCheckpoint` rows (possibly empty).
        """
        with Session(self._engine) as session:
            rows = session.scalars(
                select(GepaCheckpointModel).where(GepaCheckpointModel.optimization_id == optimization_id)
            ).all()
            return [self._to_checkpoint(row) for row in rows]

    def delete(self, optimization_id: str, pair_index: int = SINGLE_RUN_PAIR_INDEX) -> None:
        """Remove one run/pair's checkpoint if present.

        Args:
            optimization_id: Owning job id.
            pair_index: Grid pair index, or ``-1`` for a single run.
        """
        with Session(self._engine) as session:
            session.execute(
                delete(GepaCheckpointModel).where(
                    GepaCheckpointModel.optimization_id == optimization_id,
                    GepaCheckpointModel.pair_index == pair_index,
                )
            )
            session.commit()

    def delete_all(self, optimization_id: str) -> None:
        """Remove every checkpoint for a job (e.g. once a grid succeeds).

        Args:
            optimization_id: Job whose checkpoints are dropped.
        """
        with Session(self._engine) as session:
            session.execute(
                delete(GepaCheckpointModel).where(GepaCheckpointModel.optimization_id == optimization_id)
            )
            session.commit()

    def has_any(self, optimization_id: str) -> bool:
        """Return whether the job has any saved checkpoint (single run or any pair).

        Args:
            optimization_id: Job to test.

        Returns:
            ``True`` when at least one checkpoint row exists.
        """
        with Session(self._engine) as session:
            row = (
                session.query(GepaCheckpointModel.pair_index)
                .filter(GepaCheckpointModel.optimization_id == optimization_id)
                .first()
            )
            return row is not None

    def has_any_batch(self, optimization_ids: list[str]) -> set[str]:
        """Return the subset of ids that have at least one saved checkpoint.

        One ``IN (...)`` round trip replaces a per-row :meth:`has_any` probe
        on list pages.

        Args:
            optimization_ids: Job ids to test.

        Returns:
            The ids with at least one checkpoint row.
        """
        if not optimization_ids:
            return set()
        with Session(self._engine) as session:
            rows = (
                session.query(GepaCheckpointModel.optimization_id)
                .filter(GepaCheckpointModel.optimization_id.in_(optimization_ids))
                .distinct()
                .all()
            )
            return {row[0] for row in rows}

    @staticmethod
    def _to_checkpoint(row: GepaCheckpointModel) -> GepaCheckpoint:
        """Project an ORM row onto an immutable :class:`GepaCheckpoint`."""
        return GepaCheckpoint(
            optimization_id=row.optimization_id,
            pair_index=row.pair_index,
            iteration=row.iteration,
            data=row.data,
            stored_bytes=row.stored_bytes,
        )


class PostgresGridPairResultStore:
    """Completed grid-pair results in ``grid_pair_results``, keyed by ``(optimization_id, pair_index)``."""

    def __init__(self, engine: Engine) -> None:
        """Bind the pair-result store to a SQLAlchemy engine.

        Args:
            engine: Engine whose schema carries ``grid_pair_results``.
        """
        self._engine = engine

    def put(self, optimization_id: str, pair_index: int, result: dict[str, Any]) -> None:
        """Insert or replace the result for one completed grid pair.

        Args:
            optimization_id: Owning grid job id.
            pair_index: The completed pair's index.
            result: The pair's serialized :class:`PairResult`.
        """
        now = datetime.now(UTC)
        with Session(self._engine) as session:
            existing = session.get(GridPairResultModel, (optimization_id, pair_index))
            if existing is None:
                session.add(
                    GridPairResultModel(
                        optimization_id=optimization_id,
                        pair_index=pair_index,
                        result=result,
                        stored_bytes=_json_bytes(result),
                        updated_at=now,
                    )
                )
            else:
                existing.result = result
                existing.stored_bytes = _json_bytes(result)
                existing.updated_at = now
            session.commit()

    def get_all(self, optimization_id: str) -> dict[int, dict[str, Any]]:
        """Return ``{pair_index: result}`` for every completed pair of a grid.

        Args:
            optimization_id: Grid job whose finished pairs are read.

        Returns:
            Mapping of completed pair index to its serialized result (possibly empty).
        """
        with Session(self._engine) as session:
            rows = session.scalars(
                select(GridPairResultModel).where(GridPairResultModel.optimization_id == optimization_id)
            ).all()
            return {row.pair_index: row.result for row in rows}

    def delete_all(self, optimization_id: str) -> None:
        """Remove every stored pair result for a grid (e.g. once it succeeds).

        Args:
            optimization_id: Grid job whose pair results are dropped.
        """
        with Session(self._engine) as session:
            session.execute(
                delete(GridPairResultModel).where(GridPairResultModel.optimization_id == optimization_id)
            )
            session.commit()

    def delete_one(self, optimization_id: str, pair_index: int) -> None:
        """Remove a single pair's stored result (e.g. before re-running it).

        Args:
            optimization_id: Grid job owning the pair.
            pair_index: The pair whose result is dropped.
        """
        with Session(self._engine) as session:
            session.execute(
                delete(GridPairResultModel).where(
                    GridPairResultModel.optimization_id == optimization_id,
                    GridPairResultModel.pair_index == pair_index,
                )
            )
            session.commit()

    def has_any(self, optimization_id: str) -> bool:
        """Return whether the grid has any completed-pair result stored.

        Args:
            optimization_id: Grid job to test.

        Returns:
            ``True`` when at least one pair result exists.
        """
        with Session(self._engine) as session:
            row = (
                session.query(GridPairResultModel.pair_index)
                .filter(GridPairResultModel.optimization_id == optimization_id)
                .first()
            )
            return row is not None

    def has_any_batch(self, optimization_ids: list[str]) -> set[str]:
        """Return the subset of ids that have at least one stored pair result.

        One ``IN (...)`` round trip replaces a per-row :meth:`has_any` probe
        on list pages.

        Args:
            optimization_ids: Grid job ids to test.

        Returns:
            The ids with at least one pair-result row.
        """
        if not optimization_ids:
            return set()
        with Session(self._engine) as session:
            rows = (
                session.query(GridPairResultModel.optimization_id)
                .filter(GridPairResultModel.optimization_id.in_(optimization_ids))
                .distinct()
                .all()
            )
            return {row[0] for row in rows}
