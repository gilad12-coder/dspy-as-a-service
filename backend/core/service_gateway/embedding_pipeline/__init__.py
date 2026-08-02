"""Embedding pipelines feeding explore search and the agent-history search.

:mod:`core` runs the per-job embedding (``embed_finished_job``) and the
continuous repair loop (``EmbeddingIndexSweeper``). :mod:`conversations` is the
companion pipeline for agent conversations — same embedder, same dispatch
shape, different source table. :mod:`embeddings` wraps the OpenAI-compatible
vector-model client and :mod:`summarizer` turns a finished run into the
prose chunk that gets embedded.
"""

from __future__ import annotations

from .conversations import (
    backfill_missing_conversation_embeddings,
    embed_conversation,
    purge_orphan_conversation_embeddings,
    queue_conversation_embed,
)
from .core import (
    backfill_missing_embeddings,
    embed_finished_job,
    purge_orphan_embeddings,
    set_embedding_privacy,
    set_embedding_task_name,
    start_embedding_index_sweeper,
)

__all__ = [
    "backfill_missing_conversation_embeddings",
    "backfill_missing_embeddings",
    "embed_conversation",
    "embed_finished_job",
    "purge_orphan_conversation_embeddings",
    "purge_orphan_embeddings",
    "queue_conversation_embed",
    "set_embedding_privacy",
    "set_embedding_task_name",
    "start_embedding_index_sweeper",
]
