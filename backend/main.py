"""Application entrypoint for the Skynet backend.

Loads environment variables from ``backend/.env``, configures logging, builds
the ``ServiceRegistry`` and the FastAPI ``app`` object, and exposes
``run_server`` as the script entrypoint that boots Uvicorn.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import dspy
import uvicorn
from dotenv import load_dotenv

from core.api.app import create_app
from core.api.observability import configure_logging
from core.registry import ServiceRegistry

load_dotenv(Path(__file__).parent / ".env")

# Serve/share/workflow inference runs dspy inside this process with client
# caching on, and dspy's default cache pins every response in a process-wide
# in-memory LRU (1M-entry cap — unbounded in practice) for the pod's whole
# lifetime. Disk-only caching keeps retry dedup at near-zero resident cost;
# an unwritable cache dir degrades to no caching at all. Mirrors the per-job
# child setup in core/worker/subprocess_runner.py.
try:
    dspy.configure_cache(enable_memory_cache=False)
except Exception:
    with contextlib.suppress(Exception):
        dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

# Must run before create_app() so loggers acquired during router import
# inherit the configured formatter, not Uvicorn's default.
configure_logging()

registry = ServiceRegistry()
app = create_app(registry=registry)


def run_server() -> None:
    """Boot the FastAPI application using Uvicorn, honouring API_HOST / API_PORT."""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run_server()
