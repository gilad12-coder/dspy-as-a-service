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
from core.service_gateway.react_compat import configure_native_tool_calling

load_dotenv(Path(__file__).parent / ".env")

# Disable dspy caching entirely. The in-memory cache pins every response in a
# process-wide LRU (1M-entry cap — unbounded in practice) for the pod's whole
# lifetime, and the disk cache routes through diskcache, whose pickle-backed
# store has an unpatched deserialization flaw (GHSA-w8v5-vhqr-4h9v — no fixed
# release exists). Dropping the disk cache keeps that path out of the process,
# trading away retry dedup. suppress() so an incompatible dspy build can't
# abort startup. Mirrors the per-job child in core/worker/subprocess_runner.py.
with contextlib.suppress(Exception):
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

# Install the native-function-calling ChatAdapter process-wide when
# REACT_NATIVE_TOOL_CALLING is set, so every ReAct serve path routes tools
# through the provider API. A no-op when the flag is off; suppress() so an
# incompatible dspy build can't abort startup. Mirrored in the optimize child.
with contextlib.suppress(Exception):
    configure_native_tool_calling()

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
