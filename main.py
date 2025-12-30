from __future__ import annotations
import uvicorn
from core import ServiceRegistry, create_app

registry = ServiceRegistry()
app = create_app(registry=registry)


def run_server() -> None:
    """Start the FastAPI server via Uvicorn."""

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run_server()
