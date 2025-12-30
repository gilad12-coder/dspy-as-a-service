# DSPy as a Service

FastAPI wrapper that turns [DSPy](https://github.com/stanfordnlp/dspy) optimizers into a managed HTTP service. You deploy the server once, operators configure LLM credentials and the optimizer registry, and clients submit datasets + signatures over HTTP. This document targets **service operators**—people running and customizing the API. If you are a client who wants to call the service, read [`usage_guide/README.md`](usage_guide/README.md).

## Contents
- [Prerequisites](#prerequisites)
- [Install & Run](#install--run)
- [Architecture](#architecture)
- [Configuration & Credentials](#configuration--credentials)
- [Built-in Registry & Extensibility](#built-in-registry--extensibility)
- [Payload Requirements](#payload-requirements)
- [API Reference](#api-reference)
- [Artifacts & Logs](#artifacts--logs)
- [Monitoring & Operations](#monitoring--operations)
- [Client Usage Guide](#client-usage-guide)

## Prerequisites
- Python ≥ 3.11
- DSPy-compatible optimizer packages available on the server (`pip install dspy`)
- Provider credentials (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) exported in the environment
- Optional: git, virtualenv, uvicorn/gunicorn for production deployment

## Install & Run
### Option A: `uv` (recommended)
[`uv`](https://github.com/astral-sh/uv) handles virtualenv creation, dependency resolution, and script execution from the same CLI.
```bash
# 0) install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 1) create / reuse a local virtualenv
uv venv .venv
source .venv/bin/activate

# 2) install the service_gateway + dev tools (pytest, httpx, etc.)
# note the quotes so zsh doesn't glob the extras spec
uv pip install -e '.[dev]'

# 3) provide your provider key(s)
export OPENAI_API_KEY=sk-...

# 4) run the server (uv wraps python so PYTHONPATH is already set)
uv run python main.py
```
Need fully pinned builds? Run `uv pip compile pyproject.toml -o uv.lock` after updating dependencies and commit the generated lock.
### Option B: plain `python -m venv`
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
export OPENAI_API_KEY=sk-...
python main.py
```

In both cases the server listens on `http://0.0.0.0:8000` by default. Adjust host/port or TLS by editing `main.py` or running `uvicorn dspy_service.app:create_app --host ... --port ...`.

## Architecture

The service uses a distributed task processing architecture:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   FastAPI   │────▶│    Redis    │◀────│   Celery    │
│   (API)     │     │  (Broker)   │     │  (Workers)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       │         ┌─────────────┐               │
       └────────▶│  Artifacts  │◀──────────────┘
                 │   (Disk)    │
                 └─────────────┘
```

**Components:**
- **FastAPI API** (`dspy_service/app.py`): HTTP endpoints for job submission, status polling, and artifact retrieval
- **Redis**: Message broker for Celery tasks and job state storage
- **Celery Workers** (`dspy_service/tasks.py`): Execute DSPy optimization jobs asynchronously
- **Job Store** (`dspy_service/jobs.py`): Redis-backed storage for job metadata, progress events, and logs
- **Artifact Store** (`dspy_service/artifacts.py`): Persists optimized programs to disk and Base64-encodes for API responses

**Docker Deployment:**
```bash
docker-compose up --build
```
This starts three containers: `api` (FastAPI), `worker` (Celery), and `redis`. Flower dashboard available on port 5555.

## Configuration & Credentials
- **Provider keys**: set environment vars before launch; payloads never include secrets.
- **Artifacts**: optimized programs are written under `./artifacts/<job_id>` and also returned inline via Base64.
- **Log level**: server logs plus `dspy.*` INFO logs are captured per job and exposed through the API.
- **Resource limits**: `MAX_PROGRESS_EVENTS`/`MAX_LOG_ENTRIES` in `dspy_service/jobs.py` guard memory usage.

## Built-in Registry & Extensibility
Aliases provided by default:
- Modules: `predict` → `dspy.Predict`, `cot` → `dspy.modules.ChainOfThought`
- Optimizers: `miprov2`, `gepa`

You can:
- Register additional factories in `configure_registry` (see `main.py`).
- Reference any DSPy callable via dotted path, e.g., `dspy.teleprompt.GEPA`.
- Extend `resolver.py` if you need custom resolution logic.

## Payload Requirements
Every `POST /run` submission must include:
1. `signature_code`: serialized `dspy.Signature` class definition.
2. `metric_code`: serialized metric function (callable returning float).
3. `dataset`: list of records; at least one row.
4. `column_mapping`: maps signature inputs/outputs to dataset keys.
5. `optimizer_name`, `optimizer_kwargs`: aliases or dotted paths; kwargs are validated against the optimizer signature (unsupported keys raise HTTP 400).
6. `compile_kwargs`: forwarded directly.
7. `model_config`: default LM for DSPy modules/baseline/test scoring.
8. Optional `prompt_model_config`/`task_model_config`: override MiPro v2 prompt/task LMs (fall back to `model_config` when omitted).
9. Optional `reflection_model_config`: required by GEPA.
10. Split fractions + shuffle flag + seed (defaults provided).

## API Reference

`POST /run`
- Validates payload schema, signature↔column mappings, optimizer kwargs, and split fractions before enqueueing work.
- **Response (`200 OK`)**
  ```json
  {
    "job_id": "uuid",
    "status": "pending",
    "estimated_total_seconds": null
  }
  ```

`GET /jobs/{job_id}` *(detailed stream)*
- Returns the live job record, including all captured progress events and logs:
  ```json
  {
    "job_id": "uuid",
    "status": "running",
    "message": "optimizer_progress",
    "created_at": "2025-11-30T07:15:00Z",
    "started_at": "2025-11-30T07:15:05Z",
    "completed_at": null,
    "latest_metrics": {"baseline_test_metric": 0.75, ...},
    "progress_events": [
      {"timestamp": "...", "event": "dataset_splits_ready", "metrics": {...}},
      {"timestamp": "...", "event": "optimizer_progress", "metrics": {"tqdm_percent": 55.0, ...}},
      ...
    ],
    "logs": [
      {"timestamp": "...", "level": "INFO", "logger": "dspy.evaluate", "message": "Average Metric: ..."},
      ...
    ],
    "estimated_seconds_remaining": 42.3,
    "result": null
  }
  ```
- After success/failure the `result` field holds the full `RunResponse` with metrics, metadata, Base64 artifact, and `run_log` (identical to `/jobs/{id}/logs`).

`GET /jobs/{job_id}/summary` *(aggregated view)*
- Lightweight snapshot for dashboards:
  ```json
  {
    "job_id": "uuid",
    "status": "running",
    "message": "optimizer_progress",
    "created_at": "...",
    "started_at": "...",
    "completed_at": null,
    "elapsed_seconds": 128.4,
    "estimated_seconds_remaining": 35.6,
    "module_name": "cot",
    "optimizer_name": "miprov2",
    "dataset_rows": 120,
    "split_fractions": {"train": 0.5, "val": 0.3, "test": 0.2},
    "shuffle": true,
    "seed": 7,
    "optimizer_kwargs": {"auto": "light", ...},
    "compile_kwargs": {},
    "latest_metrics": {"baseline_test_metric": 0.75, "optimized_test_metric": 0.9}
  }
  ```

`GET /jobs/{job_id}/logs`
- Entire chronological log history:
  ```json
  [
    {"timestamp": "...", "level": "INFO", "logger": "dspy.evaluate", "message": "Average Metric: 3.0 / 4 (75.0%)"},
    {"timestamp": "...", "level": "INFO", "logger": "dspy.teleprompt.mipro_optimizer_v2", "message": "Running with ..."},
    ...
  ]
  ```
- Same entries are embedded in `RunResponse.run_log` so clients always receive the full run transcript when the job completes.

`GET /health`
- Readiness/status probe:
  ```json
  {
    "status": "ok",
    "registered_assets": {
      "modules": ["predict", "cot"],
      "optimizers": ["gepa", "miprov2"]
    }
  }
  ```

`GET /jobs/{job_id}/artifact`
- Returns the serialized artifact after successful completion (409 while running, 404 if the job never produced one):
  ```json
  {
    "program_artifact_path": "artifacts/uuid",
    "program_artifact": {
      "metadata": {...},
      "program_pickle_base64": "..."
    }
  }
  ```

## Artifacts & Logs
- Artifacts live under `./artifacts/<job_id>` and are also returned inline (`RunResponse.program_artifact`).
- Use the helper below to materialize artifacts elsewhere:
```python
import base64, json
from pathlib import Path

def materialize_artifact(bundle, destination):
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metadata.json").write_text(json.dumps(bundle["metadata"], indent=2))
    (dest / "program.pkl").write_bytes(base64.b64decode(bundle["program_pickle_base64"]))
    return dest
```
- Every INFO log emitted during optimization is stored per job. `/jobs/{id}/logs` returns them in submission order; successful jobs also embed the same list in `result.run_log`.
- For ad-hoc inspection, edit `inspect_artifact.py` (set `ARTIFACT_PATH` and optional `TEST_INPUTS`) and run `python inspect_artifact.py` to print metadata/signature details and exercise the compiled module.

## Monitoring & Operations
- Progress events come from a patched `tqdm` that emits GEPA/MiPro rollouts, percent complete, elapsed time, and ETA; clients can render their own dashboards without tailing stdout.
- Job summary endpoint is ideal for dashboards that poll occasionally rather than streaming.
- Failure handling:
  - Schema errors → HTTP 422 with friendly `{field, message}` entries.
  - Unsupported optimizer kwargs → HTTP 400 before execution.
  - Runtime errors → job transitions to `failed` with stack trace captured in `logs` and `message`.

## Client Usage Guide
If you are consuming the API (building payloads, polling, downloading artifacts), read [`usage_guide/README.md`](usage_guide/README.md). It explains the helper notebooks, demonstrates the polling helper, and shows how to call the new summary/log endpoints from Python. The `usage_guide/` folder now includes its own `pyproject.toml` (plus a regeneratable `uv.lock`) so you can spin up a clean notebook environment:

```bash
cd usage_guide
uv venv .venv && source .venv/bin/activate
uv pip install -r pyproject.toml
# optionally: uv pip compile pyproject.toml -o uv.lock
```

This keeps the service runtime and the example notebooks isolated from each other.
