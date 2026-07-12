<p align="center">
  <img src="docs/assets/skynet-wordmark.gif" width="480" alt="SKYNET" />
</p>

**A self-hostable platform for building, optimizing, and serving LLM programs — with prompt optimization (GEPA) at its core, priced at break-even.**

Skynet turns "I have a dataset and a task" into an optimized, deployable LLM program. Upload data, describe the task, and the platform compiles a [DSPy](https://github.com/stanfordnlp/dspy) program, evolves its prompts with GEPA against your own metric, shows you the held-out lift it earned, and serves the result for inference — all through a web UI a non-engineer can drive, or a REST API.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

## Features

- **Optimization runs** — a guided wizard (or an agent) builds the signature, metric, and column mapping; GEPA evolves the prompt with live streaming progress, resumable checkpoints, and a baseline-vs-optimized held-out score. Grid search compares model pairs side by side.
- **AI co-tagging** — the tagger interviews you about your dataset, distills an editable labeling guide, calibrates against ~30 of your own labels (the AI guesses blind and reveals only after you commit), then earns the right to tag the rest through agreement-gated review rounds. Every label carries provenance (`human` / `ai_confirmed` / `ai_auto`), and one click turns your labels into a real optimized classifier.
- **Dataset library** — save, share, clone, and edit datasets in place with a spreadsheet editor; hand any dataset to the tagger or the wizard by reference.
- **Agents** — a generalist assistant (Cmd/Ctrl+J) that operates the whole wizard through tools with configurable trust modes, and a code agent that authors signatures, metrics, and multi-step workflow graphs on a visual canvas.
- **Serving** — every successful run yields a program artifact: inspect the evolved instructions and demos, run inference against it, or export a runnable program.
- **24 locales, RTL-first** — Hebrew is the base language; Arabic and Persian are first-class; the rest overlay with graceful fallback.
- **Break-even pricing** — credits map to raw provider cost plus payment-processing fees only (markup 1.09, zero profit). Bring your own API key and runs are **free**. A "no lift, no charge" guarantee refunds runs that don't beat their baseline. Without Stripe keys, billing is simply off.

## Quick Start (local)

Prerequisites: Python 3.11, Node 20+, PostgreSQL 15+, [`just`](https://github.com/casey/just), [`uv`](https://github.com/astral-sh/uv) (or pip), Docker (for the LiteLLM model gateway).

```bash
git clone https://github.com/gilad12-coder/skynet.git && cd skynet

# 1. Database
createdb skynet

# 2. Configure
cp backend/.env.example backend/.env        # set REMOTE_DB_URL + model keys
cp frontend/.env.example frontend/.env.local

# 3. Model gateway (routes all LLM traffic; holds provider keys)
cd deploy/litellm && docker compose up -d && cd ../..

# 4. Install + run
just install
just backend    # FastAPI on :8000 — migrations apply automatically at boot
just frontend   # Next.js on :3000
```

Open http://localhost:3000. The API reference lives at http://localhost:8000/scalar.

Useful recipes: `just test`, `just lint`, `just check-i18n`, `just --list` for everything.

## Architecture

```
frontend/   Next.js (App Router) · Tailwind v4 · shadcn/radix · SSE streaming UI
backend/    FastAPI · SQLAlchemy + Alembic (boot-time migrations) · DSPy 3.2
            └─ worker: multi-pod job fleet over Postgres (SELECT … FOR UPDATE
               SKIP LOCKED leases, orphan recovery, resumable GEPA checkpoints)
deploy/     litellm proxy (compose) · helm chart for Kubernetes
i18n/       Hebrew base catalog + 23 overlay locales → generated typed catalogs
docs/       operator guides (Stripe setup, design briefs)
```

All model traffic flows through a LiteLLM proxy, so any OpenAI-compatible provider works and keys live in one place. Billing (optional) is Stripe: prepaid credit packs, metered usage at $0.01/credit, and a per-user encrypted BYOK vault.

## Configuration

### Backend (`backend/.env`)

```bash
# ── Required ──
REMOTE_DB_URL=postgresql://user@localhost:5432/skynet
LITELLM_PROXY_URL=http://localhost:4000/v1     # the model gateway
LITELLM_PROXY_API_KEY=...                      # its master key

# ── Server ──
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_ORIGINS=http://localhost:3000          # comma-separated CORS origins

# ── Worker ──
WORKER_CONCURRENCY=4                           # parallel background jobs

# ── Billing (optional — omit to disable charging entirely) ──
# STRIPE_SECRET_KEY=...                        # see docs/stripe-setup.md
```

See `backend/.env.example` for the full annotated list (agents' models, tagger assist models, notifications, air-gap gateways, and more).

### Frontend (`frontend/.env.local`)

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
AUTH_SECRET=generate-with-openssl-rand-base64-32

# Without SSO configured, the login page offers email/password signup.
# ADFS/OIDC SSO and Google/GitHub OAuth: see frontend/.env.example.
```

## Serving optimized programs

```bash
# What inputs does the program expect?
curl http://localhost:8000/serve/{optimization_id}/info

# Run inference
curl -X POST http://localhost:8000/serve/{optimization_id} \
  -H 'Content-Type: application/json' \
  -d '{"inputs": {"question": "What is 7+3?"}}'
```

The job detail page includes a built-in inference playground and a program export (runnable zip).

## Deployment

- **Anywhere with Postgres** — the backend migrates its own schema at boot and the worker fleet scales horizontally via DB-lease job claims (no external queue).
- **Kubernetes** — Helm chart in `deploy/helm`.
- **Docker** — `cd backend && docker compose up --build` starts API + Postgres.
- **Billing** — optional; follow `docs/stripe-setup.md` to enable credit packs and metered usage.

## Extensibility

Register custom modules and optimizers in `main.py`:

```python
from core import ServiceRegistry, create_app

registry = ServiceRegistry()
registry.register_module("my_module", my_module_factory)
registry.register_optimizer("my_optimizer", my_optimizer_factory)

app = create_app(registry=registry)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for dev setup, the test-suite layout, i18n rules, and the migration discipline. PRs welcome.

## License

[AGPL-3.0](LICENSE). Run it, fork it, self-host it — and if you offer a modified Skynet as a service, share your modifications back.
