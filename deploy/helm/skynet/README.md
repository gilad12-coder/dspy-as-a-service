# Skynet on-prem Helm chart

The chart deploys the Hebrew-only Skynet frontend, FastAPI backend with PostgreSQL-lease workers, a migration Job, and either bundled or external PostgreSQL. It targets OpenShift and standard Kubernetes primitives where possible. No Redis, Stripe, external queue, or cloud observability service is required.

## Render first

```bash
helm lint deploy/helm/skynet
helm template skynet deploy/helm/skynet -n skynet -f values-production.yaml > rendered.yaml
```

## Install

```bash
helm upgrade --install skynet deploy/helm/skynet \
  --namespace skynet --create-namespace \
  --set backend.image.repository=registry.example.internal/skynet/backend \
  --set frontend.image.repository=registry.example.internal/skynet/frontend \
  --set backend.image.tag=1.0.0 \
  --set frontend.image.tag=1.0.0 \
  --set openshift.routes.backend.host=skynet-api.example.internal \
  --set openshift.routes.frontend.host=skynet.example.internal
```

Use existing Secrets in production. The backend Secret needs `BACKEND_AUTH_SECRET`, `BYOK_VAULT_KEY`, database/model credentials, and optional transcription/log-shipping credentials. The frontend Secret needs `AUTH_SECRET`, the same `BACKEND_AUTH_SECRET`, and `AUTH_SSO_CLIENT_SECRET`.

## Components

| Component | Default | Scaling/state |
|---|---:|---|
| backend API + workers | 2 pods | HPA 2-8; PostgreSQL queue leases |
| frontend | 2 pods | HPA 2-4; stateless JWT sessions |
| migration | one-shot | Fresh on-prem baseline only |
| PostgreSQL | optional 1 pod | PersistentVolume; external DB preferred for production |
| PgBouncer | optional 2 pods | Stateless transaction pooler |

## Production checklist

- Mirror backend, frontend, PostgreSQL, and PgBouncer images into the internal registry.
- Use an external PostgreSQL service with backups and tested restores when available.
- Set `frontend.env.API_URL` to the browser-reachable backend URL. `BACKEND_INTERNAL_URL` defaults to the in-cluster Service.
- Set `frontend.env.AUTH_URL`, all `AUTH_SSO_*` ADFS values, username/group claims, and admin groups.
- Set matching backend/frontend `BACKEND_AUTH_SECRET` values and a separate stable `AUTH_SECRET`.
- Set `backend.env.ADMIN_USERNAMES` and frontend `AUTH_ADMINS` for one break-glass identity.
- Set `backend.env.ALLOWED_ORIGINS` to the frontend HTTPS origin.
- Keep OpenShift Route TLS enabled with redirect, or provide equivalent TLS ingress.
- Restrict NetworkPolicy egress to PostgreSQL, IdP, LLM, embedding, transcription, SMTP, and local observability endpoints.
- Confirm no empty allowlist leaves the chart's documented `0.0.0.0/0` fallback in production.
- Start at two worker threads per backend pod and tune from memory/CPU metrics.
- Set `BYOK_VAULT_KEY` before enabling provider-key storage.

## Important values

| Value | Purpose |
|---|---|
| `backend.env.WORKER_CONCURRENCY` | Job threads per pod; default 2 for predictable memory use. |
| `backend.env.JOB_ADMISSION_MAX_MEMORY_FRACTION` | Defers new claims near the cgroup memory limit. |
| `backend.env.USER_STORAGE_QUOTA_BYTES` | Default per-user unified quota; UI overrides are stored in PostgreSQL. |
| `backend.env.SEARCH_BACKEND` | `lexical` (plain PostgreSQL), `bm25` (`pg_search`), or `semantic` (`pgvector`). |
| `backend.autoscaling.queueDepth` | Optional `skynet_jobs_pending` external metric. |
| `pgbouncer.enabled` | PostgreSQL transaction pooling at high replica counts. |
| `frontend.env.API_URL` | Browser-reachable backend origin. |
| `frontend.env.BACKEND_INTERNAL_URL` | Server-only backend Service used during authentication. |
| `frontend.env.TRANSCRIPTION_ENABLED` | Shows dictation only when backend transcription is also configured. |

## Database lifecycle

The migration hook runs `alembic upgrade head` against a fresh database. This private edition intentionally does not upgrade hosted or older private schemas. PostgreSQL PVCs survive Helm uninstall; delete them only as an explicit data-destruction operation.

```bash
helm uninstall skynet -n skynet
```

See `docs/ON_PREM_DEPLOYMENT.md` for secrets, ADFS, observability, smoke tests, and deployment examples.
