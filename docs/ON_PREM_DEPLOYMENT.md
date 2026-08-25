# On-prem deployment runbook

This distribution targets a new PostgreSQL database. It does not migrate hosted Skynet billing, login, or legacy account data.

## Required infrastructure

- PostgreSQL 16. Vanilla PostgreSQL supports the default lexical Explorer. `pgvector` is needed only for `SEARCH_BACKEND=semantic`; `pg_search` is needed only for `SEARCH_BACKEND=bm25`.
- An internal or explicitly approved OpenAI-compatible model endpoint, or user-supplied BYOK connections.
- ADFS/OIDC for primary identity in production.
- HTTPS certificates for the frontend and backend endpoints.

Redis, Stripe, Sentry, and an external queue are not required. The optional LiteLLM example also uses PostgreSQL without Redis.

## Before first start

1. Create a fresh database and database user.
2. Generate `AUTH_SECRET`, `BACKEND_AUTH_SECRET`, and `BYOK_VAULT_KEY`.
3. Set the same `BACKEND_AUTH_SECRET` in frontend and backend.
4. Configure `ADMIN_USERNAMES` and the matching frontend `AUTH_ADMINS` bootstrap identity.
5. Configure a browser-reachable `API_URL`, a server-only `BACKEND_INTERNAL_URL`, and `ALLOWED_ORIGINS`.
6. Configure ADFS/OIDC and register `/api/auth/callback/adfs`.
7. Point central model, agent, embedding, and optional transcription URLs only at approved internal endpoints.

The migration Job or backend startup applies the single on-prem baseline schema automatically. Do not stamp or reuse a database created by an older hosted/private build.

## Docker Compose

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env.local
cd backend
docker compose config
docker compose up --build -d
docker compose ps
curl -fsS http://localhost:8000/health
```

Compose runs PostgreSQL, an API with embedded workers, and the frontend. For multiple hosts or automatic scaling, use the Helm chart.

## OpenShift/Kubernetes

Render and inspect before installation:

```bash
helm lint deploy/helm/skynet
helm template skynet deploy/helm/skynet -n skynet -f values-production.yaml > rendered.yaml
```

At minimum, a production override should set:

```yaml
backend:
  env:
    ALLOWED_ORIGINS: https://skynet.example.internal
    ADMIN_USERNAMES: break-glass-admin
    ADMIN_GROUPS: Skynet Admins
    CODE_AGENT_BASE_URL: https://llm.example.internal/v1
    CODE_AGENT_MODEL: openai/internal-model
    GENERALIST_AGENT_BASE_URL: https://llm.example.internal/v1
    GENERALIST_AGENT_MODEL: openai/internal-model
  secrets:
    existingSecret: skynet-backend-secrets

frontend:
  env:
    API_URL: https://skynet-api.example.internal
    AUTH_URL: https://skynet.example.internal
    AUTH_SSO_ISSUER: https://adfs.example.internal/adfs
    AUTH_SSO_CLIENT_ID: skynet
    AUTH_USERNAME_CLAIM: upn
    AUTH_GROUP_CLAIM: groups
    AUTH_ADMIN_GROUPS: Skynet Admins
    AUTH_ADMINS: break-glass-admin
  secrets:
    existingSecret: skynet-frontend-secrets

externalDatabase:
  enabled: true
  host: postgres.example.internal
  database: skynet
  user: skynet
  existingSecret: skynet-database

postgres:
  enabled: false

openshift:
  routes:
    backend:
      host: skynet-api.example.internal
    frontend:
      host: skynet.example.internal
```

The backend secret must contain `REMOTE_DB_URL` when URL composition is not used, `BACKEND_AUTH_SECRET`, `BYOK_VAULT_KEY`, and central model credentials. The frontend secret must contain `AUTH_SECRET`, the same `BACKEND_AUTH_SECRET`, and the ADFS client secret.

## Scaling and resource use

Backend replicas combine API and workers. PostgreSQL leases make horizontal claims safe without Redis. The default two worker threads per pod limits memory amplification; increase it only after measuring real jobs. `JOB_ADMISSION_MAX_MEMORY_FRACTION` stops idle workers from claiming new work near their cgroup limit.

Keep total SQLAlchemy connections below PostgreSQL `max_connections`:

```text
backend replicas * (DB_POOL_SIZE + DB_POOL_MAX_OVERFLOW)
```

Enable the chart's PgBouncer transaction pool when replica count makes direct connection budgets inefficient. It is a stateless process backed by PostgreSQL, not another data store.

The backend HPA scales on CPU and can also use the `skynet_jobs_pending` metric through Prometheus Adapter. Start with the defaults, then tune pod memory and worker concurrency from observed job size.

## Observability

- Structured JSON application and user-decision logs go to stdout.
- Prometheus metrics are exposed at `/metrics`; the chart adds scrape annotations.
- Product telemetry is stored in PostgreSQL when `TELEMETRY_ENABLED=true`.
- Job progress and job logs are retained in PostgreSQL with per-job caps.
- Optional `LOG_SHIP_URL` and `ALERT_WEBHOOK_URL` may point at internal collectors.
- PostHog export is disabled unless its key is explicitly configured.

No observability data silently leaves the private network.

## Security checks

- Keep TLS route redirects enabled; only opt out for an isolated test network.
- Restrict NetworkPolicy egress to PostgreSQL, ADFS, model, embedding, transcription, SMTP, and logging endpoints actually in use.
- Keep `DISCOVER_ALLOW_PRIVATE=false` unless model discovery of trusted RFC1918 endpoints is required.
- Store secrets in an external Secret manager; never in a committed values file.
- Back up PostgreSQL and test restores. User deletion is a hard cascade and cannot be undone from the application.

## Acceptance smoke test

1. Sign in through ADFS and confirm automatic account creation.
2. Sign in through `/login?local=1` with an administrator-created username.
3. Create, promote/demote, quota, and delete a test user in Settings > Administration.
4. Submit a private run and confirm it is absent from Explorer.
5. Publish the run and confirm another authenticated user can find it in Explorer.
6. Share a private dataset/run with a named user and confirm an ungranted user is denied.
7. Save a BYOK connection, execute a run, and confirm the stored secret is encrypted.
8. Confirm `/metrics`, structured logs, PostgreSQL telemetry, and worker lease recovery.
9. Verify there are no Redis, Stripe, Sentry, or anonymous-link dependencies in the deployed manifests.
