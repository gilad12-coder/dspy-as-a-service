# Managed-inference gateway (LiteLLM proxy)

The self-hosted [LiteLLM proxy](https://docs.litellm.ai/docs/simple_proxy) that
fronts Skynet's **managed** inference. It routes all managed runs through one
OpenRouter master account, meters them, and enforces per-key spend caps. This is
**Phase E** of the provider-connections plan — additive and last: managed + BYOK
both work without it, and turning it on is a single backend config change.

## What it is (and isn't)

- **Managed runs** flow `backend → LiteLLM proxy → OpenRouter → provider`. One
  master account, 0% inference markup + the 5.5% deposit fee folded into the
  credit markup.
- **BYOK runs never touch this proxy.** The run-path bridge
  (`backend/core/billing/byok_bridge.py`) stamps the user's own key onto the
  request, which goes straight to the provider. The proxy only ever sees managed
  traffic.
- **The Skynet credit ledger stays the source of truth** for billing
  (`backend/core/billing/service.py`). The proxy's budgets are a *backstop* —
  a circuit breaker against runaway upstream spend — not the accounting system.
  Reconcile periodically: the proxy's `/spend/logs` should track the ledger's
  managed debits; a sustained divergence means a metering bug to investigate.

## Deploy

```bash
cd deploy/litellm
cp .env.example .env        # fill in OPENROUTER_API_KEY, LITELLM_MASTER_KEY, …
docker compose up -d
curl -s http://localhost:4000/health   # should report healthy
```

## Deploy on Railway (hosted)

The same `config.yaml` runs unchanged on Railway. Railway can't bind-mount a
file into the stock image, so `Dockerfile` here bakes `config.yaml` in; the proxy
still listens on a fixed `4000`. The proxy keeps its own **dedicated** Postgres
and Redis, separate from the Skynet app database.

```bash
# 1. Dedicated backends for the proxy (isolated from the app DB)
railway add -d postgres        # virtual-key + spend tables
railway add -d redis           # prompt cache

# 2. The proxy service, wired via cross-service variable references so the
#    OpenRouter master key is never copied around. <Postgres>/<Redis> are the
#    generated service names (e.g. "Postgres-ZPeG", "Redis").
railway add -s litellm \
  -v 'OPENROUTER_API_KEY=${{backend.OPENROUTER_API_KEY}}' \
  -v 'LITELLM_DATABASE_URL=${{<Postgres>.DATABASE_URL}}' \
  -v 'REDIS_HOST=${{<Redis>.REDISHOST}}' \
  -v 'REDIS_PORT=${{<Redis>.REDISPORT}}' \
  -v 'REDIS_PASSWORD=${{<Redis>.REDISPASSWORD}}'
# Strong master key, kept out of shell history:
printf 'sk-%s' "$(openssl rand -hex 24)" | \
  railway variable set LITELLM_MASTER_KEY --stdin --service litellm --skip-deploys

# 3. Build the Dockerfile here (--path-as-root makes this dir the build context)
railway up . --path-as-root --service litellm --ci

# 4. Keep the proxy private — its only caller is the backend, in the same
#    project, so it needs NO public domain. The backend reaches it over Railway's
#    internal network. Verify from inside that network:
railway ssh --service backend -- \
  curl -s http://litellm.railway.internal:4000/health/liveliness   # -> "I'm alive!"
```

The proxy binds `::` (see the Dockerfile) because Railway's private network is
IPv6-only. Admin endpoints (`/key/generate`, `/spend/logs`) take the master key;
run them from inside the network too — never expose them publicly:

```bash
# Mint a virtual key (the master key already lives in the litellm container env):
railway ssh --service litellm -- sh -c \
  'curl -s http://localhost:4000/key/generate \
     -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
     -d "{\"max_budget\": 50, \"duration\": \"30d\"}"'             # returns {"key": "sk-..."}

# Point the backend at the proxy over the internal domain — plain http, since TLS
# is unnecessary inside the private network:
railway variable set 'LITELLM_PROXY_URL=http://litellm.railway.internal:4000/v1' --service backend --skip-deploys
echo "<virtual-key>" | railway variable set LITELLM_PROXY_API_KEY --stdin --service backend --skip-deploys
railway redeploy --service backend --yes
```

If you ever need to reach the proxy from outside the project (local dev, a
separate worker), attach a domain temporarily and delete it when done:
`railway domain --port 4000 --service litellm` … `railway domain delete <id>`.

The `litellm` service is deployed from a local build via `railway up`. To switch
it to GitHub autodeploys once this is on `main`, connect the repo with this
directory as the root:

```bash
railway service source connect --repo <owner>/<repo> --branch main --service litellm
# then set the service's Root Directory to deploy/litellm in the dashboard
```

## Enable it on the backend

The backend stays dormant toward the proxy until both env vars are set:

| Env var | Value |
|---|---|
| `LITELLM_PROXY_URL` | The proxy base URL, e.g. `http://litellm-proxy:4000/v1` |
| `LITELLM_PROXY_API_KEY` | A **virtual key** minted from the master key (below) |

Mint a virtual key (optionally per-user or per-job, with its own budget):

```bash
curl -s http://localhost:4000/key/generate \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"max_budget": 50, "duration": "30d"}'
```

Once `LITELLM_PROXY_URL` is set, `build_language_model`
(`backend/core/service_gateway/language_models.py::_apply_managed_gateway`)
points every managed call at the proxy and authenticates with the virtual key.
BYOK calls (which already carry the user's `api_key`) and endpoint-pinned calls
are never rerouted.

## Rollback

Unset `LITELLM_PROXY_URL` on the backend and redeploy. Managed runs immediately
revert to calling providers directly via process env keys; no data migration,
no code change.
