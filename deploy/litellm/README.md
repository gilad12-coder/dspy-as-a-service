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
