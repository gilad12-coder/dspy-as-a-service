# Optional LiteLLM gateway

This optional gateway centralizes organization-managed model traffic. Skynet can also call a configured OpenAI-compatible endpoint directly, so the gateway is not required.

The example uses PostgreSQL for gateway state and no Redis. BYOK requests bypass it and use the encrypted connection selected by the user.

```bash
cp .env.example .env
docker compose up -d
curl -fsS http://localhost:4000/health/liveliness
```

Point the backend at it:

```dotenv
LITELLM_PROXY_URL=http://litellm-proxy:4000/v1
LITELLM_PROXY_API_KEY=...
```

Keep the admin endpoint private. `LITELLM_MASTER_KEY` belongs only in the gateway environment. The PostgreSQL database may live on the same PostgreSQL cluster as Skynet while using separate credentials and a separate database or schema.

To roll back, unset `LITELLM_PROXY_URL`; central-model calls return to the endpoint/provider configuration in the backend environment.
