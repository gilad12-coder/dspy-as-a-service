# On-prem authentication setup

Skynet supports one identity model with two entry paths: ADFS/OIDC and an administrator-managed local username fallback. There are no passwords, signup forms, reset flows, social providers, email verification, passkeys, TOTP, or anonymous sessions.

## Shared backend authentication secret

Generate a secret and put the exact same value in the backend and frontend environments:

```bash
openssl rand -base64 48
```

```dotenv
# backend/.env
BACKEND_AUTH_SECRET=...

# frontend/.env.local
BACKEND_AUTH_SECRET=...
AUTH_SECRET=...another independently generated value...
```

The frontend uses `BACKEND_AUTH_SECRET` to call the internal provisioning endpoints and sign short-lived backend bearer tokens. Never expose it to browser JavaScript or commit it.

## Bootstrap administrator

Set at least one normalized username before first start:

```dotenv
# backend
ADMIN_USERNAMES=admin

# frontend
AUTH_ADMINS=admin
```

Sign in at `/login?local=1` as `admin`, open Settings > Administration, and create the other local usernames. The UI can promote/demote database-managed administrators and set each user's storage allowance. Environment usernames and configured ADFS groups remain administrators even if the database role is demoted.

Local login is passwordless but not open registration: the backend accepts only active usernames already present in PostgreSQL or declared in `ADMIN_USERNAMES`.

## ADFS/OIDC

Configure the frontend runtime:

```dotenv
AUTH_URL=https://skynet.example.internal
AUTH_SSO_ISSUER=https://adfs.example.internal/adfs
AUTH_SSO_CLIENT_ID=skynet
AUTH_SSO_CLIENT_SECRET=...
AUTH_SSO_SCOPE=openid profile email groups
AUTH_USERNAME_CLAIM=upn
AUTH_GROUP_CLAIM=groups
AUTH_ADMIN_GROUPS=Skynet Admins,Platform Ops
AUTH_ADMINS=break-glass-admin
```

Register this callback in the IdP:

```text
https://skynet.example.internal/api/auth/callback/adfs
```

Use the issuer URL exposed by the ADFS OpenID Connect discovery document. If the IdP uses an internal CA, mount its bundle and set `NODE_EXTRA_CA_CERTS` in the frontend container.

When all three `AUTH_SSO_ISSUER`, `AUTH_SSO_CLIENT_ID`, and `AUTH_SSO_CLIENT_SECRET` values exist, the login page immediately leads with ADFS. A failed SSO screen reveals the local fallback; `/login?local=1` is the direct break-glass path.

Every valid ADFS login auto-provisions the user. The selected username claim is lowercased and trimmed. A matching local username resolves to the same account, data, role, sharing grants, and storage quota.

## Internal and browser backend URLs

The frontend needs two backend addresses in a clustered deployment:

```dotenv
API_URL=https://skynet-api.example.internal
BACKEND_INTERNAL_URL=http://skynet-backend:8000
```

`API_URL` is injected into the browser and must be reachable from user workstations. `BACKEND_INTERNAL_URL` is server-only and should use the private cluster Service. Both must target the same backend deployment.

## Deletion behavior

Deleting a user from the administration UI hard-deletes that user's jobs, artifacts, datasets, tagging sessions, conversations, API token, notification settings, quotas, share grants, and encrypted BYOK connections. It also removes inbound grants tied to the username. The operation is intentionally not reversible.

If the deleted username later completes a valid ADFS login, Skynet auto-creates a fresh empty account.
