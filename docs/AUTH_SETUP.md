# Auth setup — hosted Skynet

Skynet's hosted login offers three ways in:

- **Google** (OAuth)
- **GitHub** (OAuth)
- **Email + password** — a real account created in Skynet itself

Identity is the **email** for every provider, so one person is one account no
matter how they sign in. SSO (ADFS/OIDC) still takes over when configured; the
providers below are the default when SSO is unset.

All secrets go in `frontend/.env.local` (never commit it). After editing env,
restart `next dev` so the server picks them up.

---

## 0. Shared prerequisites (required for email/password)

The email/password path stores accounts in the backend's Postgres and is gated
by a shared secret so only the frontend can reach the register/login endpoints.

1. Generate a secret and set the **same value** on both sides:
   ```bash
   openssl rand -base64 32
   ```
   - `frontend/.env.local` → `BACKEND_AUTH_SECRET=<value>`
   - backend env → `BACKEND_AUTH_SECRET=<value>`
2. Also set `AUTH_SECRET` (NextAuth session secret) in `frontend/.env.local`:
   ```
   AUTH_SECRET=<another `openssl rand -base64 32`>
   ```
3. Run the DB migration that creates the `users` table:
   ```bash
   cd backend && alembic upgrade head
   ```
4. (Optional) Make yourself admin — works for every provider:
   ```
   AUTH_ADMINS=you@example.com
   ```

That's enough for **email/password** to work end-to-end. Social providers are
additive — set up either or both below.

---

## 1. Google

1. Go to <https://console.cloud.google.com> → create/select a project.
2. **APIs & Services → OAuth consent screen**: choose **External**, fill app
   name + support email, save. (While "Testing", add your Google account under
   **Test users**.)
3. **APIs & Services → Credentials → Create credentials → OAuth client ID**:
   - Application type: **Web application**
   - **Authorized redirect URIs** — add the exact callback for each origin you
     run on:
     - `http://localhost:3000/api/auth/callback/google`
     - `https://YOUR_DOMAIN/api/auth/callback/google` (production)
4. Copy the **Client ID** and **Client secret** into `frontend/.env.local`:
   ```
   AUTH_GOOGLE_ID=...
   AUTH_GOOGLE_SECRET=...
   ```

The "Continue with Google" button appears automatically once both are set.

---

## 2. GitHub

1. Go to <https://github.com/settings/developers> → **OAuth Apps → New OAuth App**
   (or an org's Developer settings for an org-owned app).
2. Fill in:
   - **Homepage URL**: `http://localhost:3000` (or your production URL)
   - **Authorization callback URL**: `http://localhost:3000/api/auth/callback/github`
     (add a second OAuth App for production with the prod callback — GitHub
     allows only one callback per app)
3. **Generate a new client secret**, then copy both into `frontend/.env.local`:
   ```
   AUTH_GITHUB_ID=...
   AUTH_GITHUB_SECRET=...
   ```

The "Continue with GitHub" button appears automatically once both are set.

---

## 3. Email / password

Nothing more to configure beyond **section 0**. On the login screen, the
**יצירת חשבון** (Create account) tab registers an account (min 8-char password,
stored only as a salted scrypt hash — never the plaintext) and signs in
immediately. The **התחברות** (Sign in) tab logs an existing account back in.

---

## Production notes

- Set `AUTH_SECRET` and `BACKEND_AUTH_SECRET` to strong, unique values.
- Register the production callback URLs (`https://YOUR_DOMAIN/api/auth/callback/<provider>`)
  in each OAuth app.
- Serve over HTTPS so session cookies and the backend token aren't exposed.
- Apple Sign-In is intentionally not wired up (it needs a paid Apple Developer
  Program membership and a rotating JWT client secret); add it later the same
  way if needed.
