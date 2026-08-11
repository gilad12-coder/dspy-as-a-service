import NextAuth, { CredentialsSignin } from "next-auth";
import type { Provider } from "next-auth/providers";
import Credentials from "next-auth/providers/credentials";
import Google from "next-auth/providers/google";
import GitHub from "next-auth/providers/github";
import { createHmac, randomUUID } from "crypto";

/**
 * Authentication configuration.
 *
 * On-prem SSO (ADFS / any OIDC):
 *   Set AUTH_SSO_ISSUER, AUTH_SSO_CLIENT_ID, AUTH_SSO_CLIENT_SECRET. When set,
 *   users auto-redirect to the IdP and the social/local providers below are not
 *   offered — an on-prem deployment keeps its single auth path.
 *
 * Hosted (default when ADFS is not configured):
 *   Social sign-in — Google (AUTH_GOOGLE_ID/SECRET) and GitHub
 *   (AUTH_GITHUB_ID/SECRET), each shown only when its credentials are present —
 *   plus email/password accounts ("create an account in Skynet"), verified by
 *   the backend /auth endpoints over the shared BACKEND_AUTH_SECRET.
 *
 * Identity is the email across every provider, so one person maps to one backend
 * identity however they sign in (see signBackendToken).
 *
 * Optional env vars:
 *   AUTH_SSO_SCOPE        — OIDC scopes (default: "openid profile email groups")
 *   AUTH_ADMIN_GROUPS     — comma-separated IdP groups that grant admin access
 *   AUTH_ADMINS           — comma-separated admin emails/usernames
 *   AUTH_GROUP_CLAIM      — profile claim containing groups (default: "groups")
 *   BACKEND_AUTH_SECRET   — shared secret for backend bearer tokens and the
 *                           internal /auth/register|login calls
 *   API_URL               — backend base URL the credentials provider calls
 *   NODE_EXTRA_CA_CERTS   — path to CA bundle .pem for self-signed certs
 */

const issuer = process.env.AUTH_SSO_ISSUER;
const clientId = process.env.AUTH_SSO_CLIENT_ID;
const clientSecret = process.env.AUTH_SSO_CLIENT_SECRET;
const adfsConfigured = !!issuer && !!clientId && !!clientSecret;
const devAdminFallback = adfsConfigured ? "" : "admin";

// Deploy manifests (Helm values, docker-compose) ship AUTH_ADMINS="" as an
// explicit empty string, so `??` would never reach the fallback — an empty
// var must be treated as unset for the dev-admin fallback to apply.
function envOrFallback(value: string | undefined, fallback: string) {
  return value && value.trim() ? value : fallback;
}

const ADMIN_LIST = new Set(
  envOrFallback(process.env.AUTH_ADMINS, devAdminFallback)
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean),
);
const ADMIN_GROUPS = new Set(
  envOrFallback(process.env.AUTH_ADMIN_GROUPS, "")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean),
);

const scope = process.env.AUTH_SSO_SCOPE ?? "openid profile email groups";
const groupClaim = process.env.AUTH_GROUP_CLAIM ?? "groups";
const backendAuthSecret = process.env.BACKEND_AUTH_SECRET ?? process.env.AUTH_SECRET;
const backendTokenTtlSeconds = Number.parseInt(
  process.env.BACKEND_AUTH_TOKEN_TTL_SECONDS ?? "900",
  10,
);

// The credentials provider reaches the backend over this base URL; API_URL is
// the runtime-overridable server-side value, with the build-time
// NEXT_PUBLIC_API_URL as the fallback (mirrors shared/lib/api.ts).
const backendBaseUrl =
  process.env.API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const googleConfigured = !!process.env.AUTH_GOOGLE_ID && !!process.env.AUTH_GOOGLE_SECRET;
const githubConfigured = !!process.env.AUTH_GITHUB_ID && !!process.env.AUTH_GITHUB_SECRET;

type BackendAccount = { email: string; name: string; role: string };

type CredentialsOutcome =
  | { kind: "ok"; account: BackendAccount }
  | { kind: "2fa_required"; methods: string }
  | { kind: "2fa_invalid" }
  | { kind: "email_unverified" }
  | { kind: "failed" };

/**
 * Verify email/password (+ optional second-factor code) against the backend's
 * internal /auth/login. Distinguishes the two 2FA outcomes — code missing
 * (with the account's usable methods) vs code wrong — plus an unverified email,
 * so the login form can pivot to its code or verify step; every other failure
 * collapses into a generic "failed" that never leaks which case occurred.
 */
async function verifyBackendCredentials(
  email: string,
  password: string,
  codes: { totpCode: string; emailCode: string; recoveryCode: string },
): Promise<CredentialsOutcome> {
  if (!backendAuthSecret) return { kind: "failed" };
  try {
    const res = await fetch(`${backendBaseUrl}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Auth": backendAuthSecret },
      body: JSON.stringify({
        email,
        password,
        totp_code: codes.totpCode,
        email_code: codes.emailCode,
        recovery_code: codes.recoveryCode,
      }),
    });
    if (res.ok) return { kind: "ok", account: (await res.json()) as BackendAccount };
    const body = (await res.json().catch(() => ({}))) as {
      code?: unknown;
      params?: { methods?: unknown };
    };
    if (body.code === "accounts.two_factor_required") {
      return { kind: "2fa_required", methods: String(body.params?.methods ?? "totp") };
    }
    if (body.code === "accounts.invalid_second_factor") return { kind: "2fa_invalid" };
    if (body.code === "accounts.email_not_verified") return { kind: "email_unverified" };
    return { kind: "failed" };
  } catch {
    return { kind: "failed" };
  }
}

/**
 * Verify a WebAuthn assertion against the backend's internal /auth/webauthn/verify.
 * The backend checks the signature against the stored credential and resolves
 * the owning identity; null on any failure.
 */
async function verifyBackendPasskey(assertion: string): Promise<BackendAccount | null> {
  if (!backendAuthSecret) return null;
  let credential: unknown;
  try {
    credential = JSON.parse(assertion);
  } catch {
    return null;
  }
  try {
    const res = await fetch(`${backendBaseUrl}/auth/webauthn/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Internal-Auth": backendAuthSecret },
      body: JSON.stringify({ credential }),
    });
    if (!res.ok) return null;
    return (await res.json()) as BackendAccount;
  } catch {
    return null;
  }
}

/** Build a CredentialsSignin whose ``code`` survives to the client signIn result. */
function credentialsError(code: string): CredentialsSignin {
  const error = new CredentialsSignin();
  error.code = code;
  return error;
}

const providers: Provider[] = [];

function readClaim(profile: Record<string, unknown>, path: string): unknown {
  return path.split(".").reduce<unknown>((acc, part) => {
    if (!acc || typeof acc !== "object" || Array.isArray(acc)) return undefined;
    return (acc as Record<string, unknown>)[part];
  }, profile);
}

function normalizeStringList(value: unknown): string[] {
  if (Array.isArray(value))
    return value
      .map(String)
      .map((s) => s.trim())
      .filter(Boolean);
  if (typeof value === "string") {
    return value
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }
  return [];
}

function profileGroups(profile: Record<string, unknown>): string[] {
  return normalizeStringList(readClaim(profile, groupClaim));
}

function isAdmin(identifier: string, groups: string[]) {
  if (ADMIN_LIST.has(identifier.toLowerCase())) return true;
  return groups.some((group) => ADMIN_GROUPS.has(group.toLowerCase()));
}

function base64url(value: Buffer | string) {
  return Buffer.from(value).toString("base64url");
}

function signBackendToken(token: {
  name?: unknown;
  email?: unknown;
  role?: unknown;
  groups?: unknown;
}) {
  if (!backendAuthSecret) return undefined;
  const displayName = typeof token.name === "string" ? token.name : undefined;
  const email = typeof token.email === "string" ? token.email : undefined;
  // Identity across the backend (job / dataset / share ownership) is the email,
  // never the display name: two OAuth users can share a name but never an email.
  // The backend reads its identity from the `name` claim first, so the stable
  // subject is sent there; the human-facing display name rides the session JWT.
  const subject = email || displayName;
  if (!subject) return undefined;
  const now = Math.floor(Date.now() / 1000);
  const groups = normalizeStringList(token.groups);
  const header = { alg: "HS256", typ: "JWT" };
  const payload = {
    aud: "skynet-backend",
    iss: "skynet-frontend",
    sub: subject,
    name: subject,
    email,
    role: typeof token.role === "string" ? token.role : "user",
    groups,
    iat: now,
    exp: now + Math.max(60, backendTokenTtlSeconds || 900),
    jti: randomUUID(),
  };
  const encodedHeader = base64url(JSON.stringify(header));
  const encodedPayload = base64url(JSON.stringify(payload));
  const signature = createHmac("sha256", backendAuthSecret)
    .update(`${encodedHeader}.${encodedPayload}`)
    .digest("base64url");
  return `${encodedHeader}.${encodedPayload}.${signature}`;
}

if (adfsConfigured) {
  providers.push({
    id: "adfs",
    name: "ADFS",
    type: "oidc",
    issuer,
    clientId,
    clientSecret,
    authorization: { params: { scope } },
    profile(profile) {
      const groups = profileGroups(profile as Record<string, unknown>);
      const identifier = String(
        profile.name ??
          profile.unique_name ??
          profile.upn ??
          profile.preferred_username ??
          profile.email ??
          profile.sub ??
          "",
      ).toLowerCase();
      return {
        id: profile.sub,
        name:
          profile.name ??
          profile.unique_name ??
          profile.upn ??
          profile.preferred_username ??
          profile.sub,
        email: profile.email ?? profile.upn ?? profile.preferred_username,
        groups,
        role: isAdmin(identifier, groups) ? "admin" : "user",
      };
    },
  });
} else {
  if (googleConfigured) {
    providers.push(
      Google({
        clientId: process.env.AUTH_GOOGLE_ID,
        clientSecret: process.env.AUTH_GOOGLE_SECRET,
        // One person, one identity: if they later sign in with another provider
        // asserting the same verified email, link to the existing account
        // instead of forking a second one.
        allowDangerousEmailAccountLinking: true,
      }),
    );
  }
  if (githubConfigured) {
    providers.push(
      GitHub({
        clientId: process.env.AUTH_GITHUB_ID,
        clientSecret: process.env.AUTH_GITHUB_SECRET,
        allowDangerousEmailAccountLinking: true,
      }),
    );
  }
  providers.push(
    Credentials({
      id: "credentials",
      name: "Email and password",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" },
        totpCode: { label: "Authenticator code", type: "text" },
        emailCode: { label: "Email code", type: "text" },
        recoveryCode: { label: "Recovery code", type: "text" },
      },
      async authorize(credentials) {
        const email = (credentials?.email as string)?.trim().toLowerCase();
        const password = credentials?.password as string;
        if (!email || !password) return null;
        const str = (v: unknown): string => (typeof v === "string" ? v : "");
        const result = await verifyBackendCredentials(email, password, {
          totpCode: str(credentials?.totpCode),
          emailCode: str(credentials?.emailCode),
          recoveryCode: str(credentials?.recoveryCode),
        });
        // The 2FA outcomes ride the CredentialsSignin ``code`` so the login
        // form can pivot to (or stay on) its verification-code step; the
        // methods list travels inside the code string.
        if (result.kind === "2fa_required") {
          throw credentialsError(`2fa_required:${result.methods}`);
        }
        if (result.kind === "2fa_invalid") throw credentialsError("2fa_invalid");
        // Surfaced so the login form can pivot to its email-confirmation step
        // instead of showing a dead-end "sign-in failed".
        if (result.kind === "email_unverified") throw credentialsError("email_unverified");
        if (result.kind !== "ok") return null;
        return {
          id: result.account.email,
          name: result.account.name,
          email: result.account.email,
          groups: [],
          role: result.account.role,
        };
      },
    }),
  );
  providers.push(
    Credentials({
      id: "passkey",
      name: "Passkey",
      credentials: {
        assertion: { label: "Assertion", type: "text" },
      },
      async authorize(credentials) {
        const assertion = credentials?.assertion;
        if (typeof assertion !== "string" || !assertion) return null;
        const account = await verifyBackendPasskey(assertion);
        if (!account) return null;
        return {
          id: account.email,
          name: account.name,
          email: account.email,
          groups: [],
          role: account.role,
        };
      },
    }),
  );
}

export const { handlers, auth } = NextAuth({
  providers,
  session: { strategy: "jwt" },
  pages: { signIn: "/login" },
  callbacks: {
    authorized({ auth: session }) {
      return !!session?.user;
    },
    jwt({ token, user }) {
      if (user) {
        token.name = user.name ?? token.name;
        token.email = user.email ?? token.email;
        token.groups = normalizeStringList(user.groups);
        // OAuth providers carry no role/groups; derive admin from the email
        // allowlist so AUTH_ADMINS grants admin for Google/GitHub exactly as
        // it does for SSO and email/password accounts.
        const identity = String(user.email ?? user.name ?? "").toLowerCase();
        token.role = user.role ?? (isAdmin(identity, token.groups) ? "admin" : "user");
      }
      return token;
    },
    session({ session, token }) {
      if (typeof token.name === "string") session.user.name = token.name;
      if (typeof token.email === "string") session.user.email = token.email;
      session.user.role = token.role ?? "user";
      session.user.groups = normalizeStringList(token.groups);
      session.backendAccessToken = signBackendToken(token);
      return session;
    },
  },
  trustHost: true,
});
