import { createHmac, randomUUID } from "crypto";
import NextAuth from "next-auth";
import type { Provider } from "next-auth/providers";
import Credentials from "next-auth/providers/credentials";

const issuer = process.env.AUTH_SSO_ISSUER;
const clientId = process.env.AUTH_SSO_CLIENT_ID;
const clientSecret = process.env.AUTH_SSO_CLIENT_SECRET;
const adfsConfigured = Boolean(issuer && clientId && clientSecret);
const scope = process.env.AUTH_SSO_SCOPE ?? "openid profile email groups";
const groupClaim = process.env.AUTH_GROUP_CLAIM ?? "groups";
const usernameClaim = process.env.AUTH_USERNAME_CLAIM?.trim();
const backendAuthSecret = process.env.BACKEND_AUTH_SECRET ?? process.env.AUTH_SECRET;
const backendTokenTtlSeconds = Number.parseInt(
  process.env.BACKEND_AUTH_TOKEN_TTL_SECONDS ?? "900",
  10,
);
const backendBaseUrl =
  process.env.BACKEND_INTERNAL_URL ??
  process.env.API_URL ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000";

function envSet(value: string | undefined): Set<string> {
  return new Set(
    (value ?? "")
      .split(",")
      .map((entry) => entry.trim().toLocaleLowerCase())
      .filter(Boolean),
  );
}

const ADMIN_LIST = envSet(process.env.AUTH_ADMINS);
const ADMIN_GROUPS = envSet(process.env.AUTH_ADMIN_GROUPS);

type BackendAccount = {
  username: string;
  display_name: string;
  role: "admin" | "user";
};

type AuthUser = {
  id?: string;
  name?: string | null;
  email?: string | null;
  username?: string;
  displayName?: string;
  groups?: string[];
  role?: "admin" | "user";
  externalAdmin?: boolean;
};

function readClaim(profile: Record<string, unknown>, path: string): unknown {
  return path.split(".").reduce<unknown>((value, part) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
    return (value as Record<string, unknown>)[part];
  }, profile);
}

function normalizeStringList(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value
      .map(String)
      .map((entry) => entry.trim())
      .filter(Boolean);
  }
  if (typeof value !== "string") return [];
  return value
    .split(",")
    .map((entry) => entry.trim())
    .filter(Boolean);
}

function normalizeUsername(value: unknown): string {
  return typeof value === "string" ? value.trim().toLocaleLowerCase() : "";
}

function profileUsername(profile: Record<string, unknown>): string {
  const configured = usernameClaim ? readClaim(profile, usernameClaim) : undefined;
  return normalizeUsername(
    configured ??
      profile.name ??
      profile.unique_name ??
      profile.upn ??
      profile.preferred_username ??
      profile.email ??
      profile.sub,
  );
}

function profileGroups(profile: Record<string, unknown>): string[] {
  return normalizeStringList(readClaim(profile, groupClaim));
}

function isExternalAdmin(username: string, groups: string[]): boolean {
  if (ADMIN_LIST.has(username)) return true;
  return groups.some((group) => ADMIN_GROUPS.has(group.toLocaleLowerCase()));
}

async function resolveBackendAccount(
  path: "/auth/local/login" | "/auth/sso/provision",
  body: Record<string, unknown>,
): Promise<BackendAccount | null> {
  if (!backendAuthSecret) return null;
  try {
    const response = await fetch(`${backendBaseUrl}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Internal-Auth": backendAuthSecret,
      },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    if (!response.ok) return null;
    return (await response.json()) as BackendAccount;
  } catch {
    return null;
  }
}

function base64url(value: Buffer | string): string {
  return Buffer.from(value).toString("base64url");
}

function signBackendToken(token: {
  username?: unknown;
  email?: unknown;
  role?: unknown;
  groups?: unknown;
}): string | undefined {
  if (!backendAuthSecret) return undefined;
  const username = normalizeUsername(token.username);
  if (!username) return undefined;
  const now = Math.floor(Date.now() / 1000);
  const header = { alg: "HS256", typ: "JWT" };
  const payload = {
    aud: "skynet-backend",
    iss: "skynet-frontend",
    sub: username,
    name: username,
    email: typeof token.email === "string" ? token.email : undefined,
    role: token.role === "admin" ? "admin" : "user",
    groups: normalizeStringList(token.groups),
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

const providers: Provider[] = [];

if (adfsConfigured) {
  providers.push({
    id: "adfs",
    name: "ADFS",
    type: "oidc",
    issuer,
    clientId,
    clientSecret,
    authorization: { params: { scope } },
    profile(rawProfile) {
      const profile = rawProfile as Record<string, unknown>;
      const username = profileUsername(profile);
      const groups = profileGroups(profile);
      const displayName = String(profile.display_name ?? profile.name ?? username);
      return {
        id: username,
        name: username,
        email:
          typeof profile.email === "string"
            ? profile.email
            : typeof profile.upn === "string"
              ? profile.upn
              : null,
        username,
        displayName,
        groups,
        role: isExternalAdmin(username, groups) ? "admin" : "user",
        externalAdmin: isExternalAdmin(username, groups),
      };
    },
  });
}

providers.push(
  Credentials({
    id: "local",
    name: "Local username",
    credentials: {
      username: { label: "Username", type: "text" },
    },
    async authorize(credentials) {
      const username = normalizeUsername(credentials?.username);
      if (!username) return null;
      const account = await resolveBackendAccount("/auth/local/login", { username });
      if (!account) return null;
      return {
        id: account.username,
        name: account.username,
        username: account.username,
        displayName: account.display_name,
        groups: [],
        role: account.role,
      };
    },
  }),
);

export const { handlers, auth } = NextAuth({
  providers,
  session: { strategy: "jwt" },
  pages: { signIn: "/login" },
  callbacks: {
    authorized({ auth: session }) {
      return Boolean(session?.user);
    },
    async signIn({ user, account }) {
      if (account?.provider !== "adfs") return true;
      const resolved = user as AuthUser;
      const username = normalizeUsername(resolved.username ?? resolved.id ?? resolved.name);
      if (!username) return false;
      const provisioned = await resolveBackendAccount("/auth/sso/provision", {
        username,
        display_name: resolved.displayName ?? resolved.name ?? username,
        external_admin: resolved.externalAdmin === true,
      });
      if (!provisioned) return false;
      resolved.id = provisioned.username;
      resolved.name = provisioned.username;
      resolved.username = provisioned.username;
      resolved.displayName = provisioned.display_name;
      resolved.role =
        resolved.externalAdmin || provisioned.role === "admin" ? "admin" : "user";
      return true;
    },
    jwt({ token, user }) {
      if (user) {
        const resolved = user as AuthUser;
        token.username = normalizeUsername(resolved.username ?? resolved.id ?? resolved.name);
        token.name = String(token.username ?? "");
        token.email = resolved.email ?? token.email;
        token.groups = resolved.groups ?? [];
        token.role = resolved.role === "admin" ? "admin" : "user";
      }
      return token;
    },
    session({ session, token }) {
      const username = normalizeUsername(token.username ?? token.name);
      session.user.name = username;
      if (typeof token.email === "string") session.user.email = token.email;
      session.user.role = token.role === "admin" ? "admin" : "user";
      session.user.groups = normalizeStringList(token.groups);
      session.backendAccessToken = signBackendToken(token);
      return session;
    },
  },
  trustHost: true,
});
