/**
 * Canonical backend identity for the signed-in user.
 *
 * The backend keys every piece of ownership — jobs, datasets, share grants —
 * on the value `signBackendToken` (in `auth.ts`) writes to the JWT `name`
 * claim: `email || displayName`, lowercased server-side. Any request that
 * scopes to the caller's own data — `owner_username` / `shared_with_username`
 * on `/dashboard/search` and `/dashboard/facets`, the sidebar `username`
 * filter, the analytics owner filter, the share dialogs' "is this me?" check —
 * must send THIS value. The human-facing display name (`session.user.name`)
 * is not the identity: `/dashboard/search`'s strict owner-match rejects a
 * display-name scope with a 403 ("auth.owner_mismatch").
 */

import type { Session } from "next-auth";

/**
 * The lowercased identity the backend authorizes ownership against. Mirrors the
 * backend's `subject = email || displayName`. Returns "" when signed out.
 */
export function sessionIdentity(session: Session | null | undefined): string {
  return (session?.user?.email ?? session?.user?.name ?? "").trim().toLowerCase();
}
