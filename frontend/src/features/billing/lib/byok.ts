/**
 * Bring-your-own-key (BYOK) provider-key domain model.
 *
 * When an account runs in `byok` token mode, jobs are billed to the user's own
 * provider key instead of Skynet credits. A key is saved once per provider,
 * shown only masked afterwards, and carries a verification state so the UI can
 * tell a typo'd key from a working one before a job ever runs.
 *
 * The store is backed by the real encrypt-at-rest vault (`/billing/byok/keys`):
 * the secret is encrypted before it touches the database and verified on entry,
 * so the UI only ever holds the masked tail + verification state — never the
 * plaintext. No React / `next/*` imports so it's safe from server and client.
 */

/** Whether a saved key has been checked against its provider. */
export type KeyStatus = "verified" | "unverified" | "invalid";

/** A provider a user can bring their own key for. `placeholder` hints the key shape. */
export interface ByokProviderInfo {
  slug: string;
  label: string;
  placeholder: string;
}

/** A saved provider key as the UI sees it — never the secret, only its tail + state. */
export interface ProviderKey {
  /** Matches a `ByokProviderInfo.slug`. */
  provider: string;
  /** Last 4 characters of the secret, for recognition without revealing it. */
  last4: string;
  status: KeyStatus;
  /** ISO-8601 instant the key was saved. */
  addedAt: string;
}

/**
 * Providers offered for BYOK. Curated to the major vendors a DSPy job is likely
 * to target; the backend will eventually drive this off the live catalog.
 */
export const BYOK_PROVIDERS: ByokProviderInfo[] = [
  { slug: "openai", label: "OpenAI", placeholder: "sk-…" },
  { slug: "anthropic", label: "Anthropic", placeholder: "sk-ant-…" },
  { slug: "google", label: "Google AI", placeholder: "AIza…" },
  { slug: "mistral", label: "Mistral", placeholder: "…" },
  { slug: "openrouter", label: "OpenRouter", placeholder: "sk-or-…" },
];

/** Take the recognizable tail of a secret for masked display. */
export function keyLast4(secret: string): string {
  return secret.slice(-4) || "····";
}
