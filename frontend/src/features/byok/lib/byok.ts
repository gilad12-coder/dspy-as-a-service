/**
 * Bring-your-own-key (BYOK) provider-key domain model.
 *
 * When an account runs in `byok` token mode, the model call uses the user's own
 * provider connection. A key is saved once per provider, shown only masked
 * afterwards, and carries the result of the most recent verification attempt.
 *
 * The store is backed by the encrypt-at-rest vault (`/byok/keys`):
 * the secret is encrypted before it touches the database and verified on entry,
 * so the UI only ever holds the masked tail + verification state — never the
 * plaintext. No React / `next/*` imports so it's safe from server and client.
 */

/** Whether a saved key has been checked against its provider. */
export type KeyStatus = "verified" | "unverified" | "invalid";

/** A saved provider connection as the UI sees it — never the secret, only its tail + state. */
export interface ProviderKey {
  /** Stable handle for the connection. */
  id: string;
  /** Arbitrary connection identifier supplied by the user. */
  provider: string;
  /** Optional user-facing name for the connection. */
  label?: string | null;
  /** Last 4 characters of the secret, for recognition without revealing it. */
  last4: string;
  /** Optional custom endpoint the connection targets. */
  apiBase?: string | null;
  /** Provider-specific LiteLLM arguments carried through without allowlisting. */
  params?: Record<string, unknown>;
  status: KeyStatus;
  /** ISO-8601 instant the key was saved. */
  addedAt: string;
}

/**
 * Maps a connection slug to the LiteLLM provider prefix its models carry.
 * Unknown providers pass through unchanged.
 */
const BYOK_TO_LITELLM_PROVIDER: Record<string, string> = {
  google: "gemini",
  together: "together_ai",
  fireworks: "fireworks_ai",
  cohere: "cohere_chat",
};

/** The LiteLLM provider prefix a BYOK provider slug's catalog models carry. */
export function litellmProviderForByok(slug: string): string {
  return BYOK_TO_LITELLM_PROVIDER[slug] ?? slug;
}
