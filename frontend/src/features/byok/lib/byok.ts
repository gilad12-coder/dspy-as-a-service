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

/** Whether model credentials come from the deployment or the user's vault. */
export type TokenSourceMode = "managed" | "byok";

/** A provider a user can bring their own key for. `placeholder` hints the key shape. */
export interface ByokProviderInfo {
  slug: string;
  label: string;
  placeholder: string;
}

/** A saved provider connection as the UI sees it — never the secret, only its tail + state. */
export interface ProviderKey {
  /** Stable handle for the connection. */
  id: string;
  /** Matches a `ByokProviderInfo.slug`. */
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
 * Convenient provider shortcuts. Saved arbitrary providers are appended at
 * runtime, and the JSON importer accepts any provider slug and parameter map.
 */
export const BYOK_PROVIDERS: ByokProviderInfo[] = [
  { slug: "openai", label: "OpenAI", placeholder: "sk-…" },
  { slug: "anthropic", label: "Anthropic", placeholder: "sk-ant-…" },
  { slug: "google", label: "Google Gemini", placeholder: "AIza…" },
  { slug: "groq", label: "Groq", placeholder: "gsk_…" },
  { slug: "deepseek", label: "DeepSeek", placeholder: "sk-…" },
  { slug: "xai", label: "xAI", placeholder: "xai-…" },
  { slug: "together", label: "Together AI", placeholder: "…" },
  { slug: "openrouter", label: "OpenRouter", placeholder: "sk-or-…" },
  { slug: "cerebras", label: "Cerebras", placeholder: "csk-…" },
  { slug: "fireworks", label: "Fireworks AI", placeholder: "fw_…" },
  { slug: "cohere", label: "Cohere", placeholder: "…" },
  { slug: "mistral", label: "Mistral", placeholder: "…" },
  { slug: "moonshot", label: "Moonshot", placeholder: "…" },
  { slug: "volcengine", label: "Volcengine", placeholder: "…" },
  { slug: "novita", label: "Novita AI", placeholder: "…" },
  { slug: "ollama", label: "Ollama", placeholder: "local" },
];

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
