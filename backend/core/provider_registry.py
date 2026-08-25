"""Define convenient BYOK providers and their LiteLLM prefix mappings.

These entries populate the manual provider picker and bundled model registry.
They are not an allowlist: the JSON connection importer and backend vault also
accept arbitrary provider slugs, endpoints, and pass-through parameter maps.
"""

from __future__ import annotations

# Ordered ``(vault slug, LiteLLM provider prefix)`` conveniences. Custom slugs
# still pass through unchanged.
BYOK_PROVIDER_SLUGS: tuple[tuple[str, str], ...] = (
    ("openai", "openai"),
    ("anthropic", "anthropic"),
    ("google", "gemini"),
    ("groq", "groq"),
    ("deepseek", "deepseek"),
    ("xai", "xai"),
    ("together", "together_ai"),
    ("openrouter", "openrouter"),
    ("cerebras", "cerebras"),
    ("fireworks", "fireworks_ai"),
    ("cohere", "cohere_chat"),
    ("mistral", "mistral"),
    ("moonshot", "moonshot"),
    ("volcengine", "volcengine"),
    ("novita", "novita"),
    ("ollama", "ollama"),
)

# vault slug -> LiteLLM prefix, listing only the providers whose two names differ
# (identity for every other slug). Used to resolve a saved key for a model id.
BYOK_TO_LITELLM_PROVIDER: dict[str, str] = {
    slug: prefix for slug, prefix in BYOK_PROVIDER_SLUGS if slug != prefix
}

# The reverse: LiteLLM prefix -> vault slug, for going from a model id back to
# the slug the user saved their key under.
LITELLM_TO_BYOK_PROVIDER: dict[str, str] = {
    prefix: slug for slug, prefix in BYOK_PROVIDER_SLUGS if slug != prefix
}

# The LiteLLM provider prefixes whose registry models the BYOK catalog offers.
BYOK_CATALOG_PREFIXES: frozenset[str] = frozenset(prefix for _, prefix in BYOK_PROVIDER_SLUGS)
