"""Canonical registry of bring-your-own-key (BYOK) providers.

Single source of truth for the set of providers a user may bring a key for and
the bridge between *where the key is saved* (the vault slug) and *what prefix
that provider's model ids carry* (the LiteLLM prefix). Both the vault — which
stores and verifies keys, keyed by slug — and the model catalog — which offers
each provider's registry models, keyed by LiteLLM prefix — derive their provider
sets from here, so the two can never drift: a divergence would let a key be saved
for a provider whose models aren't offered, or a model be offered for a provider
no key can be saved for (the exact failure class this registry exists to rule
out).

Deliberately a stdlib-only leaf with no imports back into ``core.api`` or
``core.billing``: the model catalog must reach it without dragging in the Stripe
import chain that ``core.billing`` pulls at package-import time, and the vault
must reach it without a cycle. The frontend keeps a parallel copy in
``frontend/src/features/billing/lib/byok.ts`` (it also carries UI-only labels and
placeholders); a parity test pins that copy's slugs and bridge to this registry.
"""

from __future__ import annotations

# Ordered ``(vault slug, LiteLLM provider prefix)`` for every BYOK provider. The
# slug is what a user saves a key under (and how the vault keys it); the prefix
# is what that provider's model ids carry in the catalog. The platform brokers
# every LLM call through OpenRouter, so OpenRouter is the only key worth
# bringing — a direct-provider key would pay for models the catalog never
# offers. Self-hosted/on-prem gateways remain reachable through the vault's
# custom ``api_base`` path, which accepts any slug. The bridge maps below stay
# derived (empty today) so a future slug≠prefix provider needs no new plumbing.
BYOK_PROVIDER_SLUGS: tuple[tuple[str, str], ...] = (
    ("openrouter", "openrouter"),
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
