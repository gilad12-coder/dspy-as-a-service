"""Pin BYOK provider shortcuts to the canonical ``core.provider_registry``.

The registry is the source of truth for the manual picker and bundled catalog.
Arbitrary JSON-imported connections remain valid outside this convenience set.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.api.model_catalog import _BYOK_CATALOG_PROVIDERS
from core.byok.vault import _PROVIDER_PROBES
from core.provider_registry import (
    BYOK_CATALOG_PREFIXES,
    BYOK_PROVIDER_SLUGS,
    BYOK_TO_LITELLM_PROVIDER,
)

_FRONTEND_BYOK_TS = (
    Path(__file__).resolve().parents[3]
    / "frontend"
    / "src"
    / "features"
    / "byok"
    / "lib"
    / "byok.ts"
)


def _frontend_byok_source() -> str:
    """Return the text of the frontend BYOK catalog module.

    Returns:
        The full source of ``frontend/src/features/byok/lib/byok.ts``.
    """
    return _FRONTEND_BYOK_TS.read_text(encoding="utf-8")


def test_vault_probe_table_only_uses_registry_slugs() -> None:
    """Every built-in verification probe belongs to a provider shortcut."""
    registry_slugs = {slug for slug, _ in BYOK_PROVIDER_SLUGS}
    assert set(_PROVIDER_PROBES) <= registry_slugs


def test_catalog_prefixes_match_the_registry() -> None:
    """The model catalog offers exactly the registry's LiteLLM prefixes."""
    assert _BYOK_CATALOG_PROVIDERS == BYOK_CATALOG_PREFIXES


def test_frontend_catalog_slugs_match_the_registry_in_order() -> None:
    """The frontend ``BYOK_PROVIDERS`` slugs equal the registry slugs, in order."""
    slugs = re.findall(r'slug:\s*"([^"]+)"', _frontend_byok_source())
    assert tuple(slugs) == tuple(slug for slug, _ in BYOK_PROVIDER_SLUGS)


def test_frontend_bridge_matches_the_registry() -> None:
    """The frontend slug->prefix bridge equals the registry's, exactly."""
    source = _frontend_byok_source()
    body = source[source.index("BYOK_TO_LITELLM_PROVIDER") :]
    body = body[body.index("{") + 1 : body.index("}")]
    bridge = dict(re.findall(r'(\w+):\s*"([^"]+)"', body))
    assert bridge == BYOK_TO_LITELLM_PROVIDER
