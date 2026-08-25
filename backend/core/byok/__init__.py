"""Provider-agnostic bring-your-own-key connection support."""

from .bridge import (
    inject_byok_connections,
    payload_uses_token_source,
    provider_slug_for_model,
    resolve_byok_model_config,
)
from .vault import (
    ProviderKeyVault,
    ProviderKeyView,
    VaultSnapshot,
    byok_provider_for_litellm,
)

__all__ = [
    "ProviderKeyVault",
    "ProviderKeyView",
    "VaultSnapshot",
    "byok_provider_for_litellm",
    "inject_byok_connections",
    "payload_uses_token_source",
    "provider_slug_for_model",
    "resolve_byok_model_config",
]
