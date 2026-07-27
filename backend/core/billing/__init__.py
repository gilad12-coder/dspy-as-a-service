"""Stripe-backed managed-credit billing.

Owns customers, credit-pack checkout, and webhook reconciliation. The rest of
the app reaches billing only through :class:`StripeBillingService`; nothing
else imports ``stripe``.
"""

from __future__ import annotations

from .byok_bridge import inject_byok_connections, provider_slug_for_model
from .byok_vault import (
    ProviderKeyVault,
    ProviderKeyView,
    VaultSnapshot,
    byok_provider_for_litellm,
)
from .service import (
    FREE_GRANT_CREDITS,
    PACK_CREDITS,
    LedgerRow,
    StripeBillingService,
    WalletSnapshot,
    cost_ceiling_budget,
)

__all__ = [
    "FREE_GRANT_CREDITS",
    "PACK_CREDITS",
    "LedgerRow",
    "ProviderKeyVault",
    "ProviderKeyView",
    "StripeBillingService",
    "VaultSnapshot",
    "WalletSnapshot",
    "byok_provider_for_litellm",
    "cost_ceiling_budget",
    "inject_byok_connections",
    "provider_slug_for_model",
]
