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
from .openrouter_float import FloatStatus, check_float, read_account_balance_credits
from .openrouter_keys import OpenRouterKeyProvisioner, inject_provisioned_openrouter_key
from .service import (
    FREE_GRANT_CREDITS,
    PACK_CREDITS,
    LedgerRow,
    StripeBillingService,
    WalletSnapshot,
    committed_spend_credits,
    cost_ceiling_budget,
)

__all__ = [
    "FREE_GRANT_CREDITS",
    "PACK_CREDITS",
    "FloatStatus",
    "LedgerRow",
    "OpenRouterKeyProvisioner",
    "ProviderKeyVault",
    "ProviderKeyView",
    "StripeBillingService",
    "VaultSnapshot",
    "WalletSnapshot",
    "byok_provider_for_litellm",
    "check_float",
    "committed_spend_credits",
    "cost_ceiling_budget",
    "inject_byok_connections",
    "inject_provisioned_openrouter_key",
    "provider_slug_for_model",
    "read_account_balance_credits",
]
