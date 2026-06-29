"""Stripe-backed managed-credit billing.

Owns customers, credit-pack checkout, the Premium subscription, the customer
portal, and webhook reconciliation. The rest of the app reaches billing only
through :class:`StripeBillingService`; nothing else imports ``stripe``.
"""

from __future__ import annotations

from .byok_bridge import inject_byok_connections, provider_slug_for_model
from .byok_vault import ProviderKeyVault, ProviderKeyView, VaultSnapshot
from .service import (
    FREE_GRANT_CREDITS,
    PACK_CREDITS,
    LedgerRow,
    StripeBillingService,
    WalletSnapshot,
    tokens_for_credits,
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
    "inject_byok_connections",
    "provider_slug_for_model",
    "tokens_for_credits",
]
