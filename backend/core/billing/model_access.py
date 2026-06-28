"""Server-side frontier/mini model-access policy for managed runs.

Mirrors the frontend ``features/billing/lib/model-access.ts`` so the frontier
lock the wizard shows is also *enforced* at submit, not merely advisory. In
managed mode an account without a purchased balance (or active Premium) may run
mini models on its free grant but not the frontier task/reflection models; BYOK
runs on the user's own key and never lock. Tier is inferred from the model id by
family hint until the catalog tags a real tier — unknown ids stay accessible (we
never lock what we can't confidently classify).
"""

from __future__ import annotations

from ..constants import TOKEN_SOURCE_MANAGED

# Small/cheap families — always runnable on the free grant. Checked first so a
# "gpt-5.5-mini" resolves to mini even though it also matches a frontier family.
_MINI_HINTS = ("mini", "haiku", "flash", "small", "lite", "nano", "gemma", "8b", "7b")

# Premium families gated behind a purchased balance in managed mode.
_FRONTIER_HINTS = (
    "opus",
    "sonnet",
    "gpt-5",
    "gpt-4o",
    "o3",
    "o4",
    "gemini-3-pro",
    "grok",
    "deepseek-r1",
    "405b",
)

MODEL_TIER_MINI = "mini"
MODEL_TIER_FRONTIER = "frontier"


def model_tier(model_value: str) -> str:
    """Classify a model id into a coarse capability/price tier.

    Args:
        model_value: The fully-qualified model id (e.g. ``openai/gpt-5.5-mini``).

    Returns:
        ``"mini"`` or ``"frontier"``; unknown ids default to ``"mini"`` so they
        are never gated.
    """
    v = model_value.lower()
    if any(hint in v for hint in _MINI_HINTS):
        return MODEL_TIER_MINI
    if any(hint in v for hint in _FRONTIER_HINTS):
        return MODEL_TIER_FRONTIER
    return MODEL_TIER_MINI


def is_model_locked(model_value: str, token_source: str, frontier_unlocked: bool) -> bool:
    """Return whether a model is locked for the given mode + entitlement.

    Only managed mode locks anything, and only the frontier tier. BYOK never
    locks (the run bills the user's own provider). An account is entitled to
    frontier when ``frontier_unlocked`` is true (it holds purchased credits or an
    active Premium subscription).

    Args:
        model_value: The fully-qualified model id.
        token_source: ``"managed"`` or ``"byok"``.
        frontier_unlocked: Whether the account may run frontier models.

    Returns:
        ``True`` when the run must be refused because the model is frontier-locked.
    """
    if token_source != TOKEN_SOURCE_MANAGED:
        return False
    if frontier_unlocked:
        return False
    return model_tier(model_value) == MODEL_TIER_FRONTIER
