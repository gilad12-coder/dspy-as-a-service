"""Bridge that hands a BYOK user's stored provider key to the run path.

The run path turns a model string into a ``dspy.LM`` in
``service_gateway.language_models.build_language_model``, which authenticates
with whatever ``api_key`` rides in the ModelConfig's ``extra``. A managed run
gets that key from process env vars; a BYOK run must get it from the user's
encrypted vault. This module is the single seam that injects it: the parent
worker calls :func:`inject_byok_connections` on the raw payload dict just before
it spawns the run subprocess, so the resolved key travels into the child — and
the grid threads it fans out — as ordinary pickled data. A contextvar could not
cross either the process or the thread-pool boundary.

Secrets are injected into the in-memory payload only, after it is loaded from
storage and before the subprocess starts, so a plaintext key is never persisted:
the stored job overview keeps the ``strip_api_key`` invariant.
"""

from __future__ import annotations

from typing import Any

from .byok_vault import ProviderKeyVault

# Payload keys holding ModelConfig blocks. Runs persist their configs under the
# field *aliases* (``model_settings`` → ``"model_config"``); grids use the plain
# list field names. ``task_model_config`` never reaches build_language_model, so
# it is intentionally omitted.
_RUN_MODEL_KEYS = ("model_config", "reflection_model_config")
_GRID_MODEL_LIST_KEYS = ("generation_models", "reflection_models")


def provider_slug_for_model(name: str) -> str | None:
    """Return the provider slug prefixing a LiteLLM model string, or ``None``.

    ``"openai/gpt-4o"`` → ``"openai"``; ``"anthropic/claude-3-5-sonnet"`` →
    ``"anthropic"``. A bare model with no ``provider/`` prefix yields ``None``,
    since there is no provider to resolve a BYOK key for.

    Args:
        name: The model identifier from a ModelConfig.

    Returns:
        The leading provider segment, or ``None`` when the string carries none.
    """
    stripped = (name or "").strip().strip("/")
    if "/" not in stripped:
        return None
    return stripped.split("/", 1)[0] or None


def _model_config_dicts(payload_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect the ModelConfig sub-dicts a run/grid payload carries.

    Args:
        payload_dict: The raw persisted payload for a run or grid submission.

    Returns:
        Every ModelConfig block present, across the run and grid shapes.
    """
    configs: list[dict[str, Any]] = []
    for key in _RUN_MODEL_KEYS:
        cfg = payload_dict.get(key)
        if isinstance(cfg, dict):
            configs.append(cfg)
    for key in _GRID_MODEL_LIST_KEYS:
        entries = payload_dict.get(key)
        if isinstance(entries, list):
            configs.extend(cfg for cfg in entries if isinstance(cfg, dict))
    return configs


def inject_byok_connections(
    payload_dict: dict[str, Any], *, username: str, vault: ProviderKeyVault
) -> None:
    """Stamp the user's vault key onto every ModelConfig in a BYOK payload, in place.

    For each ModelConfig, resolve the provider from its model string and look up
    the account's connection in the vault, then set ``extra['api_key']`` (plus the
    connection's ``api_base`` and extra ``params`` when present).
    ``build_language_model`` already forwards ``extra`` into the ``dspy.LM`` call,
    so the key reaches the provider with no further wiring. A config whose model
    carries no ``provider/`` prefix is left untouched (nothing to resolve).

    Args:
        payload_dict: The raw run/grid payload, mutated in place.
        username: Account the run bills to (the vault owner).
        vault: The provider-key vault to resolve connections from.

    Raises:
        ValueError: When a model's provider has no saved connection — the run
            cannot authenticate, so it is failed with a clear message rather than
            silently falling back to a platform key.
    """
    for cfg in _model_config_dicts(payload_dict):
        provider = provider_slug_for_model(cfg.get("name", ""))
        if provider is None:
            continue
        resolved = vault.resolve_connection(username, provider)
        if resolved is None:
            raise ValueError(
                f"No saved {provider} connection for this account. "
                "Add one in Settings → Providers to run with your own key."
            )
        extra = cfg.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            cfg["extra"] = extra
        extra.setdefault("api_key", resolved.secret)
        if resolved.api_base and not cfg.get("base_url"):
            cfg["base_url"] = resolved.api_base
        for key, value in resolved.params.items():
            extra.setdefault(key, value)
