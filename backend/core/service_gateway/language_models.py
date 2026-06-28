"""DSPy language model factory.

Builds ``dspy.LM`` instances from ``ModelConfig`` while filtering out
``None`` optional fields so LiteLLM does not reject the call.
"""

import dspy

from ..config import settings
from ..exceptions import ServiceError
from ..models import ModelConfig

_DEFAULT_REASONING_MAX_TOKENS = 4000
"""Floor on ``max_tokens`` for chat-style replies. Below this a reasoning model
can truncate mid-``tool_calls`` and emit a malformed call that dspy's ToolCalls
parser rejects with a ValidationError."""

_OPENAI_REASONING_MAX_TOKENS = 16000
"""Mandatory ``max_tokens`` floor for OpenAI reasoning models — dspy validates
``max_tokens >= 16000`` (and ``temperature == 1.0``) at ``dspy.LM`` init."""


def _is_openai_reasoning_model(model_name: str) -> bool:
    """Detect OpenAI reasoning models (gpt-5.x, o1/o3/o4 series).

    These require ``temperature=1.0`` and ``max_tokens >= 16000`` at ``dspy.LM``
    init; they also emit thinking on the ``reasoning_content`` channel when
    ``reasoning_effort`` is set. Fireworks/OpenRouter hosts of these models
    don't share the same constraints, so we scope to the ``openai/`` prefix.

    Args:
        model_name: The fully-qualified model identifier.

    Returns:
        True when ``model_name`` is an OpenAI-hosted reasoning model.
    """
    lower = model_name.lower()
    if not lower.startswith("openai/"):
        return False
    tail = lower.removeprefix("openai/")
    return tail.startswith(("gpt-5", "o1", "o3", "o4"))


def apply_model_reasoning_config(config: ModelConfig) -> ModelConfig:
    """Return a copy of ``config`` with model-specific reasoning defaults applied.

    Mirrors the provider knobs the production generalist agent relies on so any
    code path that builds a student/agent LM from a bare ``ModelConfig`` gets a
    safe ``max_tokens`` floor and the right reasoning extras — without which a
    minimax/reasoning model with no ``max_tokens`` truncates into malformed
    ``tool_calls`` (a dspy ToolCalls ValidationError).

    Defaults, by provider:

    - **Native MiniMax** (``minimax/...``): ``extra_body={"reasoning_split": true}``
      surfaces the interleaved ``<think>`` channel; ``max_tokens`` floored at 4000.
    - **OpenAI reasoning models** (``openai/gpt-5.*``, ``openai/o1|o3|o4*``):
      ``reasoning_effort="medium"``, ``temperature=1.0``, ``max_tokens`` floored
      at 16000.
    - **Everything else** (incl. Fireworks/OpenRouter MiniMax): ``max_tokens``
      floored at 4000, no reasoning knob.

    Caller-supplied values win: a larger ``max_tokens`` is never shrunk, an
    explicit ``temperature`` is never overwritten, and ``config.extra`` overrides
    the model-specific extras on conflict.

    Args:
        config: Provider-agnostic model configuration to normalize.

    Returns:
        A new ``ModelConfig`` with the reasoning defaults merged in.
    """
    lower = config.name.lower()
    model_extra: dict[str, object] = {}
    floor = _DEFAULT_REASONING_MAX_TOKENS
    temperature = config.temperature

    is_native_minimax = lower.startswith("minimax/") or (
        "minimax" in lower and "fireworks" not in lower and "openrouter" not in lower
    )
    if is_native_minimax:
        model_extra["extra_body"] = {"reasoning_split": True}
    elif _is_openai_reasoning_model(config.name):
        model_extra["reasoning_effort"] = "medium"
        floor = _OPENAI_REASONING_MAX_TOKENS
        if temperature is None:
            temperature = 1.0

    max_tokens = floor if config.max_tokens is None else max(config.max_tokens, floor)
    return config.model_copy(
        update={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra": {**model_extra, **config.extra},
        }
    )


def build_language_model(config: ModelConfig, *, disable_cache: bool = False) -> dspy.LM:
    """Construct a DSPy language model from a ModelConfig.

    Only non-None optional fields (temperature, base_url, max_tokens, top_p) are
    forwarded to ``dspy.LM`` to avoid LiteLLM rejecting unexpected None values.
    Extra kwargs from ``config.extra`` are merged in last.

    Args:
        config: Provider-agnostic model configuration.
        disable_cache: When True, force ``cache=False`` so retries always hit
            the provider. Used for user-facing surfaces (agents, serve) where
            replaying a cached response would defeat the regenerate action.

    Returns:
        A configured ``dspy.LM`` ready for use by an optimizer.

    Raises:
        ServiceError: When ``dspy.LM`` rejects the configuration.
    """

    model_name = config.name.strip("/")
    # Default per-request timeout guards against a provider that accepts the
    # connection but never sends a response — without it the SSL socket read
    # blocks forever and wedges the whole optimization run. Set before the
    # config.extra merge below so an explicit per-model timeout still wins.
    # dspy defaults to ``num_retries=3``; with our per-call timeout that lets a
    # hung provider burn up to ``(retries + 1) * timeout`` seconds of silence,
    # which meets or exceeds ``job_stall_timeout_seconds`` and trips the run's
    # stall watchdog *before* the call itself errors — turning a recoverable
    # per-call timeout into an opaque whole-run failure (observed: a GEPA
    # reflection call wedged ~1800s and the watchdog killed a 9-hour run). Cap
    # retries so the worst-case attempt sequence finishes under the watchdog
    # with one timeout of margin, keeping the watchdog's documented invariant
    # ("a hung call times out first") true even with retries.
    safe_attempts = max(1, int(settings.job_stall_timeout_seconds // settings.lm_request_timeout_seconds) - 1)
    lm_kwargs: dict[str, object] = {
        "model": model_name,
        "timeout": settings.lm_request_timeout_seconds,
        "num_retries": safe_attempts - 1,
    }
    if config.temperature is not None:
        lm_kwargs["temperature"] = config.temperature
    if config.base_url:
        lm_kwargs["base_url"] = config.base_url
    if config.max_tokens is not None:
        lm_kwargs["max_tokens"] = config.max_tokens
    if config.top_p is not None:
        lm_kwargs["top_p"] = config.top_p
    lm_kwargs.update(config.extra)
    if disable_cache:
        lm_kwargs["cache"] = False
    try:
        language_model = dspy.LM(**lm_kwargs)
    except ValueError as exc:
        raise ServiceError(f"Failed to build language model '{config.name}': {exc}") from exc

    return language_model


def _usage_total_tokens(usage: object) -> int | None:
    """Extract a non-negative total-token count from one history ``usage`` block.

    Args:
        usage: The ``usage`` value from a ``dspy.LM`` history entry — a mapping,
            a provider usage object, or ``None``.

    Returns:
        The entry's total token count, or ``None`` when ``usage`` carries no
        recognizable token fields.
    """
    if usage is None:
        return None
    get = usage.get if isinstance(usage, dict) else lambda key: getattr(usage, key, None)
    total = get("total_tokens")
    if isinstance(total, (int, float)) and total > 0:
        return int(total)
    parts = [int(p) for p in (get("prompt_tokens"), get("completion_tokens")) if isinstance(p, (int, float))]
    return sum(parts) if parts else None


def total_tokens_from_history(*language_models: object) -> int | None:
    """Sum the total token usage recorded across one or more LMs' call histories.

    Reads the ``usage`` block each ``dspy.LM`` stamps onto every ``history``
    entry (the same source ``num_lm_calls`` counts), summing ``total_tokens`` —
    or ``prompt_tokens + completion_tokens`` when a provider omits the total.
    This is the per-run token figure the billing worker meters to Stripe.

    Args:
        *language_models: LMs whose histories to total; ``None`` entries and LMs
            without a ``history`` list are skipped.

    Returns:
        The summed token count, or ``None`` when no usage information is present
        (e.g. mocked LMs in tests) so callers can tell "zero usage" apart from
        "usage not tracked" and skip metering rather than bill nothing.
    """
    total = 0
    found = False
    for language_model in language_models:
        history = getattr(language_model, "history", None)
        if not isinstance(history, list):
            continue
        for entry in history:
            tokens = _usage_total_tokens(entry.get("usage") if isinstance(entry, dict) else None)
            if tokens is not None:
                total += tokens
                found = True
    return total if found else None
