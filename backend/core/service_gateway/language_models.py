"""DSPy language model factory.

Builds ``dspy.LM`` instances from ``ModelConfig`` while filtering out
``None`` optional fields so LiteLLM does not reject the call.
"""

import logging
import threading
from dataclasses import dataclass

import dspy

from ..config import settings
from ..exceptions import ServiceError
from ..models import ModelConfig

try:
    from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
except ImportError:  # pragma: no cover - depends on litellm internals
    CustomStreamWrapper = None

logger = logging.getLogger("skynet.service_gateway.language_models")

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


def apply_reasoning_effort(config: ModelConfig, effort: str | None) -> ModelConfig:
    """Stamp a caller-chosen reasoning effort onto ``config``.

    The effort lands in ``extra`` so it survives
    :func:`apply_model_reasoning_config` — caller extras win over the
    model-specific defaults there, so an explicit choice overrides the
    ``"medium"`` default applied to OpenAI reasoning models.

    Args:
        config: The model configuration to annotate.
        effort: LiteLLM ``reasoning_effort`` level chosen by the user, or
            ``None``/empty to leave the model's default behaviour untouched.

    Returns:
        ``config`` unchanged when ``effort`` is falsy, else a copy with the
        effort merged into ``extra``.
    """
    if not effort:
        return config
    return config.model_copy(update={"extra": {**config.extra, "reasoning_effort": effort}})


def _apply_managed_gateway(lm_kwargs: dict[str, object]) -> None:
    """Route a managed call through the self-hosted LiteLLM proxy when configured.

    A BYOK call already carries the user's ``api_key`` (stamped onto the
    ModelConfig by the run-path bridge); a managed call does not — so the
    presence of ``api_key`` distinguishes the two here without threading the
    token source down to the LM factory. When a proxy URL is configured and the
    call is managed (no ``api_key``) and not already pinned to a specific
    endpoint (no ``base_url``), point it at the proxy via litellm's
    ``litellm_proxy/`` provider and authenticate with the managed virtual key,
    so all platform inference flows through one metered seam. A no-op when no
    proxy is configured, leaving the default dspy → provider path unchanged;
    BYOK and endpoint-pinned calls are never rerouted.

    Args:
        lm_kwargs: The ``dspy.LM`` kwargs assembled so far, mutated in place.
    """
    if "api_key" in lm_kwargs or "base_url" in lm_kwargs:
        return
    proxy_url = settings.litellm_proxy_url
    if not proxy_url:
        return
    lm_kwargs["base_url"] = proxy_url
    if settings.litellm_proxy_api_key is not None:
        lm_kwargs["api_key"] = settings.litellm_proxy_api_key.get_secret_value()
    # Address the proxy through litellm's dedicated ``litellm_proxy/`` provider so
    # the OpenRouter slug reaches it intact. Without the prefix litellm resolves
    # the bare provider segment itself (``openai/gpt-4o-mini`` -> openai provider,
    # sending just ``gpt-4o-mini``), and the proxy's ``*`` -> ``openrouter/*``
    # wildcard then can't reconstruct a real slug. A leading ``openrouter/`` is
    # dropped first because that wildcard re-adds it — otherwise an already
    # OpenRouter-prefixed id (``openrouter/minimax/...``) would double-prefix.
    model = lm_kwargs.get("model")
    if isinstance(model, str):
        lm_kwargs["model"] = f"litellm_proxy/{model.removeprefix('openrouter/')}"


def _translate_gateway_reasoning(lm_kwargs: dict[str, object]) -> None:
    """Mirror ``reasoning_effort`` into OpenRouter's native ``reasoning`` param.

    Calls that reach OpenRouter (directly via an ``openrouter/`` id, or through
    the LiteLLM proxy whose wildcard fronts OpenRouter) lose the OpenAI-style
    ``reasoning_effort`` kwarg: LiteLLM doesn't map it onto OpenRouter's
    ``reasoning`` request param — so a user-picked effort was a no-op and
    opt-in thinking models (Anthropic, Gemini) never streamed reasoning.
    OpenRouter's ceiling vocabulary is ``xhigh``, so Anthropic's ``max`` maps
    down to it.

    The kwarg is kept (aligned to the mapped value) rather than popped: dspy's
    ``Reasoning`` signature field injects ``reasoning_effort="low"`` at call
    time whenever the LM carries no effort of its own, and OpenRouter rejects
    requests whose ``reasoning_effort`` and ``reasoning.effort`` disagree.

    Args:
        lm_kwargs: The ``dspy.LM`` kwargs assembled so far, mutated in place.
    """
    model = lm_kwargs.get("model")
    if not isinstance(model, str) or not model.startswith(("litellm_proxy/", "openrouter/")):
        return
    effort = lm_kwargs.get("reasoning_effort")
    if not isinstance(effort, str) or not effort:
        return
    mapped = "xhigh" if effort == "max" else effort
    body = lm_kwargs.get("extra_body")
    merged = dict(body) if isinstance(body, dict) else {}
    native = merged.setdefault("reasoning", {"effort": mapped})
    if isinstance(native, dict) and isinstance(native.get("effort"), str) and native["effort"]:
        mapped = native["effort"]
    lm_kwargs["reasoning_effort"] = mapped
    lm_kwargs["extra_body"] = merged


# One lock for every MeteredLM: ``LM.copy()`` shallow-copies instances, so an
# instance-level lock would be shared anyway, and contention is a few dozen
# nanoseconds per LM call against a network round-trip.
_USAGE_LOCK = threading.Lock()


@dataclass
class LmUsageTotals:
    """Running usage aggregate for one job LM and every internal copy of it.

    ``total_found``/``split_found`` distinguish "zero usage" from "usage not
    tracked" (mocked providers), mirroring the ``None`` contract of
    :func:`total_tokens_from_history` / :func:`usage_by_model_from_history`.
    """

    calls: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_found: bool = False
    split_found: bool = False


class MeteredLM(dspy.LM):
    """``dspy.LM`` that aggregates usage instead of retaining call history.

    A single optimization makes thousands of LM calls, and stock ``dspy.LM``
    keeps each one's full prompt + raw response three times over (per-LM
    ``history``, module-level ``GLOBAL_HISTORY``, per-module history) for up
    to 10,000 entries — tens to hundreds of MB pinned for the whole run, and
    permanently in the API process for agent/tagging calls. The repo only
    ever consumed history as aggregate counts, so ``update_history`` reduces
    each entry to running totals and drops it.

    Because ``LM.copy()`` is a shallow copy, internal copies dspy makes (e.g.
    rollout clones) share the same ``usage_totals`` object — their calls are
    counted too, which plain history misses (``copy()`` resets ``history``).
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        """Construct the LM and attach a fresh usage aggregate."""
        super().__init__(*args, **kwargs)
        self.usage_totals = LmUsageTotals()
        self.last_request_model: str | None = None
        self.last_response_model: str | None = None

    def update_history(self, entry: object) -> None:
        """Fold one call's usage into the running totals and discard the entry.

        Args:
            entry: The history dict stock dspy would have appended; only its
                ``usage`` block and model ids are read.
        """
        # The served-model reveal reads just these two fields of the entry
        # this class otherwise drops (see served_model_from).
        if isinstance(entry, dict):
            request_model = entry.get("model")
            response_model = entry.get("response_model")
            self.last_request_model = request_model if isinstance(request_model, str) else None
            self.last_response_model = (
                response_model if isinstance(response_model, str) else None
            )
        usage = entry.get("usage") if isinstance(entry, dict) else None
        total = _usage_total_tokens(usage)
        split = _usage_in_out_tokens(usage)
        with _USAGE_LOCK:
            totals = self.usage_totals
            totals.calls += 1
            if total is not None:
                totals.total_tokens += total
                totals.total_found = True
            if split is not None:
                totals.input_tokens += split[0]
                totals.output_tokens += split[1]
                totals.split_found = True


def lm_call_count(language_model: object) -> int | None:
    """Count the LM calls ``language_model`` has made.

    Prefers the ``MeteredLM`` aggregate; falls back to ``len(history)`` for
    plain/mocked LMs (and for a dspy line whose LM never invokes the
    ``update_history`` hook, where history still accumulates).

    Args:
        language_model: The LM whose calls to count.

    Returns:
        The call count, or ``None`` when the object tracks neither form.
    """
    totals = getattr(language_model, "usage_totals", None)
    if isinstance(totals, LmUsageTotals) and totals.calls:
        return totals.calls
    history = getattr(language_model, "history", None)
    return len(history) if isinstance(history, list) else None


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
    _apply_managed_gateway(lm_kwargs)
    _translate_gateway_reasoning(lm_kwargs)
    if disable_cache:
        lm_kwargs["cache"] = False
        # ``cache=False`` only disables the client-side cache — the LiteLLM
        # proxy keeps its own Redis cache whose key ignores params like
        # ``reasoning``, so a user-facing turn could replay a stale cached
        # answer (and cached replays drop reasoning deltas entirely). The
        # body-level directive opts this request out server-side too.
        model = lm_kwargs.get("model")
        if isinstance(model, str) and model.startswith("litellm_proxy/"):
            body = lm_kwargs.get("extra_body")
            merged = dict(body) if isinstance(body, dict) else {}
            merged.setdefault("cache", {"no-cache": True})
            lm_kwargs["extra_body"] = merged
    try:
        language_model = MeteredLM(**lm_kwargs)
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
    """Sum the total token usage recorded across one or more LMs.

    Prefers each LM's ``MeteredLM`` running aggregate; otherwise reads the
    ``usage`` block each stock ``dspy.LM`` stamps onto every ``history`` entry,
    summing ``total_tokens`` — or ``prompt_tokens + completion_tokens`` when a
    provider omits the total. This is the per-run token figure the billing
    worker meters to Stripe.

    Args:
        *language_models: LMs whose usage to total; ``None`` entries and LMs
            tracking neither form are skipped.

    Returns:
        The summed token count, or ``None`` when no usage information is present
        (e.g. mocked LMs in tests) so callers can tell "zero usage" apart from
        "usage not tracked" and skip metering rather than bill nothing.
    """
    total = 0
    found = False
    for language_model in language_models:
        totals = getattr(language_model, "usage_totals", None)
        if isinstance(totals, LmUsageTotals):
            if totals.total_found:
                total += totals.total_tokens
                found = True
            # A metered LM with recorded calls keeps no history; skipping the
            # fallback also guards against double-counting on a dspy line
            # that appends history outside the update_history hook.
            if totals.calls:
                continue
        history = getattr(language_model, "history", None)
        if not isinstance(history, list):
            continue
        for entry in history:
            tokens = _usage_total_tokens(entry.get("usage") if isinstance(entry, dict) else None)
            if tokens is not None:
                total += tokens
                found = True
    return total if found else None


def _usage_in_out_tokens(usage: object) -> tuple[int, int] | None:
    """Split one history ``usage`` block into ``(input, output)`` token counts.

    Returns ``prompt_tokens`` / ``completion_tokens`` when either is present.
    When a provider reports only a combined ``total_tokens``, the whole amount is
    attributed to input — the cheaper side — so the fallback under-charges output
    rather than inventing a split that over-bills. Returns ``None`` when the block
    carries no recognizable token fields.

    Args:
        usage: The ``usage`` value from a ``dspy.LM`` history entry — a mapping,
            a provider usage object, or ``None``.

    Returns:
        The ``(input_tokens, output_tokens)`` pair, or ``None`` when untracked.
    """
    if usage is None:
        return None
    get = usage.get if isinstance(usage, dict) else lambda key: getattr(usage, key, None)
    prompt = get("prompt_tokens")
    completion = get("completion_tokens")
    if isinstance(prompt, (int, float)) or isinstance(completion, (int, float)):
        in_tokens = int(prompt) if isinstance(prompt, (int, float)) and prompt > 0 else 0
        out_tokens = int(completion) if isinstance(completion, (int, float)) and completion > 0 else 0
        if in_tokens or out_tokens:
            return in_tokens, out_tokens
    total = get("total_tokens")
    if isinstance(total, (int, float)) and total > 0:
        return int(total), 0
    return None


def usage_by_model_from_history(*language_models: object) -> dict[str, tuple[int, int]] | None:
    """Aggregate per-model ``(input, output)`` token usage across LMs' histories.

    The per-model companion to :func:`total_tokens_from_history`: it keys usage
    by each ``dspy.LM``'s ``model`` id and preserves the input/output split that
    per-model pricing needs, folding several LMs on the same model together.
    Prefers the ``MeteredLM`` aggregate, falling back to history entries.
    Stays billing-agnostic — it returns plain token counts, leaving the
    cost conversion to :mod:`core.billing.pricing`.

    Args:
        *language_models: LMs whose usage to total; ``None`` entries and LMs
            tracking neither form are skipped. An LM with no ``model``
            attribute buckets under ``"unknown"``.

    Returns:
        A ``model id → (input_tokens, output_tokens)`` mapping, or ``None`` when
        no usage information is present anywhere — mirroring
        :func:`total_tokens_from_history` so callers can tell "zero usage" from
        "usage not tracked" and skip charging rather than bill nothing.
    """
    by_model: dict[str, list[int]] = {}
    found = False
    for language_model in language_models:
        model = getattr(language_model, "model", None) or "unknown"
        totals = getattr(language_model, "usage_totals", None)
        if isinstance(totals, LmUsageTotals):
            if totals.split_found:
                accumulator = by_model.setdefault(model, [0, 0])
                accumulator[0] += totals.input_tokens
                accumulator[1] += totals.output_tokens
                found = True
            # Same skip rationale as total_tokens_from_history.
            if totals.calls:
                continue
        history = getattr(language_model, "history", None)
        if not isinstance(history, list):
            continue
        for entry in history:
            split = _usage_in_out_tokens(entry.get("usage") if isinstance(entry, dict) else None)
            if split is None:
                continue
            accumulator = by_model.setdefault(model, [0, 0])
            accumulator[0] += split[0]
            accumulator[1] += split[1]
            found = True
    if not found:
        return None
    return {model: (in_out[0], in_out[1]) for model, in_out in by_model.items()}


_SERVED_MODEL_PATCH_FLAG = "_skynet_served_model_patch"


def install_openrouter_served_model_patch() -> None:
    """Make LiteLLM streams report the model OpenRouter actually served.

    LiteLLM's ``CustomStreamWrapper`` overwrites every streamed chunk's
    ``model`` with the request id, discarding the concrete model OpenRouter's
    Auto Router names in its raw SSE chunks. Adopt the provider-reported
    model for openrouter calls — the same special-case LiteLLM already ships
    for Azure — so ``lm.history`` records the served model under
    ``response_model`` instead of echoing the router's own id. Idempotent;
    called once from ``create_app``.
    """
    if CustomStreamWrapper is None:
        logger.warning("litellm streaming internals moved; served-model reveal disabled")
        return
    handler = CustomStreamWrapper.handle_openai_chat_completion_chunk
    if getattr(handler, _SERVED_MODEL_PATCH_FLAG, False):
        return

    def _adopting_handler(self, chunk):
        """Adopt the provider-reported model before normal chunk handling."""
        served = getattr(chunk, "model", None)
        # litellm_proxy is included for when the platform fronts OpenRouter
        # with a self-hosted proxy: today the proxy's own litellm strips the
        # routed model (chunks echo the request group, which the suffix-echo
        # guard in served_model_from suppresses), but a passthrough-capable
        # proxy lights the reveal up with no backend change.
        if served and getattr(self, "custom_llm_provider", None) in ("openrouter", "litellm_proxy"):
            self.model = served
        return handler(self, chunk)

    setattr(_adopting_handler, _SERVED_MODEL_PATCH_FLAG, True)
    CustomStreamWrapper.handle_openai_chat_completion_chunk = _adopting_handler


def served_model_from(language_model: object) -> str | None:
    """Concrete model that answered ``language_model``'s latest call, if newsworthy.

    Reads the last ``lm.history`` entry and returns its provider-reported
    ``response_model`` only when it differs from the requested id — i.e. an
    auto-routed call resolved to a concrete model. Explicit picks and
    providers that merely echo the request return ``None`` (LiteLLM strips
    the transport prefix, so a bare suffix echo is not news either).

    Args:
        language_model: A ``dspy.LM``-like object — either a ``MeteredLM``
            (which drops history but stashes the last call's model ids) or a
            stock LM with a ``history`` list.

    Returns:
        The served model id, or ``None`` when there is nothing to reveal.
    """
    served = getattr(language_model, "last_response_model", None)
    requested = getattr(language_model, "last_request_model", None)
    if not served:
        history = getattr(language_model, "history", None)
        if not history:
            return None
        entry = history[-1]
        if not isinstance(entry, dict):
            return None
        served = entry.get("response_model")
        requested = entry.get("model")
    if not served or not isinstance(served, str):
        return None
    if served == requested:
        return None
    if isinstance(requested, str) and requested.endswith(f"/{served}"):
        return None
    return served
