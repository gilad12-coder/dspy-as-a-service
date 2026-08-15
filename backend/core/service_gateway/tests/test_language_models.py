"""Tests for ``core.service_gateway.language_models.build_language_model``."""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from core.config import settings
from core.exceptions import ServiceError
from core.models import ModelConfig
from core.service_gateway.language_models import (
    CustomStreamWrapper,
    MeteredLM,
    _apply_managed_gateway,
    _translate_gateway_reasoning,
    apply_model_reasoning_config,
    apply_reasoning_effort,
    build_language_model,
    install_openrouter_served_model_patch,
    served_model_from,
    total_tokens_from_history,
    usage_by_model_from_history,
)


@pytest.fixture(autouse=True)
def _no_local_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the managed gateway off so results don't depend on the local ``.env``.

    A developer machine with ``LITELLM_PROXY_URL`` set would otherwise reroute
    every ``build_language_model`` call through ``litellm_proxy/`` and fail the
    plain-path assertions. Gateway-specific tests re-enable the proxy with
    their own ``monkeypatch.setattr`` calls, which run after this fixture.
    """
    monkeypatch.setattr(settings, "litellm_proxy_url", None)
    monkeypatch.setattr(settings, "litellm_proxy_api_key", None)


class _FakeLM:
    """Minimal stand-in exposing ``history`` (and ``model``) like ``dspy.LM``."""

    def __init__(self, history: list[dict[str, Any]], model: str = "unknown") -> None:
        """Store the canned history entries and model id the LM should report.

        Args:
            history: Entries shaped like ``dspy.LM.history`` rows.
            model: Model id the LM reports (read by per-model usage aggregation).
        """
        self.history = history
        self.model = model


def test_total_tokens_from_history_sums_total_tokens() -> None:
    """``total_tokens`` from each entry's usage is summed across all LMs."""
    gen = _FakeLM([{"usage": {"total_tokens": 120}}, {"usage": {"total_tokens": 30}}])
    refl = _FakeLM([{"usage": {"total_tokens": 50}}])
    assert total_tokens_from_history(gen, refl) == 200


def test_total_tokens_from_history_falls_back_to_prompt_plus_completion() -> None:
    """A missing ``total_tokens`` is recovered from prompt + completion tokens."""
    lm = _FakeLM([{"usage": {"prompt_tokens": 10, "completion_tokens": 5}}])
    assert total_tokens_from_history(lm) == 15


def test_total_tokens_from_history_returns_none_when_untracked() -> None:
    """No usage info anywhere yields ``None`` (so callers skip metering, not bill zero)."""
    assert total_tokens_from_history(None) is None
    assert total_tokens_from_history(_FakeLM([{"response": "hi"}])) is None
    assert total_tokens_from_history(MagicMock(spec=[])) is None


def test_usage_by_model_splits_input_and_output_per_model() -> None:
    """Usage is keyed by each LM's model id, preserving the input/output split."""
    gen = _FakeLM(
        [{"usage": {"prompt_tokens": 100, "completion_tokens": 40}}],
        model="openai/gpt-4o-mini",
    )
    refl = _FakeLM(
        [{"usage": {"prompt_tokens": 200, "completion_tokens": 80}}],
        model="anthropic/claude-opus-4-8",
    )
    assert usage_by_model_from_history(gen, refl) == {
        "openai/gpt-4o-mini": (100, 40),
        "anthropic/claude-opus-4-8": (200, 80),
    }


def test_usage_by_model_folds_same_model_across_lms_and_entries() -> None:
    """Multiple LMs (and entries) on the same model accumulate into one bucket."""
    a = _FakeLM(
        [
            {"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            {"usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        ],
        model="openai/gpt-4o-mini",
    )
    b = _FakeLM([{"usage": {"prompt_tokens": 1, "completion_tokens": 2}}], model="openai/gpt-4o-mini")
    assert usage_by_model_from_history(a, b) == {"openai/gpt-4o-mini": (18, 10)}


def test_usage_by_model_total_only_attributes_to_input() -> None:
    """A provider reporting only total_tokens books it all as input (cheaper side)."""
    lm = _FakeLM([{"usage": {"total_tokens": 90}}], model="x/y")
    assert usage_by_model_from_history(lm) == {"x/y": (90, 0)}


def test_usage_by_model_returns_none_when_untracked() -> None:
    """No usage anywhere yields ``None`` so callers skip charging, not bill zero."""
    assert usage_by_model_from_history(None) is None
    assert usage_by_model_from_history(_FakeLM([{"response": "hi"}], model="x/y")) is None


def _cfg(**kwargs: Any) -> ModelConfig:
    """Build a ``ModelConfig`` with a default model name and optional overrides."""
    base: dict[str, Any] = {"name": "openai/gpt-4o-mini"}
    base.update(kwargs)
    return ModelConfig(**base)


def test_build_language_model_passes_model_name_to_dspy() -> None:
    """The model name is passed through to ``dspy.LM`` and the LM is returned."""
    mock_lm = MagicMock()

    with patch("core.service_gateway.language_models.MeteredLM", return_value=mock_lm) as mock_cls:
        result = build_language_model(_cfg(name="openai/gpt-4o-mini"))

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    assert result is mock_lm


def test_build_language_model_strips_leading_slash_from_name() -> None:
    """A leading slash in the model name is stripped before forwarding."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg(name="/openai/gpt-4o-mini"))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-4o-mini"


def test_build_language_model_includes_temperature() -> None:
    """The ``temperature`` value is forwarded to ``dspy.LM``."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg(temperature=0.7))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["temperature"] == 0.7


def test_build_language_model_omits_base_url_when_none() -> None:
    """``base_url`` is omitted from the call when unset."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg(base_url=None))

    call_kwargs = mock_cls.call_args[1]
    assert "base_url" not in call_kwargs


def test_build_language_model_passes_base_url_when_set() -> None:
    """A configured ``base_url`` is forwarded to ``dspy.LM``."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg(base_url="http://localhost:8080"))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["base_url"] == "http://localhost:8080"


def test_build_language_model_omits_max_tokens_when_none() -> None:
    """``max_tokens`` is omitted when not configured."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg(max_tokens=None))

    call_kwargs = mock_cls.call_args[1]
    assert "max_tokens" not in call_kwargs


def test_build_language_model_passes_max_tokens_when_set() -> None:
    """A configured ``max_tokens`` is forwarded to ``dspy.LM``."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg(max_tokens=512))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["max_tokens"] == 512


def test_build_language_model_applies_default_request_timeout() -> None:
    """A default ``timeout`` from settings guards against hung provider reads."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg())

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["timeout"] == settings.lm_request_timeout_seconds


def test_build_language_model_caps_num_retries_under_stall_watchdog() -> None:
    """Retries are capped so the worst-case attempt sequence finishes before the
    stall watchdog — otherwise dspy's default num_retries=3 lets a hung call burn
    the whole watchdog budget and fail the run instead of timing out first."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg())

    call_kwargs = mock_cls.call_args[1]
    num_retries = call_kwargs["num_retries"]
    worst_case = (num_retries + 1) * settings.lm_request_timeout_seconds
    assert worst_case < settings.job_stall_timeout_seconds


def test_build_language_model_extra_overrides_num_retries() -> None:
    """A per-model ``num_retries`` in ``extra`` wins over the watchdog-derived cap."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg(extra={"num_retries": 5}))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["num_retries"] == 5


def test_build_language_model_merges_extra_kwargs() -> None:
    """Entries in the ``extra`` mapping are merged into the call kwargs, overriding the default timeout."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg(extra={"api_key": "sk-test", "timeout": 30}))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["api_key"] == "sk-test"
    assert call_kwargs["timeout"] == 30


def test_build_language_model_all_optional_fields_combined() -> None:
    """All optional fields can be combined and are forwarded together."""
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(
            _cfg(
                name="openai/gpt-4o",
                base_url="http://proxy",
                temperature=0.5,
                max_tokens=256,
                extra={"logit_bias": {}},
            )
        )

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-4o"
    assert call_kwargs["base_url"] == "http://proxy"
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["max_tokens"] == 256
    assert "logit_bias" in call_kwargs


def test_build_language_model_value_error_from_dspy_raises_service_error() -> None:
    """A ``ValueError`` from ``dspy.LM`` is wrapped into a ``ServiceError``."""
    with (
        patch("core.service_gateway.language_models.MeteredLM", side_effect=ValueError("unsupported model")),
        pytest.raises(ServiceError, match="Failed to build language model"),
    ):
        build_language_model(_cfg(name="bad/model"))


def test_build_language_model_service_error_message_contains_model_name() -> None:
    """The wrapped ``ServiceError`` message includes the offending model name."""
    with patch("core.service_gateway.language_models.MeteredLM", side_effect=ValueError("nope")), pytest.raises(ServiceError, match="my-model-name"):
        build_language_model(_cfg(name="my-model-name"))


def test_apply_reasoning_config_native_minimax_sets_extra_and_floor() -> None:
    """A native ``minimax/`` model gets the reasoning-split extra and a 4000 floor."""
    out = apply_model_reasoning_config(ModelConfig(name="minimax/abc"))

    assert out.max_tokens == 4000
    assert out.extra["extra_body"] == {"reasoning_split": True}


def test_apply_reasoning_config_shipped_default_floors_without_extra() -> None:
    """The shipped auto-router default gets max_tokens>=4000 and no extra."""
    out = apply_model_reasoning_config(ModelConfig(name=settings.generalist_agent_model))

    assert settings.generalist_agent_model == "openrouter/openrouter/auto-beta"
    assert out.max_tokens == 4000
    assert out.extra == {}


def test_apply_reasoning_config_openai_reasoning_sets_temperature_and_floor() -> None:
    """An OpenAI reasoning model gets temperature=1.0 and max_tokens>=16000."""
    out = apply_model_reasoning_config(ModelConfig(name="openai/gpt-5"))

    assert out.temperature == 1.0
    assert out.max_tokens == 16000
    assert out.extra["reasoning_effort"] == "medium"


def test_apply_reasoning_config_does_not_shrink_caller_max_tokens() -> None:
    """A caller-supplied larger ``max_tokens`` is preserved, never clamped to the floor."""
    out = apply_model_reasoning_config(ModelConfig(name="minimax/abc", max_tokens=8000))

    assert out.max_tokens == 8000


def test_apply_reasoning_config_caller_extra_wins_on_conflict() -> None:
    """``config.extra`` overrides the model-specific extras on key conflict."""
    out = apply_model_reasoning_config(ModelConfig(name="minimax/abc", extra={"extra_body": {"custom": 1}}))

    assert out.extra["extra_body"] == {"custom": 1}


def test_apply_reasoning_config_non_reasoning_model_is_floored_plain() -> None:
    """A plain model gets the 4000 floor, no temperature, and an empty extra."""
    out = apply_model_reasoning_config(ModelConfig(name="openai/gpt-4o-mini"))

    assert out.max_tokens == 4000
    assert out.temperature is None
    assert out.extra == {}


def test_apply_reasoning_effort_none_is_a_no_op() -> None:
    """No chosen effort returns the config unchanged."""
    config = ModelConfig(name="openai/gpt-5")

    assert apply_reasoning_effort(config, None) is config


def test_apply_reasoning_effort_overrides_reasoning_model_default() -> None:
    """A chosen effort survives the reasoning defaults and beats the "medium" one."""
    config = apply_reasoning_effort(ModelConfig(name="openai/gpt-5"), "high")
    out = apply_model_reasoning_config(config)

    assert out.extra["reasoning_effort"] == "high"


def test_managed_gateway_routes_managed_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A managed call (no api_key / base_url) routes through the configured proxy."""
    monkeypatch.setattr(settings, "litellm_proxy_url", "https://proxy.internal/v1")
    monkeypatch.setattr(settings, "litellm_proxy_api_key", SecretStr("sk-proxy"))
    kwargs: dict[str, object] = {"model": "openai/gpt-4o"}
    _apply_managed_gateway(kwargs)
    assert kwargs["base_url"] == "https://proxy.internal/v1"
    assert kwargs["api_key"] == "sk-proxy"
    # Addressed via the litellm_proxy provider so the OpenRouter slug reaches the
    # proxy intact — a bare ``openai/`` prefix would otherwise be stripped to a
    # slug the proxy's ``*`` -> ``openrouter/*`` wildcard can't reconstruct.
    assert kwargs["model"] == "litellm_proxy/openai/gpt-4o"


def test_managed_gateway_strips_openrouter_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """An openrouter-prefixed id is reduced to its slug before the proxy prefix.

    The proxy's ``*`` -> ``openrouter/*`` wildcard re-adds ``openrouter/``, so
    passing the id through verbatim would double-prefix the model upstream.
    """
    monkeypatch.setattr(settings, "litellm_proxy_url", "https://proxy.internal/v1")
    monkeypatch.setattr(settings, "litellm_proxy_api_key", SecretStr("sk-proxy"))
    kwargs: dict[str, object] = {"model": "openrouter/minimax/minimax-m3"}
    _apply_managed_gateway(kwargs)
    assert kwargs["model"] == "litellm_proxy/minimax/minimax-m3"


def test_managed_gateway_skips_byok_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A BYOK call (api_key already stamped on) bypasses the proxy, key untouched."""
    monkeypatch.setattr(settings, "litellm_proxy_url", "https://proxy.internal/v1")
    monkeypatch.setattr(settings, "litellm_proxy_api_key", SecretStr("sk-proxy"))
    kwargs: dict[str, object] = {"model": "openai/gpt-4o", "api_key": "sk-user"}
    _apply_managed_gateway(kwargs)
    assert kwargs["api_key"] == "sk-user"
    assert "base_url" not in kwargs
    assert kwargs["model"] == "openai/gpt-4o"  # not rewritten for BYOK


def test_managed_gateway_skips_endpoint_pinned_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A call already pinned to a base_url is left on that endpoint, not rerouted."""
    monkeypatch.setattr(settings, "litellm_proxy_url", "https://proxy.internal/v1")
    kwargs: dict[str, object] = {"model": "x", "base_url": "https://custom/v1"}
    _apply_managed_gateway(kwargs)
    assert kwargs["base_url"] == "https://custom/v1"
    assert "api_key" not in kwargs
    assert kwargs["model"] == "x"  # endpoint-pinned call not rewritten


def test_managed_gateway_noop_without_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no proxy configured, the default dspy → provider path is unchanged."""
    monkeypatch.setattr(settings, "litellm_proxy_url", None)
    kwargs: dict[str, object] = {"model": "openai/gpt-4o"}
    _apply_managed_gateway(kwargs)
    assert "base_url" not in kwargs
    assert "api_key" not in kwargs
    assert kwargs["model"] == "openai/gpt-4o"  # untouched without a proxy


def test_gateway_reasoning_translates_effort_to_native_param() -> None:
    """On a gateway-bound call ``reasoning_effort`` is mirrored into ``reasoning``.

    The kwarg must survive (not be popped): dspy's ``Reasoning`` field injects
    ``reasoning_effort="low"`` at call time when the LM carries none, and
    OpenRouter rejects requests where the two forms disagree.
    """
    kwargs: dict[str, object] = {
        "model": "litellm_proxy/google/gemini-3.6-flash",
        "reasoning_effort": "low",
    }
    _translate_gateway_reasoning(kwargs)
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["extra_body"] == {"reasoning": {"effort": "low"}}


def test_gateway_reasoning_maps_max_to_openrouter_ceiling() -> None:
    """Anthropic's ``max`` maps to ``xhigh`` on both wire forms, kept in agreement."""
    kwargs: dict[str, object] = {
        "model": "openrouter/anthropic/claude-fable-5",
        "reasoning_effort": "max",
    }
    _translate_gateway_reasoning(kwargs)
    assert kwargs["reasoning_effort"] == "xhigh"
    assert kwargs["extra_body"] == {"reasoning": {"effort": "xhigh"}}


def test_gateway_reasoning_leaves_direct_provider_calls_alone() -> None:
    """A direct (non-gateway) call keeps the LiteLLM-native ``reasoning_effort``."""
    kwargs: dict[str, object] = {"model": "openai/gpt-5.6-sol", "reasoning_effort": "low"}
    _translate_gateway_reasoning(kwargs)
    assert kwargs["reasoning_effort"] == "low"
    assert "extra_body" not in kwargs


def test_gateway_reasoning_preserves_existing_extra_body() -> None:
    """Translation merges into an existing ``extra_body`` without clobbering it."""
    kwargs: dict[str, object] = {
        "model": "litellm_proxy/openrouter/auto-beta",
        "reasoning_effort": "high",
        "extra_body": {"plugins": [{"id": "auto-router"}]},
    }
    _translate_gateway_reasoning(kwargs)
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"] == {
        "plugins": [{"id": "auto-router"}],
        "reasoning": {"effort": "high"},
    }


def test_gateway_reasoning_aligns_kwarg_to_caller_supplied_body() -> None:
    """A caller-set ``extra_body.reasoning`` wins and the kwarg is aligned to it."""
    kwargs: dict[str, object] = {
        "model": "openrouter/deepseek/deepseek-v4-pro",
        "reasoning_effort": "max",
        "extra_body": {"reasoning": {"effort": "high"}},
    }
    _translate_gateway_reasoning(kwargs)
    assert kwargs["reasoning_effort"] == "high"
    assert kwargs["extra_body"] == {"reasoning": {"effort": "high"}}


def test_disable_cache_sends_proxy_no_cache_directive(monkeypatch: pytest.MonkeyPatch) -> None:
    """``disable_cache`` opts a proxied call out of the proxy's server-side cache too."""
    monkeypatch.setattr(settings, "litellm_proxy_url", "https://proxy.internal/v1")
    monkeypatch.setattr(settings, "litellm_proxy_api_key", SecretStr("sk-proxy"))
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg(name="openrouter/minimax/minimax-m3"), disable_cache=True)

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["cache"] is False
    assert call_kwargs["extra_body"]["cache"] == {"no-cache": True}


def test_disable_cache_keeps_direct_provider_body_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a proxy, ``disable_cache`` stays client-side — no foreign body keys."""
    monkeypatch.setattr(settings, "litellm_proxy_url", None)
    with patch("core.service_gateway.language_models.MeteredLM") as mock_cls:
        build_language_model(_cfg(name="openai/gpt-4o-mini"), disable_cache=True)

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["cache"] is False
    assert "extra_body" not in call_kwargs


class _FakeLm:
    """Minimal stand-in exposing the ``history`` list dspy LMs record."""

    def __init__(self, history: list[dict[str, Any]]) -> None:
        """Store the canned history.

        Args:
            history: Entries mimicking dspy's per-call records.
        """
        self.history = history


def test_served_model_from_reveals_auto_routed_pick() -> None:
    """An auto-routed call resolving to a concrete model is revealed."""
    lm = _FakeLm(
        [
            {
                "model": "openrouter/openrouter/auto-beta",
                "response_model": "google/gemini-3.6-flash",
            }
        ]
    )
    assert served_model_from(lm) == "google/gemini-3.6-flash"


def test_served_model_from_reads_metered_lm_attributes() -> None:
    """MeteredLM drops history, so the reveal reads the stashed model ids."""
    lm = MagicMock(spec=[])
    lm.last_request_model = "openrouter/openrouter/auto-beta"
    lm.last_response_model = "deepseek/deepseek-v4-flash"
    assert served_model_from(lm) == "deepseek/deepseek-v4-flash"

    echo = MagicMock(spec=[])
    echo.last_request_model = "openrouter/openai/gpt-5.6-terra"
    echo.last_response_model = "openai/gpt-5.6-terra"
    assert served_model_from(echo) is None


def test_metered_lm_update_history_stashes_model_ids() -> None:
    """update_history keeps the served-model fields of the entry it discards."""
    lm = MagicMock()
    lm.last_request_model = None
    lm.last_response_model = None
    MeteredLM.update_history(
        lm,
        {
            "model": "openrouter/openrouter/auto-beta",
            "response_model": "google/gemini-3.6-flash",
            "usage": {},
        },
    )
    assert lm.last_request_model == "openrouter/openrouter/auto-beta"
    assert lm.last_response_model == "google/gemini-3.6-flash"


def test_served_model_from_suppresses_non_news() -> None:
    """Echoes, prefix-stripped echoes, and empty history all return None."""
    assert served_model_from(_FakeLm([])) is None
    assert served_model_from(object()) is None
    same = {"model": "openai/gpt-4o-mini", "response_model": "openai/gpt-4o-mini"}
    assert served_model_from(_FakeLm([same])) is None
    stripped = {
        "model": "openrouter/openai/gpt-5.6-terra",
        "response_model": "openai/gpt-5.6-terra",
    }
    assert served_model_from(_FakeLm([stripped])) is None
    assert served_model_from(_FakeLm([{"model": "x", "response_model": None}])) is None


def test_openrouter_patch_adopts_provider_chunk_model() -> None:
    """The patched chunk handler adopts the reported model for openrouter-addressed calls only."""
    assert CustomStreamWrapper is not None, "litellm streaming internals moved"
    install_openrouter_served_model_patch()
    install_openrouter_served_model_patch()
    handler = CustomStreamWrapper.handle_openai_chat_completion_chunk
    assert getattr(handler, "_skynet_served_model_patch", False)

    # Stub self/chunk exercise only the adoption prologue; litellm's real
    # parsing then fails on the stub, which is irrelevant to this assertion.
    target = MagicMock(custom_llm_provider="openrouter", model="openrouter/auto-beta")
    chunk = MagicMock(model="google/gemini-3.6-flash")
    with contextlib.suppress(Exception):
        handler(target, chunk)
    assert target.model == "google/gemini-3.6-flash"

    # The managed proxy path surfaces as an openai-compatible passthrough but
    # keeps the "openrouter/" marker in the requested model.
    proxied = MagicMock(custom_llm_provider="openai", model="openrouter/auto-beta")
    with contextlib.suppress(Exception):
        handler(proxied, chunk)
    assert proxied.model == "google/gemini-3.6-flash"

    other = MagicMock(custom_llm_provider="openai", model="openai/gpt-4o-mini")
    with contextlib.suppress(Exception):
        handler(other, chunk)
    assert other.model == "openai/gpt-4o-mini"
