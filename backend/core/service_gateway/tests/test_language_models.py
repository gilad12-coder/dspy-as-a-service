"""Tests for ``core.service_gateway.language_models.build_language_model``."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from core.config import settings
from core.exceptions import ServiceError
from core.models import ModelConfig
from core.service_gateway.language_models import (
    _apply_managed_gateway,
    apply_model_reasoning_config,
    build_language_model,
    total_tokens_from_history,
)


class _FakeLM:
    """Minimal stand-in exposing a ``history`` list like ``dspy.LM``."""

    def __init__(self, history: list[dict[str, Any]]) -> None:
        """Store the canned history entries the LM should report.

        Args:
            history: Entries shaped like ``dspy.LM.history`` rows.
        """
        self.history = history


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


def _cfg(**kwargs: Any) -> ModelConfig:
    """Build a ``ModelConfig`` with a default model name and optional overrides."""
    base: dict[str, Any] = {"name": "openai/gpt-4o-mini"}
    base.update(kwargs)
    return ModelConfig(**base)


def test_build_language_model_passes_model_name_to_dspy() -> None:
    """The model name is passed through to ``dspy.LM`` and the LM is returned."""
    mock_lm = MagicMock()

    with patch("dspy.LM", return_value=mock_lm) as mock_cls:
        result = build_language_model(_cfg(name="openai/gpt-4o-mini"))

    mock_cls.assert_called_once()
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-4o-mini"
    assert result is mock_lm


def test_build_language_model_strips_leading_slash_from_name() -> None:
    """A leading slash in the model name is stripped before forwarding."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg(name="/openai/gpt-4o-mini"))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-4o-mini"


def test_build_language_model_includes_temperature() -> None:
    """The ``temperature`` value is forwarded to ``dspy.LM``."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg(temperature=0.7))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["temperature"] == 0.7


def test_build_language_model_omits_base_url_when_none() -> None:
    """``base_url`` is omitted from the call when unset."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg(base_url=None))

    call_kwargs = mock_cls.call_args[1]
    assert "base_url" not in call_kwargs


def test_build_language_model_passes_base_url_when_set() -> None:
    """A configured ``base_url`` is forwarded to ``dspy.LM``."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg(base_url="http://localhost:8080"))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["base_url"] == "http://localhost:8080"


def test_build_language_model_omits_max_tokens_when_none() -> None:
    """``max_tokens`` is omitted when not configured."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg(max_tokens=None))

    call_kwargs = mock_cls.call_args[1]
    assert "max_tokens" not in call_kwargs


def test_build_language_model_passes_max_tokens_when_set() -> None:
    """A configured ``max_tokens`` is forwarded to ``dspy.LM``."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg(max_tokens=512))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["max_tokens"] == 512


def test_build_language_model_omits_top_p_when_none() -> None:
    """``top_p`` is omitted when not configured."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg(top_p=None))

    call_kwargs = mock_cls.call_args[1]
    assert "top_p" not in call_kwargs


def test_build_language_model_passes_top_p_when_set() -> None:
    """A configured ``top_p`` is forwarded to ``dspy.LM``."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg(top_p=0.9))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["top_p"] == 0.9


def test_build_language_model_applies_default_request_timeout() -> None:
    """A default ``timeout`` from settings guards against hung provider reads."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg())

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["timeout"] == settings.lm_request_timeout_seconds


def test_build_language_model_caps_num_retries_under_stall_watchdog() -> None:
    """Retries are capped so the worst-case attempt sequence finishes before the
    stall watchdog — otherwise dspy's default num_retries=3 lets a hung call burn
    the whole watchdog budget and fail the run instead of timing out first."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg())

    call_kwargs = mock_cls.call_args[1]
    num_retries = call_kwargs["num_retries"]
    worst_case = (num_retries + 1) * settings.lm_request_timeout_seconds
    assert worst_case < settings.job_stall_timeout_seconds


def test_build_language_model_extra_overrides_num_retries() -> None:
    """A per-model ``num_retries`` in ``extra`` wins over the watchdog-derived cap."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg(extra={"num_retries": 5}))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["num_retries"] == 5


def test_build_language_model_merges_extra_kwargs() -> None:
    """Entries in the ``extra`` mapping are merged into the call kwargs, overriding the default timeout."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(_cfg(extra={"api_key": "sk-test", "timeout": 30}))

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["api_key"] == "sk-test"
    assert call_kwargs["timeout"] == 30


def test_build_language_model_all_optional_fields_combined() -> None:
    """All optional fields can be combined and are forwarded together."""
    with patch("dspy.LM") as mock_cls:
        build_language_model(
            _cfg(
                name="openai/gpt-4o",
                base_url="http://proxy",
                temperature=0.5,
                max_tokens=256,
                top_p=0.95,
                extra={"logit_bias": {}},
            )
        )

    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["model"] == "openai/gpt-4o"
    assert call_kwargs["base_url"] == "http://proxy"
    assert call_kwargs["temperature"] == 0.5
    assert call_kwargs["max_tokens"] == 256
    assert call_kwargs["top_p"] == 0.95
    assert "logit_bias" in call_kwargs


def test_build_language_model_value_error_from_dspy_raises_service_error() -> None:
    """A ``ValueError`` from ``dspy.LM`` is wrapped into a ``ServiceError``."""
    with (
        patch("dspy.LM", side_effect=ValueError("unsupported model")),
        pytest.raises(ServiceError, match="Failed to build language model"),
    ):
        build_language_model(_cfg(name="bad/model"))


def test_build_language_model_service_error_message_contains_model_name() -> None:
    """The wrapped ``ServiceError`` message includes the offending model name."""
    with patch("dspy.LM", side_effect=ValueError("nope")), pytest.raises(ServiceError, match="my-model-name"):
        build_language_model(_cfg(name="my-model-name"))


def test_apply_reasoning_config_native_minimax_sets_extra_and_floor() -> None:
    """A native ``minimax/`` model gets the reasoning-split extra and a 4000 floor."""
    out = apply_model_reasoning_config(ModelConfig(name="minimax/abc"))

    assert out.max_tokens == 4000
    assert out.extra["extra_body"] == {"reasoning_split": True}


def test_apply_reasoning_config_fireworks_minimax_floors_without_extra() -> None:
    """The shipped Fireworks minimax default gets max_tokens>=4000 and no extra."""
    out = apply_model_reasoning_config(
        ModelConfig(name=settings.generalist_agent_model)
    )

    assert "minimax" in settings.generalist_agent_model.lower()
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
    out = apply_model_reasoning_config(
        ModelConfig(name="minimax/abc", max_tokens=8000)
    )

    assert out.max_tokens == 8000


def test_apply_reasoning_config_caller_extra_wins_on_conflict() -> None:
    """``config.extra`` overrides the model-specific extras on key conflict."""
    out = apply_model_reasoning_config(
        ModelConfig(name="minimax/abc", extra={"extra_body": {"custom": 1}})
    )

    assert out.extra["extra_body"] == {"custom": 1}


def test_apply_reasoning_config_non_reasoning_model_is_floored_plain() -> None:
    """A plain model gets the 4000 floor, no temperature, and an empty extra."""
    out = apply_model_reasoning_config(ModelConfig(name="openai/gpt-4o-mini"))

    assert out.max_tokens == 4000
    assert out.temperature is None
    assert out.extra == {}


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
