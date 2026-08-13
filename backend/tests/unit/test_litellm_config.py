"""Regression tests for production LiteLLM cost and concurrency limits."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "deploy" / "litellm" / "config.yaml"


def test_proxy_has_global_spend_and_parallel_request_backstops() -> None:
    """Keep the proxy's independent launch circuit breakers enabled."""
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["litellm_settings"]["max_budget"] == 50
    assert config["litellm_settings"]["budget_duration"] == "1d"
    assert config["general_settings"]["global_max_parallel_requests"] == 64
