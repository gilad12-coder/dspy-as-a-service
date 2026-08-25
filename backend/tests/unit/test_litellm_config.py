"""Regression tests for the optional on-prem LiteLLM gateway."""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "deploy" / "litellm" / "config.yaml"


def test_proxy_has_no_spend_limit_and_keeps_parallel_request_backstop() -> None:
    """Keep licensing limits out while bounding per-replica provider work."""
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))

    assert "max_budget" not in config["litellm_settings"]
    assert "budget_duration" not in config["litellm_settings"]
    assert config["general_settings"]["global_max_parallel_requests"] == 64
    assert config["general_settings"]["database_url"] == "os.environ/LITELLM_DATABASE_URL"
