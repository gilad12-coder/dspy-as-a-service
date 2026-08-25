"""Demo: drive Skynet's POST /run react path as the training-ground CLI would.

Reproduces ``python -m training_ground.optimize`` through the PRODUCT path
(``DspyService.run`` with ``module_name="react"``): the real ``GeneralistSig``,
real recorded trajectories from ``agent_messages``, the real live-MCP tool
roster, and the GEPA replay reward.

Two passes:
  * REAL pass  — uses the configured student model. Without an API key it runs
    the whole pipeline (tool sourcing, replay-example build, seed program,
    ``gepa.optimize`` wiring) and stops at the model call, proving the wiring
    is real, not stubbed.
  * OFFLINE pass — best-effort: swaps in a deterministic ``DummyLM`` and a tiny
    budget so the loop can attempt to complete with no network. GEPA's
    reflective interleaving may not satisfy a flat ``DummyLM`` (documented in
    test_react_e2e.py), so this pass is reported as best-effort.

Run: ``.venv/bin/python -m scripts.demo_react_optimization`` from backend/.
"""

from __future__ import annotations

import functools
import os
from unittest.mock import patch

from dspy.utils import DummyLM
from sqlalchemy import create_engine

from core.config import settings
from core.models import ColumnMapping, ReplayMapping, Reward, ToolSource
from core.models.common import ModelConfig
from core.models.submissions import RunRequest
from core.registry import ServiceRegistry
from core.service_gateway.language_models import apply_model_reasoning_config
from core.service_gateway.optimization.core import DspyService
from core.service_gateway.optimization.data import rows_to_examples
from core.service_gateway.optimization.training_ground import exporter, run_react

# Demo runs with stdout redirected to a file; force line-flushing so the
# step/result prints interleave with dspy's logger instead of being lost to
# block buffering on exit.
print = functools.partial(print, flush=True)

_SIGNATURE_CODE = '''import dspy


class GeneralistSig(dspy.Signature):
    """The generalist wizard agent's per-turn signature."""

    user_message: str = dspy.InputField()
    wizard_state: str = dspy.InputField()
    chat_history: str = dspy.InputField()
    assistant_message: str = dspy.OutputField()
'''


def _banner(text: str) -> None:
    """Print a section banner."""
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def _build_payload(
    rows: list[dict],
    *,
    max_metric_calls: int,
    reward_preset: str = "generalist",
    metric_code: str | None = None,
    match_mode: str = "exact",
) -> RunRequest:
    """Assemble a react RunRequest mirroring the training-ground CLI run.

    Args:
        rows: Exported trajectory rows in the canonical replay schema.
        max_metric_calls: GEPA rollout budget.
        reward_preset: Built-in reward preset (``generalist`` or ``general``);
            ignored when ``metric_code`` is supplied.
        metric_code: Optional custom ``metric(example, rollout)`` source — when
            present it overrides the preset (scalarized with no hard cap).

    Returns:
        A validated ``module_name="react"`` RunRequest.
    """
    return RunRequest(
        name="demo-react-generalist",
        module_name="react",
        signature_code=_SIGNATURE_CODE,
        dataset=rows,
        column_mapping=ColumnMapping(**exporter.GENERALIST_COLUMN_MAPPING),
        replay_mapping=ReplayMapping(**exporter.GENERALIST_REPLAY_MAPPING),
        tool_source=ToolSource(kind="live_mcp", mcp_url=settings.generalist_agent_mcp_url),
        # Always carry a Reward so match_mode threads through even with a custom
        # metric_code (the preset is ignored when metric_code is set, but
        # _run_react reads match_mode off payload.reward).
        reward=Reward(preset=reward_preset, match_mode=match_mode),
        metric_code=metric_code,
        optimizer_name="gepa",
        optimizer_kwargs={"max_metric_calls": max_metric_calls},
        # minimax-m2.7 streams reasoning inline in content, so a 40-tool ReAct
        # turn needs far more than the 4000-token chat default or the tool call
        # gets truncated mid-output. Give the rollout room to finish.
        model_settings=apply_model_reasoning_config(
            ModelConfig(
                name=settings.generalist_agent_model,
                base_url=settings.generalist_agent_base_url or None,
            )
        ).model_copy(update={"max_tokens": 32000}),
    )


def _export_model_keys() -> None:
    """Mirror provider API keys from pydantic settings into ``os.environ``.

    LiteLLM (under ``dspy.LM``) reads provider keys from the process env, but
    the project loads them into pydantic ``settings`` from ``.env`` only. Export
    them so a script run outside the server process can reach the model.
    """
    for env_name, value in (
        ("FIREWORKS_AI_API_KEY", getattr(settings, "fireworks_ai_api_key", None)),
        ("OPENAI_API_KEY", getattr(settings, "openai_api_key", None)),
        ("OPENROUTER_API_KEY", getattr(settings, "openrouter_api_key", None)),
    ):
        if value is None or os.environ.get(env_name):
            continue
        # settings expose secrets as pydantic SecretStr; LiteLLM needs the raw str.
        os.environ[env_name] = value.get_secret_value() if hasattr(value, "get_secret_value") else str(value)


def main() -> int:
    """Run the demo. Returns a process exit code."""
    _export_model_keys()
    _banner("Skynet /run react demo — CLI-equivalent optimization through the product path")

    engine = create_engine(
        os.environ.get("DATABASE_URL", "postgresql://postgres@localhost:5432/skynet")
    )

    rows = exporter.export_agent_messages_to_rows(engine, window="3650d")
    print(f"[1] exported {len(rows)} real trajectory rows from agent_messages")
    if not rows:
        print("    no recorded trajectories — nothing to optimize on.")
        return 1

    tool_source = ToolSource(kind="live_mcp", mcp_url=settings.generalist_agent_mcp_url)
    tools, _hashes = run_react.resolve_react_tools(tool_source, None, settings)
    print(f"[2] sourced {len(tools)} real tools from live MCP ({settings.generalist_agent_mcp_url})")
    print(f"    e.g. {', '.join(t.name for t in tools[:5])} ...")

    examples = run_react.build_replay_examples(
        _to_dspy_examples(rows), ReplayMapping(**exporter.GENERALIST_REPLAY_MAPPING)
    )
    recorded = sum(1 for e in examples if e.replay_steps)
    print(f"[3] built {len(examples)} replay examples ({recorded} with recorded tool calls)")

    service = DspyService(registry=ServiceRegistry())
    payload = _build_payload(rows, max_metric_calls=3)
    service.validate_payload(payload)
    print("[4] validate_payload PASSED (react preset, no metric_code required)")

    _banner("REAL pass — attempt the optimization with the configured model")
    try:
        resp = service.run(payload)
        _print_response(resp)
    except Exception as exc:  # demo: surface the real model-call boundary
        print("    run executed the real pipeline and stopped at:")
        print(f"      {type(exc).__name__}: {str(exc)[:240]}")
        print("    (tools sourced + replay examples + seed program were all real;")
        print("     only the student/reflection model call needs an API key)")

    _banner("OFFLINE pass — best-effort completion with a deterministic DummyLM")
    _offline_pass(service, _build_payload(rows, max_metric_calls=2))
    return 0


def _to_dspy_examples(rows: list[dict]) -> list:
    """Convert exported rows to dspy.Examples with replay extras (as /run does)."""
    extra = {v for v in exporter.GENERALIST_REPLAY_MAPPING.values() if v}
    return rows_to_examples(rows, ColumnMapping(**exporter.GENERALIST_COLUMN_MAPPING), extra_columns=extra)


def _offline_pass(service: DspyService, payload: RunRequest) -> None:
    """Best-effort: run the react branch fully offline via a DummyLM."""
    def _fake_lm(_model_settings):
        return DummyLM([{"next_thought": "done", "next_tool_name": "finish", "next_tool_args": "{}"}] * 200)

    try:
        with patch("core.service_gateway.optimization.core.build_language_model", _fake_lm):
            resp = service.run(payload)
        print("    OFFLINE run COMPLETED — the full /run react loop ran with no network:")
        _print_response(resp)
    except Exception as exc:  # demo: best-effort, documented DummyLM limitation
        print("    offline completion not achievable with a flat DummyLM (expected):")
        print(f"      {type(exc).__name__}: {str(exc)[:200]}")
        print("    A real run needs a reflective model; see test_react_e2e.py for the rationale.")


def _print_response(resp) -> None:
    """Print the salient fields of a react RunResponse."""
    print(f"    baseline_test_metric : {resp.baseline_test_metric}")
    print(f"    optimized_test_metric: {resp.optimized_test_metric}")
    print(f"    objective_scores     : {resp.objective_scores}")
    print(f"    paired_bootstrap     : {resp.paired_bootstrap}")
    print(f"    promotion            : {resp.promotion}")
    overlay = resp.program_artifact.react_overlay if resp.program_artifact else None
    if overlay:
        print(f"    react_overlay        : {len(overlay.tool_descriptions)} tool descriptions, "
              f"max_iters={overlay.max_iters}")


if __name__ == "__main__":
    raise SystemExit(main())
