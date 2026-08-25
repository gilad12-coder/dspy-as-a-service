"""Run a real high-budget GEPA optimization of the generalist agent via /run.

Exports recorded trajectories from ``agent_messages``, builds a
``module_name="react"`` RunRequest (correct fireworks-minimax config +
generous ``max_tokens``), runs the real GEPA replay optimization through
``DspyService.run``, and reports baseline vs optimized score + the optimized
system prompt / tool names / descriptions. Saves the full result to ``--out``.

Run: ``.venv/bin/python -m scripts.run_react_optimization --budget 300``
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sqlalchemy import create_engine

from core.registry import ServiceRegistry
from core.service_gateway.optimization.core import DspyService
from core.service_gateway.optimization.training_ground import exporter
from scripts.demo_react_optimization import _build_payload, _export_model_keys

# Custom reward that rewards reproducing the recorded tool calls (trajectory
# coverage + call precision), with NO termination/gate hard cap. Unlike the
# generalist preset — whose score is dominated by saturated and data-constant
# dims (gate_progress, submit_clean) — this gives GEPA a gradient it can move:
# pick the right tool with the right args -> score rises.
MATCHING_METRIC = '''
def metric(example, rollout):
    """Replay-matching reward (tool-selection primary, arg fidelity secondary).

    With tool_name match mode, a "hit" means the agent called the right tool
    in order; coverage = hits/recorded_steps is the reliable, improvable signal.
    Exact-argument fidelity (where reproducible) is a smaller bonus. In-scope
    (no out-of-phase / drift calls) rounds it out. No hard cap.
    """
    steps = len(example.replay_steps)
    events = list(rollout.events)
    hits = [e for e in events if e.outcome == "hit"]
    bad = any(e.outcome in ("tool_not_allowed", "schema_drift") for e in events)
    if steps == 0:
        return 1.0 if not events else 0.3
    coverage = len(hits) / steps
    arg_ok = sum(
        1 for e in hits
        if e.matched_step is not None and e.candidate_argument_hash == e.matched_step.argument_hash
    )
    arg_fidelity = (arg_ok / len(hits)) if hits else 0.0
    score = 0.7 * coverage + 0.2 * arg_fidelity + (0.0 if bad else 0.1)
    return max(0.0, min(1.0, score))
'''


def main() -> int:
    """Run the optimization and report whether the score improved."""
    parser = argparse.ArgumentParser(description="High-budget react optimization of the generalist agent.")
    parser.add_argument("--budget", type=int, default=300, help="GEPA max_metric_calls (rollout budget).")
    parser.add_argument("--window", default="3650d", help="Trajectory lookback window.")
    parser.add_argument(
        "--reward",
        default="match",
        choices=["match", "general", "generalist"],
        help="Reward: 'match' = custom replay-matching metric (no cap); else a built-in preset.",
    )
    parser.add_argument("--out", default="/tmp/react_run_result.json")
    args = parser.parse_args()

    _export_model_keys()
    engine = create_engine(os.environ.get("DATABASE_URL", "postgresql://postgres@localhost:5432/skynet"))
    rows = exporter.export_agent_messages_to_rows(engine, window=args.window)
    print(f"[1] exported {len(rows)} trajectory rows from agent_messages", flush=True)
    if not rows:
        print("    no trajectories; nothing to optimize.", flush=True)
        return 1

    if args.reward == "match":
        payload = _build_payload(
            rows, max_metric_calls=args.budget, metric_code=MATCHING_METRIC, match_mode="tool_name"
        )
    else:
        payload = _build_payload(rows, max_metric_calls=args.budget, reward_preset=args.reward)
    print(f"    reward = {args.reward}, match_mode = {payload.reward.match_mode}", flush=True)
    service = DspyService(registry=ServiceRegistry())
    service.validate_payload(payload)
    print(
        f"[2] validate_payload OK — running GEPA: budget={args.budget}, "
        f"model={payload.model_settings.name}, max_tokens={payload.model_settings.max_tokens}",
        flush=True,
    )

    resp = service.run(payload)

    overlay = resp.program_artifact.react_overlay if resp.program_artifact else None
    baseline = resp.baseline_test_metric or 0.0
    optimized = resp.optimized_test_metric or 0.0
    print(f"[3] baseline={baseline:.4f}  optimized={optimized:.4f}  improvement={resp.metric_improvement}", flush=True)
    print(f"    objective_scores = {resp.objective_scores}", flush=True)
    print(f"    paired_bootstrap = {resp.paired_bootstrap}", flush=True)
    print(f"    promotion        = {resp.promotion}", flush=True)
    if overlay:
        print(
            f"    react_overlay    = {len(overlay.tool_descriptions)} tool descs, "
            f"tool_names={overlay.tool_names}",
            flush=True,
        )

    result = {
        "n_trajectories": len(rows),
        "budget": args.budget,
        "baseline_test_metric": resp.baseline_test_metric,
        "optimized_test_metric": resp.optimized_test_metric,
        "metric_improvement": resp.metric_improvement,
        "objective_scores": resp.objective_scores,
        "paired_bootstrap": resp.paired_bootstrap.model_dump() if resp.paired_bootstrap else None,
        "promotion": resp.promotion.model_dump() if resp.promotion else None,
        "react_overlay": (
            {
                "tool_descriptions": overlay.tool_descriptions,
                "tool_arg_descriptions": overlay.tool_arg_descriptions,
                "tool_names": overlay.tool_names,
                "max_iters": overlay.max_iters,
            }
            if overlay
            else None
        ),
        "program_state": resp.program_artifact.program_state_json if resp.program_artifact else None,
    }
    with Path(args.out).open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(f"[4] saved full result to {args.out}", flush=True)

    improved = optimized > baseline
    print(f"[5] SCORE IMPROVED: {improved}  ({baseline:.4f} -> {optimized:.4f})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
