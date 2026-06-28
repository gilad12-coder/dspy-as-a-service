# ReAct-Agent Optimization — Design Spec (DRAFT)

Status: **draft for review.** No code yet. This document is the contract the
implementation will follow; it supersedes and generalizes the (referenced but
absent) `training_ground_SPEC.md`. Line references are `file:line` against the
current working tree and may drift.

---

## 1. Goal

Make **optimizing a tool-using ReAct agent a first-class Skynet capability** —
available to anyone, on the same footing Skynet already gives simpler
Predict/CoT pipelines. A user brings their own agent signature, their own
tools, their own recorded-trajectory dataset, and their own reward, and gets
back an optimized, **servable** program. Nothing in the core path is specific
to any one agent.

Two production agents are already ReAct and become the first consumers:

- the generalist wizard agent — `dspy.ReActV2(GeneralistSig, …)`, `core/service_gateway/agents/generalist.py:1195`
- the code agent — `dspy.ReActV2(CodeAssistant, …)`, `core/service_gateway/agents/code.py:1335`

The eventual payoff (out of scope for this spec, see §3) is recursive: Skynet
optimizing the agents that power Skynet.

### Design principle: symmetry with the simple-pipeline run

Every knob a normal optimization run exposes (`signature_code`, `dataset` /
`staged_dataset_id`, `column_mapping`, `metric_code`, `optimizer_kwargs`,
`model_settings`, `reflection_model_settings`, `split_fractions`) keeps the same
meaning here. ReAct optimization **adds** fields; it never forks a parallel
product. It is `module_name="react"` on the existing `POST /run`.

---

## 2. Why this is tractable (current state)

A complete, working reference implementation already exists as a **CLI-only,
generalist-specific** harness in `core/service_gateway/optimization/training_ground/`
(currently **untracked** in git). It builds a `ReActV2` seed, scores candidates
by replaying recorded tool calls, and drives `gepa.optimize(adapter=…)`. This
spec is largely: **generalize that harness and wire it into `DspyService.run`.**

What the exploration established:

- **The replay engine is already domain-agnostic.** `TraceConditionedMCPMock`
  (`training_ground/replay.py:145`) matches a candidate's `(tool_name,
  argument_hash)` against recorded steps in order; divergence ends the rollout.
  No wizard knowledge. `submit` is ReActV2's *general* reserved terminal
  (`replay.py:28`), not a generalist concept.
- **The serve/runtime half is reusable.** `registry.fresh_program_for_bundle`
  (`training_ground/registry.py:104`) already takes an **injected**
  `seed_signature`, rebuilds `ReActV2(sig, tools=…)`, re-applies optimized tool
  descriptions, and runs drift + version guards. The `Bundle` model
  (`training_ground/types.py:165`) has **zero** wizard-specific fields.
- **Specificity is concentrated in exactly three plug-points** (see §9):
  1. the `agent_messages` trajectory loader,
  2. `phase_of` (the wizard FSM used for stratification),
  3. the MiniMax/Fireworks grounding concretes.
- **The dataset path already carries arbitrary columns.** Staging stores rows
  as a free-form JSON blob (`AgentStagedDatasetModel.rows`,
  `core/storage/models.py:286`), and the only place columns are dropped is
  `rows_to_examples` (`data.py:103-115`), which copies only *mapped* fields.

---

## 3. Scope

**In scope (full arc, approved):**

1. Carry replay side-columns from dataset rows to the reward.
2. `react` module type + tool sourcing/rendering.
3. Replay-based reward as the run's metric (general default + generalist preset
   + custom `metric_code` override), routed through `gepa.optimize(adapter=…)`.
4. Result envelope: per-objective vector + paired-bootstrap CI + promotion verdict.
5. Serve a tool-bearing ReAct program through the existing `/serve/{id}`.
6. Trajectory exporter (`agent_messages` → stage-ready rows) as the generalist's
   reference on-ramp.

**Out of scope (deliberately deferred):**

- Hot-wiring the *live* production agents to auto-load their optimized program
  on every user turn. (This spec produces a *servable* run; flipping the live
  agent to consume it is a separate decision.)
- **Live-rollout** evaluation (actually executing the agent's real tools during
  optimization). Replay only. Live rollout needs a sandbox + programmatic
  success signal and is a future mode behind a flag.
- Optimizers other than GEPA for the replay path (see §6).

---

## 4. Public contract — the request

`module_name="react"` on `POST /run` (`RunRequest`, `core/models/submissions.py:85`).
The run still goes through `_optimization_type = "run"` →
`DspyService.run` (the worker dispatch at `subprocess_runner.py:158` is
unchanged). The branch is internal, keyed on the module being a ReAct module.

### 4.1 Reused fields (unchanged meaning)

| Field | Use for a ReAct run |
|---|---|
| `signature_code` | The agent's `dspy.Signature` (e.g. `GeneralistSig`). Defines the per-turn input/output fields. |
| `module_name` | `"react"` (new alias → `dspy.ReActV2`). |
| `module_kwargs` | May carry `max_iters` (default 8). **Not** `tools` — tools are sourced separately (§4.2). |
| `dataset` / `staged_dataset_id` | The recorded-trajectory rows (§5). |
| `column_mapping` | Maps the **signature** input/output fields to dataset columns (as today). |
| `split_fractions` | train/val/test (defaults 0.7/0.15/0.15). val is GEPA's reflective/holdout set; test is the promotion-gate holdout. |
| `model_settings` | The **student** LM (the model being optimized — must be 1:1 with what serves the agent). |
| `reflection_model_settings` | GEPA's reflective proposer LM (e.g. `openai/gpt-5.5`). |
| `optimizer_name` | `"gepa"`. Replay routes through `gepa.optimize(adapter=…)` (§6). |
| `metric_code` | **Now optional** for ReAct runs (see §4.4). When present, it is the custom reward hook. |

### 4.2 New field — `tool_source`

Both modes are supported (per decision). Shape:

```
tool_source:
  kind: "live_mcp" | "dataset_snapshot"     # required
  # kind == live_mcp:
  mcp_url: str | None                        # default settings.generalist_agent_mcp_url
  mcp_auth_header: str | None
  tool_filter: list[str] | None              # restrict/order the roster; null = all agent-tagged tools
  # kind == dataset_snapshot:
  #   tool specs (name, description, arg schema) travel inside the dataset
  #   (a dataset-level sidecar row or a reserved column); the run rebuilds the
  #   roster from the snapshot. Self-contained + reproducible; can go stale.
```

`live_mcp` reuses the existing roster machinery (`_list_live_tools`,
`training_ground/optimize.py:464`) and `dspy.Tool.from_mcp_tool`. The **same**
`tool_source` is recorded on the artifact so `/serve` rebuilds tools identically
(§8). Schema drift is detected via `hash_tool_schema` (`registry.py:186`).

### 4.3 New field — `replay_mapping`

Per decision ("Mapping in the request"), an explicit map (parallel to
`column_mapping`) declaring which dataset columns hold the replay metadata.
Keeps the feature general — any column names work.

```
replay_mapping:
  steps: str               # column holding the recorded tool-call list (required)
  allowed_tools: str       # column holding the per-turn allowed-tool set (required)
  tool_schema_hashes: str  # column holding {tool: schema_hash} (required for drift checks)
  state_before: str | None # opaque per-turn state snapshot (was wizard_state_before)
  state_after: str | None  # opaque per-turn state snapshot (was wizard_state_after)
  chat_history: str | None # prior {role, content} turns
```

The named columns are carried onto the `dspy.Example` as **non-input,
non-output** side data (§5.2) and reconstructed into a generalized
`ReplayExample` (§7) for the reward. The signature's *input* fields (via
`column_mapping`) are what the candidate program actually receives each turn.

### 4.4 New field — `reward`

```
reward:
  preset: "general" | "generalist"   # default "general"
  grounding_weight: float            # default 0.05 (ECHO λ); auto-disabled when
                                     # no echo-capable scorer is configured (§6.3)
```

- **Default = combined** (`preset` task dims + `grounding_weight`·grounding),
  matching the validated recipe — but grounding **gracefully degrades to
  task-only** when the student model/provider can't produce prompt-echo
  logprobs (§6.3), so "combined" never breaks for an arbitrary agent.
- `metric_code` (optional) **overrides** the preset entirely. Contract (§6.4):
  `metric(example, rollout, dimensions) -> float | dict[str, float]`.

### 4.5 Validation changes

- `metric_code` becomes optional when `module_name="react"` and a built-in
  `reward.preset` is selected. (Base model currently requires it,
  `submissions.py:37`.)
- The GEPA **metric-arity check** (`_require_metric_compatible_with_optimizer`,
  `core.py:105`, invoked from the `validate_payload` gate — *not* from `run`)
  is skipped for the replay path, which has no 5-arg scalar metric. A custom
  `metric_code` here uses the replay contract (§6.4), not the DSPi `(gold, pred,
  trace, …)` contract.
- `column_mapping` validation is unchanged (signature ⊆ mapping). `replay_mapping`
  gets an analogous validator: every required role present, every named column
  exists in the dataset.

---

## 5. Dataset contract

A ReAct run's dataset is a list of **recorded turns**. One row = one turn.

### 5.1 Columns

- **Signature I/O columns** — mapped by `column_mapping` to the agent's
  signature fields (e.g. `user_message`, `wizard_state`, `chat_history` →
  inputs; `assistant_message` → output). These feed the candidate rollout.
- **Replay columns** — mapped by `replay_mapping`: `steps`, `allowed_tools`,
  `tool_schema_hashes`, optional `state_before/after`, `chat_history`.

`steps` is a list of recorded calls; each entry (the v1 shape produced by
`generalist_agent.py::_wrap_with_persistence`, documented at `replay.py:66-76`):

```
{ "tool": str, "reason": str|null, "status": "done"|"error",
  "startedAt": ms|null, "endedAt": ms|null,
  "payload": { "arguments": {...}, "result": {...} } }
```

Nested values travel natively through JSON/JSONL upload (CSV would force
JSON-encoded strings — discouraged for trajectories).

### 5.2 Carry-through (the one keystone change)

`rows_to_examples` (`data.py:71`) today copies only `column_mapping` fields onto
the `dspy.Example`. Add an `extra_columns: set[str]` parameter (the
`replay_mapping` columns); fold those onto the Example **without**
`with_inputs`/`with_outputs` marking them — so they're readable by the reward
(`gold.<col>`) but never fed to the program at predict time. Verified safe:
DSPy `Example` retains all constructed fields; split + GEPA `gold` delivery
preserve them.

### 5.3 The reference on-ramp (generalist) — `agent_messages` exporter

A converter that reads `agent_messages` (reusing the loader logic at
`training_ground/persistence.py:61`) and emits stage-ready rows in the schema
above, postable to `POST /datasets/stage-for-agent`
(`datasets.py:374`). This is the **generalist's** convenience on-ramp and the
canonical reference for the row schema — see **Open Question Q1** on whether
general users get more than BYO-dataset + this exporter.

---

## 6. Reward design

### 6.1 General default preset (domain-agnostic)

Computed purely from the replay (no domain semantics). Each dim ∈ [0,1]:

| Dim | Meaning | Derived from |
|---|---|---|
| `tool_selection` | Right tools, right order | hit events / step pointer |
| `argument_fidelity` | Exact-argument matches | hit = `(name, arg_hash)` match |
| `trajectory_coverage` | Fraction of recorded steps matched before divergence | pointer / len(steps) |
| `in_scope_tools` | Never called a tool outside `allowed_tools` | `tool_not_allowed` outcome |
| `clean_termination` | Submit (clean) > forced_submit > early-divergence | ReActV2 termination_reason |
| `no_schema_drift` | No `schema_drift` outcome | rollout events |
| `observation_threading` | Threads prior results into later args | consecutive hit pairs |
| `engaged_when_expected` | Didn't sit idle when the recorded turn used tools | steps present + zero hits |

Critical set (trips the hard-cap when any < floor): `trajectory_coverage`,
`in_scope_tools`, `clean_termination`. These guard reward-hacking-by-doing-nothing
in a domain-neutral way.

### 6.2 Generalist preset (reproduces the validated 12-dim reward)

The existing `vector_reward` (`metrics.py:270`) verbatim — `submit_clean`,
`gate_progress`, `one_call_compliance` (update_wizard_state), `missing_reflection_model`,
`no_repeated_dataset_upload`, `no_hallucinated_ids`, etc. — with its current
weights and critical set (`metrics.py:22-54`). Selecting `preset: "generalist"`
reproduces exactly what we validated end-to-end.

### 6.3 Aggregation + grounding

- A `RewardSpec` declares `{weights, critical_set, critical_floor, hard_cap}`.
  `scalar_with_hard_caps` (`metrics.py:301`) generalizes to take a `RewardSpec`.
  General default and generalist are two `RewardSpec`s.
- **Grounding is optional and Protocol-based.** The whole grounding stack
  (`training_ground/grounding.py`) already programs against `ChatTemplate` +
  `PromptScorer` Protocols; `MiniMaxChatTemplate` + `FireworksEchoScorer` are
  reference concretes. Grounding requires prompt-echo logprobs, which **most
  providers cannot produce** (Anthropic, most chat-only endpoints). So:
  `grounding_weight > 0` is honored only when an echo-capable scorer is
  configured for the student model; otherwise it degrades to task-only with a
  logged notice. Default `0.05` is therefore "combined when possible, task-only
  otherwise."

### 6.4 Custom reward hook (`metric_code` override)

Per decision ("Example + rollout + 12 dims"):

```python
def metric(example, rollout, dimensions):
    # example:    generalized ReplayExample (state_before/after, allowed_tools,
    #             replay_steps, tool_schema_hashes, chat_history, user inputs)
    # rollout:    ReplayRollout (events, submit_called, forced_submit, …)
    # dimensions: dict[str, float] — the precomputed preset dims
    # return:     float (scalar) OR dict[str, float] (full objective vector;
    #             aggregated via the preset's RewardSpec)
    ...
```

Loaded by the existing `load_metric_from_code` (`data.py:253`) sandbox, but
introspected for the replay arity, not the DSPi 5-arg arity.

---

## 7. Replay mechanism (generalization)

- `EvaluationExample` (`types.py:108`) → rename `wizard_state_before/after` to
  **`state_before/after: dict[str, Any]`** (opaque). Everything else
  (`turn_id`, `user_message`, `allowed_tools`, `tool_schema_hashes`,
  `replay_steps`, `chat_history`) is already generic. (Keep a thin
  `wizard_state_*` alias if needed to avoid churn in the generalist preset.)
- `TraceConditionedMCPMock` (`replay.py:145`) is reused unchanged — it already
  matches purely on `(tool_name, argument_hash)` and `allowed_tools`.
- **Rollout inputs become signature-driven.** Today `_run_candidate`
  (`gepa_adapter.py:619`) hardcodes `{user_message, wizard_state, chat_history}`.
  Generalize to build the input dict from the **signature's input fields**,
  populated from the row's `column_mapping` inputs. The agent's own signature
  decides what it receives.
- **Candidate key** `tool_module:generalist` (`gepa_adapter.py:37`) → derive a
  neutral key from the program (e.g. `tool_module:react`); the blob shape
  (`{<predictor>: instructions, "tools": {name: {desc, args}}}`) is unchanged.
- GEPA mutates **predictor instructions + tool descriptions + tool arg
  descriptions** (matches today). Signature field descriptions are out of scope
  for mutation in the first cut.

---

## 8. Execution path & the seams that diverge

Within `DspyService.run` (`core.py:470`), branch when the resolved module is a
ReAct module. Divergences from the scalar `compile()` path:

1. **Module + tools.** Add `react` to `MODULE_ALIASES` (`resolvers.py:31`) →
   `dspy.ReActV2`. **Inject `tools`** into `module_kwargs` before
   `module_factory(**module_kwargs)` (`core.py:514`) — the current path injects
   only `signature`. Tools come from `tool_source` (§4.2).
2. **Skip `load_metric_from_code`** (`core.py:516`) when using a built-in
   preset; the reward is the adapter's. A custom `metric_code` is loaded but
   under the replay contract (§6.4).
3. **Route to `gepa.optimize(adapter=…)`**, not the teleprompter. The regular
   path uses `dspy.teleprompt.GEPA(...).compile(...)` which exposes **no
   `adapter=`** and is metric/scalar driven. The replay path calls the
   lower-level `gepa.optimize(seed_candidate=seed_candidate_from_program(program),
   trainset=splits.train, valset=splits.val, adapter=<DspyAdapter subclass>,
   reflection_lm=…, …)` (reference: `optimize.py:958`). This replaces
   `instantiate_optimizer` + `compile_program` (`optimizers.py:264`, `:32`) for
   this branch.
   *Verified API (gepa 0.1.1):* `optimize(seed_candidate: dict[str,str],
   trainset, valset=None, adapter=None, reflection_lm=None,
   frontier_type='instance'|'objective'|'hybrid'|'cartesian',
   batch_sampler='epoch_shuffled'|BatchSampler, reflection_minibatch_size=None,
   max_metric_calls=None, seed=0, track_best_outputs=False, …) -> GEPAResult`.
   We pass `frontier_type='objective'` (per-objective frontier) and
   `max_metric_calls` as the budget (the `--auto light/medium/heavy` → 500/2000/8000
   mapping).
4. **Realize the result.** `gepa.optimize` returns a `GEPAResult` whose
   `best_candidate` is a `dict[str,str]` of component text — convert to a
   program via `adapter.build_program(best_candidate)` (`build_program` is a
   **confirmed public method** on the parent `gepa.adapters…DspyAdapter`) before
   eval/persist; reference `_program_state_from`, `optimize.py:807`.
   *Verified GEPAResult surface:* `best_candidate`, `best_idx`,
   `per_objective_best_candidates`, `objective_pareto_front`,
   `val_aggregate_subscores`, `total_metric_calls` — so the result envelope's
   per-objective vector (§9) comes directly from GEPA, not only from a re-score.
5. **Eval via the adapter reward**, not `dspy.Evaluate` + scalar metric
   (reference `_evaluate_candidate_on_examples`, `optimize.py:583`), for both
   baseline (seed) and optimized candidate on the test split.
6. **Persist tool metadata.** `persist_program` (`artifacts.py:159`) writes only
   `program_state_json`; `ProgramArtifact` (`models/artifacts.py:32`) carries no
   tools. It does have a free-form `metadata: dict | None` field (verified), so
   the overlay *can* ride there — but for a first-class feature add a typed
   optional `react_overlay` sub-model carrying `tool_descriptions`,
   `tool_arg_descriptions`, `tool_schema_hashes`, `max_iters`, and the resolved
   `tool_source` (mirrors the `Bundle` fields, `types.py:179-188`). `/serve`
   reads it to rebuild (§10).

The adapter itself (`TrainingGroundDspyAdapter`, `gepa_adapter.py:78`) is reused;
the reward becomes a `RewardSpec`/custom-hook parameter instead of the hardcoded
`vector_reward`.

---

## 9. Result envelope

The scalar metrics (`baseline_test_metric`, `optimized_test_metric`,
`metric_improvement`) live on both `RunResponse` (`models/results.py:31`) and
`OptimizationSummaryResponse` (`models/optimizations.py:108`) — keep them as the
headline for UI parity. The per-objective detail belongs on **`RunResponse`**
(returned inside `OptimizationStatusResponse.result`). Add (additive, optional):

- `objective_scores: dict[str, float]` — the per-dimension vector (seed + best).
- `paired_bootstrap: {resamples, mean_delta, ci95_lower, ci95_upper}` —
  reuse `paired_bootstrap_ci` (`persistence.py:272`, fully generic).
- `promotion: {promotable: bool, reasons: [...]}` — the §11-style gate verdict.

`RunResponse` already has free-form `optimization_metadata: dict` and
`details: dict` (verified, `results.py:41-42`) — acceptable as a zero-schema-change
fallback, but typed fields are preferred for OpenAPI/UI. The vector is sourced
from GEPA's `per_objective_best_candidates` / `val_aggregate_subscores` (§8.4)
plus a final re-score on the test split.

---

## 10. Serving

Generalize `/serve/{id}` (`_helpers._materialize_program`, `_helpers.py:438`)
for ReAct modules — and **reuse `registry.py` rather than duplicate it**:

1. `react` alias resolves the module.
2. Re-source tools from the persisted `tool_source` (live MCP or dataset
   snapshot), exactly as the run did.
3. `_assert_tool_set_matches` (`registry.py:221`) drift-checks against the
   persisted `tool_schema_hashes`; `_apply_bundle_tool_overrides` (`registry.py:153`)
   re-applies the optimized tool descriptions.
4. Build `dspy.ReActV2(signature, tools=…, max_iters=…)`, then `load_state`.
5. Cache key must include the tool roster identity (the current `_program_cache`
   keys on artifact id only).

Factor `fresh_program_for_bundle`'s helpers out of `training_ground/` into a
shared module both the bundle loader and `/serve` import.

---

## 11. Promotion gate

Reuse the paired-bootstrap acceptance test, generalized:

- Per-bucket holdout floors via the **injected stratifier** (§9 below):
  `phase_of` becomes a strategy; the default is a single bucket (plain split),
  the generalist supplies its wizard-phase stratifier.
- Gate is **advisory** on the result by default; **promotion stays an explicit
  human action**. (The generalist's data volume won't pass the strict floors yet
  — already accepted; the run still produces a servable artifact + verdict.)

---

## 12. Generalization refactor (where specificity moves)

Three injectable seams; the generalist becomes one implementation of each:

1. **`TrajectoryLoader`** — `load_trajectories` + `_fetch_assistant_rows` +
   `_row_to_example` (`persistence.py:61-211`) hardcode the `agent_messages`
   Postgres schema. Behind an interface, this trio is "the v1 agent_messages
   adapter." General users supply rows directly (the dataset), so the loader is
   only needed by the exporter (§5.3).
2. **Stratifier** — `phase_of` (`persistence.py:214`) is pure wizard FSM. Make it
   an injected key on `split_stratified`; default single-bucket.
3. **`ChatTemplate` + `PromptScorer`** — MiniMax/Fireworks concretes
   (`grounding.py:214`, `:257`). Ship as reference impls; the grounding reward
   needs no change. New providers register their own (or run task-only).

Already-clean / reuse-as-is: `paired_bootstrap_ci`, `write_bundle`,
`extract_program_state`, `parse_window`, all of `registry.py`, the `Bundle`
model, the grounding reward stack above the concretes, and `EvaluationExample` /
`ReplayStep` / `ReplayRollout` (after the `state_*` rename).

---

## 13. Module registry, CLI, compatibility

- **`MODULE_ALIASES`** (`resolvers.py:31`): add `"react": ModuleAlias(("dspy.ReActV2",
  "dspy.ReAct"), auto_signature=True)`. Tool injection (§8.1) is the real work;
  the alias alone is insufficient (ReActV2 needs `tools=`).
- **CLI** (`training_ground/optimize.py`) is refactored to call the shared
  general core (extract a `run_react_optimization(...)`); it keeps working and
  stops duplicating logic. Existing `optimize.py` behavior is preserved.
- **Backward compatibility:** all new request fields are optional; existing
  Predict/CoT runs are untouched (no `react` module → no branch). Existing
  generalist bundles still load through `registry.py` unchanged.
- **OpenAPI/Pydantic contract:** new fields on `RunRequest` / `RunResponse` /
  `ProgramArtifact` must be optional with defaults so existing payloads validate
  unchanged. Do **not** add or remove class-level docstrings on these
  `BaseModel`s (per AGENTS.md they are part of the OpenAPI contract — use a
  comment above the class). The change is contract-visible via
  `/openapi.public.json` (`core/api/app.py:100`).

---

## 14. Testing strategy

- **Unit:** carry-through (`extra_columns` stored, not in `.inputs()`); general
  RewardSpec on a non-wizard rollout (no critical-floor misfire); generalist
  preset reproduces known scalars; replay matching/divergence; `/serve` ReAct
  rebuild with tool overlays + drift.
- **Integration:** synthetic trajectory rows → stage → `module=react` run →
  scored result with vector + CI; then `/serve` the result and invoke it.
- **Parity:** the general path on the generalist preset must match the CLI's
  scalars on identical rows (guards the refactor).

---

## 15. Open questions / decisions needed

- **Q1 — Trajectory provenance — RESOLVED: BYO dataset + exporter.** General
  users upload recorded-trajectory rows in the §5 replay schema (symmetric with
  simple-pipeline dataset upload); the `agent_messages` exporter is the
  generalist's reference on-ramp. A generic auto-recording capability is a
  possible later layer, explicitly out of scope for the first cut.
- **Q2 — Spec location/format.** This file lives at
  `core/service_gateway/optimization/REACT_AGENT_OPTIMIZATION_SPEC.md`. Move it
  (e.g. `docs/`) or keep it co-located?
- **Q3 — Confirm during implementation:** exact `validate_payload` call site
  (the arity gate is not in `run`); whether the code agent's signature needs any
  special handling vs the generalist's.

---

## 16. Decisions log (answered)

- General capability (any ReAct agent); generalist + code agent are first
  consumers. Nothing generalist-specific in the core path.
- It's a **regular `run`** (`module="react"`), not a new optimization type.
- **Replay** evaluation (not live rollout).
- Dataset = recorded trajectories, ridden through the **existing dataset
  upload/staging** infra.
- Trajectory provenance: **BYO dataset** + the `agent_messages` exporter
  (generic auto-recording is a possible later layer, not in the first cut).
- Tool source: **both** live MCP and dataset snapshot, per-run.
- Replay columns: **explicit mapping in the request** (`replay_mapping`).
- Reward: **combined default** (auto-degrades to task-only without echo);
  domain-agnostic **general** preset + selectable **generalist** preset; fully
  overridable via `metric_code` receiving `(example, rollout, dimensions)`.
- Optimizer: **GEPA** for replay; **grounding optional**; **models per-run**.
- Scope: **full arc 1-6**, excluding hot-wiring the live agents.
