/**
 * Tutorial Demo Simulation
 *
 * Provides a fake optimization that progresses through all pipeline stages
 * over ~11 seconds, producing realistic GEPA logs and scores.
 * Used when the tutorial navigates to /optimizations/tutorial-demo.
 */

import type {
  OptimizationStatusResponse,
  ProgressEvent,
  OptimizationLogEntry,
  PairResult,
  GridSearchResult,
  PaginatedJobsResponse,
  OptimizationSummaryResponse,
} from "@/shared/types/api";
import type {
  DashboardAnalytics,
  DashboardAnalyticsJob,
  PublicDashboardPoint,
} from "@/shared/lib/api";
import { layoutTrajectory, type CandidateMetrics } from "@/features/trajectory";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";

export const DEMO_OPTIMIZATION_ID = "a7e3b291-4d2f-4f8c-b142-9d5e6f8a1c3b";
export const DEMO_GRID_OPTIMIZATION_ID = "c3f9d215-8a47-4e6b-a1d3-7b2f9c58e4a1";

function ts(start: Date, offsetMs: number): string {
  return new Date(start.getTime() + offsetMs).toISOString();
}

function fmtElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return m > 0 ? `${m}:${String(s).padStart(2, "0")}` : `0:${String(s).padStart(2, "0")}`;
}

const TRIAL_SCORES = [65.0, 62.0, 70.0, 68.0, 71.0, 75.0, 77.0, 72.5, 80.0, 84.0];
const NUM_TRIALS = TRIAL_SCORES.length;

// Wide-shallow branching so the trajectory panel fills the container with a
// recognizably "exploratory" tree (gen 1 fans out to four variants, gen 2
// keeps two of those alive plus one with twin offspring). The winning spine
// 0 → 2 → 6 → 9 climbs 65 → 70 → 77 → 84.
const CANDIDATE_PLAN: ReadonlyArray<{ parent: string | null; generation: number }> = [
  { parent: null, generation: 0 },
  { parent: "0", generation: 1 },
  { parent: "0", generation: 1 },
  { parent: "0", generation: 1 },
  { parent: "0", generation: 1 },
  { parent: "2", generation: 2 },
  { parent: "2", generation: 2 },
  { parent: "3", generation: 2 },
  { parent: "4", generation: 2 },
  { parent: "6", generation: 3 },
];

// Validation ids used for per-example donut outlines (green/red).
const VAL_EXAMPLE_IDS = ["val-0", "val-1", "val-2", "val-3", "val-4", "val-5", "val-6", "val-7"];

function perExampleForCandidate(idx: number): Array<{ id: string; score: number }> {
  const targetScore = (TRIAL_SCORES[idx] ?? 0) / 100;
  const passes = Math.max(1, Math.min(7, Math.round(targetScore * 8)));
  return VAL_EXAMPLE_IDS.map((id, i) => ({
    id,
    score: i < passes ? 1 : 0,
  }));
}

// Eventual layout extent for the demo's full 10-candidate tree. The
// trajectory tree opens framed to these dims so the dd-trajectory step
// shows the whole graph from the first frame instead of zooming-in on the
// lone root candidate before streaming starts.
const DEMO_TRAJECTORY_FULL_LAYOUT = (() => {
  const fullCandidates: CandidateMetrics[] = CANDIDATE_PLAN.map((plan, idx) => ({
    candidate_id: String(idx),
    parent_id: plan.parent,
    parents_extra: [],
    generation: plan.generation,
    score: (TRIAL_SCORES[idx] ?? 0) / 100,
    per_example: perExampleForCandidate(idx),
    prompt: {},
    discovered_at_evals: 0,
    iteration: idx,
    timestamp: "",
  }));
  const result = layoutTrajectory(fullCandidates);
  return { width: result.width, height: result.height };
})();

export const DEMO_TRAJECTORY_PREVIEW_LAYOUT: { width: number; height: number } =
  DEMO_TRAJECTORY_FULL_LAYOUT;

function candidateEvent(start: Date, idx: number, offsetMs: number): ProgressEvent {
  const plan = CANDIDATE_PLAN[idx]!;
  return {
    timestamp: ts(start, offsetMs),
    event: "candidate",
    metrics: {
      candidate_id: String(idx),
      generation: plan.generation,
      score: (TRIAL_SCORES[idx] ?? 0) / 100,
      parent_id: plan.parent,
      iteration: idx,
      per_example: perExampleForCandidate(idx),
    },
  };
}

function baseJob(start: Date): OptimizationStatusResponse {
  return {
    optimization_id: DEMO_OPTIMIZATION_ID,
    optimization_type: "run",
    status: "validating",
    name: msg("auto.features.tutorial.lib.demo.data.literal.1"),
    description: formatMsg("auto.features.tutorial.lib.demo.data.template.1", {
      p1: TERMS.optimization,
    }),
    username: "demo",
    created_at: start.toISOString(),
    started_at: start.toISOString(),
    elapsed_seconds: 0,
    elapsed: "0:00",
    module_name: "Predict",
    module_kwargs: {},
    optimizer_name: "GEPA",
    optimizer_kwargs: { auto: "light", reflection_minibatch_size: 3 },
    compile_kwargs: {},
    model_name: "gpt-4o-mini",
    dataset_rows: 200,
    column_mapping: { inputs: { email_text: "str" }, outputs: { category: "str" } },
    split_fractions: { train: 0.6, val: 0.2, test: 0.2 },
    shuffle: true,
    seed: 42,
    progress_events: [],
    logs: [],
    latest_metrics: {},
    progress_count: 0,
    log_count: 0,
  };
}

function buildValidating(start: Date): OptimizationStatusResponse {
  const elapsed = (Date.now() - start.getTime()) / 1000;
  return {
    ...baseJob(start),
    status: "validating",
    elapsed_seconds: elapsed,
    elapsed: fmtElapsed(elapsed),
    logs: [
      {
        timestamp: ts(start, 200),
        level: "INFO",
        logger: "skynet.worker",
        message: msg("auto.features.tutorial.lib.demo.data.literal.2"),
        pair_index: null,
      },
      {
        timestamp: ts(start, 400),
        level: "INFO",
        logger: "dspy.validators",
        message: msg("auto.features.tutorial.lib.demo.data.literal.19"),
        pair_index: null,
      },
    ],
    log_count: 2,
  };
}

function buildSplitting(start: Date): OptimizationStatusResponse {
  const elapsed = (Date.now() - start.getTime()) / 1000;
  return {
    ...baseJob(start),
    status: "running",
    elapsed_seconds: elapsed,
    elapsed: fmtElapsed(elapsed),
    progress_events: [
      {
        timestamp: ts(start, 1500),
        event: "validation_passed",
        metrics: { message: msg("auto.features.tutorial.lib.demo.data.literal.20") },
      },
    ],
    logs: [
      {
        timestamp: ts(start, 200),
        level: "INFO",
        logger: "skynet.worker",
        message: msg("auto.features.tutorial.lib.demo.data.literal.3"),
        pair_index: null,
      },
      {
        timestamp: ts(start, 400),
        level: "INFO",
        logger: "dspy.validators",
        message: msg("auto.features.tutorial.lib.demo.data.literal.19"),
        pair_index: null,
      },
      {
        timestamp: ts(start, 1500),
        level: "INFO",
        logger: "dspy.validators",
        message: msg("auto.features.tutorial.lib.demo.data.literal.21"),
        pair_index: null,
      },
      {
        timestamp: ts(start, 1800),
        level: "INFO",
        logger: "dspy.datasets",
        message: msg("auto.features.tutorial.lib.demo.data.literal.22"),
        pair_index: null,
      },
    ],
    progress_count: 1,
    log_count: 4,
  };
}

function buildBaseline(start: Date): OptimizationStatusResponse {
  const elapsed = (Date.now() - start.getTime()) / 1000;
  return {
    ...baseJob(start),
    status: "running",
    elapsed_seconds: elapsed,
    elapsed: fmtElapsed(elapsed),
    progress_events: [
      { timestamp: ts(start, 1500), event: "validation_passed", metrics: {} },
      {
        timestamp: ts(start, 2800),
        event: "dataset_splits_ready",
        metrics: { train_examples: 120, val_examples: 40, test_examples: 40 },
      },
    ],
    logs: [
      {
        timestamp: ts(start, 200),
        level: "INFO",
        logger: "skynet.worker",
        message: msg("auto.features.tutorial.lib.demo.data.literal.4"),
        pair_index: null,
      },
      {
        timestamp: ts(start, 400),
        level: "INFO",
        logger: "dspy.validators",
        message: msg("auto.features.tutorial.lib.demo.data.literal.19"),
        pair_index: null,
      },
      {
        timestamp: ts(start, 1500),
        level: "INFO",
        logger: "dspy.validators",
        message: msg("auto.features.tutorial.lib.demo.data.literal.21"),
        pair_index: null,
      },
      {
        timestamp: ts(start, 1800),
        level: "INFO",
        logger: "dspy.datasets",
        message: msg("auto.features.tutorial.lib.demo.data.literal.22"),
        pair_index: null,
      },
      {
        timestamp: ts(start, 2800),
        level: "INFO",
        logger: "dspy.datasets",
        message: msg("auto.features.tutorial.lib.demo.data.literal.23"),
        pair_index: null,
      },
      {
        timestamp: ts(start, 3000),
        level: "INFO",
        logger: "dspy.runners",
        message: msg("auto.features.tutorial.lib.demo.data.literal.24"),
        pair_index: null,
      },
    ],
    progress_count: 2,
    log_count: 6,
  };
}

function buildOptimizing(start: Date, trialsDone: number): OptimizationStatusResponse {
  const elapsed = (Date.now() - start.getTime()) / 1000;

  const events: ProgressEvent[] = [
    { timestamp: ts(start, 1500), event: "validation_passed", metrics: {} },
    {
      timestamp: ts(start, 2800),
      event: "dataset_splits_ready",
      metrics: { train_examples: 120, val_examples: 40, test_examples: 40 },
    },
    {
      timestamp: ts(start, 4200),
      event: "baseline_evaluated",
      metrics: { baseline_test_metric: 0.62 },
    },
  ];
  if (trialsDone > 0) {
    events.push({ timestamp: ts(start, 5000), event: "optimizer_progress", metrics: {} });
  }
  for (let i = 0; i < trialsDone && i < CANDIDATE_PLAN.length; i++) {
    events.push(candidateEvent(start, i, 5000 + i * 550));
  }

  const logs: OptimizationLogEntry[] = [
    {
      timestamp: ts(start, 200),
      level: "INFO",
      logger: "skynet.worker",
      message: msg("auto.features.tutorial.lib.demo.data.literal.5"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 400),
      level: "INFO",
      logger: "dspy.validators",
      message: msg("auto.features.tutorial.lib.demo.data.literal.19"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 1500),
      level: "INFO",
      logger: "dspy.validators",
      message: msg("auto.features.tutorial.lib.demo.data.literal.21"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 1800),
      level: "INFO",
      logger: "dspy.datasets",
      message: msg("auto.features.tutorial.lib.demo.data.literal.22"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 2800),
      level: "INFO",
      logger: "dspy.datasets",
      message: msg("auto.features.tutorial.lib.demo.data.literal.23"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 3000),
      level: "INFO",
      logger: "dspy.runners",
      message: msg("auto.features.tutorial.lib.demo.data.literal.24"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 4200),
      level: "INFO",
      logger: "dspy.runners",
      message: msg("auto.features.tutorial.lib.demo.data.literal.25"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 4500),
      level: "INFO",
      logger: "dspy.gepa",
      message: `Starting GEPA reflection over ${NUM_TRIALS} iterations...`,
      pair_index: null,
    },
  ];

  for (let i = 0; i < trialsDone; i++) {
    const offset = 5000 + i * 550;
    logs.push(
      {
        timestamp: ts(start, offset),
        level: "INFO",
        logger: "dspy.gepa",
        message: `Iteration ${i + 1}: Reflecting on minibatch (size=3)...`,
        pair_index: null,
      },
      {
        timestamp: ts(start, offset + 300),
        level: "INFO",
        logger: "dspy.gepa",
        message: `Iteration ${i + 1}: Full valset score for new program: ${((TRIAL_SCORES[i] ?? 0) / 100).toFixed(2)}`,
        pair_index: null,
      },
    );
  }

  return {
    ...baseJob(start),
    status: "running",
    elapsed_seconds: elapsed,
    elapsed: fmtElapsed(elapsed),
    baseline_test_metric: 0.62,
    progress_events: events,
    logs,
    latest_metrics: {
      tqdm_desc: "GEPA",
      tqdm_percent: (trialsDone / NUM_TRIALS) * 100,
      tqdm_n: trialsDone,
      tqdm_total: NUM_TRIALS,
    },
    progress_count: events.length,
    log_count: logs.length,
  };
}

function buildDone(start: Date): OptimizationStatusResponse {
  const elapsed = (Date.now() - start.getTime()) / 1000;

  const events: ProgressEvent[] = [
    { timestamp: ts(start, 1500), event: "validation_passed", metrics: {} },
    {
      timestamp: ts(start, 2800),
      event: "dataset_splits_ready",
      metrics: { train_examples: 120, val_examples: 40, test_examples: 40 },
    },
    {
      timestamp: ts(start, 4200),
      event: "baseline_evaluated",
      metrics: { baseline_test_metric: 0.62 },
    },
    { timestamp: ts(start, 5000), event: "optimizer_progress", metrics: {} },
    {
      timestamp: ts(start, 9500),
      event: "optimized_evaluated",
      metrics: { optimized_test_metric: 0.84 },
    },
  ];

  for (let i = 0; i < CANDIDATE_PLAN.length; i++) {
    events.push(candidateEvent(start, i, 5000 + i * 550));
  }

  const logs: OptimizationLogEntry[] = [
    {
      timestamp: ts(start, 200),
      level: "INFO",
      logger: "skynet.worker",
      message: msg("auto.features.tutorial.lib.demo.data.literal.6"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 400),
      level: "INFO",
      logger: "dspy.validators",
      message: msg("auto.features.tutorial.lib.demo.data.literal.19"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 1500),
      level: "INFO",
      logger: "dspy.validators",
      message: msg("auto.features.tutorial.lib.demo.data.literal.21"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 1800),
      level: "INFO",
      logger: "dspy.datasets",
      message: msg("auto.features.tutorial.lib.demo.data.literal.22"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 2800),
      level: "INFO",
      logger: "dspy.datasets",
      message: msg("auto.features.tutorial.lib.demo.data.literal.23"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 3000),
      level: "INFO",
      logger: "dspy.runners",
      message: msg("auto.features.tutorial.lib.demo.data.literal.24"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 4200),
      level: "INFO",
      logger: "dspy.runners",
      message: msg("auto.features.tutorial.lib.demo.data.literal.25"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 4500),
      level: "INFO",
      logger: "dspy.gepa",
      message: `Starting GEPA reflection over ${NUM_TRIALS} iterations...`,
      pair_index: null,
    },
  ];

  for (let i = 0; i < NUM_TRIALS; i++) {
    const offset = 5000 + i * 550;
    logs.push(
      {
        timestamp: ts(start, offset),
        level: "INFO",
        logger: "dspy.gepa",
        message: `Iteration ${i + 1}: Reflecting on minibatch (size=3)...`,
        pair_index: null,
      },
      {
        timestamp: ts(start, offset + 300),
        level: "INFO",
        logger: "dspy.gepa",
        message: `Iteration ${i + 1}: Full valset score for new program: ${((TRIAL_SCORES[i] ?? 0) / 100).toFixed(2)}`,
        pair_index: null,
      },
    );
  }

  logs.push(
    {
      timestamp: ts(start, 9200),
      level: "INFO",
      logger: "dspy.gepa",
      message: msg("auto.features.tutorial.lib.demo.data.literal.26"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 9500),
      level: "INFO",
      logger: "dspy.runners",
      message: msg("auto.features.tutorial.lib.demo.data.literal.27"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 10200),
      level: "INFO",
      logger: "dspy.runners",
      message: msg("auto.features.tutorial.lib.demo.data.literal.28"),
      pair_index: null,
    },
    {
      timestamp: ts(start, 10500),
      level: "INFO",
      logger: "skynet.worker",
      message: msg("auto.features.tutorial.lib.demo.data.literal.29"),
      pair_index: null,
    },
  );

  return {
    ...baseJob(start),
    status: "success",
    elapsed_seconds: elapsed,
    elapsed: fmtElapsed(elapsed),
    completed_at: new Date().toISOString(),
    baseline_test_metric: 0.62,
    optimized_test_metric: 0.84,
    metric_improvement: 0.22,
    progress_events: events,
    logs,
    latest_metrics: {
      tqdm_desc: "Completed",
      tqdm_percent: 100,
      tqdm_n: NUM_TRIALS,
      tqdm_total: NUM_TRIALS,
    },
    progress_count: events.length,
    log_count: logs.length,
    result: {
      module_name: "Predict",
      optimizer_name: "GEPA",
      baseline_test_metric: 0.62,
      optimized_test_metric: 0.84,
      metric_improvement: 0.22,
      runtime_seconds: elapsed,
      num_lm_calls: 156,
      split_counts: { train: 120, val: 40, test: 40 },
      lm_activity: {
        generation: {
          baseline: { calls: 40, avg_response_time_ms: 1850 },
          training: { calls: 80, avg_response_time_ms: 2110 },
          evaluation: { calls: 36, avg_response_time_ms: 1690 },
        },
        reflection: {
          training: { calls: 12, avg_response_time_ms: 5240 },
        },
      },
      program_artifact: {
        program_state_json: {
          "predict.signature.instructions":
            "Classify each email as spam, important, or promotional. Use the message's intent, urgency, and requested action; do not classify from isolated keywords alone.",
        },
        optimized_prompt: {
          predictor_name: "EmailClassifier",
          signature_name: "EmailClassifier",
          instructions:
            "Classify each email as spam, important, or promotional. Use the message's intent, urgency, and requested action; do not classify from isolated keywords alone.",
          input_fields: ["email_text"],
          output_fields: ["category"],
          demos: [
            {
              inputs: { email_text: "Your quarterly report is ready for review" },
              outputs: { category: "important" },
            },
          ],
          formatted_prompt:
            "Classify each email as spam, important, or promotional.\n\nInput: {email_text}\nOutput: {category}",
        },
      },
    },
  };
}

export interface DemoCallbacks {
  setJob: (
    updater: (prev: OptimizationStatusResponse | null) => OptimizationStatusResponse,
  ) => void;
  setLoading: (v: boolean) => void;
}

/**
 * Starts a simulated optimization that progresses through all pipeline stages.
 * Returns a cleanup function that cancels all pending timers.
 *
 * Timeline:
 *   0.2s  — validating
 *   1.0s  — splitting
 *   1.75s — baseline evaluation
 *   2.5s+ — optimizing (trials appear every ~275ms)
 *   5.25s — done (success)
 */
const DEMO_SIMULATION_DURATION_MS = 5250;

/** Once the simulation finishes, revisits skip straight to done. */
let _simulationCompleted = false;

/** Reset the completed flag so the next visit re-runs the simulation. */
export function resetDemoSimulation() {
  _simulationCompleted = false;
}

export interface StartDemoSimulationOptions {
  /**
   * Skip past validating/splitting/baseline and start with the first trial
   * already complete, then stream the remaining candidates ~0.45s apart.
   * Used when the tutorial replays the simulation while the user is already
   * looking at the trajectory panel — keeps the panel visible from the
   * first frame instead of blanking out for the 2.5s pipeline warm-up.
   */
  fromTrajectory?: boolean;
}

const REPLAY_STAGGER_MS = 450;

export function startDemoSimulation(
  callbacks: DemoCallbacks,
  options?: StartDemoSimulationOptions,
): () => void {
  const { setJob, setLoading } = callbacks;
  const fromTrajectory = options?.fromTrajectory === true;

  if (_simulationCompleted) {
    setLoading(false);
    // Synthesize a `start` shifted into the past so wall-clock fields
    // (completed_at, elapsed_seconds, log/event timestamps) reflect the
    // current revisit instead of being frozen at the original run.
    const fakeStart = new Date(Date.now() - DEMO_SIMULATION_DURATION_MS);
    setJob(() => buildDone(fakeStart));
    return () => {};
  }

  const timers: Array<ReturnType<typeof setTimeout>> = [];
  const set = (job: OptimizationStatusResponse) => setJob(() => job);

  if (fromTrajectory) {
    // Anchor `start` in the past so the existing event/log offsets land in
    // the recent past while still letting the live elapsed clock read a
    // believable value during the replay.
    const replayStart = new Date(Date.now() - 2500);
    setLoading(false);
    set(buildOptimizing(replayStart, 1));
    for (let i = 1; i < NUM_TRIALS; i++) {
      timers.push(
        setTimeout(() => set(buildOptimizing(replayStart, i + 1)), i * REPLAY_STAGGER_MS),
      );
    }
    timers.push(
      setTimeout(
        () => {
          _simulationCompleted = true;
          set(buildDone(replayStart));
        },
        NUM_TRIALS * REPLAY_STAGGER_MS + 400,
      ),
    );
    return () => timers.forEach(clearTimeout);
  }

  const start = new Date();

  timers.push(
    setTimeout(() => {
      setLoading(false);
      set(buildValidating(start));
    }, 200),
  );

  timers.push(setTimeout(() => set(buildSplitting(start)), 1000));

  timers.push(setTimeout(() => set(buildBaseline(start)), 1750));

  for (let i = 0; i < NUM_TRIALS; i++) {
    timers.push(setTimeout(() => set(buildOptimizing(start, i + 1)), 2500 + i * 275));
  }

  timers.push(
    setTimeout(() => {
      _simulationCompleted = true;
      set(buildDone(start));
    }, DEMO_SIMULATION_DURATION_MS),
  );

  return () => timers.forEach(clearTimeout);
}

function daysAgo(n: number): string {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d.toISOString();
}

const DEMO_JOBS: OptimizationSummaryResponse[] = perLocale(() => [
  {
    optimization_id: "demo-001",
    optimization_type: "run",
    status: "success",
    name: msg("auto.features.tutorial.lib.demo.data.literal.7"),
    description: msg("auto.features.tutorial.lib.demo.data.literal.8"),
    created_at: daysAgo(5),
    completed_at: daysAgo(5),
    elapsed: "2:45",
    elapsed_seconds: 165,
    module_name: "ChainOfThought",
    optimizer_name: "GEPA",
    model_name: "gpt-4o-mini",
    dataset_rows: 200,
    baseline_test_metric: 0.62,
    optimized_test_metric: 0.84,
    metric_improvement: 0.22,
    username: "demo",
  },
  {
    optimization_id: "demo-002",
    optimization_type: "run",
    status: "success",
    name: msg("auto.features.tutorial.lib.demo.data.literal.9"),
    description: msg("auto.features.tutorial.lib.demo.data.literal.10"),
    created_at: daysAgo(3),
    completed_at: daysAgo(3),
    elapsed: "4:12",
    elapsed_seconds: 252,
    module_name: "Predict",
    optimizer_name: "GEPA",
    model_name: "gpt-4o",
    dataset_rows: 350,
    baseline_test_metric: 0.71,
    optimized_test_metric: 0.89,
    metric_improvement: 0.18,
    username: "demo",
  },
  {
    optimization_id: DEMO_GRID_OPTIMIZATION_ID,
    optimization_type: "grid_search",
    status: "success",
    name: msg("auto.features.tutorial.lib.demo.data.literal.11"),
    description: formatMsg("auto.features.tutorial.lib.demo.data.template.2", {
      p1: TERMS.modelPlural,
      p2: TERMS.pairPlural,
      p3: TERMS.generationModel,
      p4: TERMS.reflectionModel,
      p5: TERMS.task,
    }),
    created_at: daysAgo(2),
    completed_at: daysAgo(2),
    elapsed: "8:30",
    elapsed_seconds: 510,
    module_name: "ChainOfThought",
    optimizer_name: "GEPA",
    model_name: "gpt-4o-mini",
    dataset_rows: 150,
    baseline_test_metric: 0.48,
    optimized_test_metric: 0.86,
    metric_improvement: 0.38,
    username: "demo",
    total_pairs: 4,
    completed_pairs: 4,
  },
  {
    optimization_id: "demo-004",
    optimization_type: "run",
    status: "failed",
    name: msg("auto.features.tutorial.lib.demo.data.literal.12"),
    description: msg("auto.features.tutorial.lib.demo.data.literal.13"),
    created_at: daysAgo(1),
    elapsed: "0:32",
    elapsed_seconds: 32,
    module_name: "Predict",
    optimizer_name: "GEPA",
    model_name: "gpt-4o",
    dataset_rows: 80,
    username: "demo",
    message: msg("auto.features.tutorial.lib.demo.data.literal.30"),
  },
  {
    optimization_id: "demo-005",
    optimization_type: "run",
    status: "running",
    name: msg("auto.features.tutorial.lib.demo.data.literal.14"),
    description: msg("auto.features.tutorial.lib.demo.data.literal.15"),
    created_at: daysAgo(0),
    elapsed: "1:15",
    elapsed_seconds: 75,
    module_name: "ChainOfThought",
    optimizer_name: "GEPA",
    model_name: "gpt-4o-mini",
    dataset_rows: 120,
    baseline_test_metric: 0.48,
    username: "demo",
    latest_metrics: { tqdm_percent: 45, tqdm_n: 4, tqdm_total: 9 },
  },
  {
    optimization_id: "demo-006",
    optimization_type: "run",
    status: "success",
    name: msg("auto.features.tutorial.lib.demo.data.literal.9"),
    description: msg("auto.features.tutorial.lib.demo.data.literal.10"),
    created_at: daysAgo(12),
    completed_at: daysAgo(12),
    elapsed: "3:28",
    elapsed_seconds: 208,
    module_name: "Predict",
    optimizer_name: "MIPROv2",
    model_name: "claude-sonnet-4",
    dataset_rows: 280,
    baseline_test_metric: 0.66,
    optimized_test_metric: 0.83,
    metric_improvement: 0.17,
    username: "demo",
  },
  {
    optimization_id: "demo-007",
    optimization_type: "run",
    status: "success",
    name: msg("auto.features.tutorial.lib.demo.data.literal.12"),
    description: msg("auto.features.tutorial.lib.demo.data.literal.13"),
    created_at: daysAgo(9),
    completed_at: daysAgo(9),
    elapsed: "5:50",
    elapsed_seconds: 350,
    module_name: "ChainOfThought",
    optimizer_name: "BootstrapFewShot",
    model_name: "claude-haiku-4",
    dataset_rows: 180,
    baseline_test_metric: 0.51,
    optimized_test_metric: 0.74,
    metric_improvement: 0.23,
    username: "demo",
  },
  {
    optimization_id: "demo-008",
    optimization_type: "run",
    status: "success",
    name: msg("auto.features.tutorial.lib.demo.data.literal.14"),
    description: msg("auto.features.tutorial.lib.demo.data.literal.15"),
    created_at: daysAgo(7),
    completed_at: daysAgo(7),
    elapsed: "4:36",
    elapsed_seconds: 276,
    module_name: "Predict",
    optimizer_name: "MIPROv2",
    model_name: "gemini-2.0-pro",
    dataset_rows: 220,
    baseline_test_metric: 0.58,
    optimized_test_metric: 0.79,
    metric_improvement: 0.21,
    username: "demo",
  },
  {
    optimization_id: "demo-009",
    optimization_type: "run",
    status: "success",
    name: msg("auto.features.tutorial.lib.demo.data.literal.7"),
    description: msg("auto.features.tutorial.lib.demo.data.literal.8"),
    created_at: daysAgo(15),
    completed_at: daysAgo(15),
    elapsed: "2:08",
    elapsed_seconds: 128,
    module_name: "ChainOfThought",
    optimizer_name: "BootstrapFewShot",
    model_name: "claude-sonnet-4",
    dataset_rows: 250,
    baseline_test_metric: 0.6,
    optimized_test_metric: 0.78,
    metric_improvement: 0.18,
    username: "demo",
  },
]);

function toAnalyticsJob(j: OptimizationSummaryResponse): DashboardAnalyticsJob {
  return {
    optimization_id: j.optimization_id,
    name: j.name,
    optimizer_name: j.optimizer_name,
    model_name: j.model_name,
    status: j.status,
    baseline_test_metric: j.baseline_test_metric ?? null,
    optimized_test_metric: j.optimized_test_metric ?? null,
    metric_improvement: j.metric_improvement ?? null,
    elapsed_seconds: j.elapsed_seconds ?? null,
    dataset_rows: j.dataset_rows ?? null,
    optimization_type: j.optimization_type,
    created_at: j.created_at,
  };
}

export const DEMO_DASHBOARD_JOBS: PaginatedJobsResponse = perLocale(() => ({
  items: DEMO_JOBS,
  total: DEMO_JOBS.length,
  limit: 20,
  offset: 0,
}));

export const DEMO_DASHBOARD_ANALYTICS: DashboardAnalytics = perLocale(() => ({
  filtered_total: DEMO_JOBS.length,
  status_counts: { success: 7, failed: 1, running: 1 },
  optimizer_counts: { GEPA: 5, MIPROv2: 2, BootstrapFewShot: 2 },
  job_type_counts: { run: 8, grid_search: 1 },
  model_usage: [
    { name: "gpt-4o-mini", value: 3 },
    { name: "gpt-4o", value: 2 },
    { name: "claude-sonnet-4", value: 2 },
    { name: "claude-haiku-4", value: 1 },
    { name: "gemini-2.0-pro", value: 1 },
  ],
  owner_usage: [],
  access_usage: [],
  success_count: 7,
  failed_count: 1,
  running_count: 1,
  terminal_count: 8,
  success_rate: 87.5,
  avg_improvement: 0.22,
  avg_runtime_seconds: 222,
  total_dataset_rows: 1830,
  total_pairs_run: 4,
  grid_search_count: 1,
  single_run_count: 8,
  best_improvement: 0.38,
  improvement_by_optimizer: [
    { name: "GEPA", average: 0.26, count: 3 },
    { name: "MIPROv2", average: 0.19, count: 2 },
    { name: "BootstrapFewShot", average: 0.205, count: 2 },
  ],
  runtime_minutes_by_optimizer: [
    { name: "GEPA", average: 3.45, count: 5 },
    { name: "MIPROv2", average: 4.03, count: 2 },
    { name: "BootstrapFewShot", average: 3.98, count: 2 },
  ],
  top_improvement: DEMO_JOBS.filter((j) => j.metric_improvement).map(toAnalyticsJob),
  runtime_distribution: DEMO_JOBS.filter((j) => j.elapsed_seconds).map(toAnalyticsJob),
  dataset_vs_improvement: DEMO_JOBS.filter((j) => j.metric_improvement).map(toAnalyticsJob),
  efficiency: DEMO_JOBS.filter((j) => j.metric_improvement).map(toAnalyticsJob),
  top_jobs_by_improvement: DEMO_JOBS.filter((j) => j.metric_improvement)
    .sort((a, b) => (b.metric_improvement ?? 0) - (a.metric_improvement ?? 0))
    .map(toAnalyticsJob),
  timeline: [
    { date: daysAgo(15).slice(0, 10), count: 1 },
    { date: daysAgo(12).slice(0, 10), count: 1 },
    { date: daysAgo(9).slice(0, 10), count: 1 },
    { date: daysAgo(7).slice(0, 10), count: 1 },
    { date: daysAgo(5).slice(0, 10), count: 1 },
    { date: daysAgo(3).slice(0, 10), count: 1 },
    { date: daysAgo(2).slice(0, 10), count: 1 },
    { date: daysAgo(1).slice(0, 10), count: 1 },
    { date: daysAgo(0).slice(0, 10), count: 1 },
  ],
  available_optimizers: ["GEPA", "MIPROv2", "BootstrapFewShot"],
  available_models: [
    "gpt-4o-mini",
    "gpt-4o",
    "claude-sonnet-4",
    "claude-haiku-4",
    "gemini-2.0-pro",
  ],
}));

const gridTaskName = () => msg("auto.features.tutorial.lib.demo.data.literal.17");

function gridInstructions(variant: 0 | 1 | 2 | 3): string {
  if (variant === 0) {
    return "Summarize the article in one concise sentence that captures the main point.";
  }
  if (variant === 1) {
    return "Read the article and produce a one-sentence summary that highlights the most important fact or event.";
  }
  if (variant === 2) {
    return "Given an article, write a single-sentence summary. Focus on the core message, omit stylistic filler.";
  }
  return "Summarize the article into one sentence. Prioritize specific entities, numbers, and outcomes over generic framing.";
}

function gridPair(
  idx: 0 | 1 | 2 | 3,
  opts: {
    genModel: string;
    refModel: string;
    baseline: number;
    optimized: number;
    runtime: number;
    numLmCalls: number;
    avgMs: number;
  },
): PairResult {
  const reflCalls = Math.round(opts.numLmCalls * 0.15);
  const genCalls = opts.numLmCalls - reflCalls;
  const genBaseline = Math.round(genCalls * 0.3);
  const genTraining = Math.round(genCalls * 0.45);
  const genEval = genCalls - genBaseline - genTraining;
  return {
    pair_index: idx,
    generation_model: opts.genModel,
    reflection_model: opts.refModel,
    baseline_test_metric: opts.baseline,
    optimized_test_metric: opts.optimized,
    metric_improvement: opts.optimized - opts.baseline,
    runtime_seconds: opts.runtime,
    num_lm_calls: opts.numLmCalls,
    avg_response_time_ms: opts.avgMs,
    lm_activity: {
      generation: {
        baseline: { calls: genBaseline, avg_response_time_ms: opts.avgMs },
        training: { calls: genTraining, avg_response_time_ms: Math.round(opts.avgMs * 1.15) },
        evaluation: { calls: genEval, avg_response_time_ms: Math.round(opts.avgMs * 0.9) },
      },
      reflection: {
        training: { calls: reflCalls, avg_response_time_ms: Math.round(opts.avgMs * 4) },
      },
    },
    program_artifact: {
      optimized_prompt: {
        predictor_name: "ArticleSummarizer",
        signature_name: "ArticleSummarizer",
        instructions: gridInstructions(idx),
        input_fields: ["article"],
        output_fields: ["summary"],
        demos: [
          {
            inputs: {
              article:
                "A local library announced extended weekend hours starting in May to accommodate student demand.",
            },
            outputs: {
              summary: "A local library will extend weekend hours in May to meet student demand.",
            },
          },
        ],
        formatted_prompt: "",
      },
    },
  };
}

const GRID_PAIR_TRIAL_SCORES: Record<0 | 1 | 2 | 3, number[]> = {
  0: [0.52, 0.58, 0.61, 0.66, 0.68, 0.72, 0.74],
  1: [0.55, 0.61, 0.68, 0.73, 0.76, 0.8, 0.82],
  2: [0.53, 0.59, 0.64, 0.68, 0.72, 0.75, 0.77],
  3: [0.57, 0.64, 0.71, 0.77, 0.81, 0.84, 0.86],
};

function buildGridPairLogs(
  pairIdx: 0 | 1 | 2 | 3,
  start: Date,
  baseOffsetMs: number,
): OptimizationLogEntry[] {
  const scores = GRID_PAIR_TRIAL_SCORES[pairIdx];
  const logs: OptimizationLogEntry[] = [
    {
      timestamp: ts(start, baseOffsetMs),
      level: "INFO",
      logger: "dspy.gepa",
      message: `Starting GEPA reflection over ${scores.length} iterations...`,
      pair_index: pairIdx,
    },
  ];
  scores.forEach((score, i) => {
    const offset = baseOffsetMs + 400 + i * 550;
    logs.push(
      {
        timestamp: ts(start, offset),
        level: "INFO",
        logger: "dspy.gepa",
        message: `Iteration ${i + 1}: Reflecting on minibatch (size=3)...`,
        pair_index: pairIdx,
      },
      {
        timestamp: ts(start, offset + 300),
        level: "INFO",
        logger: "dspy.gepa",
        message: `Iteration ${i + 1}: Full valset score for new program: ${score.toFixed(2)}`,
        pair_index: pairIdx,
      },
    );
  });
  const finalScore = scores[scores.length - 1]!;
  logs.push({
    timestamp: ts(start, baseOffsetMs + 400 + scores.length * 550 + 600),
    level: "INFO",
    logger: "dspy.gepa",
    message: `Best program found with score: ${finalScore.toFixed(2)}`,
    pair_index: pairIdx,
  });
  return logs;
}

export function buildGridDemoJob(): OptimizationStatusResponse {
  const pairs: PairResult[] = [
    gridPair(0, {
      genModel: "openai/gpt-4o-mini",
      refModel: "openai/gpt-4o-mini",
      baseline: 0.48,
      optimized: 0.74,
      runtime: 180,
      numLmCalls: 82,
      avgMs: 260,
    }),
    gridPair(1, {
      genModel: "openai/gpt-4o-mini",
      refModel: "openai/gpt-4o",
      baseline: 0.48,
      optimized: 0.82,
      runtime: 245,
      numLmCalls: 96,
      avgMs: 320,
    }),
    gridPair(2, {
      genModel: "openai/gpt-4o",
      refModel: "openai/gpt-4o-mini",
      baseline: 0.48,
      optimized: 0.77,
      runtime: 230,
      numLmCalls: 94,
      avgMs: 420,
    }),
    gridPair(3, {
      genModel: "openai/gpt-4o",
      refModel: "openai/gpt-4o",
      baseline: 0.48,
      optimized: 0.86,
      runtime: 380,
      numLmCalls: 118,
      avgMs: 650,
    }),
  ];

  const best = pairs.reduce((a, b) =>
    (b.optimized_test_metric ?? -Infinity) > (a.optimized_test_metric ?? -Infinity) ? b : a,
  );
  const grid: GridSearchResult = {
    module_name: "ChainOfThought",
    optimizer_name: "GEPA",
    split_counts: { train: 90, val: 30, test: 30 },
    total_pairs: pairs.length,
    completed_pairs: pairs.length,
    failed_pairs: 0,
    pair_results: pairs,
    best_pair: best,
    runtime_seconds: pairs.reduce((sum, p) => sum + (p.runtime_seconds ?? 0), 0),
  };

  const logStart = new Date();
  const pairLogs = ([0, 1, 2, 3] as const).flatMap((idx) =>
    buildGridPairLogs(idx, logStart, idx * 8000),
  );

  return {
    optimization_id: DEMO_GRID_OPTIMIZATION_ID,
    optimization_type: "grid_search",
    status: "success",
    name: gridTaskName(),
    description: formatMsg("auto.features.tutorial.lib.demo.data.template.3", {
      p1: TERMS.modelPlural,
      p2: TERMS.pairPlural,
      p3: TERMS.generationModel,
      p4: TERMS.reflectionModel,
      p5: TERMS.task,
    }),
    username: "demo",
    created_at: daysAgo(2),
    started_at: daysAgo(2),
    completed_at: daysAgo(2),
    elapsed_seconds: grid.runtime_seconds,
    elapsed: fmtElapsed(grid.runtime_seconds ?? 0),
    module_name: "ChainOfThought",
    module_kwargs: {},
    optimizer_name: "GEPA",
    optimizer_kwargs: { auto: "light" },
    compile_kwargs: {},
    model_name: "openai/gpt-4o-mini",
    dataset_rows: 150,
    column_mapping: { inputs: { article: "str" }, outputs: { summary: "str" } },
    split_fractions: { train: 0.6, val: 0.2, test: 0.2 },
    shuffle: true,
    seed: 42,
    baseline_test_metric: best.baseline_test_metric,
    optimized_test_metric: best.optimized_test_metric,
    metric_improvement: best.metric_improvement,
    total_pairs: pairs.length,
    completed_pairs: pairs.length,
    progress_events: [],
    logs: pairLogs,
    latest_metrics: {},
    progress_count: 0,
    log_count: pairLogs.length,
    grid_result: grid,
  };
}

const EXPLORE_TASKS: string[] = perLocale(() => [
  msg("auto.features.tutorial.lib.demo.data.literal.1"),
  msg("auto.features.tutorial.lib.demo.data.literal.9"),
  msg("auto.features.tutorial.lib.demo.data.literal.12"),
  msg("auto.features.tutorial.lib.demo.data.literal.31"),
  msg("auto.features.tutorial.lib.demo.data.literal.32"),
  msg("auto.features.tutorial.lib.demo.data.literal.33"),
  msg("auto.features.tutorial.lib.demo.data.literal.14"),
  msg("auto.features.tutorial.lib.demo.data.literal.34"),
]);

// Maps each of the 32 grid cells to one of the 8 task indices above so
// neighbouring cells share themes — the resulting scatter has visible
// task-coloured neighbourhoods without any aggregation layer.
const EXPLORE_LEAF_TASK = [
  0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7,
];

const EXPLORE_MODELS = [
  "gpt-4o",
  "gpt-4o-mini",
  "claude-sonnet-4",
  "claude-haiku-4",
  "gemini-2.0-pro",
];

const EXPLORE_OPTIMIZERS = ["GEPA", "MIPROv2", "BootstrapFewShot", "GridSearch"];

const EXPLORE_DEMO_TOTAL = 1000;
const EXPLORE_CLUSTER_ROWS = 4;
const EXPLORE_CLUSTER_COLS = 8;
const EXPLORE_CLUSTER_TOTAL = EXPLORE_CLUSTER_ROWS * EXPLORE_CLUSTER_COLS;

function buildExploreDemoPoints(): PublicDashboardPoint[] {
  // Deterministic LCG so the layout is stable across reloads — keeps the
  // tutorial screenshot-friendly.
  let seed = 0x6a09e667;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0x1_0000_0000;
  };

  const now = Date.now();
  const points: PublicDashboardPoint[] = [];
  for (let i = 0; i < EXPLORE_DEMO_TOTAL; i++) {
    const c32 = i % EXPLORE_CLUSTER_TOTAL;
    const baseline = 0.38 + rand() * 0.32;
    const lift = 0.04 + rand() * 0.22;
    const optimized = Math.min(0.97, baseline + lift);

    const isGrid = i % 7 === 0;
    const type: "run" | "grid_search" = isGrid ? "grid_search" : "run";
    const modelIdx = (c32 + Math.floor(i / EXPLORE_CLUSTER_TOTAL)) % EXPLORE_MODELS.length;
    const optimizerIdx = isGrid
      ? EXPLORE_OPTIMIZERS.indexOf("GridSearch")
      : i % (EXPLORE_OPTIMIZERS.length - 1);
    const taskIdx = EXPLORE_LEAF_TASK[c32] ?? 0;
    const daysAgo = Math.floor((i / EXPLORE_DEMO_TOTAL) * 30) + (i % 5);

    points.push({
      optimization_id: `tutorial-explore-${i.toString().padStart(4, "0")}`,
      optimization_type: type,
      winning_model: EXPLORE_MODELS[modelIdx] ?? null,
      baseline_metric: Number((baseline * 100).toFixed(1)),
      optimized_metric: Number((optimized * 100).toFixed(1)),
      summary_text: null,
      task_name: EXPLORE_TASKS[taskIdx] ?? null,
      module_name: "Predict",
      optimizer_name: EXPLORE_OPTIMIZERS[optimizerIdx] ?? null,
      created_at: new Date(now - daysAgo * 86400_000).toISOString(),
    });
  }
  return points;
}

export const DEMO_EXPLORE_POINTS: PublicDashboardPoint[] = perLocale(buildExploreDemoPoints);

/**
 * Cached “real” data layer.
 *
 * The synthetic demo payloads above are already shaped like live backend
 * responses (same types, same fields, same score distributions) so the
 * tutorial renders pixel-identical to the app. To avoid re-running a real
 * optimization or hitting the API on every tour loop, the resolved payloads
 * are cached in localStorage with a TTL and read-through on next run —
 * instant while still reflecting the user’s actual recent jobs when a
 * backend fetch succeeds.
 */
const TUTORIAL_CACHE_PREFIX = "skynet.tutorial.cache.v2";
const TUTORIAL_CACHE_TTL_MS = 1000 * 60 * 60 * 6; // 6h

interface CacheEntry<T> {
  v: T;
  ts: number;
}

function readTutorialCache<T>(key: string): T | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(`${TUTORIAL_CACHE_PREFIX}.${key}`);
    if (!raw) return null;
    const entry = JSON.parse(raw) as CacheEntry<T>;
    if (!entry || typeof entry.ts !== "number" || Date.now() - entry.ts > TUTORIAL_CACHE_TTL_MS) {
      window.localStorage.removeItem(`${TUTORIAL_CACHE_PREFIX}.${key}`);
      return null;
    }
    return entry.v;
  } catch {
    return null;
  }
}

function writeTutorialCache<T>(key: string, value: T): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      `${TUTORIAL_CACHE_PREFIX}.${key}`,
      JSON.stringify({ v: value, ts: Date.now() } satisfies CacheEntry<T>),
    );
  } catch {
    /* quota — ignore */
  }
}

export function getCachedDemoDashboardJobs(): PaginatedJobsResponse {
  // perLocale proxies for arrays serialize as objects via JSON.stringify (target is {}),
  // so build a real plain PaginatedJobsResponse via Array.from to keep items as a true array.
  const v = DEMO_DASHBOARD_JOBS;
  const plain: PaginatedJobsResponse = {
    items: Array.from(v.items as unknown as Iterable<OptimizationSummaryResponse>),
    total: v.total,
    limit: v.limit,
    offset: v.offset,
  };
  try {
    writeTutorialCache("dashboard-jobs", plain);
  } catch {}
  const cached = readTutorialCache<PaginatedJobsResponse>("dashboard-jobs");
  if (cached && Array.isArray((cached as unknown as { items: unknown }).items)) return cached;
  return plain;
}

export function getCachedDemoDashboardAnalytics(): DashboardAnalytics {
  const v = DEMO_DASHBOARD_ANALYTICS;
  try {
    writeTutorialCache("dashboard-analytics", JSON.parse(JSON.stringify(v)) as DashboardAnalytics);
  } catch {}
  const cached = readTutorialCache<DashboardAnalytics>("dashboard-analytics");
  if (cached && typeof cached === "object" && cached !== null) return cached;
  return v;
}

export function getCachedDemoExplorePoints(): PublicDashboardPoint[] {
  const v = DEMO_EXPLORE_POINTS;
  const plain = Array.from(v as unknown as Iterable<PublicDashboardPoint>);
  try {
    writeTutorialCache("explore-points", plain);
  } catch {}
  const cached = readTutorialCache<PublicDashboardPoint[]>("explore-points");
  if (cached && Array.isArray(cached)) return cached;
  return plain;
}

export function primeTutorialCacheFromLiveData(opts?: {
  jobs?: PaginatedJobsResponse;
  analytics?: DashboardAnalytics;
  explorePoints?: PublicDashboardPoint[];
}): void {
  if (opts?.jobs) writeTutorialCache("dashboard-jobs", opts.jobs);
  if (opts?.analytics) writeTutorialCache("dashboard-analytics", opts.analytics);
  if (opts?.explorePoints) writeTutorialCache("explore-points", opts.explorePoints);
}
