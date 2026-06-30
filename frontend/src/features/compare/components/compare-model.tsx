import type { OptimizationStatusResponse, OptimizedPredictor } from "@/shared/types/api";
import { computePairScores } from "@/features/optimizations";

export type RunInfo = {
  job: OptimizationStatusResponse;
  label: string;
  baseline: number | null;
  optimized: number | null;
  improvement: number | null;
  runtime: number | null;
  avgResponseMs: number | null;
  prompt: OptimizedPredictor | null;
  isGrid: boolean;
  pairLabel: string | null;
  winnerPairIndex: number | null;
  moduleName: string | null;
  modelName: string | null;
  reasoningEffort: string | null;
  reflectionModelName: string | null;
  reflectionReasoningEffort: string | null;
  datasetRows: number | null;
};

// 8 warm-brown values (one per run up to COMPARE_MAX in DashboardView).
// Ordered so each neighbor alternates dark/light for high adjacent contrast
// at small N (2-4, the common case); later slots fill in remaining L gaps.
const COL_HUES = [
  "#2B1E14", // A · deep espresso   (L≈22)
  "#D8BC9A", // B · light cream     (L≈78)
  "#6B4F38", // C · coffee          (L≈40)
  "#B5A08A", // D · warm mushroom   (L≈65)
  "#3D2E22", // E · cocoa           (L≈28)
  "#C8A882", // F · warm tan        (L≈72)
  "#8C7A6B", // G · dove            (L≈52)
  "#A48B75", // H · sandy           (L≈60)
];
export const colorFor = (i: number) => COL_HUES[i % COL_HUES.length];
// A, B, C … through Z. Past 26 (caller passes more runs than the 8-hue
// palette can distinguish anyway) fall back to "27", "28" so we don't render
// the punctuation chars (`[`, `\`, `]` …) that follow Z in ASCII.
export const runToken = (i: number) => (i < 26 ? String.fromCharCode(65 + i) : String(i + 1));

export function RunChip({
  index,
  label,
  winner,
  size = "sm",
}: {
  index: number;
  label: string;
  winner?: boolean;
  size?: "sm" | "md";
}) {
  const fontSize = size === "md" ? "text-sm" : "text-xs";
  const tokenSize = size === "md" ? "size-6 text-[0.6875rem]" : "size-5 text-[0.625rem]";
  return (
    <span className={`inline-flex min-w-0 items-center gap-1.5 ${fontSize}`}>
      <span
        className={`${tokenSize} rounded-md flex items-center justify-center font-bold text-white tabular-nums shrink-0`}
        style={{ background: colorFor(index) }}
      >
        {runToken(index)}
      </span>
      <span
        className={`min-w-0 overflow-hidden text-ellipsis whitespace-nowrap font-mono tabular-nums ${
          winner ? "text-primary font-bold" : ""
        }`}
        dir="ltr"
        title={label}
      >
        {label}
      </span>
    </span>
  );
}

export function fmt(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${v.toFixed(1)}%`;
}

export function fmtImprovement(v: number | null | undefined): string {
  if (v == null) return "—";
  return v >= 0 ? `+${v.toFixed(1)}%` : `${v.toFixed(1)}%`;
}

export function fmtElapsed(s: number | null | undefined): string {
  if (s == null) return "—";
  const hrs = Math.floor(s / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const secs = Math.floor(s % 60);
  const pad = (n: number) => String(n).padStart(2, "0");
  if (hrs > 0) return `${hrs}:${pad(mins)}:${pad(secs)}`;
  return `${mins}:${pad(secs)}`;
}

export function fmtLatency(ms: number | null | undefined): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

export function deriveRunInfo(job: OptimizationStatusResponse): RunInfo {
  const label = job.optimization_id;
  if (job.grid_result) {
    const scoring = computePairScores(job.grid_result.pair_results);
    const winnerIdx = scoring.overallWinner;
    const winnerPair =
      winnerIdx != null
        ? job.grid_result.pair_results.find((p) => p.pair_index === winnerIdx)
        : undefined;
    const baseline = winnerPair?.baseline_test_metric ?? null;
    const optimized = winnerPair?.optimized_test_metric ?? null;
    const improvement = baseline != null && optimized != null ? optimized - baseline : null;
    return {
      job,
      label,
      baseline,
      optimized,
      improvement,
      runtime: job.grid_result.runtime_seconds ?? winnerPair?.runtime_seconds ?? null,
      avgResponseMs: winnerPair?.avg_response_time_ms ?? null,
      prompt: winnerPair?.program_artifact?.optimized_prompt ?? null,
      isGrid: true,
      pairLabel: winnerPair
        ? `${winnerPair.generation_model} × ${winnerPair.reflection_model}`
        : null,
      winnerPairIndex: winnerIdx ?? null,
      moduleName: job.module_name ?? null,
      modelName: winnerPair?.generation_model ?? null,
      reasoningEffort: winnerPair?.generation_reasoning_effort ?? null,
      reflectionModelName: winnerPair?.reflection_model ?? null,
      reflectionReasoningEffort: winnerPair?.reflection_reasoning_effort ?? null,
      datasetRows: job.dataset_rows ?? null,
    };
  }
  const modelExtra = (job.model_settings as { extra?: { reasoning_effort?: string } } | undefined)
    ?.extra;
  return {
    job,
    label,
    baseline: job.result?.baseline_test_metric ?? null,
    optimized: job.result?.optimized_test_metric ?? null,
    improvement: job.result?.metric_improvement ?? null,
    runtime: job.result?.runtime_seconds ?? null,
    avgResponseMs: job.result?.avg_response_time_ms ?? null,
    prompt: job.result?.program_artifact?.optimized_prompt ?? null,
    isGrid: false,
    pairLabel: null,
    winnerPairIndex: null,
    moduleName: job.module_name ?? null,
    modelName: job.model_name ?? null,
    reasoningEffort: modelExtra?.reasoning_effort ?? null,
    reflectionModelName: null,
    reflectionReasoningEffort: null,
    datasetRows: job.dataset_rows ?? null,
  };
}

export function bestIndexOf(values: Array<number | null>, prefer: "max" | "min"): number | null {
  let bestIdx: number | null = null;
  let bestVal: number | null = null;
  values.forEach((v, i) => {
    if (v == null) return;
    if (bestVal == null || (prefer === "max" ? v > bestVal : v < bestVal)) {
      bestVal = v;
      bestIdx = i;
    }
  });
  return bestIdx;
}

export function isRowTie(values: Array<number | null>): boolean {
  const nonNull = values.filter((v): v is number => v != null);
  if (nonNull.length < 2) return false;
  return nonNull.every((v) => v === nonNull[0]);
}

export function winnerIndexOf(runs: RunInfo[]): number | null {
  return bestIndexOf(
    runs.map((r) => r.optimized),
    "max",
  );
}
