"use client";

import { memo, type ReactNode } from "react";
import dynamic from "next/dynamic";
import { Gauge, Hourglass, MessageSquare, Timer, TrendingUp } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { FadeIn, StaggerContainer, StaggerItem, TiltCard } from "@/shared/ui/motion";
import { HelpTip } from "@/shared/ui/help-tip";
import type { LMActivity, OptimizationStatusResponse, PairResult } from "@/shared/types/api";
import { type PipelineStage } from "../constants";
import { detectPairStage, detectStage } from "../lib/detect-stage";
import { formatDuration, formatImprovement, formatPercent } from "@/shared/lib";
import { tip } from "@/shared/lib/tooltips";
import { TERMS } from "@/shared/lib/terms";
import type { ScorePoint } from "../lib/extract-scores";
import { InfoCard } from "./ui-primitives";
import { PipelineStages, computeStageTimestamps } from "./PipelineStages";
import { TrajectoryPanel } from "@/features/trajectory";
import { formatMsg, msg } from "@/shared/lib/messages";

const ScoreChart = dynamic(() => import("@/shared/ui/score-chart").then((m) => m.ScoreChart), {
  ssr: false,
  loading: () => <div className="h-full" />,
});

// Lazy-loaded so the recharts/d3 vendor chunk stays out of the first-load JS on
// /optimizations/[id] (the default eager tab). Both only render for grid runs.
const GridLiveChart = dynamic(() => import("./GridLiveChart").then((m) => m.GridLiveChart), {
  ssr: false,
  loading: () => <div className="h-[300px]" />,
});

const GridOverview = dynamic(() => import("./GridOverview").then((m) => m.GridOverview), {
  ssr: false,
  loading: () => <div className="h-[300px]" />,
});

/**
 * One live-telemetry metric: a muted icon + label over the value, sized to fill
 * an equal share of the panel width. The icon inherits the label's muted color
 * so it reads as quiet wayfinding, not decoration.
 */
function LiveStat({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="flex items-center gap-1.5 text-[#A89680]">
        {icon}
        <span className="truncate text-[0.625rem] font-semibold uppercase tracking-[0.08em]">
          {label}
        </span>
      </span>
      <span className="truncate text-sm font-semibold tabular-nums text-[#1C1612]">{value}</span>
    </div>
  );
}

function OverviewTabImpl({
  job,
  isActive,
  scorePoints,
  activePairIndex,
  activePair,
  onStageClick,
  onPairSelect,
  onPairDeleted,
  trajectoryPreviewLayout,
}: {
  job: OptimizationStatusResponse;
  isActive: boolean;
  scorePoints: ScorePoint[];
  activePairIndex: number | null;
  activePair?: PairResult | null;
  onStageClick: (stage: PipelineStage) => void;
  onPairSelect: (pairIndex: number) => void;
  onPairDeleted?: (pairIndex: number) => void;
  trajectoryPreviewLayout?: { width: number; height: number };
}) {
  const metrics = job.latest_metrics ?? {};
  const isPairContext = activePair != null;
  // Render the single-run overview blocks for both a standalone run AND a
  // grid-search pair view — the goal is "exactly identical components", so a
  // pair is just a run scoped by pair_index plus aggregation around it.
  const renderRunBlocks = job.optimization_type === "run" || isPairContext;
  const renderGridAgg = job.optimization_type === "grid_search" && !isPairContext;

  const pairIndex = isPairContext ? activePair.pair_index : undefined;
  const currentStage = isPairContext
    ? detectPairStage(job, activePair.pair_index)
    : job.status === "success"
      ? "done"
      : detectStage(job);
  const stageTs = computeStageTimestamps(
    job.progress_events ?? [],
    job.started_at,
    job.completed_at,
    pairIndex,
  );
  const stagesActive = isPairContext
    ? isActive && currentStage !== "done" && !activePair.error
    : isActive;
  const stagesFailed = isPairContext
    ? !!activePair.error || (!isActive && currentStage !== "done")
    : job.status === "failed" || job.status === "cancelled";

  // Score values — pair view picks up the pair's own metrics with event
  // fallback (pair_index-filtered events arrive before grid_result is
  // finalized); standalone run uses job.result with global event fallback.
  const baselineFromEvents = isPairContext
    ? (job.progress_events?.find(
        (e) =>
          e.event === "baseline_evaluated" && e.metrics?.pair_index === activePair.pair_index,
      )?.metrics?.baseline_test_metric as number | undefined)
    : (job.progress_events?.find((e) => e.event === "baseline_evaluated")?.metrics
        ?.baseline_test_metric as number | undefined);
  const optimizedFromEvents = isPairContext
    ? (job.progress_events?.find(
        (e) =>
          e.event === "optimized_evaluated" && e.metrics?.pair_index === activePair.pair_index,
      )?.metrics?.optimized_test_metric as number | undefined)
    : (job.progress_events?.find((e) => e.event === "optimized_evaluated")?.metrics
        ?.optimized_test_metric as number | undefined);
  const runResult = isPairContext ? activePair : job.result;
  const baseline = runResult?.baseline_test_metric ?? baselineFromEvents;
  const optimized = runResult?.optimized_test_metric ?? optimizedFromEvents;
  const improvement =
    runResult?.metric_improvement ??
    (baseline != null && optimized != null ? optimized - baseline : undefined);
  const scoresReady =
    runResult != null && baseline != null && optimized != null && !activePair?.error;
  const lmActivity: LMActivity | null = (runResult?.lm_activity as LMActivity | undefined) ?? null;

  // The score cards stream the real evaluated metrics as they land — the
  // baseline from baseline_evaluated, the optimized score from
  // optimized_evaluated — and show "—" until each metric is genuinely
  // evaluated. Never a stale or interpolated value, so a stalled or
  // unfinished run reads honestly. Improvement only resolves once both the
  // baseline and the optimized score exist, keeping the three cards coherent.
  const displayImprovement =
    baseline != null && optimized != null ? (improvement ?? optimized - baseline) : undefined;
  const showScoreCards = scoresReady || (stagesActive && baseline != null);

  // Status text reflects what the user is looking at — for a pair, that is
  // the pair's own state (running/done/failed), not the parent grid.
  const viewStatus: "running" | "success" | "failed" | "cancelled" | "other" = (() => {
    if (isPairContext) {
      if (activePair.error) return "failed";
      if (currentStage === "done") return "success";
      if (job.status === "cancelled") return "cancelled";
      if (stagesActive) return "running";
      return "other";
    }
    if (isActive) return "running";
    if (job.status === "cancelled") return "cancelled";
    if (job.status === "failed") return "failed";
    return "other";
  })();

  return (
    <>
      {renderRunBlocks && (
        <FadeIn>
          <p className="text-sm text-muted-foreground">
            {viewStatus === "running"
              ? formatMsg("auto.features.optimizations.components.overviewtab.template.1", {
                  p1: TERMS.optimization,
                })
              : viewStatus === "cancelled"
                ? formatMsg("auto.features.optimizations.components.overviewtab.template.2", {
                    p1: TERMS.optimization,
                  })
                : viewStatus === "failed"
                  ? formatMsg("auto.features.optimizations.components.overviewtab.template.3", {
                      p1: TERMS.optimization,
                    })
                  : formatMsg("auto.features.optimizations.components.overviewtab.template.4", {
                      p1: TERMS.optimization,
                    })}
          </p>
        </FadeIn>
      )}

      {renderRunBlocks &&
        stagesActive &&
        (() => {
          const tqdmPercent = metrics.tqdm_percent as number | undefined;
          if (tqdmPercent == null) return null;
          const tqdmN = metrics.tqdm_n as number | undefined;
          const tqdmTotal = metrics.tqdm_total as number | undefined;
          const tqdmElapsed = metrics.tqdm_elapsed as number | undefined;
          const tqdmRemaining = metrics.tqdm_remaining as number | undefined;
          const tqdmRate = metrics.tqdm_rate as number | undefined;
          const stats: Array<{ key: string; icon: ReactNode; label: string; value: string }> = [];
          if (tqdmElapsed != null)
            stats.push({
              key: "elapsed",
              icon: <Timer className="size-3.5 shrink-0" />,
              label: msg("auto.features.optimizations.components.overviewtab.literal.2"),
              value: formatDuration(tqdmElapsed),
            });
          if (tqdmRemaining != null)
            stats.push({
              key: "remaining",
              icon: <Hourglass className="size-3.5 shrink-0" />,
              label: msg("auto.features.optimizations.components.overviewtab.literal.3"),
              value: formatDuration(Number(tqdmRemaining)),
            });
          if (tqdmRate != null)
            stats.push({
              key: "rate",
              icon: <Gauge className="size-3.5 shrink-0" />,
              label: msg("auto.features.optimizations.components.overviewtab.literal.4"),
              value: formatMsg("auto.features.optimizations.components.overviewtab.template.5", {
                p1: tqdmRate.toFixed(2),
              }),
            });

          return (
            <FadeIn>
              <div className="rounded-xl border border-[#E3DCD0] bg-[#FBF9F4] px-4 py-3.5">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="flex items-center gap-2 text-sm font-semibold text-[#1C1612]">
                    <span
                      className="size-1.5 shrink-0 rounded-full bg-[var(--warning)] motion-safe:animate-pulse"
                      aria-hidden="true"
                    />
                    {msg("optimization.progress.gepa")}
                  </span>
                  <span dir="ltr" className="flex items-baseline gap-1.5 font-mono tabular-nums">
                    <span className="text-sm font-semibold text-[#1C1612]">
                      {tqdmPercent.toFixed(0)}%
                    </span>
                    <span className="text-[0.6875rem] text-[#A89680]">
                      {tqdmN ?? 0}/{tqdmTotal ?? "?"}
                    </span>
                  </span>
                </div>
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[#E3DCD0]/70">
                  <div
                    className="h-full rounded-full bg-primary transition-[width] duration-500 ease-out"
                    style={{ width: `${tqdmPercent}%` }}
                  />
                </div>
                {stats.length > 0 && (
                  <div className="mt-3.5 grid grid-cols-2 gap-x-4 gap-y-4 border-t border-[#E3DCD0]/70 pt-3.5 sm:grid-cols-3">
                    {stats.map((s) => (
                      <LiveStat key={s.key} icon={s.icon} label={s.label} value={s.value} />
                    ))}
                  </div>
                )}
              </div>
            </FadeIn>
          );
        })()}

      {renderRunBlocks && (
        <FadeIn delay={0.05}>
          <PipelineStages
            currentStage={currentStage}
            stageTs={stageTs}
            isActive={stagesActive}
            isFailed={stagesFailed}
            onStageClick={onStageClick}
            dataTutorial={isPairContext ? undefined : "pipeline-stages"}
          />
        </FadeIn>
      )}

      {renderRunBlocks &&
        runResult &&
        !lmActivity &&
        (runResult.num_lm_calls != null || runResult.avg_response_time_ms != null) && (
          <FadeIn delay={0.1}>
            <div className="grid grid-cols-2 gap-2.5">
              {runResult.num_lm_calls != null && (
                <InfoCard
                  label={
                    <HelpTip text={tip("lm.calls_count")}>
                      {msg("auto.features.optimizations.components.overviewtab.1")}
                    </HelpTip>
                  }
                  value={formatMsg(
                    "auto.features.optimizations.components.overviewtab.template.6",
                    { p1: runResult.num_lm_calls },
                  )}
                  icon={<MessageSquare className="size-3.5" />}
                />
              )}
              {runResult.avg_response_time_ms != null && (
                <InfoCard
                  label={
                    <HelpTip text={tip("lm.avg_response_time")}>
                      {msg("auto.features.optimizations.components.overviewtab.2")}
                    </HelpTip>
                  }
                  value={formatMsg(
                    "auto.features.optimizations.components.overviewtab.template.7",
                    { p1: (runResult.avg_response_time_ms / 1000).toFixed(1) },
                  )}
                  icon={<Timer className="size-3.5" />}
                />
              )}
            </div>
          </FadeIn>
        )}

      {renderRunBlocks && showScoreCards && (
        <div data-tutorial="score-cards">
          <StaggerContainer className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <StaggerItem>
              <TiltCard className=" rounded-xl border border-border/50 bg-card p-6 text-center">
                <p className="text-[0.6875rem] text-muted-foreground mb-2 font-medium tracking-wide">
                  <HelpTip text={tip("score.baseline")}>{TERMS.baselineScore}</HelpTip>
                </p>
                <p className="text-3xl font-mono font-bold tabular-nums">
                  {formatPercent(baseline)}
                </p>
              </TiltCard>
            </StaggerItem>
            <StaggerItem>
              <TiltCard className="rounded-xl border border-primary/30 bg-gradient-to-br from-primary/5 to-primary/10 p-6 text-center shadow-[0_0_20px_rgba(var(--primary),0.08)]">
                <p className="text-[0.6875rem] text-muted-foreground mb-2 font-medium tracking-wide">
                  <HelpTip text={tip("score.optimized")}>{TERMS.optimizedScore}</HelpTip>
                </p>
                <p className="text-3xl font-mono font-bold text-primary tabular-nums">
                  {formatPercent(optimized)}
                </p>
              </TiltCard>
            </StaggerItem>
            <StaggerItem>
              <TiltCard
                className={`rounded-xl border p-6 text-center ${(displayImprovement ?? 0) >= 0 ? "border-stone-400/50 bg-gradient-to-br from-stone-100/50 to-stone-200/30" : "border-red-300/50 bg-gradient-to-br from-red-50/50 to-red-100/30"}`}
              >
                <p className="text-[0.6875rem] text-muted-foreground mb-2 font-medium tracking-wide">
                  <HelpTip text={tip("score.improvement")}>
                    {msg("auto.features.optimizations.components.overviewtab.3")}
                  </HelpTip>
                </p>
                <p
                  className={`text-3xl font-mono font-bold tabular-nums ${(displayImprovement ?? 0) >= 0 ? "text-stone-600" : "text-red-600"}`}
                >
                  {formatImprovement(displayImprovement)}
                </p>
              </TiltCard>
            </StaggerItem>
          </StaggerContainer>
        </div>
      )}

      {renderRunBlocks && (
        <TrajectoryPanel
          job={job}
          pairIndex={pairIndex}
          previewLayout={trajectoryPreviewLayout}
          toolSeverities={runResult?.program_artifact?.react_overlay?.tool_severities}
        />
      )}

      {renderRunBlocks && scorePoints.length > 1 && (
        <FadeIn delay={0.1}>
          <Card
            className="relative overflow-hidden shadow-[0_1px_3px_rgba(28,22,18,0.04),inset_0_1px_0_rgba(255,255,255,0.5)]"
            data-tutorial="score-chart"
          >
            <div
              className="absolute inset-x-0 top-0 h-px bg-gradient-to-l from-transparent via-[#C8A882]/40 to-transparent"
              aria-hidden="true"
            />
            <CardHeader>
              <CardTitle className="text-base flex items-center gap-2">
                <TrendingUp className="size-4 text-[#7C6350]" aria-hidden="true" />
                <HelpTip text={tip("score.progression")}>
                  <span className="font-bold tracking-tight">
                    {msg("auto.features.optimizations.components.overviewtab.4")}
                  </span>
                </HelpTip>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[220px] min-w-0">
                <ScoreChart data={scorePoints} />
              </div>
            </CardContent>
          </Card>
        </FadeIn>
      )}

      {renderGridAgg && !job.grid_result && activePairIndex === null && (
        <FadeIn delay={0.1}>
          <GridLiveChart job={job} />
        </FadeIn>
      )}

      {renderGridAgg && job.grid_result && activePairIndex === null && (
        <GridOverview job={job} onPairSelect={onPairSelect} onPairDeleted={onPairDeleted} />
      )}
    </>
  );
}

// Memoized so unrelated parent state ticks (live elapsed badge, SSE in-place
// patches) don't re-render the whole overview — props are now stable identities
// (memoized job/scorePoints, useCallback'd handlers).
export const OverviewTab = memo(OverviewTabImpl);
