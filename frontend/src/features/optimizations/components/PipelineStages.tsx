"use client";

/**
 * Visual pipeline timeline — renders the 5 DSPy stages (validating →
 * splitting → baseline → optimizing → evaluating) as a connected row of
 * nodes with per-stage timestamps.
 *
 * Used by OverviewTab for both the job-wide pipeline and (when scoped by
 * pair_index) the per-pair pipeline inside a grid_search. Stage detection
 * + timestamp derivation live in the caller; this component is a pure
 * renderer.
 */

import { useEffect, useRef, useState } from "react";
import { CheckCircle2, Circle, Loader2, XCircle } from "lucide-react";
import { PIPELINE_STAGES, type PipelineStage } from "../constants";
import type { ProgressEvent } from "@/shared/types/api";

const VERTICAL_BREAKPOINT_PX = 600;

interface StageTs {
  date: string;
  time: string;
}

function fmtTs(iso: string): StageTs {
  const d = new Date(iso);
  return {
    date: d.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
    time: d.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }),
  };
}

/**
 * Build `{stage → timestamp}` from a progress-event stream. When
 * `pairIndex` is provided, only events tagged with that pair are used
 * for pair-scoped stages (baseline/optimizing/evaluating); pre-pair
 * stages (validating/splitting) still fall back to the global events.
 */
export function computeStageTimestamps(
  events: ProgressEvent[],
  startedAt: string | null | undefined,
  completedAt: string | null | undefined,
  pairIndex?: number,
): Partial<Record<PipelineStage, StageTs>> {
  const globalMap: Record<string, PipelineStage> = {
    validation_passed: "validating",
    dataset_splits_ready: "splitting",
  };
  const pairMap: Record<string, PipelineStage> = {
    grid_pair_started: "baseline",
    baseline_evaluated: "baseline",
    optimizer_progress: "optimizing",
    optimized_evaluated: "evaluating",
    grid_pair_completed: "evaluating",
  };

  const stageTs: Partial<Record<PipelineStage, StageTs>> = {};
  for (const ev of events) {
    if (!ev.event || !ev.timestamp) continue;
    const evPairIndex = ev.metrics?.pair_index;
    const isPairScoped = ev.event in pairMap;

    if (pairIndex != null && isPairScoped) {
      if (typeof evPairIndex !== "number" || evPairIndex !== pairIndex) continue;
    }

    const sk = globalMap[ev.event] ?? pairMap[ev.event];
    if (!sk) continue;
    stageTs[sk] = fmtTs(ev.timestamp);
  }
  if (startedAt && !stageTs.validating) {
    stageTs.validating = fmtTs(startedAt);
  }
  if (completedAt && !stageTs.evaluating && pairIndex == null) {
    stageTs.evaluating = fmtTs(completedAt);
  }
  return stageTs;
}

export function PipelineStages({
  currentStage,
  stageTs,
  isActive,
  isFailed,
  onStageClick,
  dataTutorial,
}: {
  currentStage: PipelineStage | "done";
  stageTs: Partial<Record<PipelineStage, StageTs>>;
  isActive: boolean;
  isFailed: boolean;
  onStageClick: (stage: PipelineStage) => void;
  dataTutorial?: string;
}) {
  const completedStageIdx =
    currentStage === "done"
      ? PIPELINE_STAGES.length
      : PIPELINE_STAGES.findIndex((s) => s.key === currentStage);

  const containerRef = useRef<HTMLDivElement>(null);
  const [isVertical, setIsVertical] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        setIsVertical(e.contentRect.width < VERTICAL_BREAKPOINT_PX);
      }
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  return (
    <div
      ref={containerRef}
      className={
        isVertical ? "relative flex flex-col gap-3" : "relative flex items-start justify-between"
      }
      data-tutorial={dataTutorial}
    >
      {!isVertical && (
        <>
          <div className="absolute top-[14px] start-[14px] end-[14px] h-[2px] bg-border/50 rounded-full" />
          <div
            className={`absolute top-[14px] start-[14px] h-[2px] rounded-full transition-all duration-700 ease-out ${isFailed ? "bg-destructive/60" : "bg-[#3D2E22]"}`}
            style={{
              width: `calc(${(Math.min(completedStageIdx, PIPELINE_STAGES.length - 1) / (PIPELINE_STAGES.length - 1)) * 100}% - 28px)`,
            }}
          />
        </>
      )}
      {PIPELINE_STAGES.map((s, i) => {
        const isDone = i < completedStageIdx;
        const isCurrent = isActive && i === completedStageIdx;
        const isStopped = isFailed && i === completedStageIdx;
        const ts = stageTs[s.key];
        return (
          <div
            key={s.key}
            className={
              isVertical
                ? "relative z-10 flex flex-row items-center gap-3 min-w-0 group/node cursor-pointer"
                : "relative z-10 flex flex-col items-center gap-2 min-w-0 group/node cursor-pointer"
            }
            onClick={() => onStageClick(s.key)}
          >
            <div
              className={`size-7 rounded-full flex items-center justify-center transition-all duration-300 group-hover/node:scale-125 group-hover/node:shadow-[0_0_0_6px_rgba(61,46,34,0.1)] ${
                isStopped
                  ? "bg-destructive text-white shadow-[0_0_0_4px_rgba(176,64,48,0.15)]"
                  : isCurrent
                    ? "bg-[#3D2E22] text-white shadow-[0_0_0_4px_rgba(61,46,34,0.15)]"
                    : isDone
                      ? "bg-[#3D2E22] text-white"
                      : "bg-[#E5DDD4] text-[#8C7A6B]"
              }`}
            >
              {isDone ? (
                <CheckCircle2 className="size-3.5" />
              ) : isCurrent ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : isStopped ? (
                <XCircle className="size-3.5" />
              ) : (
                <Circle className="size-3" />
              )}
            </div>
            <span
              className={`text-[0.6875rem] whitespace-nowrap transition-colors duration-200 group-hover/node:text-[#3D2E22] ${
                isCurrent
                  ? "text-[#3D2E22] font-semibold"
                  : isStopped
                    ? "text-destructive font-semibold"
                    : isDone
                      ? "text-[#3D2E22]/80"
                      : "text-muted-foreground/40"
              }`}
            >
              {s.label}
            </span>
            {ts && isDone && (
              <div
                className={
                  isVertical
                    ? "flex flex-row items-baseline gap-1.5 ms-auto"
                    : "flex flex-col items-center -mt-0.5"
                }
                dir="ltr"
              >
                <span className="text-[0.625rem] text-muted-foreground/50 tracking-wide uppercase">
                  {ts.date}
                </span>
                <span className="text-[0.6875rem] text-muted-foreground/70 font-mono tabular-nums">
                  {ts.time}
                </span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
