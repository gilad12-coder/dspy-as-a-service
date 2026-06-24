"use client";

import type { ReactNode } from "react";
import { Activity, MessageSquare, Timer } from "lucide-react";
import { FadeIn } from "@/shared/ui/motion";
import { HelpTip } from "@/shared/ui/help-tip";
import { msg } from "@/shared/lib/messages";
import { tip } from "@/shared/lib/tooltips";
import type { LMActivity, LMStageStats } from "@/shared/types/api";

const STAGE_KEYS = ["baseline", "training", "evaluation"] as const;
type StageKey = (typeof STAGE_KEYS)[number];

const STAGE_MESSAGE_KEYS: Record<
  StageKey,
  | "auto.features.optimizations.components.lmactivitytab.stage_baseline"
  | "auto.features.optimizations.components.lmactivitytab.stage_training"
  | "auto.features.optimizations.components.lmactivitytab.stage_evaluation"
> = {
  baseline: "auto.features.optimizations.components.lmactivitytab.stage_baseline",
  training: "auto.features.optimizations.components.lmactivitytab.stage_training",
  evaluation: "auto.features.optimizations.components.lmactivitytab.stage_evaluation",
};

const STAGE_TIP_KEYS: Record<
  StageKey,
  "lm_activity.stage.baseline" | "lm_activity.stage.training" | "lm_activity.stage.evaluation"
> = {
  baseline: "lm_activity.stage.baseline",
  training: "lm_activity.stage.training",
  evaluation: "lm_activity.stage.evaluation",
};

function aggregateColumn(
  perStage: Record<string, LMStageStats>,
): { calls: number; avg_response_time_ms: number | null } {
  // Weighted mean across stages: ``avg_ms`` per stage is itself a mean, so
  // multiply by per-stage call count before reducing.
  let totalCalls = 0;
  let weighted = 0;
  let weighted_n = 0;
  for (const cell of Object.values(perStage)) {
    const calls = cell?.calls ?? 0;
    totalCalls += calls;
    if (calls > 0 && typeof cell?.avg_response_time_ms === "number") {
      weighted += cell.avg_response_time_ms * calls;
      weighted_n += calls;
    }
  }
  return {
    calls: totalCalls,
    avg_response_time_ms: weighted_n > 0 ? weighted / weighted_n : null,
  };
}

function formatCalls(n: number): string {
  return n.toLocaleString("he-IL");
}

// Unified to seconds so the eye doesn't re-parse units row to row. Sub-100ms
// shows ``<0.1s`` rather than ``42ms`` to keep the column dimension uniform —
// the diagnostic question is "where did time go", so consistent scale wins
// over per-cell precision at the millisecond floor.
function formatSeconds(ms: number | null | undefined): string {
  if (ms == null) return "—";
  const seconds = ms / 1000;
  if (seconds < 0.1) return "<0.1s";
  return `${seconds.toFixed(1)}s`;
}

function MetricCells({
  calls,
  ms,
  emphasized = false,
  groupStart = false,
}: {
  calls: number;
  ms: number | null;
  emphasized?: boolean;
  groupStart?: boolean;
}) {
  const empty = calls === 0;
  const numeric = emphasized ? "font-semibold text-foreground" : "font-normal text-foreground";
  const numericMuted = emphasized ? "text-foreground" : "text-muted-foreground";
  return (
    <>
      <td
        className={`px-3 py-2.5 text-end align-middle ${
          groupStart ? "border-s border-border/50" : ""
        }`}
      >
        {empty ? (
          <span className="text-[var(--text-3)]">—</span>
        ) : (
          <span className={`font-mono tabular-nums text-sm ${numeric}`} dir="ltr">
            {formatCalls(calls)}
          </span>
        )}
      </td>
      <td className="px-3 py-2.5 text-end align-middle">
        {empty ? (
          <span className="text-[var(--text-3)]">—</span>
        ) : (
          <span className={`font-mono tabular-nums text-sm ${numericMuted}`} dir="ltr">
            {formatSeconds(ms)}
          </span>
        )}
      </td>
    </>
  );
}

function StageRow({
  stage,
  generation,
  reflection,
  hasReflection,
}: {
  stage: StageKey;
  generation: LMStageStats | undefined;
  reflection: LMStageStats | undefined;
  hasReflection: boolean;
}) {
  return (
    <tr className="border-t border-border/40 transition-colors hover:bg-muted/30">
      <th
        scope="row"
        className="px-3 py-2.5 text-start text-sm font-medium text-foreground whitespace-nowrap"
      >
        <HelpTip text={tip(STAGE_TIP_KEYS[stage])}>{msg(STAGE_MESSAGE_KEYS[stage])}</HelpTip>
      </th>
      <MetricCells
        calls={generation?.calls ?? 0}
        ms={generation?.avg_response_time_ms ?? null}
        groupStart
      />
      {hasReflection && (
        <MetricCells
          calls={reflection?.calls ?? 0}
          ms={reflection?.avg_response_time_ms ?? null}
          groupStart
        />
      )}
    </tr>
  );
}

function TotalRow({
  generation,
  reflection,
  hasReflection,
}: {
  generation: { calls: number; avg_response_time_ms: number | null };
  reflection: { calls: number; avg_response_time_ms: number | null };
  hasReflection: boolean;
}) {
  return (
    <tr className="border-t border-border bg-muted/20">
      <th
        scope="row"
        className="px-3 py-3 text-start text-sm font-bold text-foreground whitespace-nowrap"
      >
        <HelpTip text={tip("lm_activity.total_row")}>
          {msg("auto.features.optimizations.components.lmactivitytab.row_total")}
        </HelpTip>
      </th>
      <MetricCells
        calls={generation.calls}
        ms={generation.avg_response_time_ms}
        emphasized
        groupStart
      />
      {hasReflection && (
        <MetricCells
          calls={reflection.calls}
          ms={reflection.avg_response_time_ms}
          emphasized
          groupStart
        />
      )}
    </tr>
  );
}

function SubHeader({
  tipKey,
  icon,
  label,
  groupStart = false,
}: {
  tipKey: "lm_activity.cell.calls" | "lm_activity.cell.avg_ms";
  icon: ReactNode;
  label: string;
  groupStart?: boolean;
}) {
  return (
    <th
      scope="col"
      className={`px-3 pb-2 pt-1 text-end text-[11px] font-medium uppercase tracking-wide text-muted-foreground whitespace-nowrap ${
        groupStart ? "border-s border-border/50" : ""
      }`}
    >
      <HelpTip text={tip(tipKey)}>
        <span className="inline-flex items-center gap-1.5">
          <span aria-hidden="true" className="text-muted-foreground/70">
            {icon}
          </span>
          {label}
        </span>
      </HelpTip>
    </th>
  );
}

export function LMActivityTab({ lmActivity }: { lmActivity: LMActivity | null | undefined }) {
  const generation = lmActivity?.generation ?? {};
  const reflection = lmActivity?.reflection ?? {};
  // Hide the reflection column entirely when no reflection traffic was recorded —
  // unoptimized runs and grid pairs that didn't reach reflection should not show
  // a column full of em-dashes.
  const hasReflection = Object.values(reflection).some((c) => (c?.calls ?? 0) > 0);
  const hasAnyCalls =
    hasReflection || Object.values(generation).some((c) => (c?.calls ?? 0) > 0);

  const genTotal = aggregateColumn(generation);
  const reflTotal = aggregateColumn(reflection);

  const callsLabel = msg("auto.features.optimizations.components.lmactivitytab.cell_calls");
  const avgMsLabel = msg("auto.features.optimizations.components.lmactivitytab.cell_avg_ms");
  const stageLabel = msg("auto.features.optimizations.components.lmactivitytab.col_stage");
  const genLabel = msg("auto.features.optimizations.components.lmactivitytab.col_generation");
  const reflLabel = msg("auto.features.optimizations.components.lmactivitytab.col_reflection");

  const callsIcon = <MessageSquare className="size-3" strokeWidth={1.75} />;
  const avgIcon = <Timer className="size-3" strokeWidth={1.75} />;

  // Plain ``<section>`` instead of ``<Card>`` to opt out of the global
  // ``[data-slot="card"]`` chrome (gradient bg, backdrop-blur, mouse-spotlight,
  // hover-lift, top hairline). This card is a quiet diagnostic surface; the
  // ornamental defaults fight that intent.
  return (
    <FadeIn>
      <p className="text-sm text-muted-foreground mb-4">
        {msg("optimizations.lmactivity.description")}
      </p>
      <section
        aria-labelledby="lm-activity-title"
        data-tutorial="lm-activity"
        className="rounded-2xl border border-border bg-card text-card-foreground shadow-[var(--shadow-sm)]"
      >
        <header className="flex items-center gap-2 px-6 pt-5 pb-3">
          <Activity
            className="size-4 text-muted-foreground"
            strokeWidth={1.75}
            aria-hidden="true"
          />
          <HelpTip text={tip("lm_activity.section")}>
            <h3
              id="lm-activity-title"
              className="m-0 text-lg font-bold text-foreground"
            >
              {msg("auto.features.optimizations.components.lmactivitytab.title")}
            </h3>
          </HelpTip>
        </header>
        <div className="px-6 pb-5">
          {!hasAnyCalls ? (
            <p className="text-sm text-muted-foreground">
              {msg("auto.features.optimizations.components.lmactivitytab.no_data")}
            </p>
          ) : (
            <div className="overflow-x-auto -mx-2 px-2">
              <table className="guide-table w-full text-sm">
                <thead>
                  <tr className="bg-muted/20">
                    <th
                      scope="col"
                      rowSpan={2}
                      className="px-3 py-2 text-start text-[11px] font-semibold uppercase tracking-wide text-muted-foreground align-bottom whitespace-nowrap"
                    >
                      <span className="sr-only">{stageLabel}</span>
                    </th>
                    <th
                      scope="colgroup"
                      colSpan={2}
                      className="px-3 pt-2 pb-1 text-center text-xs font-semibold text-foreground border-s border-border/50 whitespace-nowrap"
                    >
                      <HelpTip text={tip("lm_activity.column.generation")}>{genLabel}</HelpTip>
                    </th>
                    {hasReflection && (
                      <th
                        scope="colgroup"
                        colSpan={2}
                        className="px-3 pt-2 pb-1 text-center text-xs font-semibold text-foreground border-s border-border/50 whitespace-nowrap"
                      >
                        <HelpTip text={tip("lm_activity.column.reflection")}>{reflLabel}</HelpTip>
                      </th>
                    )}
                  </tr>
                  <tr className="bg-muted/20">
                    <SubHeader
                      tipKey="lm_activity.cell.calls"
                      icon={callsIcon}
                      label={callsLabel}
                      groupStart
                    />
                    <SubHeader
                      tipKey="lm_activity.cell.avg_ms"
                      icon={avgIcon}
                      label={avgMsLabel}
                    />
                    {hasReflection && (
                      <>
                        <SubHeader
                          tipKey="lm_activity.cell.calls"
                          icon={callsIcon}
                          label={callsLabel}
                          groupStart
                        />
                        <SubHeader
                          tipKey="lm_activity.cell.avg_ms"
                          icon={avgIcon}
                          label={avgMsLabel}
                        />
                      </>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {STAGE_KEYS.map((stage) => (
                    <StageRow
                      key={stage}
                      stage={stage}
                      generation={generation[stage]}
                      reflection={reflection[stage]}
                      hasReflection={hasReflection}
                    />
                  ))}
                  <TotalRow
                    generation={genTotal}
                    reflection={reflTotal}
                    hasReflection={hasReflection}
                  />
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>
    </FadeIn>
  );
}
