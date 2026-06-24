"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useMemo, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { HelpTip } from "@/shared/ui/help-tip";
import { tip } from "@/shared/lib/tooltips";
import { TERMS } from "@/shared/lib/terms";
import { ChartTable } from "@/shared/charts/chart-table";
import { useLiteMode } from "@/features/settings";
import type { OptimizationStatusResponse } from "@/shared/types/api";
import { formatMsg, msg } from "@/shared/lib/messages";

type PairSnapshot = {
  pair_index: number;
  name: string;
  baselineScore: number | null;
  optimizedScore: number | null;
};

function shortName(model: unknown): string {
  if (typeof model !== "string") return "?";
  return model.split("/").pop() ?? model;
}

function toPct(v: unknown): number | null {
  if (typeof v !== "number" || !Number.isFinite(v)) return null;
  return Math.round(v);
}

function buildPairs(job: OptimizationStatusResponse): PairSnapshot[] {
  const events = job.progress_events ?? [];
  const byIndex = new Map<number, PairSnapshot>();
  const ensure = (pair_index: number, gen?: unknown, ref?: unknown) => {
    const existing = byIndex.get(pair_index);
    if (existing) return existing;
    const entry: PairSnapshot = {
      pair_index,
      name: `${shortName(gen)} × ${shortName(ref)}`,
      baselineScore: null,
      optimizedScore: null,
    };
    byIndex.set(pair_index, entry);
    return entry;
  };
  for (const e of events) {
    const m = e.metrics ?? {};
    const pi = m.pair_index;
    if (typeof pi !== "number") continue;
    if (e.event === "baseline_evaluated") {
      const entry = ensure(pi);
      entry.baselineScore = toPct(m.baseline_test_metric);
    } else if (e.event === "optimized_evaluated") {
      const entry = ensure(pi);
      entry.optimizedScore = toPct(m.optimized_test_metric);
    } else if (e.event === "grid_pair_completed") {
      const entry = ensure(pi, m.generation_model, m.reflection_model);
      if (entry.baselineScore == null) entry.baselineScore = toPct(m.baseline_test_metric);
      if (entry.optimizedScore == null) entry.optimizedScore = toPct(m.optimized_test_metric);
      if (entry.name.startsWith("?")) {
        entry.name = `${shortName(m.generation_model)} × ${shortName(m.reflection_model)}`;
      }
    }
  }
  return Array.from(byIndex.values())
    .filter((p) => p.baselineScore != null || p.optimizedScore != null)
    .sort((a, b) => a.pair_index - b.pair_index);
}

type TipPayload = Array<{ value: number; dataKey?: string; color?: string }>;

function LiveTip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TipPayload;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const nameMap: Record<string, string> = {
    baselineScore: formatMsg("auto.features.optimizations.components.gridlivechart.template.1", {
      p1: TERMS.score,
      p2: TERMS.optimization,
    }),
    optimizedScore: formatMsg("auto.features.optimizations.components.gridlivechart.template.2", {
      p1: TERMS.score,
      p2: TERMS.optimization,
    }),
  };
  return (
    <div className="rounded-xl border border-border/60 bg-background/95 backdrop-blur-sm p-3 shadow-lg">
      {label && <p className="font-semibold mb-1.5 text-foreground text-xs">{label}</p>}
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 text-xs text-muted-foreground">
          {p.color && (
            <span className="size-2 rounded-full shrink-0" style={{ backgroundColor: p.color }} />
          )}
          <span>{nameMap[String(p.dataKey)] ?? String(p.dataKey)}</span>
          <span className="font-mono font-semibold text-foreground ms-auto">{p.value}%</span>
        </div>
      ))}
    </div>
  );
}

function formatLivePct(value: unknown): string {
  return typeof value === "number" ? `${value}%` : "—";
}

export function GridLiveChart({ job }: { job: OptimizationStatusResponse }) {
  const lite = useLiteMode();
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  // Recompute only when new progress events land, not on every parent tick —
  // this chart lives under the live-updating OverviewTab.
  const pairs = useMemo(() => buildPairs(job), [job.progress_events]);
  const totalPairs = useMemo(() => {
    const events = job.progress_events ?? [];
    for (let i = events.length - 1; i >= 0; i--) {
      const t = events[i]?.metrics?.total_pairs;
      if (typeof t === "number") return t;
    }
    return null;
  }, [job.progress_events]);
  if (pairs.length === 0) return null;
  const toggle = (key: string) => {
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold flex items-center justify-between gap-2">
          <HelpTip text={tip("grid.score_comparison")}>
            {msg("auto.features.optimizations.components.gridlivechart.1")}
          </HelpTip>
          {totalPairs != null && (
            <span className="text-[0.6875rem] font-normal text-muted-foreground tabular-nums">
              {pairs.length}/{totalPairs}
              {msg("auto.features.optimizations.components.gridlivechart.2")}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-0">
        {lite ? (
          <div className="h-[220px] min-w-0">
            <ChartTable
              rows={pairs}
              columns={[
                {
                  key: "name",
                  label: msg("auto.features.optimizations.components.gridlivechart.literal.1"),
                },
                {
                  key: "baselineScore",
                  label: msg("auto.features.optimizations.components.gridlivechart.literal.3"),
                  align: "end",
                  format: formatLivePct,
                },
                {
                  key: "optimizedScore",
                  label: msg("auto.features.optimizations.components.gridlivechart.literal.5"),
                  align: "end",
                  format: formatLivePct,
                },
              ]}
            />
          </div>
        ) : (
        <div className="h-[220px] min-w-0" dir="ltr">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={pairs}
              layout="vertical"
              margin={{ left: 10, right: 20, top: 5, bottom: 5 }}
            >
              <CartesianGrid horizontal={false} strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                type="number"
                domain={[0, 105]}
                tickLine={false}
                axisLine={false}
                tick={{ fontSize: 10 }}
                className="fill-muted-foreground"
                label={{
                  value: msg("auto.features.optimizations.components.gridlivechart.literal.1"),
                  position: "insideBottom",
                  offset: -2,
                  fontSize: 10,
                }}
              />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 10 }}
                width={100}
                className="fill-muted-foreground"
                tickLine={false}
                axisLine={false}
              />
              <Tooltip content={<LiveTip />} />
              {!hidden.has(
                msg("auto.features.optimizations.components.gridlivechart.literal.2"),
              ) && (
                <Bar
                  dataKey="baselineScore"
                  name={msg("auto.features.optimizations.components.gridlivechart.literal.3")}
                  fill="var(--color-chart-4)"
                  radius={[0, 3, 3, 0]}
                  barSize={12}
                  isAnimationActive={false}
                />
              )}
              {!hidden.has(
                msg("auto.features.optimizations.components.gridlivechart.literal.4"),
              ) && (
                <Bar
                  dataKey="optimizedScore"
                  name={msg("auto.features.optimizations.components.gridlivechart.literal.5")}
                  fill="var(--color-chart-2)"
                  radius={[0, 3, 3, 0]}
                  barSize={12}
                  isAnimationActive={false}
                />
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
        )}
        <div className="flex justify-center gap-4 mt-1">
          {[
            {
              key: msg("auto.features.optimizations.components.gridlivechart.literal.6"),
              color: "var(--color-chart-4)",
            },
            {
              key: msg("auto.features.optimizations.components.gridlivechart.literal.7"),
              color: "var(--color-chart-2)",
            },
          ].map(({ key, color }) => {
            const isHidden = hidden.has(key);
            return (
              <button
                key={key}
                type="button"
                onClick={() => toggle(key)}
                className={`flex items-center gap-1.5 text-[0.625rem] cursor-pointer transition-colors ${isHidden ? "text-muted-foreground/50" : "text-muted-foreground hover:text-foreground"}`}
                aria-pressed={!isHidden}
              >
                <span
                  className="size-2 rounded-full transition-all"
                  style={
                    isHidden
                      ? {
                          backgroundColor: "transparent",
                          boxShadow: `inset 0 0 0 1.5px ${color}`,
                        }
                      : { backgroundColor: color }
                  }
                />
                {key}
              </button>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}
