"use client";

import { useCallback, useMemo, useState } from "react";
import { BarChart3 } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartTooltip } from "@/shared/charts/chart-utils";
import { ChartTable } from "@/shared/charts/chart-table";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { HelpTip } from "@/shared/ui/help-tip";
import { useLiteMode } from "@/features/settings";
import { msg } from "@/shared/lib/messages";
import { RunChip, colorFor, runToken, type RunInfo } from "./compare-model";

type ChartRow = {
  metric: string;
  [runKey: string]: string | number | null;
};

export function PerformanceChart({ runs }: { runs: RunInfo[] }) {
  const lite = useLiteMode();
  const [hiddenRuns, setHiddenRuns] = useState<Set<string>>(new Set());
  const toggleRun = useCallback(
    (id: string) => {
      setHiddenRuns((prev) => {
        const next = new Set(prev);
        if (next.has(id)) next.delete(id);
        else next.add(id);
        // Prevent hiding every single run — always keep at least one visible.
        if (next.size >= prev.size + 1 && next.size === runs.length) {
          return prev;
        }
        return next;
      });
    },
    [runs.length],
  );
  const chartData = useMemo(() => {
    const latencies = runs
      .map((r) => r.avgResponseMs)
      .filter((v): v is number => v != null && v > 0);
    const minLatency = latencies.length ? Math.min(...latencies) : null;
    const hasSpeed = minLatency != null;

    const quality = (r: RunInfo) => {
      const v = r.optimized;
      if (v == null) return null;
      return v / 100;
    };
    const speed = (r: RunInfo) => {
      if (!hasSpeed || r.avgResponseMs == null || r.avgResponseMs <= 0) return null;
      return minLatency! / r.avgResponseMs;
    };

    const toPct = (v: number | null) => (v == null ? null : Math.round(v * 1000) / 10);

    const qualityRow: ChartRow = { metric: msg("auto.app.compare.page.literal.9") };
    const speedRow: ChartRow = { metric: msg("auto.app.compare.page.literal.10") };
    runs.forEach((r, i) => {
      const key = runToken(i);
      qualityRow[key] = toPct(quality(r));
      speedRow[key] = toPct(speed(r));
    });

    const rows: ChartRow[] = [qualityRow];
    if (hasSpeed) {
      rows.push(speedRow);
    }
    return { rows, hasSpeed };
  }, [runs]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <BarChart3 className="size-4" />
          <HelpTip text={msg("auto.app.compare.page.literal.12")}>
            {msg("auto.app.compare.page.6")}
          </HelpTip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[280px] min-w-0" dir="ltr">
          {lite ? (
            <ChartTable
              rows={chartData.rows}
              columns={[
                { key: "metric", label: msg("auto.app.compare.page.6") },
                ...runs.map((run, i) => ({
                  key: runToken(i),
                  label: `${runToken(i)} · ${run.label}`,
                  align: "end" as const,
                  format: (value: unknown) =>
                    typeof value === "number" ? `${value.toFixed(1)}%` : "—",
                })),
              ]}
            />
          ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={chartData.rows}
              margin={{ top: 16, right: 16, bottom: 8, left: 0 }}
              barCategoryGap="20%"
              barGap={4}
            >
              <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted" />
              <XAxis
                dataKey="metric"
                tickLine={false}
                axisLine={false}
                tick={{ fontSize: 12 }}
                className="fill-muted-foreground"
                reversed
              />
              <YAxis
                domain={[0, 100]}
                tickLine={false}
                axisLine={false}
                tick={{ fontSize: 11 }}
                className="fill-muted-foreground"
                ticks={[0, 25, 50, 75, 100]}
                tickFormatter={(v) => `${v}%`}
                orientation="right"
                width={40}
              />
              <RechartsTooltip
                content={<ChartTooltip />}
                cursor={{ fill: "var(--muted)", opacity: 0.3 }}
              />
              {runs.map((run, i) => {
                if (hiddenRuns.has(run.job.optimization_id)) return null;
                return (
                  <Bar
                    key={run.job.optimization_id}
                    dataKey={runToken(i)}
                    name={`${runToken(i)} · ${run.label}`}
                    radius={[4, 4, 0, 0]}
                    animationDuration={400}
                  >
                    {chartData.rows.map((_, idx) => (
                      <Cell key={idx} fill={colorFor(i)} />
                    ))}
                  </Bar>
                );
              })}
            </BarChart>
          </ResponsiveContainer>
          )}
        </div>
        <div className="flex flex-wrap justify-center gap-4 mt-3">
          {runs.map((run, i) => {
            const isHidden = hiddenRuns.has(run.job.optimization_id);
            const color = colorFor(i);
            return (
              <button
                key={run.job.optimization_id}
                type="button"
                onClick={() => toggleRun(run.job.optimization_id)}
                aria-pressed={!isHidden}
                className={`flex items-center gap-1.5 text-xs cursor-pointer transition-opacity ${
                  isHidden ? "opacity-45 hover:opacity-75" : "hover:opacity-80"
                }`}
              >
                <span
                  className="size-2.5 rounded-full shrink-0 transition-all"
                  style={
                    isHidden
                      ? { backgroundColor: "transparent", boxShadow: `inset 0 0 0 1.5px ${color}` }
                      : { backgroundColor: color }
                  }
                />
                <RunChip index={i} label={run.label} />
              </button>
            );
          })}
        </div>
        {!chartData.hasSpeed && (
          <p className="text-[0.6875rem] text-muted-foreground text-center mt-2">
            {msg("auto.app.compare.page.7")}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
