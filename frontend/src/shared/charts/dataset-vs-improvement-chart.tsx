"use client";

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";
import { useLiteMode } from "@/features/settings";
import { ChartTable } from "@/shared/charts/chart-table";
import { getActiveDir } from "@/shared/lib/runtime-locale";

interface DatasetVsImprovementChartProps {
  data: Array<{ rows: number; improvement: number; name: string }>;
  optimizationIds?: string[];
  onDotClick?: (optimizationId: string) => void;
}

export function DatasetVsImprovementChart({
  data,
  optimizationIds,
  onDotClick,
}: DatasetVsImprovementChartProps) {
  const lite = useLiteMode();

  if (data.length === 0) return null;

  if (lite) {
    return (
      <ChartTable
        rows={data}
        columns={[
          {
            key: "rows",
            label: formatMsg("auto.shared.charts.dataset.vs.improvement.chart.template.1", {
              p1: TERMS.dataset,
            }),
            align: "end",
            format: (value) => (typeof value === "number" ? String(value) : "—"),
          },
          {
            key: "improvement",
            label: msg("auto.shared.charts.dataset.vs.improvement.chart.literal.1"),
            align: "end",
            format: (value) => (typeof value === "number" ? String(value) : "—"),
          },
          { key: "name", label: TERMS.optimization },
        ]}
        onRowClick={
          onDotClick
            ? (_, index) => {
                if (optimizationIds?.[index]) onDotClick(optimizationIds[index]);
              }
            : undefined
        }
      />
    );
  }

  return (
    <div className="h-[250px] min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ left: 10, right: 20, top: 10, bottom: 25 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            type="number"
            dataKey="rows"
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 10 }}
            className="fill-muted-foreground"
            label={{
              value: formatMsg("auto.shared.charts.dataset.vs.improvement.chart.template.1", {
                p1: TERMS.dataset,
              }),
              position: "insideBottom",
              offset: -10,
              fontSize: 10,
            }}
          />
          <YAxis
            type="number"
            dataKey="improvement"
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 10 }}
            className="fill-muted-foreground"
            label={{
              value: msg("auto.shared.charts.dataset.vs.improvement.chart.literal.1"),
              angle: -90,
              position: "center",
              dx: -15,
              fontSize: 10,
            }}
          />
          <ZAxis range={[40, 40]} />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const d = payload[0]?.payload as
                | { name?: string; rows?: number; improvement?: number }
                | undefined;
              return (
                <div
                  className="rounded-xl border border-border/60 bg-background/95 backdrop-blur-sm p-3 shadow-lg text-sm"
                  dir={getActiveDir()}
                >
                  {d?.name && (
                    <p className="font-semibold mb-1 text-foreground font-mono" dir="ltr">
                      {d.name}
                    </p>
                  )}
                  <div className="space-y-0.5 text-xs text-muted-foreground">
                    <div>
                      {msg("auto.shared.charts.dataset.vs.improvement.chart.1")}{" "}
                      <span className="font-mono font-semibold text-foreground">{d?.rows}</span>
                    </div>
                    <div>
                      {msg("auto.shared.charts.dataset.vs.improvement.chart.2")}{" "}
                      <span className="font-mono font-semibold text-foreground">
                        {d?.improvement}
                      </span>
                    </div>
                  </div>
                </div>
              );
            }}
          />
          <Scatter
            data={data}
            fill="var(--color-chart-2)"
            cursor={onDotClick ? "pointer" : "default"}
            onClick={(_entry, index) => {
              if (
                onDotClick &&
                optimizationIds &&
                typeof index === "number" &&
                optimizationIds[index]
              )
                onDotClick(optimizationIds[index]);
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
