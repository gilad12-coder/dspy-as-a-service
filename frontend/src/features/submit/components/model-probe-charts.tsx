import * as React from "react";
import {
  LineChart,
  Line as RLine,
  XAxis,
  YAxis,
  CartesianGrid,
  ReferenceLine,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
} from "recharts";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLiteMode } from "@/features/settings";
import { ChartTable } from "@/shared/charts/chart-table";
import { bestSoFar, rowColor, type ModelRow, type TrajectoryPoint } from "./model-probe-model";

export function TrajectorySparkline({
  points,
  asymptote,
  width = 120,
  height = 32,
  color = "currentColor",
}: {
  points: TrajectoryPoint[];
  asymptote: number | null;
  width?: number;
  height?: number;
  color?: string;
}) {
  if (points.length === 0) return null;
  const w = width;
  const h = height;
  const pad = 4;
  const scores = points.map((p) => p.score);
  const minY = Math.min(...scores, asymptote ?? Infinity);
  const maxY = Math.max(...scores, asymptote ?? -Infinity);
  const range = maxY - minY || 1;
  const n = points.length;
  const xAt = (i: number) => (n <= 1 ? w / 2 : pad + (i / (n - 1)) * (w - 2 * pad));
  const yAt = (v: number) => h - pad - ((v - minY) / range) * (h - 2 * pad);
  const path = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xAt(i).toFixed(1)} ${yAt(p.score).toFixed(1)}`)
    .join(" ");
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      width={w}
      height={h}
      className="overflow-visible"
      aria-hidden="true"
    >
      {asymptote !== null && (
        <line
          x1={pad}
          x2={w - pad}
          y1={yAt(asymptote)}
          y2={yAt(asymptote)}
          stroke={color}
          strokeWidth={1}
          strokeDasharray="2 3"
          opacity={0.45}
        />
      )}
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {points.map((p, i) => (
        <circle key={i} cx={xAt(i)} cy={yAt(p.score)} r={i === n - 1 ? 2.5 : 1.5} fill={color} />
      ))}
    </svg>
  );
}

function ProbeChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number; name: string; color: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border bg-background p-3 shadow-md text-sm">
      <p className="font-medium mb-1.5">
        {msg("auto.features.submit.components.modelprobedialog.20")}{" "}
        <span dir="ltr" className="font-mono">
          #{label}
        </span>
      </p>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="size-2.5 rounded-full shrink-0" style={{ backgroundColor: p.color }} />
          <span className="text-muted-foreground">{p.name}:</span>
          <span className="font-mono font-bold ms-auto" dir="ltr">
            {p.value.toFixed(1)}
          </span>
        </div>
      ))}
    </div>
  );
}

export function TrajectoryDetailChart({
  points,
  asymptote,
}: {
  points: TrajectoryPoint[];
  asymptote: number | null;
  color?: string;
}) {
  const lite = useLiteMode();
  const bsf = bestSoFar(points);
  const data = points.map((p, i) => ({
    step: p.step,
    score: p.score,
    best: bsf[i]?.score ?? p.score,
  }));

  if (lite) {
    return (
      <div dir="ltr" className="h-[340px] w-full min-w-0">
        <ChartTable
          rows={data}
          columns={[
            { key: "step", label: msg("auto.features.submit.components.modelprobedialog.literal.4") },
            {
              key: "score",
              label: msg("auto.features.submit.components.modelprobedialog.literal.6"),
              align: "end",
              format: (value: unknown) => (typeof value === "number" ? value.toFixed(1) : "—"),
            },
            {
              key: "best",
              label: msg("auto.features.submit.components.modelprobedialog.literal.7"),
              align: "end",
              format: (value: unknown) => (typeof value === "number" ? value.toFixed(1) : "—"),
            },
          ]}
        />
      </div>
    );
  }

  return (
    <div dir="ltr" className="h-[340px] w-full min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 10, left: 5, bottom: 18 }}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="step"
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 10 }}
            className="fill-muted-foreground"
            label={{
              value: msg("auto.features.submit.components.modelprobedialog.literal.4"),
              position: "insideBottom",
              offset: -12,
              fontSize: 10,
              fill: "var(--muted-foreground)",
            }}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 10 }}
            className="fill-muted-foreground"
            label={{
              value: msg("auto.features.submit.components.modelprobedialog.literal.5"),
              angle: -90,
              position: "insideLeft",
              offset: 10,
              fontSize: 10,
              fill: "var(--muted-foreground)",
            }}
            domain={[0, "auto"]}
          />
          <RechartsTooltip content={<ProbeChartTooltip />} />
          {asymptote !== null && (
            <ReferenceLine
              y={asymptote}
              stroke="var(--color-chart-2)"
              strokeDasharray="4 3"
              strokeOpacity={0.55}
            />
          )}
          <RLine
            type="monotone"
            dataKey="score"
            name={msg("auto.features.submit.components.modelprobedialog.literal.6")}
            stroke="var(--color-chart-4)"
            strokeWidth={1.5}
            dot={{
              r: 3.5,
              strokeWidth: 1.5,
              stroke: "var(--color-chart-4)",
              fill: "var(--background, #fff)",
            }}
            activeDot={{
              r: 5,
              strokeWidth: 2,
              stroke: "var(--color-chart-4)",
              fill: "var(--background, #fff)",
            }}
            isAnimationActive={false}
          />
          <RLine
            type="stepAfter"
            dataKey="best"
            name={msg("auto.features.submit.components.modelprobedialog.literal.7")}
            stroke="var(--color-chart-2)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function TrajectoryCompareChart({ rows }: { rows: ModelRow[] }) {
  const [soloModel, setSoloModel] = React.useState<string | null>(null);
  const [hovered, setHovered] = React.useState<{
    label: string;
    step: number;
    score: number;
    x: number;
    y: number;
  } | null>(null);
  const svgRef = React.useRef<SVGSVGElement>(null);

  const allSeries = rows
    .filter((r) => r.trajectory.length > 0)
    .map((r) => ({
      model: r.model,
      label: r.label,
      color: rowColor(r),
      points: bestSoFar(r.trajectory),
      raw: r.trajectory,
      asymptote:
        r.scaling?.signal === "strong" || r.scaling?.signal === "observed"
          ? (r.scaling?.asymptote ?? null)
          : null,
    }));
  if (allSeries.length === 0) return null;

  const toggleSolo = (model: string) => setSoloModel((prev) => (prev === model ? null : model));

  const series = soloModel ? allSeries.filter((s) => s.model === soloModel) : allSeries;

  const w = 800;
  const h = 240;
  const padX = 46;
  const padTop = 16;
  const padBottom = 44;
  const innerW = w - padX * 2;
  const innerH = h - padTop - padBottom;

  const allScores = series.flatMap((s) => s.points.map((p) => p.score));
  const allAsymptotes = series.map((s) => s.asymptote).filter((v): v is number => v !== null);
  const minY = Math.min(...allScores, ...allAsymptotes);
  const maxY = Math.max(...allScores, ...allAsymptotes);
  const yRange = maxY - minY || 1;
  const maxLen = Math.max(...series.map((s) => s.points.length));
  const yAt = (v: number) => padTop + innerH - ((v - minY) / yRange) * innerH;
  const xAt = (i: number) => (maxLen <= 1 ? w / 2 : padX + (i / (maxLen - 1)) * innerW);

  const tickCount = 4;
  const ticks = Array.from({ length: tickCount + 1 }, (_, i) => minY + (yRange * i) / tickCount);

  return (
    <div dir="ltr" className="min-w-0 space-y-2">
      <div className="relative min-w-0">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${w} ${h}`}
          className="h-[240px] w-full"
          preserveAspectRatio="xMidYMid meet"
          aria-hidden="true"
          onMouseLeave={() => setHovered(null)}
        >
          {ticks.map((t, i) => (
            <g key={i}>
              <line
                x1={padX}
                x2={w - padX}
                y1={yAt(t)}
                y2={yAt(t)}
                stroke="currentColor"
                strokeWidth={0.5}
                strokeDasharray="2 4"
                className="text-muted-foreground/30"
              />
              <text
                x={padX - 6}
                y={yAt(t) + 3}
                textAnchor="end"
                className="fill-muted-foreground text-[9px] font-mono tabular-nums"
              >
                {t.toFixed(1)}
              </text>
            </g>
          ))}

          {Array.from({ length: Math.min(maxLen, 12) }, (_, i) => {
            const step = maxLen <= 12 ? i : Math.round((i / 11) * (maxLen - 1));
            return (
              <text
                key={step}
                x={xAt(step)}
                y={h - 22}
                textAnchor="middle"
                className="fill-muted-foreground text-[9px] font-mono tabular-nums"
              >
                {step}
              </text>
            );
          })}

          <text
            x={w / 2}
            y={h - 4}
            textAnchor="middle"
            className="fill-muted-foreground text-[10px]"
          >
            {msg("auto.features.submit.components.modelprobedialog.21")}
          </text>
          <text
            x={12}
            y={padTop + innerH / 2}
            textAnchor="middle"
            transform={`rotate(-90, 12, ${padTop + innerH / 2})`}
            className="fill-muted-foreground text-[10px]"
          >
            {msg("auto.features.submit.components.modelprobedialog.22")}
          </text>

          {series.map((s) => {
            if (s.points.length === 0) return null;
            const path = s.points
              .map(
                (p, i) => `${i === 0 ? "M" : "L"} ${xAt(i).toFixed(1)} ${yAt(p.score).toFixed(1)}`,
              )
              .join(" ");
            return (
              <g key={s.model}>
                {s.asymptote !== null && (
                  <line
                    x1={padX}
                    x2={w - padX}
                    y1={yAt(s.asymptote)}
                    y2={yAt(s.asymptote)}
                    stroke={s.color}
                    strokeWidth={0.75}
                    strokeDasharray="3 3"
                    opacity={0.4}
                  />
                )}
                <path
                  d={path}
                  fill="none"
                  stroke={s.color}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {s.points.map((p, i) => {
                  const isLast = i === s.points.length - 1;
                  return (
                    <g key={i}>
                      <circle
                        cx={xAt(i)}
                        cy={yAt(p.score)}
                        r={isLast ? 3.5 : 2}
                        fill={s.color}
                        opacity={isLast ? 1 : 0.5}
                        stroke={isLast ? "var(--background, #fff)" : undefined}
                        strokeWidth={isLast ? 1.5 : undefined}
                      />
                      <circle
                        cx={xAt(i)}
                        cy={yAt(p.score)}
                        r={isLast ? 10 : 8}
                        fill="transparent"
                        className="cursor-pointer"
                        onClick={() => toggleSolo(s.model)}
                        onMouseEnter={() => {
                          const svg = svgRef.current;
                          if (!svg) return;
                          const rect = svg.getBoundingClientRect();
                          const px = (xAt(i) / w) * rect.width;
                          const py = (yAt(p.score) / h) * rect.height;
                          setHovered({
                            label: s.label,
                            step: p.step,
                            score: p.score,
                            x: px,
                            y: py,
                          });
                        }}
                        onMouseLeave={() => setHovered(null)}
                      />
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
        {hovered &&
          (() => {
            const flipBelow = hovered.y < 40;
            return (
              <div
                className="pointer-events-none absolute z-10 max-w-[180px] rounded-md border border-border/60 bg-popover px-2.5 py-1.5 text-xs shadow-md"
                style={{
                  left: `clamp(70px, ${hovered.x}px, calc(100% - 70px))`,
                  top: hovered.y,
                  transform: flipBelow
                    ? "translate(-50%, 16px)"
                    : "translate(-50%, calc(-100% - 10px))",
                }}
              >
                <div className="truncate font-mono font-semibold" dir="ltr">
                  {hovered.label}
                </div>
                <div className="text-muted-foreground tabular-nums">
                  {msg("auto.features.submit.components.modelprobedialog.23")}
                  <span dir="ltr">{hovered.step}</span> ·{" "}
                  <span className="font-semibold text-foreground" dir="ltr">
                    {hovered.score.toFixed(1)}
                  </span>
                </div>
              </div>
            );
          })()}
      </div>

      <div
        className="flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-[0.6875rem]"
      >
        {allSeries.map((s) => {
          const dimmed = soloModel !== null && soloModel !== s.model;
          return (
            <div
              key={s.model}
              className={cn(
                "flex items-center gap-1.5 cursor-pointer transition-opacity select-none",
                dimmed && "opacity-30",
              )}
              onClick={() => toggleSolo(s.model)}
              title={s.label}
            >
              <span
                className="inline-block h-2 w-2 shrink-0 self-start mt-1 rounded-full"
                style={{ backgroundColor: s.color }}
              />
              <span className="flex flex-col items-center min-w-0">
                <span className="max-w-[120px] truncate font-mono text-muted-foreground" dir="ltr">
                  {s.label}
                </span>
                {s.asymptote !== null && (
                  <span
                    className="font-mono text-[0.625rem] tabular-nums text-foreground"
                    dir="ltr"
                  >
                    {s.asymptote.toFixed(1)}
                  </span>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
