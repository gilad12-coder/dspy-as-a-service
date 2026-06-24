"use client";

import { memo, useMemo, useState } from "react";
import {
  ChevronLeft,
  Crown,
  Gauge,
  Loader2,
  Play,
  RotateCcw,
  Trash2,
  Trophy,
  X,
  XCircle,
} from "lucide-react";
import { AnimatePresence, motion } from "framer-motion";
import { toast } from "react-toastify";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
import { Button } from "@/shared/ui/primitives/button";
import { Dialog, DialogContent, DialogFooter } from "@/shared/ui/primitives/dialog";
import { DialogTitleRow } from "@/shared/ui/dialog-title-row";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { FadeIn, StaggerContainer, StaggerItem, TiltCard } from "@/shared/ui/motion";
import { HelpTip } from "@/shared/ui/help-tip";
import type { OptimizationStatusResponse, PairResult } from "@/shared/types/api";
import { formatPercent } from "@/shared/lib";
import { deleteGridPair, restartGridPair, resumeGridPair } from "@/shared/lib/api";
import { ChartTable } from "@/shared/charts/chart-table";
import { useLiteMode } from "@/features/settings";
import { msg } from "@/shared/lib/messages";
import { tip } from "@/shared/lib/tooltips";
import { computePairScores } from "../lib/pair-scores";
import { ReasoningPill } from "./ui-primitives";
import {
  CombinedTip,
  ScatterTip,
  ScoreTip,
  computePareto,
  pairLabel,
  type ScatterPoint,
} from "./grid-overview-helpers";

function GridOverviewImpl({
  job,
  onPairSelect,
  onPairDeleted,
}: {
  job: OptimizationStatusResponse;
  onPairSelect: (pairIndex: number) => void;
  onPairDeleted?: (pairIndex: number) => void;
}) {
  const lite = useLiteMode();
  const [pendingDelete, setPendingDelete] = useState<PairResult | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [rerunningPair, setRerunningPair] = useState<number | null>(null);
  const [pairFilter, setPairFilter] = useState<number | null>(null);
  const [hiddenPairSeries, setHiddenPairSeries] = useState<Set<string>>(new Set());
  const [hiddenCombinedSeries, setHiddenCombinedSeries] = useState<Set<string>>(new Set());
  const [hiddenScatterSeries, setHiddenScatterSeries] = useState<Set<string>>(new Set());
  const togglePairSeries = (key: string) => {
    setHiddenPairSeries((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };
  const toggleCombinedSeries = (key: string) => {
    setHiddenCombinedSeries((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };
  const toggleScatterSeries = (key: string) => {
    setHiddenScatterSeries((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };
  const handleBarClick = (data: unknown) => {
    const pi = (data as { pair_index?: number }).pair_index;
    if (typeof pi === "number") setPairFilter(pi);
  };
  const handleConfirmDelete = async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await deleteGridPair(job.optimization_id, pendingDelete.pair_index);
      onPairDeleted?.(pendingDelete.pair_index);
      setPendingDelete(null);
      window.dispatchEvent(new Event("optimizations-changed"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("optimization.delete.failed"));
    } finally {
      setDeleting(false);
    }
  };
  const handlePairRerun = async (pairIndex: number, resume: boolean) => {
    setRerunningPair(pairIndex);
    try {
      if (resume) await resumeGridPair(job.optimization_id, pairIndex);
      else await restartGridPair(job.optimization_id, pairIndex);
      toast.success(msg(resume ? "optimization.pair.resume.success" : "optimization.pair.restart.success"));
      // The grid re-enters running in place — the parent's poll picks up the change.
      window.dispatchEvent(new Event("optimizations-changed"));
    } catch (err) {
      toast.error(
        err instanceof Error
          ? err.message
          : msg(resume ? "optimization.pair.resume.failed" : "optimization.pair.restart.failed"),
      );
    } finally {
      setRerunningPair(null);
    }
  };
  const gridResult = job.grid_result;
  // Keyed on `gridResult`: the parent polls every 5s and runs a 1s elapsed
  // interval, so without memoization these datasets (and the O(n²) Pareto
  // pass) rebuild ~1Hz even when the grid payload is unchanged.
  const derived = useMemo(() => {
    if (!gridResult) return null;
    const prs = gridResult.pair_results;
    const completedPrs = prs.filter((p) => !p.error);
    const scoring = computePairScores(prs);
    const byIdx = (i: number | null | undefined) =>
      i == null ? null : (prs.find((p) => p.pair_index === i) ?? null);
    const qualityPair = byIdx(scoring.qualityWinner);
    const speedPair = byIdx(scoring.speedWinner);
    const pairScoresData = completedPrs.map((p) => ({
      pair_index: p.pair_index,
      name: pairLabel(p),
      baselineScore: Math.round(p.baseline_test_metric ?? 0),
      optimizedScore: Math.round(p.optimized_test_metric ?? 0),
      isBest: scoring.overallWinner === p.pair_index,
    }));
    const combinedScoresData = completedPrs
      .map((p) => {
        const s = scoring.byIndex[p.pair_index];
        if (!s) return null;
        return {
          pair_index: p.pair_index,
          name: pairLabel(p),
          quality: Math.round(s.quality * 100),
          speed: s.speed != null ? Math.round(s.speed * 100) : 0,
          isBest: scoring.overallWinner === p.pair_index,
        };
      })
      .filter((r): r is NonNullable<typeof r> => r != null);
    const pairRespTimeData = completedPrs
      .filter((p) => p.avg_response_time_ms)
      .map((p) => ({
        pair_index: p.pair_index,
        name: pairLabel(p),
        responseTime: +(p.avg_response_time_ms! / 1000).toFixed(1),
        isBest: scoring.speedWinner === p.pair_index,
      }));

    const scatterPoints: ScatterPoint[] = completedPrs
      .filter((p) => p.avg_response_time_ms != null && p.optimized_test_metric != null)
      .map((p) => {
        const raw = p.optimized_test_metric!;
        return {
          pair_index: p.pair_index,
          name: pairLabel(p),
          quality: +raw.toFixed(1),
          latency: +(p.avg_response_time_ms! / 1000).toFixed(2),
          isOverall: scoring.overallWinner === p.pair_index,
          isQuality: scoring.qualityWinner === p.pair_index,
          isSpeed: scoring.speedWinner === p.pair_index,
        };
      });
    const { pareto: paretoPoints, dominated: dominatedPoints } = computePareto(scatterPoints);
    return {
      prs,
      completedPrs,
      scoring,
      qualityPair,
      speedPair,
      pairScoresData,
      combinedScoresData,
      pairRespTimeData,
      scatterPoints,
      paretoPoints,
      dominatedPoints,
    };
  }, [gridResult]);

  if (!derived) return null;
  const {
    prs,
    completedPrs,
    scoring,
    qualityPair,
    speedPair,
    pairScoresData,
    combinedScoresData,
    pairRespTimeData,
    scatterPoints,
    paretoPoints,
    dominatedPoints,
  } = derived;

  const matchesFilter = (pi: number) => pairFilter == null || pi === pairFilter;
  const pairScoresFiltered = pairScoresData.filter((r) => matchesFilter(r.pair_index));
  const combinedScoresFiltered = combinedScoresData.filter((r) => matchesFilter(r.pair_index));
  const pairRespTimeFiltered = pairRespTimeData.filter((r) => matchesFilter(r.pair_index));
  const paretoFiltered = paretoPoints.filter((p) => matchesFilter(p.pair_index));
  const dominatedFiltered = dominatedPoints.filter((p) => matchesFilter(p.pair_index));
  const visiblePrs = (() => {
    const filtered = pairFilter != null ? prs.filter((p) => p.pair_index === pairFilter) : prs;
    const winnerId = scoring.overallWinner;
    if (winnerId == null) return filtered;
    const winner = filtered.find((p) => p.pair_index === winnerId);
    if (!winner || filtered[0]?.pair_index === winnerId) return filtered;
    return [winner, ...filtered.filter((p) => p.pair_index !== winnerId)];
  })();
  const selectedPair =
    pairFilter != null ? (prs.find((p) => p.pair_index === pairFilter) ?? null) : null;

  return (
    <div className="space-y-4" data-tutorial="grid-search">
      <StaggerContainer className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <StaggerItem>
          <TiltCard
            className="rounded-xl border border-border/50 bg-card/80 p-4 text-center"
            onClick={qualityPair ? () => setPairFilter(qualityPair.pair_index) : undefined}
          >
            <div className="flex items-center justify-center gap-1.5 mb-1">
              <Trophy className="size-3 text-[#C8A882]" />
              <p className="text-[0.625rem] text-muted-foreground">
                {msg("auto.features.optimizations.components.gridoverview.7")}
              </p>
            </div>
            <p
              className="font-mono text-[0.6875rem] font-semibold truncate text-foreground"
              title={qualityPair ? pairLabel(qualityPair) : undefined}
              dir="ltr"
            >
              {qualityPair ? pairLabel(qualityPair) : "—"}
            </p>
            <p className="text-sm font-mono font-bold tabular-nums text-foreground mt-0.5">
              {formatPercent(qualityPair?.optimized_test_metric)}
            </p>
          </TiltCard>
        </StaggerItem>
        <StaggerItem>
          <TiltCard
            className="rounded-xl border border-border/50 bg-card/80 p-4 text-center"
            onClick={speedPair ? () => setPairFilter(speedPair.pair_index) : undefined}
          >
            <div className="flex items-center justify-center gap-1.5 mb-1">
              <Gauge className="size-3 text-[#3D2E22]" />
              <p className="text-[0.625rem] text-muted-foreground">
                {msg("auto.features.optimizations.components.gridoverview.8")}
              </p>
            </div>
            <p
              className="font-mono text-[0.6875rem] font-semibold truncate text-foreground"
              title={speedPair ? pairLabel(speedPair) : undefined}
              dir="ltr"
            >
              {speedPair ? pairLabel(speedPair) : "—"}
            </p>
            <p className="text-sm font-mono font-bold tabular-nums text-foreground mt-0.5">
              {speedPair?.avg_response_time_ms != null
                ? `${(speedPair.avg_response_time_ms / 1000).toFixed(1)}s`
                : "—"}
            </p>
          </TiltCard>
        </StaggerItem>
      </StaggerContainer>

      {completedPrs.length > 0 && (
        <FadeIn delay={0.1}>
          {pairFilter != null && (
            <div className="flex items-center gap-2 flex-wrap mb-3">
              <span className="group inline-flex items-center gap-1.5 rounded-lg bg-[#3D2E22]/[0.06] border border-[#3D2E22]/10 pe-1 ps-2.5 py-1 transition-all duration-150 hover:bg-[#3D2E22]/[0.1] hover:border-[#3D2E22]/20">
                <span
                  className="font-mono text-[0.6875rem] font-medium text-[#3D2E22]/80"
                  dir="ltr"
                >
                  {selectedPair ? pairLabel(selectedPair) : "—"}
                </span>
                <button
                  onClick={() => setPairFilter(null)}
                  className="size-5 rounded-md flex items-center justify-center text-[#3D2E22]/40 hover:text-[#3D2E22] hover:bg-[#3D2E22]/10 transition-colors cursor-pointer"
                  aria-label={msg("auto.features.optimizations.components.gridoverview.literal.1")}
                >
                  <X className="size-3" />
                </button>
              </span>
            </div>
          )}
          {scatterPoints.length > 0 && (
            <Card className="mb-4">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-semibold">
                  <HelpTip
                    text={msg("auto.features.optimizations.components.gridoverview.literal.2")}
                  >
                    {msg("auto.features.optimizations.components.gridoverview.9")}
                  </HelpTip>
                </CardTitle>
              </CardHeader>
              <CardContent className="pt-0">
                {lite ? (
                  <div className="h-[280px] min-w-0">
                    <ChartTable
                      rows={scatterPoints.filter((p) => matchesFilter(p.pair_index))}
                      onRowClick={(row) => handleBarClick(row)}
                      columns={[
                        {
                          key: "name",
                          label: msg(
                            "auto.features.optimizations.components.gridoverview.9",
                          ),
                        },
                        {
                          key: "quality",
                          label: msg(
                            "auto.features.optimizations.components.gridoverview.literal.4",
                          ),
                          align: "end",
                          format: (value) =>
                            typeof value === "number" ? `${value.toFixed(1)}%` : "—",
                        },
                        {
                          key: "latency",
                          label: msg(
                            "auto.features.optimizations.components.gridoverview.literal.3",
                          ),
                          align: "end",
                          format: (value) =>
                            typeof value === "number" ? `${value.toFixed(2)}s` : "—",
                        },
                      ]}
                    />
                  </div>
                ) : (
                <div className="h-[280px] min-w-0" dir="ltr">
                  <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 10, right: 30, bottom: 35, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                      <XAxis
                        type="number"
                        dataKey="latency"
                        name="latency"
                        unit="s"
                        tickLine={false}
                        axisLine={false}
                        tick={{ fontSize: 10 }}
                        className="fill-muted-foreground"
                        label={{
                          value: msg(
                            "auto.features.optimizations.components.gridoverview.literal.3",
                          ),
                          position: "insideBottom",
                          offset: -15,
                          fontSize: 10,
                        }}
                      />
                      <YAxis
                        type="number"
                        dataKey="quality"
                        name="quality"
                        unit="%"
                        domain={[0, 100]}
                        tickLine={false}
                        axisLine={false}
                        tick={{ fontSize: 10 }}
                        className="fill-muted-foreground"
                        label={{
                          value: msg(
                            "auto.features.optimizations.components.gridoverview.literal.4",
                          ),
                          angle: -90,
                          position: "center",
                          dx: -15,
                          fontSize: 10,
                        }}
                      />
                      <ZAxis range={[70, 70]} />
                      <Tooltip content={<ScatterTip />} cursor={{ strokeDasharray: "3 3" }} />
                      {!hiddenScatterSeries.has("dominated") && (
                        <Scatter
                          data={dominatedFiltered}
                          fill="var(--color-muted-foreground)"
                          fillOpacity={0.35}
                          cursor="pointer"
                          onClick={handleBarClick}
                        />
                      )}
                      {!hiddenScatterSeries.has("pareto") && (
                        <Scatter
                          data={paretoFiltered}
                          fill="#3D2E22"
                          cursor="pointer"
                          onClick={handleBarClick}
                        />
                      )}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
                )}
                <div className="mt-2 flex items-center justify-center gap-4 text-[0.6875rem]">
                  {[
                    {
                      key: "pareto",
                      label: msg("auto.features.optimizations.components.gridoverview.literal.5"),
                      color: "#3D2E22",
                    },
                    {
                      key: "dominated",
                      label: msg("auto.features.optimizations.components.gridoverview.literal.6"),
                      color: "var(--color-muted-foreground)",
                    },
                  ].map(({ key, label, color }) => {
                    const isHidden = hiddenScatterSeries.has(key);
                    return (
                      <button
                        key={key}
                        type="button"
                        onClick={() => toggleScatterSeries(key)}
                        className={`inline-flex items-center gap-1.5 cursor-pointer transition-colors ${isHidden ? "text-muted-foreground/50" : "text-muted-foreground hover:text-foreground"}`}
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
                        {label}
                      </button>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}
          <AnimatePresence mode="wait">
            <motion.div
              key={pairFilter ?? "all"}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -4 }}
              transition={{ duration: 0.3, ease: [0.2, 0.8, 0.2, 1] }}
            >
              <div className="grid gap-4 md:grid-cols-2">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-semibold">
                      <HelpTip text={tip("grid.score_comparison")}>
                        {msg("auto.features.optimizations.components.gridoverview.10")}
                      </HelpTip>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    {lite ? (
                      <div className="h-[220px] min-w-0">
                        <ChartTable
                          rows={pairScoresFiltered}
                          onRowClick={(row) => handleBarClick(row)}
                          columns={[
                            {
                              key: "name",
                              label: msg(
                                "auto.features.optimizations.components.gridoverview.literal.8",
                              ),
                            },
                            {
                              key: "baselineScore",
                              label: msg(
                                "auto.features.optimizations.components.gridoverview.literal.10",
                              ),
                              align: "end",
                              format: (value) =>
                                typeof value === "number" ? `${value}%` : "—",
                            },
                            {
                              key: "optimizedScore",
                              label: msg(
                                "auto.features.optimizations.components.gridoverview.literal.12",
                              ),
                              align: "end",
                              format: (value) =>
                                typeof value === "number" ? `${value}%` : "—",
                            },
                          ]}
                        />
                      </div>
                    ) : (
                    <div className="h-[220px] min-w-0" dir="ltr">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={pairScoresFiltered}
                          layout="vertical"
                          margin={{ left: 28, right: 20, top: 5, bottom: 5 }}
                        >
                          <CartesianGrid
                            horizontal={false}
                            strokeDasharray="3 3"
                            className="stroke-muted"
                          />
                          <XAxis
                            type="number"
                            domain={[0, 105]}
                            tickLine={false}
                            axisLine={false}
                            tick={{ fontSize: 10 }}
                            className="fill-muted-foreground"
                            label={{
                              value: msg(
                                "auto.features.optimizations.components.gridoverview.literal.7",
                              ),
                              position: "insideBottom",
                              offset: -2,
                              fontSize: 10,
                            }}
                          />
                          <YAxis
                            type="category"
                            dataKey="name"
                            tick={{ fontSize: 10 }}
                            width={80}
                            tickFormatter={(v: string) =>
                              v.length > 12 ? `${v.slice(0, 11)}…` : v
                            }
                            className="fill-muted-foreground"
                            tickLine={false}
                            axisLine={false}
                            label={{
                              value: msg(
                                "auto.features.optimizations.components.gridoverview.literal.8",
                              ),
                              angle: -90,
                              position: "left",
                              offset: 10,
                              fontSize: 10,
                            }}
                          />
                          <Tooltip content={<ScoreTip />} />
                          {!hiddenPairSeries.has(
                            msg("auto.features.optimizations.components.gridoverview.literal.9"),
                          ) && (
                            <Bar
                              dataKey="baselineScore"
                              name={msg(
                                "auto.features.optimizations.components.gridoverview.literal.10",
                              )}
                              fill="var(--color-chart-4)"
                              radius={[0, 3, 3, 0]}
                              barSize={12}
                              animationDuration={400}
                              cursor="pointer"
                              onClick={handleBarClick}
                            />
                          )}
                          {!hiddenPairSeries.has(
                            msg("auto.features.optimizations.components.gridoverview.literal.11"),
                          ) && (
                            <Bar
                              dataKey="optimizedScore"
                              name={msg(
                                "auto.features.optimizations.components.gridoverview.literal.12",
                              )}
                              fill="var(--color-chart-2)"
                              radius={[0, 3, 3, 0]}
                              barSize={12}
                              animationDuration={400}
                              cursor="pointer"
                              onClick={handleBarClick}
                            />
                          )}
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    )}
                    <div className="flex justify-center gap-4 mt-1">
                      {[
                        {
                          key: msg(
                            "auto.features.optimizations.components.gridoverview.literal.13",
                          ),
                          color: "var(--color-chart-4)",
                        },
                        {
                          key: msg(
                            "auto.features.optimizations.components.gridoverview.literal.14",
                          ),
                          color: "var(--color-chart-2)",
                        },
                      ].map(({ key, color }) => {
                        const isHidden = hiddenPairSeries.has(key);
                        return (
                          <button
                            key={key}
                            type="button"
                            onClick={() => togglePairSeries(key)}
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

                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-semibold">
                      <HelpTip text={tip("grid.quality_speed_combined")}>
                        {msg("auto.features.optimizations.components.gridoverview.11")}
                      </HelpTip>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    {lite ? (
                      <div className="h-[220px] min-w-0">
                        <ChartTable
                          rows={combinedScoresFiltered}
                          onRowClick={(row) => handleBarClick(row)}
                          columns={[
                            {
                              key: "name",
                              label: msg(
                                "auto.features.optimizations.components.gridoverview.literal.16",
                              ),
                            },
                            {
                              key: "quality",
                              label: msg(
                                "auto.features.optimizations.components.gridoverview.literal.18",
                              ),
                              align: "end",
                              format: (value) =>
                                typeof value === "number" ? `${value}%` : "—",
                            },
                            {
                              key: "speed",
                              label: msg(
                                "auto.features.optimizations.components.gridoverview.literal.20",
                              ),
                              align: "end",
                              format: (value) =>
                                typeof value === "number" ? `${value}%` : "—",
                            },
                          ]}
                        />
                      </div>
                    ) : (
                    <div className="h-[220px] min-w-0" dir="ltr">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={combinedScoresFiltered}
                          layout="vertical"
                          margin={{ left: 28, right: 20, top: 5, bottom: 5 }}
                        >
                          <CartesianGrid
                            horizontal={false}
                            strokeDasharray="3 3"
                            className="stroke-muted"
                          />
                          <XAxis
                            type="number"
                            domain={[0, 100]}
                            tickLine={false}
                            axisLine={false}
                            tick={{ fontSize: 10 }}
                            className="fill-muted-foreground"
                            label={{
                              value: msg(
                                "auto.features.optimizations.components.gridoverview.literal.15",
                              ),
                              position: "insideBottom",
                              offset: -2,
                              fontSize: 10,
                            }}
                          />
                          <YAxis
                            type="category"
                            dataKey="name"
                            tick={{ fontSize: 10 }}
                            width={80}
                            tickFormatter={(v: string) =>
                              v.length > 12 ? `${v.slice(0, 11)}…` : v
                            }
                            className="fill-muted-foreground"
                            tickLine={false}
                            axisLine={false}
                            label={{
                              value: msg(
                                "auto.features.optimizations.components.gridoverview.literal.16",
                              ),
                              angle: -90,
                              position: "left",
                              offset: 10,
                              fontSize: 10,
                            }}
                          />
                          <Tooltip content={<CombinedTip />} />
                          {!hiddenCombinedSeries.has(
                            msg("auto.features.optimizations.components.gridoverview.literal.17"),
                          ) && (
                            <Bar
                              dataKey="quality"
                              name={msg(
                                "auto.features.optimizations.components.gridoverview.literal.18",
                              )}
                              fill="var(--color-chart-2)"
                              radius={[0, 3, 3, 0]}
                              barSize={8}
                              animationDuration={400}
                              cursor="pointer"
                              onClick={handleBarClick}
                            />
                          )}
                          {!hiddenCombinedSeries.has(
                            msg("auto.features.optimizations.components.gridoverview.literal.19"),
                          ) && (
                            <Bar
                              dataKey="speed"
                              name={msg(
                                "auto.features.optimizations.components.gridoverview.literal.20",
                              )}
                              fill="var(--color-chart-4)"
                              radius={[0, 3, 3, 0]}
                              barSize={8}
                              animationDuration={400}
                              cursor="pointer"
                              onClick={handleBarClick}
                            />
                          )}
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    )}
                    <div className="flex flex-wrap justify-center gap-3 mt-1">
                      {[
                        {
                          key: msg(
                            "auto.features.optimizations.components.gridoverview.literal.23",
                          ),
                          color: "var(--color-chart-2)",
                        },
                        {
                          key: msg(
                            "auto.features.optimizations.components.gridoverview.literal.24",
                          ),
                          color: "var(--color-chart-4)",
                        },
                      ].map(({ key, color }) => {
                        const isHidden = hiddenCombinedSeries.has(key);
                        return (
                          <button
                            key={key}
                            type="button"
                            onClick={() => toggleCombinedSeries(key)}
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
              </div>

              {pairRespTimeFiltered.length > 0 && (
                <Card className="mt-4">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-semibold">
                      <HelpTip text={tip("grid.avg_response_time_per_pair")}>
                        {msg("auto.features.optimizations.components.gridoverview.12")}
                      </HelpTip>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0">
                    {lite ? (
                      <div className="h-[220px] min-w-0">
                        <ChartTable
                          rows={pairRespTimeFiltered}
                          onRowClick={(row) => handleBarClick(row)}
                          columns={[
                            {
                              key: "name",
                              label: msg(
                                "auto.features.optimizations.components.gridoverview.literal.27",
                              ),
                            },
                            {
                              key: "responseTime",
                              label: msg(
                                "auto.features.optimizations.components.gridoverview.literal.28",
                              ),
                              align: "end",
                              format: (value) =>
                                typeof value === "number"
                                  ? `${value}${msg("auto.features.optimizations.components.gridoverview.14")}`
                                  : "—",
                            },
                          ]}
                        />
                      </div>
                    ) : (
                    <div className="h-[220px] min-w-0" dir="ltr">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={pairRespTimeFiltered}
                          layout="vertical"
                          margin={{ left: 28, right: 20, top: 5, bottom: 5 }}
                        >
                          <CartesianGrid
                            horizontal={false}
                            strokeDasharray="3 3"
                            className="stroke-muted"
                          />
                          <XAxis
                            type="number"
                            tickLine={false}
                            axisLine={false}
                            tick={{ fontSize: 10 }}
                            className="fill-muted-foreground"
                            label={{
                              value: msg(
                                "auto.features.optimizations.components.gridoverview.literal.26",
                              ),
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
                            label={{
                              value: msg(
                                "auto.features.optimizations.components.gridoverview.literal.27",
                              ),
                              angle: -90,
                              position: "left",
                              offset: 10,
                              fontSize: 10,
                            }}
                          />
                          <Tooltip
                            content={({ active, payload, label }) => {
                              if (!active || !payload?.length) return null;
                              return (
                                <div className="rounded-xl border border-border/60 bg-background/95 backdrop-blur-sm p-3 shadow-lg">
                                  {label && (
                                    <p className="font-semibold mb-1.5 text-foreground text-xs">
                                      {label}
                                    </p>
                                  )}
                                  <div className="flex items-center gap-2 text-xs">
                                    <span className="text-muted-foreground">
                                      {msg(
                                        "auto.features.optimizations.components.gridoverview.13",
                                      )}
                                    </span>
                                    <span className="font-mono font-semibold text-foreground ms-auto">
                                      {payload[0]!.value}
                                      {msg(
                                        "auto.features.optimizations.components.gridoverview.14",
                                      )}
                                    </span>
                                  </div>
                                </div>
                              );
                            }}
                          />
                          <Bar
                            dataKey="responseTime"
                            name={msg(
                              "auto.features.optimizations.components.gridoverview.literal.28",
                            )}
                            radius={[0, 3, 3, 0]}
                            barSize={14}
                            animationDuration={400}
                            cursor="pointer"
                            onClick={handleBarClick}
                          >
                            {pairRespTimeFiltered.map((entry, i) => (
                              <Cell
                                key={i}
                                fill={entry.isBest ? "#C8A882" : "var(--color-chart-4)"}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    )}
                  </CardContent>
                </Card>
              )}
            </motion.div>
          </AnimatePresence>
        </FadeIn>
      )}

      <div className="space-y-2" data-tutorial="grid-pair-list">
        {visiblePrs.map((pr) => {
          const isOverall = scoring.overallWinner === pr.pair_index;
          const isQuality = scoring.qualityWinner === pr.pair_index;
          const isSpeed = scoring.speedWinner === pr.pair_index;
          const s = scoring.byIndex[pr.pair_index];
          const barRatio = s?.quality ?? 0;
          // A failed pair offers Resume if it crashed mid-run (has a checkpoint),
          // else Restart — one-to-one with a single run.
          const pairResumable = job.grid_resumable_pairs?.includes(pr.pair_index) ?? false;
          const rerunBusy = rerunningPair === pr.pair_index;
          return (
            <div
              key={pr.pair_index}
              className={`group rounded-xl border p-4 transition-all duration-200 cursor-pointer hover:shadow-sm ${
                pr.error
                  ? "border-[#B04030]/30 bg-[#B04030]/[0.02] hover:border-[#B04030]/50"
                  : isOverall
                    ? "border-[#3D2E22]/40 bg-[#3D2E22]/[0.03] hover:border-[#3D2E22]/60"
                    : "border-border/50 bg-card/80 hover:border-border"
              }`}
              onClick={() => onPairSelect(pr.pair_index)}
            >
              <div className="flex items-center gap-3">
                {isOverall && <Crown className="size-4 text-[#C8A882] shrink-0" />}
                {pr.error && <XCircle className="size-4 text-[#B04030] shrink-0" />}

                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <div className="flex items-center gap-1 min-w-0">
                    <span className="font-mono text-xs truncate" title={pr.generation_model}>
                      {pr.generation_model.split("/").pop()}
                    </span>
                    {pr.generation_reasoning_effort && (
                      <ReasoningPill value={pr.generation_reasoning_effort} />
                    )}
                  </div>
                  <span className="text-[0.625rem] text-muted-foreground/50">×</span>
                  <div className="flex items-center gap-1 min-w-0">
                    <span className="font-mono text-xs truncate" title={pr.reflection_model}>
                      {pr.reflection_model.split("/").pop()}
                    </span>
                    {pr.reflection_reasoning_effort && (
                      <ReasoningPill value={pr.reflection_reasoning_effort} />
                    )}
                  </div>
                </div>

                {!pr.error ? (
                  <div className="flex items-center gap-4 shrink-0 tabular-nums font-mono text-xs">
                    <div className="text-center min-w-[44px]">
                      <div className="text-[9px] text-foreground/50 mb-0.5 flex items-center justify-center gap-1">
                        <Trophy className="size-2.5" />
                        {msg("auto.features.optimizations.components.gridoverview.15")}
                      </div>
                      <div className={isQuality ? "font-bold text-[#3D2E22]" : "text-foreground"}>
                        {s ? `${Math.round(s.quality * 100)}%` : "—"}
                      </div>
                    </div>
                    <div className="text-center min-w-[44px]">
                      <div className="text-[9px] text-foreground/50 mb-0.5 flex items-center justify-center gap-1">
                        <Gauge className="size-2.5" />
                        {msg("auto.features.optimizations.components.gridoverview.16")}
                      </div>
                      <div className={isSpeed ? "font-bold text-[#3D2E22]" : "text-foreground"}>
                        {pr.avg_response_time_ms != null
                          ? `${(pr.avg_response_time_ms / 1000).toFixed(1)}s`
                          : "—"}
                      </div>
                    </div>
                  </div>
                ) : (
                  <span
                    className="text-[0.6875rem] text-[#B04030] truncate max-w-[280px]"
                    title={pr.error}
                  >
                    {pr.error}
                  </span>
                )}

                {pr.error && (
                  <TooltipButton
                    tooltip={msg(
                      pairResumable
                        ? "optimization.pair.resume_tooltip"
                        : "optimization.pair.restart_tooltip",
                    )}
                  >
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7 text-muted-foreground/50 hover:text-foreground shrink-0"
                      disabled={rerunBusy}
                      aria-label={msg(
                        pairResumable ? "optimization.pair.resume" : "optimization.pair.restart",
                      )}
                      onClick={(e) => {
                        e.stopPropagation();
                        void handlePairRerun(pr.pair_index, pairResumable);
                      }}
                    >
                      {pairResumable ? (
                        <Play className={`size-3.5 -scale-x-100${rerunBusy ? " animate-spin" : ""}`} />
                      ) : (
                        <RotateCcw className={`size-3.5${rerunBusy ? " animate-spin" : ""}`} />
                      )}
                    </Button>
                  </TooltipButton>
                )}

                <TooltipButton
                  tooltip={msg("auto.features.optimizations.components.gridoverview.18")}
                >
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-7 text-muted-foreground/50 hover:text-red-600 shrink-0"
                    aria-label={msg(
                      "auto.features.optimizations.components.gridoverview.literal.29",
                    )}
                    onClick={(e) => {
                      e.stopPropagation();
                      setPendingDelete(pr);
                    }}
                  >
                    <Trash2 className="size-3.5" />
                  </Button>
                </TooltipButton>

                <ChevronLeft className="size-4 text-muted-foreground/30 group-hover:text-muted-foreground transition-colors shrink-0" />
              </div>

              {!pr.error && (
                <div className="mt-2.5 h-1 rounded-full bg-border/30 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${barRatio * 100}%`,
                      background: isOverall ? "#3D2E22" : "#C8A882",
                      opacity: isOverall ? 0.6 : 0.3,
                    }}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      <Dialog open={pendingDelete !== null} onOpenChange={(o) => !o && setPendingDelete(null)}>
        <DialogContent className="w-[min(28rem,92vw)] max-w-[min(28rem,92vw)] sm:max-w-md">
          <DialogTitleRow
            title={msg("auto.features.optimizations.components.gridoverview.19")}
            description={
              <>
                {msg("auto.features.optimizations.components.gridoverview.20")}{" "}
                <span className="font-mono font-semibold text-foreground break-all">
                  {pendingDelete ? pairLabel(pendingDelete) : ""}
                </span>
                {msg("auto.features.optimizations.components.gridoverview.21")}
              </>
            }
          />
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setPendingDelete(null)}
              disabled={deleting}
              className="w-full justify-center"
            >
              {msg("auto.features.optimizations.components.gridoverview.22")}
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmDelete}
              disabled={deleting}
              className="w-full justify-center"
            >
              {deleting ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                msg("auto.features.optimizations.components.gridoverview.literal.30")
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export const GridOverview = memo(GridOverviewImpl);
