import type { DashboardAnalytics, DashboardAnalyticsJob } from "@/shared/lib/api";
import { getStatusLabel } from "@/shared/constants/job-status";
import { TERMS } from "@/shared/lib/terms";
import { STATUS_COLORS } from "../constants";
import { msg } from "@/shared/lib/messages";
import { getActiveIntlLocale } from "@/shared/lib/runtime-locale";

export type ChartData = {
  status: Array<{ key: string; name: string; value: number; fill: string }>;
  improvement: Array<{
    name: string;
    optimizedScore: number;
    baselineScore: number;
    delta: number | undefined;
  }>;
  improvementJobIds: string[];
  optimizer: Array<{ name: string; value: number }>;
  kpis: null | {
    successRate: number;
    avgImprovement: number;
    avgRuntime: number;
    totalRows: number;
    successCount: number;
    terminalCount: number;
    totalPairsRun: number;
    gridSearchCount: number;
    singleRunCount: number;
    bestImprovement: number;
  };
  avgByOptimizer: Array<{ name: string; avgImprovement: number; count: number }>;
  modelUsage: Array<{ name: string; count: number }>;
  ownerUsage: Array<{ name: string; count: number }>;
  accessUsage: Array<{ name: string; count: number }>;
  topJobs: DashboardAnalyticsJob[];
  jobTypeData: Array<{ name: string; value: number }>;
  runtimeDistribution: Array<{ name: string; runtimeMinutes: number }>;
  runtimeDistributionJobIds: string[];
  datasetVsImprovement: Array<{ rows: number; improvement: number; name: string }>;
  datasetVsImprovementIds: string[];
  efficiencyData: Array<{ name: string; efficiency: number }>;
  efficiencyJobIds: string[];
  timelineData: Array<{ name: string; [valueKey: string]: string | number }>;
  timelineDates: string[];
};

const EMPTY_CHART_DATA: ChartData = {
  status: [],
  improvement: [],
  improvementJobIds: [],
  optimizer: [],
  kpis: null,
  avgByOptimizer: [],
  modelUsage: [],
  ownerUsage: [],
  accessUsage: [],
  topJobs: [],
  jobTypeData: [],
  runtimeDistribution: [],
  runtimeDistributionJobIds: [],
  datasetVsImprovement: [],
  datasetVsImprovementIds: [],
  efficiencyData: [],
  efficiencyJobIds: [],
  timelineData: [],
  timelineDates: [],
};

const shortId = (id: string) => `${id.slice(0, 8)}…`;

export function transformChartData(analyticsData: DashboardAnalytics | null): ChartData {
  if (!analyticsData) return EMPTY_CHART_DATA;

  const status = Object.entries(analyticsData.status_counts).map(([key, count]) => ({
    key,
    name: getStatusLabel(key),
    value: count,
    fill: STATUS_COLORS[key] ?? "var(--color-chart-5)",
  }));

  const improvement = analyticsData.top_improvement.map((j) => {
    const opt = j.optimized_test_metric ?? 0;
    const bl = j.baseline_test_metric ?? 0;
    return {
      name: shortId(j.optimization_id),
      optimizedScore: Math.round(opt),
      baselineScore: Math.round(bl),
      delta: j.metric_improvement ?? undefined,
    };
  });
  const improvementJobIds = analyticsData.top_improvement.map((j) => j.optimization_id);

  const optimizer = Object.entries(analyticsData.optimizer_counts).map(([name, value]) => ({
    name,
    value,
  }));

  const kpis = {
    successRate: analyticsData.success_rate * 100,
    avgImprovement: analyticsData.avg_improvement ?? 0,
    avgRuntime: analyticsData.avg_runtime_seconds ?? 0,
    totalRows: analyticsData.total_dataset_rows,
    successCount: analyticsData.success_count,
    terminalCount: analyticsData.terminal_count,
    totalPairsRun: analyticsData.total_pairs_run,
    gridSearchCount: analyticsData.grid_search_count,
    singleRunCount: analyticsData.single_run_count,
    bestImprovement: analyticsData.best_improvement ?? 0,
  };

  const avgByOptimizer = analyticsData.improvement_by_optimizer.map((o) => ({
    name: o.name,
    avgImprovement: +o.average.toFixed(1),
    count: o.count,
  }));

  const modelUsage = analyticsData.model_usage.map((m) => ({
    name: m.name,
    count: m.value,
  }));

  const ownerUsage = (analyticsData.owner_usage ?? []).map((o) => ({
    name: o.name,
    count: o.value,
  }));

  const accessUsage = (analyticsData.access_usage ?? []).map((a) => ({
    name: a.name,
    count: a.value,
  }));

  const topJobs = analyticsData.top_jobs_by_improvement;

  const jobTypeData = Object.entries(analyticsData.job_type_counts).map(([key, value]) => ({
    name:
      key === "grid_search"
        ? msg("auto.features.dashboard.lib.transform.chart.data.literal.1")
        : msg("auto.features.dashboard.lib.transform.chart.data.literal.2"),
    value,
  }));

  const runtimeDistribution = analyticsData.runtime_distribution.map((j) => ({
    name: shortId(j.optimization_id),
    runtimeMinutes: +((j.elapsed_seconds ?? 0) / 60).toFixed(1),
  }));
  const runtimeDistributionJobIds = analyticsData.runtime_distribution.map(
    (j) => j.optimization_id,
  );

  const datasetVsImprovement = analyticsData.dataset_vs_improvement.map((j) => ({
    rows: j.dataset_rows ?? 0,
    improvement: +(j.metric_improvement ?? 0).toFixed(1),
    name: shortId(j.optimization_id),
  }));
  const datasetVsImprovementIds = analyticsData.dataset_vs_improvement.map(
    (j) => j.optimization_id,
  );

  const efficiencyData = analyticsData.efficiency.map((j) => {
    const delta = j.metric_improvement ?? 0;
    const elapsed = j.elapsed_seconds ?? 0;
    const efficiency = elapsed > 0 ? +((delta / elapsed) * 60).toFixed(2) : 0;
    return { name: shortId(j.optimization_id), efficiency };
  });
  const efficiencyJobIds = analyticsData.efficiency.map((j) => j.optimization_id);

  const timelineData = analyticsData.timeline.map((t) => ({
    name: new Date(t.date).toLocaleDateString(getActiveIntlLocale(), {
      day: "numeric",
      month: "short",
    }),
    [TERMS.optimizationPlural]: t.count,
  }));
  const timelineDates = analyticsData.timeline.map((t) => t.date);

  return {
    status,
    improvement,
    improvementJobIds,
    optimizer,
    kpis,
    avgByOptimizer,
    modelUsage,
    ownerUsage,
    accessUsage,
    topJobs,
    jobTypeData,
    runtimeDistribution,
    runtimeDistributionJobIds,
    datasetVsImprovement,
    datasetVsImprovementIds,
    efficiencyData,
    efficiencyJobIds,
    timelineData,
    timelineDates,
  };
}
