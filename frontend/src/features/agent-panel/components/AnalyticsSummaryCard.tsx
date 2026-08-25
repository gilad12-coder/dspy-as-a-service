"use client";

import * as React from "react";
import { formatMsg, msg } from "@/shared/lib/messages";

import { TERMS } from "@/shared/lib/terms";
import { formatElapsed, formatImprovement, formatPercent } from "@/shared/lib/formatters";
import { getStatusLabel } from "@/shared/constants/job-status";
import { StatusBadge } from "@/shared/ui/status-badge";

import type { AgentToolCall } from "@/shared/ui/agent/types";

import { ToolCallRow } from "./ToolCallRow";
import { StatTile } from "./result-card-atoms";

interface AnalyticsSummaryCardProps {
  call: AgentToolCall;
}

interface AnalyticsResult {
  total_jobs?: number;
  success_count?: number;
  failed_count?: number;
  cancelled_count?: number;
  pending_count?: number;
  running_count?: number;
  success_rate?: number;
  avg_improvement?: number | null;
  avg_runtime?: number | null;
  total_dataset_rows?: number;
  total_pairs?: number;
  truncated?: boolean;
}

function extractResult(call: AgentToolCall): AnalyticsResult | null {
  const payload = (call.payload ?? {}) as Record<string, unknown>;
  const result = payload.result;
  if (!result || typeof result !== "object" || Array.isArray(result)) return null;
  const r = result as Record<string, unknown>;
  if (typeof r.total_jobs !== "number") return null;
  return r as AnalyticsResult;
}

/** ``success_rate`` arrives as a 0–1 fraction; the score scale is 0–100. */
function ratePercent(rate: number | undefined): string {
  return rate == null ? "—" : formatPercent(rate * 100);
}

function buildSummary(data: AnalyticsResult | null, isRunning: boolean): string | null {
  if (isRunning || !data) return null;
  return formatMsg("auto.features.agent.panel.components.analyticssummarycard.summary", {
    p1: data.total_jobs ?? 0,
    p2: TERMS.optimizationPlural,
    p3: ratePercent(data.success_rate),
  });
}

const DISTRIBUTION: Array<{ status: string; key: keyof AnalyticsResult }> = [
  { status: "success", key: "success_count" },
  { status: "failed", key: "failed_count" },
  { status: "running", key: "running_count" },
  { status: "cancelled", key: "cancelled_count" },
  { status: "pending", key: "pending_count" },
];

/**
 * Result card for ``get_analytics_summary`` — renders the KPI rollup as four
 * stat tiles (total, success rate, avg improvement, avg runtime) plus a status
 * distribution strip, instead of a flat 16-key dump.
 */
export function AnalyticsSummaryCard({ call }: AnalyticsSummaryCardProps) {
  const data = extractResult(call);
  const summary = buildSummary(data, call.status === "running");

  if (!data) {
    return <ToolCallRow call={call} summary={summary} />;
  }

  const improvement = data.avg_improvement;
  const improvementColor =
    improvement == null
      ? undefined
      : improvement > 0
        ? "var(--success)"
        : improvement < 0
          ? "var(--danger)"
          : undefined;

  const customBody = (
    <div className="space-y-3">
      <dl className="grid grid-cols-2 gap-x-3 gap-y-2.5">
        <StatTile
          label={formatMsg("auto.features.agent.panel.components.analyticssummarycard.total", {
            p1: TERMS.optimizationPlural,
          })}
          value={data.total_jobs ?? 0}
          valueDir="ltr"
        />
        <StatTile
          label={msg("auto.features.agent.panel.components.analyticssummarycard.success_rate")}
          value={ratePercent(data.success_rate)}
          valueDir="ltr"
        />
        <StatTile
          label={msg("auto.features.agent.panel.components.analyticssummarycard.avg_improvement")}
          value={
            improvement == null ? null : (
              <span style={{ color: improvementColor }}>{formatImprovement(improvement)}</span>
            )
          }
          valueDir="ltr"
        />
        <StatTile
          label={msg("auto.features.agent.panel.components.analyticssummarycard.avg_runtime")}
          value={data.avg_runtime == null ? null : formatElapsed(data.avg_runtime)}
          valueDir="ltr"
        />
      </dl>

      <div className="flex flex-wrap items-center gap-1.5">
        {DISTRIBUTION.map(({ status, key }) => {
          const n = (data[key] as number | undefined) ?? 0;
          if (n <= 0) return null;
          return (
            <span
              key={status}
              className="inline-flex items-center gap-1"
              title={getStatusLabel(status)}
            >
              <StatusBadge status={status} compact />
              <span className="font-mono text-[0.6875rem] tabular-nums text-foreground/70">{n}</span>
            </span>
          );
        })}
      </div>

      {data.truncated && (
        <p className="text-[0.625rem] italic text-muted-foreground/60">
          {msg("auto.features.agent.panel.components.analyticssummarycard.sampled")}
        </p>
      )}
    </div>
  );

  return <ToolCallRow call={call} summary={summary} customBody={customBody} />;
}
