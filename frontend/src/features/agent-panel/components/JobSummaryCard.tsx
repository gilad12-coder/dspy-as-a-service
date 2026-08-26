"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowUpRight } from "@/shared/ui/icons";
import { msg } from "@/shared/lib/messages";

import { TERMS } from "@/shared/lib/terms";
import { formatPercent, modelDisplayName, moduleLabel } from "@/shared/lib/formatters";
import { getStatusLabel } from "@/shared/constants/job-status";
import { StatusBadge } from "@/shared/ui/status-badge";
import type { OptimizationSummaryResponse } from "@/shared/types/api";

import type { AgentToolCall } from "@/shared/ui/agent/types";

import { ToolCallRow } from "./ToolCallRow";
import { GainPill, StatTile, TypeBadge, computeGain } from "./result-card-atoms";

interface JobSummaryCardProps {
  call: AgentToolCall;
}

/** A parsed summary is any object that carries the two always-present fields. */
function extractJob(call: AgentToolCall): OptimizationSummaryResponse | null {
  const payload = (call.payload ?? {}) as Record<string, unknown>;
  const result = payload.result;
  if (!result || typeof result !== "object" || Array.isArray(result)) return null;
  const r = result as Record<string, unknown>;
  if (typeof r.optimization_id !== "string" || typeof r.status !== "string") return null;
  return r as unknown as OptimizationSummaryResponse;
}

function buildSummary(job: OptimizationSummaryResponse | null, isRunning: boolean): string | null {
  if (isRunning) return null;
  if (!job) return null;
  const title = job.name?.trim() || job.optimization_id.slice(0, 8);
  const parts = [title, getStatusLabel(job.status)];
  const gain = computeGain(job.baseline_test_metric, job.optimized_test_metric);
  if (gain && gain.kind !== "neutral") parts.push(gain.text);
  return parts.join(" · ");
}

/**
 * Result card for ``get_job_summary`` — turns the ~50-field
 * ``OptimizationSummaryResponse`` blob into a scannable header (name, status,
 * type), the baseline→optimized score, a small config grid, and a link to the
 * job. Falls back to the generic row when the payload can't be parsed.
 */
export function JobSummaryCard({ call }: JobSummaryCardProps) {
  const job = extractJob(call);
  const summary = buildSummary(job, call.status === "running");

  if (!job) {
    return <ToolCallRow call={call} summary={summary} />;
  }

  const hasScore = job.baseline_test_metric != null || job.optimized_test_metric != null;

  const customBody = (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <span
          dir="auto"
          className="min-w-0 flex-1 truncate text-[0.8125rem] font-medium text-foreground/90"
        >
          {job.name?.trim() || job.optimization_id.slice(0, 8)}
        </span>
        <StatusBadge status={job.status} compact />
        <TypeBadge type={job.optimization_type} />
      </div>

      {hasScore && (
        <div className="flex items-center gap-1.5">
          <span className="text-[0.625rem] uppercase tracking-wide text-muted-foreground/60">
            {TERMS.score}
          </span>
          {job.optimized_test_metric != null ? (
            <span dir="ltr" className="text-[0.8125rem] font-medium tabular-nums text-foreground/90">
              {formatPercent(job.optimized_test_metric)}
            </span>
          ) : (
            <span dir="ltr" className="text-[0.8125rem] tabular-nums text-muted-foreground">
              {formatPercent(job.baseline_test_metric)}
            </span>
          )}
          <GainPill baseline={job.baseline_test_metric} optimized={job.optimized_test_metric} />
        </div>
      )}

      <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5">
        <StatTile
          label={msg("auto.features.agent.panel.components.resultcards.module")}
          value={job.module_name ? moduleLabel(job.module_name) : null}
          valueDir="auto"
        />
        <StatTile label={TERMS.optimizer} value={job.optimizer_name} valueDir="auto" />
        <StatTile
          label={TERMS.model}
          value={job.model_name ? modelDisplayName(job.model_name) : null}
          valueDir="ltr"
        />
        <StatTile
          label={msg("auto.features.agent.panel.components.resultcards.rows")}
          value={job.dataset_rows ?? null}
          valueDir="ltr"
        />
      </dl>

      {job.summary_text?.trim() && (
        <p dir="auto" className="line-clamp-3 text-[0.6875rem] leading-snug text-foreground/65">
          {job.summary_text.trim()}
        </p>
      )}

      <Link
        href={`/optimizations/${job.optimization_id}`}
        className="inline-flex items-center gap-1 text-[0.6875rem] font-medium text-foreground/70 transition-colors hover:text-foreground"
      >
        {msg("auto.features.agent.panel.components.resultcards.open")}
        <ArrowUpRight className="size-3" aria-hidden="true" />
      </Link>
    </div>
  );

  return <ToolCallRow call={call} summary={summary} customBody={customBody} />;
}
