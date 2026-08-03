"use client";

import * as React from "react";
import { WarningCircle } from "@/shared/ui/icons";
import { formatMsg, msg } from "@/shared/lib/messages";

import { cn } from "@/shared/lib/utils";
import { TERMS } from "@/shared/lib/terms";
import { formatPercent, modelDisplayName, moduleLabel } from "@/shared/lib/formatters";
import { StatusBadge } from "@/shared/ui/status-badge";
import { ExportTableMenu } from "@/shared/ui/export-table-menu";

import type { AgentToolCall } from "@/shared/ui/agent/types";

import { ToolCallRow } from "./ToolCallRow";
import { GainPill, TypeBadge } from "./result-card-atoms";

interface CompareJobsCardProps {
  call: AgentToolCall;
}

interface CompareSnapshot {
  optimization_id: string;
  status: string;
  name?: string | null;
  optimization_type?: string | null;
  module_name?: string | null;
  optimizer_name?: string | null;
  model_name?: string | null;
  dataset_rows?: number | null;
  baseline_test_metric?: number | null;
  optimized_test_metric?: number | null;
  metric_improvement?: number | null;
}

interface CompareResult {
  jobs?: CompareSnapshot[];
  differing_fields?: string[];
  missing_optimization_ids?: string[];
}

function extractResult(call: AgentToolCall): CompareResult | null {
  const payload = (call.payload ?? {}) as Record<string, unknown>;
  const result = payload.result;
  if (!result || typeof result !== "object" || Array.isArray(result)) return null;
  const r = result as CompareResult;
  if (!Array.isArray(r.jobs)) return null;
  return r;
}

interface FieldRow {
  /** Matches a ``differing_fields`` key so the row highlights when it differs. */
  key: string;
  label: () => string;
  render: (job: CompareSnapshot) => React.ReactNode;
}

function scoreCell(job: CompareSnapshot): React.ReactNode {
  const { baseline_test_metric: base, optimized_test_metric: opt } = job;
  if (base != null && opt != null) return <GainPill baseline={base} optimized={opt} />;
  if (opt != null) return <span dir="ltr">{formatPercent(opt)}</span>;
  return "—";
}

const FIELD_ROWS: FieldRow[] = [
  {
    key: "status",
    label: () => msg("auto.features.agent.panel.components.comparejobscard.status"),
    render: (j) => <StatusBadge status={j.status} compact />,
  },
  {
    key: "optimization_type",
    label: () => msg("auto.features.agent.panel.components.comparejobscard.type"),
    render: (j) => <TypeBadge type={j.optimization_type ?? "run"} />,
  },
  {
    key: "module_name",
    label: () => msg("auto.features.agent.panel.components.resultcards.module"),
    render: (j) => (j.module_name ? moduleLabel(j.module_name) : "—"),
  },
  {
    key: "optimizer_name",
    label: () => TERMS.optimizer,
    render: (j) => j.optimizer_name ?? "—",
  },
  {
    key: "model_name",
    label: () => TERMS.model,
    render: (j) => (
      <span dir="ltr" title={j.model_name ?? undefined}>
        {j.model_name ? modelDisplayName(j.model_name) : "—"}
      </span>
    ),
  },
  {
    key: "dataset_rows",
    label: () => msg("auto.features.agent.panel.components.resultcards.rows"),
    render: (j) => <span dir="ltr">{j.dataset_rows ?? "—"}</span>,
  },
  {
    key: "score",
    label: () => TERMS.score,
    render: scoreCell,
  },
];

/**
 * Result card for ``compare_jobs`` — lays the snapshots out as a table with one
 * column per job and one row per field, tinting the rows the backend flagged in
 * ``differing_fields`` so the actual differences pop.
 */
export function CompareJobsCard({ call }: CompareJobsCardProps) {
  const data = extractResult(call);
  const jobs = data?.jobs ?? [];
  const summary =
    call.status === "running" || !data
      ? null
      : `${jobs.length} ${TERMS.optimizationPlural}`;

  if (!data || jobs.length === 0) {
    return <ToolCallRow call={call} summary={summary} />;
  }

  const differing = new Set(data.differing_fields ?? []);
  const missing = data.missing_optimization_ids ?? [];

  const customBody = (
    <div className="space-y-2">
      <div className="flex justify-end">
        <ExportTableMenu
          iconOnly
          getData={() => ({
            columns: [
              "name",
              "status",
              "optimization_type",
              "module_name",
              "optimizer_name",
              "model_name",
              "dataset_rows",
              "baseline_test_metric",
              "optimized_test_metric",
              "metric_improvement",
            ],
            rows: jobs.map((j) => ({
              name: j.name?.trim() || j.optimization_id.slice(0, 6),
              status: j.status,
              optimization_type: j.optimization_type ?? null,
              module_name: j.module_name ?? null,
              optimizer_name: j.optimizer_name ?? null,
              model_name: j.model_name ?? null,
              dataset_rows: j.dataset_rows ?? null,
              baseline_test_metric: j.baseline_test_metric ?? null,
              optimized_test_metric: j.optimized_test_metric ?? null,
              metric_improvement: j.metric_improvement ?? null,
            })),
            filename: "compare-jobs",
          })}
        />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[0.6875rem]">
          <thead>
            <tr>
              <th className="w-px" />
              {jobs.map((j) => (
                <th
                  key={j.optimization_id}
                  dir="auto"
                  title={j.name ?? j.optimization_id}
                  className="max-w-[9rem] truncate px-2 py-1 text-start font-medium text-foreground/85"
                >
                  {j.name?.trim() || j.optimization_id.slice(0, 6)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {FIELD_ROWS.map((row) => {
              const highlight = differing.has(row.key);
              return (
                <tr key={row.key} className={cn(highlight && "bg-foreground/[0.05]")}>
                  <td className="whitespace-nowrap py-1 pe-2 text-[0.625rem] uppercase tracking-wide text-muted-foreground/60">
                    {row.label()}
                  </td>
                  {jobs.map((j) => (
                    <td
                      key={j.optimization_id}
                      dir="auto"
                      className="max-w-[9rem] truncate px-2 py-1 align-middle text-foreground/85"
                    >
                      {row.render(j)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {missing.length > 0 && (
        <div className="flex items-center gap-1.5 text-[0.625rem] text-[#9B2C1F]">
          <WarningCircle className="size-3 shrink-0" aria-hidden="true" />
          <span>
            {formatMsg("auto.features.agent.panel.components.comparejobscard.missing", {
              p1: missing.length,
            })}
          </span>
        </div>
      )}
    </div>
  );

  return <ToolCallRow call={call} summary={summary} customBody={customBody} />;
}
