import * as React from "react";
import { formatMsg, msg } from "@/shared/lib/messages";

import type { AgentToolCall } from "@/shared/ui/agent/types";
import { TERMS } from "@/shared/lib/terms";

import { UUID_RE } from "../lib/entry-format";

import { AnalyticsSummaryCard } from "./AnalyticsSummaryCard";
import { JobSummaryCard } from "./JobSummaryCard";
import { SearchResultsCard } from "./SearchResultsCard";
import { SubmitSummaryCard } from "./SubmitSummaryCard";
import { TaggingSessionsCard } from "./TaggingSessionsCard";
import { TestResultsCard } from "./TestResultsCard";
import { WalletCard } from "./WalletCard";

export interface ToolRenderer {
  card?: (call: AgentToolCall) => React.ReactNode;
  summary?: (call: AgentToolCall) => string | null;
}

function getArgs(call: AgentToolCall): Record<string, unknown> {
  const p = (call.payload ?? {}) as Record<string, unknown>;
  const a = p.arguments;
  return a && typeof a === "object" && !Array.isArray(a) ? (a as Record<string, unknown>) : {};
}

function getResult(call: AgentToolCall): unknown {
  const p = (call.payload ?? {}) as Record<string, unknown>;
  return p.result;
}

function shortId(v: unknown): string | undefined {
  if (typeof v !== "string") return undefined;
  return UUID_RE.test(v) ? v.slice(0, 8) : v;
}

function pickId(args: Record<string, unknown>): string | undefined {
  return shortId(args.id ?? args.optimization_id ?? args.job_id ?? args.sample_id);
}

function pickIds(args: Record<string, unknown>): string[] {
  const raw = args.ids ?? args.optimization_ids;
  if (!Array.isArray(raw)) return [];
  return raw.filter((x): x is string => typeof x === "string");
}

function byStatus(
  call: AgentToolCall,
  map: { running: string; done: string; error?: string },
): string {
  if (call.status === "running") return map.running;
  if (call.status === "error") return map.error ?? map.done;
  return map.done;
}

function truncate(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

const RENDERERS: Record<string, ToolRenderer> = {
  delete_job_optimizations: {
    summary: (call) => {
      const id = pickId(getArgs(call));
      return byStatus(call, {
        running: id
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.2", { p1: id })
          : formatMsg("auto.features.agent.panel.lib.tool.renderers.template.3", {
              p1: TERMS.optimization,
            }),
        done: id
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.4", { p1: id })
          : formatMsg("auto.features.agent.panel.lib.tool.renderers.template.5", {
              p1: TERMS.optimization,
            }),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.1"),
      });
    },
  },

  bulk_delete_jobs_optimizations_bulk_delete_post: {
    summary: (call) => {
      const n = pickIds(getArgs(call)).length;
      return byStatus(call, {
        running: n
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.6", {
              p1: n,
              p2: TERMS.optimizationPlural,
            })
          : formatMsg("auto.features.agent.panel.lib.tool.renderers.template.7", {
              p1: TERMS.optimizationPlural,
            }),
        done: n
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.8", {
              p1: n,
              p2: TERMS.optimizationPlural,
            })
          : formatMsg("auto.features.agent.panel.lib.tool.renderers.template.9", {
              p1: TERMS.optimizationPlural,
            }),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.2"),
      });
    },
  },

  rename_job_optimizations: {
    summary: (call) => {
      const args = getArgs(call);
      const raw =
        typeof args.new_name === "string"
          ? args.new_name
          : typeof args.name === "string"
            ? args.name
            : undefined;
      const name = raw ? truncate(raw, 26) : undefined;
      return byStatus(call, {
        running: name
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.10", { p1: name })
          : msg("auto.features.agent.panel.lib.tool.renderers.literal.3"),
        done: name
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.11", { p1: name })
          : msg("auto.features.agent.panel.lib.tool.renderers.literal.4"),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.5"),
      });
    },
  },

  toggle_pin_job_optimizations: {
    summary: (call) => {
      const args = getArgs(call);
      const pin = Boolean(args.pinned ?? args.value);
      return byStatus(call, {
        running: pin
          ? msg("auto.features.agent.panel.lib.tool.renderers.literal.6")
          : msg("auto.features.agent.panel.lib.tool.renderers.literal.7"),
        done: pin
          ? msg("auto.features.agent.panel.lib.tool.renderers.literal.8")
          : msg("auto.features.agent.panel.lib.tool.renderers.literal.9"),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.10"),
      });
    },
  },

  bulk_pin_jobs_optimizations_bulk_pin_post: {
    summary: (call) => {
      const args = getArgs(call);
      const n = pickIds(args).length;
      const pin = Boolean(args.pinned ?? args.value);
      return byStatus(call, {
        running: pin
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.12", { p1: n })
          : formatMsg("auto.features.agent.panel.lib.tool.renderers.template.13", { p1: n }),
        done: pin
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.14", { p1: n })
          : formatMsg("auto.features.agent.panel.lib.tool.renderers.template.15", { p1: n }),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.16"),
      });
    },
  },

  cancel_job_optimizations: {
    summary: (call) => {
      const id = pickId(getArgs(call));
      return byStatus(call, {
        running: msg("auto.features.agent.panel.lib.tool.renderers.literal.18"),
        done: id
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.20", { p1: id })
          : msg("auto.features.agent.panel.lib.tool.renderers.literal.19"),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.20"),
      });
    },
  },

  bulk_cancel_jobs_optimizations_bulk_cancel_post: {
    summary: (call) => {
      const n = pickIds(getArgs(call)).length;
      return byStatus(call, {
        running: n
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.39", { p1: n })
          : msg("auto.features.agent.panel.lib.tool.renderers.literal.18"),
        done: n
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.40", { p1: n })
          : msg("auto.features.agent.panel.lib.tool.renderers.literal.19"),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.63"),
      });
    },
  },

  clone_job_optimizations: {
    summary: (call) => {
      const args = getArgs(call);
      const n = typeof args.count === "number" && args.count > 0 ? args.count : 1;
      return byStatus(call, {
        running:
          n > 1
            ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.21", { p1: n })
            : msg("auto.features.agent.panel.lib.tool.renderers.literal.21"),
        done:
          n > 1
            ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.22", { p1: n })
            : formatMsg("auto.features.agent.panel.lib.tool.renderers.template.23", {
                p1: TERMS.optimization,
              }),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.22"),
      });
    },
  },

  retry_job_optimizations: {
    summary: (call) =>
      byStatus(call, {
        running: msg("auto.features.agent.panel.lib.tool.renderers.literal.23"),
        done: msg("auto.features.agent.panel.lib.tool.renderers.literal.24"),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.25"),
      }),
  },

  submit_job_run_post: {
    card: (call) => <SubmitSummaryCard call={call} />,
    summary: (call) => {
      const args = getArgs(call);
      const raw = typeof args.job_name === "string" ? args.job_name : undefined;
      const name = raw ? truncate(raw, 24) : undefined;
      return byStatus(call, {
        running: name
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.24", { p1: name })
          : formatMsg("auto.features.agent.panel.lib.tool.renderers.template.25", {
              p1: TERMS.optimizationTypeRun,
            }),
        done: name
          ? formatMsg("auto.features.agent.panel.lib.tool.renderers.template.26", {
              p1: TERMS.optimizationTypeRun,
              p2: name,
            })
          : formatMsg("auto.features.agent.panel.lib.tool.renderers.template.27", {
              p1: TERMS.optimizationTypeRun,
            }),
        error: formatMsg("auto.features.agent.panel.lib.tool.renderers.template.28", {
          p1: TERMS.optimizationTypeRun,
        }),
      });
    },
  },

  submit_grid_search_grid_search_post: {
    card: (call) => <SubmitSummaryCard call={call} />,
    summary: (call) =>
      byStatus(call, {
        running: formatMsg("auto.features.agent.panel.lib.tool.renderers.template.29", {
          p1: TERMS.optimizationTypeGrid,
        }),
        done: formatMsg("auto.features.agent.panel.lib.tool.renderers.template.30", {
          p1: TERMS.optimizationTypeGrid,
        }),
        error: formatMsg("auto.features.agent.panel.lib.tool.renderers.template.31", {
          p1: TERMS.optimizationTypeGrid,
        }),
      }),
  },

  update_user_preferences: {
    summary: (call) =>
      byStatus(call, {
        running: msg("settings.agent.settings_tool.updating"),
        done: msg("settings.agent.settings_tool.updated"),
        error: msg("settings.agent.settings_tool.update_failed"),
      }),
  },

  list_jobs_optimizations_get: {
    summary: (call) => {
      if (call.status === "running")
        return formatMsg("auto.features.agent.panel.lib.tool.renderers.template.32", {
          p1: TERMS.optimizationPlural,
        });
      if (call.status === "error")
        return msg("auto.features.agent.panel.lib.tool.renderers.literal.26");
      const result = getResult(call);
      let count: number | null = null;
      if (Array.isArray(result)) count = result.length;
      else if (result && typeof result === "object") {
        const r = result as Record<string, unknown>;
        const items = r.items ?? r.jobs ?? r.optimizations;
        if (Array.isArray(items)) count = items.length;
      }
      return count !== null
        ? `${count} ${TERMS.optimizationPlural}`
        : msg("auto.features.agent.panel.lib.tool.renderers.literal.27");
    },
  },

  validate_code_validate_code_post: {
    summary: (call) => {
      if (call.status === "running")
        return msg("auto.features.agent.panel.lib.tool.renderers.literal.43");
      const result = getResult(call);
      if (result && typeof result === "object") {
        const valid = (result as Record<string, unknown>).valid;
        if (valid === true) return msg("auto.features.agent.panel.lib.tool.renderers.literal.44");
        if (valid === false) return msg("auto.features.agent.panel.lib.tool.renderers.literal.45");
      }
      return byStatus(call, {
        running: msg("auto.features.agent.panel.lib.tool.renderers.literal.46"),
        done: msg("auto.features.agent.panel.lib.tool.renderers.literal.47"),
      });
    },
  },

  edit_code_optimizations_edit_code_post: {
    summary: (call) =>
      byStatus(call, {
        running: msg("auto.features.agent.panel.lib.tool.renderers.literal.48"),
        done: msg("auto.features.agent.panel.lib.tool.renderers.literal.49"),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.50"),
      }),
  },

  profile_datasets_profile_post: {
    summary: (call) =>
      byStatus(call, {
        running: formatMsg("auto.features.agent.panel.lib.tool.renderers.template.36", {
          p1: TERMS.dataset,
        }),
        done: msg("auto.features.agent.panel.lib.tool.renderers.literal.51"),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.52"),
      }),
  },

  discover_models_models_discover_post: {
    summary: (call) =>
      byStatus(call, {
        running: msg("auto.features.agent.panel.lib.tool.renderers.literal.53"),
        done: msg("auto.features.agent.panel.lib.tool.renderers.literal.54"),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.55"),
      }),
  },

  set_column_roles_datasets_column_roles_post: {
    summary: (call) =>
      byStatus(call, {
        running: msg("auto.features.agent.panel.lib.tool.renderers.literal.57"),
        done: msg("auto.features.agent.panel.lib.tool.renderers.literal.58"),
        error: msg("auto.features.agent.panel.lib.tool.renderers.literal.59"),
      }),
  },

  public_search_dashboard_search_post: {
    card: (call) => <SearchResultsCard call={call} />,
  },

  get_job_summary_optimizations: {
    card: (call) => <JobSummaryCard call={call} />,
  },

  get_analytics_summary_analytics_summary_get: {
    card: (call) => <AnalyticsSummaryCard call={call} />,
  },

  get_test_results_optimizations: {
    card: (call) => <TestResultsCard call={call} />,
  },

  get_wallet_for_agent: {
    card: (call) => <WalletCard call={call} />,
  },

  list_tagging_sessions_for_agent: {
    card: (call) => <TaggingSessionsCard call={call} />,
  },

};

export function getToolRenderer(tool: string): ToolRenderer | undefined {
  return RENDERERS[tool];
}
