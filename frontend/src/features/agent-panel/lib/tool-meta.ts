import {
  BarChart3,
  Check,
  CheckCircle2,
  Code2,
  Copy,
  Database,
  FileSearch,
  GitCompare,
  ListChecks,
  Pencil,
  Pin,
  Play,
  RefreshCw,
  ScanSearch,
  ScrollText,
  Search,
  Sparkles,
  Square,
  Tags,
  Trash2,
  Wand2,
  type LucideIcon,
} from "lucide-react";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";

export type ApprovalSeverity = "destructive" | "warning" | "info";

export interface ToolMeta {
  title: string;
  description: string;
  confirmLabel: string;
  severity: ApprovalSeverity;
  icon: LucideIcon;
}

// Locale strings are stored as thunks, not resolved values: `msg()` reads the
// active locale's catalog, which is delivered out of band and is empty at
// module-eval time on the server (so a bare `msg()` freezes to the raw key,
// process-wide) yet populated in the browser — resolving eagerly here would
// hydrate-mismatch. The accessors below call the thunks per request, where the
// catalog is pinned. `severity`/`icon` are locale-independent and stay concrete.
type LocaleString = () => string;

interface ToolMetaDef {
  title: LocaleString;
  description: LocaleString;
  confirmLabel: LocaleString;
  severity: ApprovalSeverity;
  icon: LucideIcon;
}

export const TOOL_META: Record<string, ToolMetaDef> = {
  delete_job_optimizations: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.1", {
      p1: TERMS.optimization,
    }),
    description: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.2", {
      p1: TERMS.optimization,
    }),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.1"),
    severity: "destructive",
    icon: Trash2,
  },
  bulk_delete_jobs_optimizations_bulk_delete_post: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.3", {
      p1: TERMS.optimizationPlural,
    }),
    description: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.4", {
      p1: TERMS.optimizationPlural,
    }),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.2"),
    severity: "destructive",
    icon: Trash2,
  },
  cancel_job_optimizations: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.5", {
      p1: TERMS.optimization,
    }),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.6"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.7"),
    severity: "warning",
    icon: Square,
  },
  bulk_cancel_jobs_optimizations_bulk_cancel_post: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.26", {
      p1: TERMS.optimizationPlural,
    }),
    description: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.27", {
      p1: TERMS.optimizationPlural,
    }),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.69"),
    severity: "warning",
    icon: Square,
  },
  submit_job_run_post: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.6", {
      p1: TERMS.optimization,
    }),
    description: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.7", {
      p1: TERMS.optimizationTypeRun,
    }),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.8"),
    severity: "warning",
    icon: Play,
  },
  submit_grid_search_grid_search_post: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.8", {
      p1: TERMS.optimizationTypeGrid,
    }),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.template.9"),
    confirmLabel: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.10", {
      p1: TERMS.optimizationTypeGrid,
    }),
    severity: "warning",
    icon: Play,
  },
  rename_job_optimizations: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.11", {
      p1: TERMS.optimization,
    }),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.9"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.10"),
    severity: "info",
    icon: Pencil,
  },
  toggle_pin_job_optimizations: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.11"),
    description: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.12", {
      p1: TERMS.optimization,
    }),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.12"),
    severity: "info",
    icon: Pin,
  },
  edit_code_optimizations_edit_code_post: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.18"),
    description: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.14", {
      p1: TERMS.signature,
      p2: TERMS.metric,
    }),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.19"),
    severity: "info",
    icon: Code2,
  },
  validate_code_validate_code_post: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.20"),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.21"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.22"),
    severity: "info",
    icon: CheckCircle2,
  },
  profile_datasets_profile_post: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.15", { p1: TERMS.dataset }),
    description: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.16", {
      p1: TERMS.dataset,
    }),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.23"),
    severity: "info",
    icon: FileSearch,
  },
  discover_models_models_discover_post: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.24"),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.25"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.26"),
    severity: "info",
    icon: Search,
  },
  clone_job_optimizations: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.17", {
      p1: TERMS.optimization,
    }),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.30"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.31"),
    severity: "warning",
    icon: Copy,
  },
  retry_job_optimizations: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.18", {
      p1: TERMS.optimization,
    }),
    description: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.19", {
      p1: TERMS.optimizationTypeRun,
    }),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.32"),
    severity: "warning",
    icon: RefreshCw,
  },
  compare_jobs_optimizations_compare_post: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.20", {
      p1: TERMS.optimizationPlural,
    }),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.33"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.34"),
    severity: "info",
    icon: GitCompare,
  },
  bulk_pin_jobs_optimizations_bulk_pin_post: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.21", {
      p1: TERMS.optimizationPlural,
    }),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.35"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.36"),
    severity: "info",
    icon: Pin,
  },
  set_column_roles_datasets_column_roles_post: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.46"),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.47"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.48"),
    severity: "info",
    icon: Tags,
  },
  list_jobs_optimizations_get: {
    title: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.25", {
      p1: TERMS.optimizationPlural,
    }),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.49"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.50"),
    severity: "info",
    icon: FileSearch,
  },
  update_wizard_state: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.54"),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.55"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.56"),
    severity: "info",
    icon: Wand2,
  },
  public_search_dashboard_search_post: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.57"),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.58"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.59"),
    severity: "info",
    icon: ScanSearch,
  },
  get_test_results_optimizations: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.60"),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.61"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.62"),
    severity: "info",
    icon: ListChecks,
  },
  get_job_logs_optimizations: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.63"),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.64"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.65"),
    severity: "info",
    icon: ScrollText,
  },
  get_analytics_summary_analytics_summary_get: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.66"),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.67"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.68"),
    severity: "info",
    icon: BarChart3,
  },
  request_user_inference: {
    title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.70"),
    description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.71"),
    confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.72"),
    severity: "info",
    icon: Sparkles,
  },
};

// Read-only / lookup tools (no TOOL_META entry, no approval card) fall through
// here. The icon is the "done" glyph rendered by ``StatusGlyph`` in
// ``ToolCallRow``; ``running`` shows a pulse and ``error`` shows its own
// triangle, so the default must communicate "completed successfully" — a plain
// check, not a warning triangle which used to mis-render every finished
// read-only call as if it were a danger pill (see PER-?? screenshots).
const DEFAULT_META: ToolMetaDef = {
  title: () => msg("auto.features.agent.panel.lib.tool.meta.literal.51"),
  description: () => msg("auto.features.agent.panel.lib.tool.meta.literal.52"),
  confirmLabel: () => msg("auto.features.agent.panel.lib.tool.meta.literal.53"),
  severity: "warning",
  icon: Check,
};

// Hebrew display labels for tools that aren't in TOOL_META — read-only
// discovery / lookup tools that never trigger an approval card, so they
// don't need icon/severity/description. Keeps tool rows from falling back
// to an English-looking prettified snake_case (e.g. "list models for agent").
const TOOL_TITLES: Record<string, LocaleString> = {
  list_models_for_agent: () => formatMsg("auto.features.agent.panel.lib.tool.meta.template.28", {
    p1: TERMS.modelPlural,
  }),
  get_registry_snapshot_registry_get: () => msg("auto.features.agent.panel.lib.tool.meta.literal.73"),
  get_optimization_counts_optimizations_counts_get: () => formatMsg(
    "auto.features.agent.panel.lib.tool.meta.template.29",
    { p1: TERMS.optimizationPlural },
  ),
  get_job_summary_optimizations: () => formatMsg(
    "auto.features.agent.panel.lib.tool.meta.template.30",
    { p1: TERMS.optimization },
  ),
  get_optimizer_stats_analytics_optimizers_get: () => formatMsg(
    "auto.features.agent.panel.lib.tool.meta.template.31",
    { p1: TERMS.optimizer },
  ),
  get_model_stats_analytics_models_get: () => formatMsg(
    "auto.features.agent.panel.lib.tool.meta.template.31",
    { p1: TERMS.modelPlural },
  ),
  serve_info_serve: () => msg("auto.features.agent.panel.lib.tool.meta.literal.74"),
  serve_pair_info_serve: () => msg("auto.features.agent.panel.lib.tool.meta.literal.75"),
  request_user_dataset_datasets_request_upload_post: () => msg(
    "auto.features.agent.panel.lib.tool.meta.literal.76",
  ),
  request_user_dataset_from_library: () =>
    msg("auto.features.agent.panel.components.librarydatasetcard.title"),
  get_grid_search_result_optimizations: () => msg("auto.features.agent.panel.lib.tool.meta.literal.77"),
  get_pair_test_results_optimizations: () => msg("auto.features.agent.panel.lib.tool.meta.literal.78"),
  get_wallet_for_agent: () => msg("auto.features.agent.panel.components.walletcard.title"),
  list_tagging_sessions_for_agent: () =>
    msg("auto.features.agent.panel.components.taggingsessionscard.title"),
};

function prettifyToolName(tool: string): string {
  return tool
    .replace(/_(post|get|put|delete|patch)$/i, "")
    .replace(/_/g, " ")
    .trim();
}

/** Resolve a def's locale thunks into concrete strings for the active request. */
function resolveMeta(def: ToolMetaDef): ToolMeta {
  return {
    title: def.title(),
    description: def.description(),
    confirmLabel: def.confirmLabel(),
    severity: def.severity,
    icon: def.icon,
  };
}

export function getToolMeta(tool: string): ToolMeta {
  return resolveMeta(TOOL_META[tool] ?? DEFAULT_META);
}

export function getToolTitle(tool: string): string {
  return TOOL_META[tool]?.title() ?? TOOL_TITLES[tool]?.() ?? prettifyToolName(tool);
}
