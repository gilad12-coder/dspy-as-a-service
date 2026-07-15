"use client";

import * as React from "react";
import { formatMsg, msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";

import { cn } from "@/shared/lib/utils";
import { TERMS } from "@/shared/lib/terms";

import { TOOL_META } from "../lib/tool-meta";
import { Carousel } from "./Carousel";
import { ToolHeader } from "./ToolHeader";

interface ToolEntry {
  key: string;
  description: string;
}

// Curated subset of TOOL_META. Bulk variants are folded into their singular
// counterparts so the user sees distinct capabilities, not API permutations.
// Sorted by severity tier (info → warning → destructive), with the most
// distinctive capabilities — auto-fill, semantic search, diagnostics — leading
// the info group so the carousel opens on something memorable rather than a
// rename action.
const FEATURED_TOOLS: readonly string[] = [
  "update_wizard_state",
  "public_search_dashboard_search_post",
  "get_analytics_summary_analytics_summary_get",
  "get_job_logs_optimizations",
  "get_test_results_optimizations",
  "compare_jobs_optimizations_compare_post",
  "list_jobs_optimizations_get",
  "rename_job_optimizations",
  "toggle_pin_job_optimizations",
  "set_column_roles_datasets_column_roles_post",
  "profile_datasets_profile_post",
  "edit_code_optimizations_edit_code_post",
  "validate_code_validate_code_post",
  "discover_models_models_discover_post",
  "submit_job_run_post",
  "submit_grid_search_grid_search_post",
  "request_user_inference",
  "clone_job_optimizations",
  "retry_job_optimizations",
  "cancel_job_optimizations",
  "delete_job_optimizations",
];

// Tour-oriented descriptions (capabilities, not approval warnings). Fallback
// to TOOL_META.description when a key is missing.
const TOUR_DESCRIPTIONS: Record<string, string> = perLocale(() => ({
  submit_job_run_post: formatMsg("auto.features.agent.panel.components.toolscarousel.template.1", {
    p1: TERMS.optimization,
  }),
  submit_grid_search_grid_search_post: msg(
    "auto.features.agent.panel.components.toolscarousel.literal.1",
  ),
  rename_job_optimizations: formatMsg(
    "auto.features.agent.panel.components.toolscarousel.template.2",
    { p1: TERMS.optimization },
  ),
  clone_job_optimizations: msg("auto.features.agent.panel.components.toolscarousel.literal.2"),
  retry_job_optimizations: msg("auto.features.agent.panel.components.toolscarousel.literal.3"),
  cancel_job_optimizations: msg("auto.features.agent.panel.components.toolscarousel.literal.4"),
  delete_job_optimizations: formatMsg(
    "auto.features.agent.panel.components.toolscarousel.template.3",
    { p1: TERMS.optimization },
  ),
  toggle_pin_job_optimizations: formatMsg(
    "auto.features.agent.panel.components.toolscarousel.template.4",
    { p1: TERMS.optimization },
  ),
  compare_jobs_optimizations_compare_post: msg(
    "auto.features.agent.panel.components.toolscarousel.literal.5",
  ),
  list_jobs_optimizations_get: formatMsg(
    "auto.features.agent.panel.components.toolscarousel.template.6",
    { p1: TERMS.optimizationPlural },
  ),
  edit_code_optimizations_edit_code_post: formatMsg(
    "auto.features.agent.panel.components.toolscarousel.template.7",
    { p1: TERMS.signature, p2: TERMS.metric },
  ),
  validate_code_validate_code_post: msg(
    "auto.features.agent.panel.components.toolscarousel.literal.8",
  ),
  profile_datasets_profile_post: formatMsg(
    "auto.features.agent.panel.components.toolscarousel.template.8",
    { p1: TERMS.dataset },
  ),
  set_column_roles_datasets_column_roles_post: msg(
    "auto.features.agent.panel.components.toolscarousel.literal.9",
  ),
  discover_models_models_discover_post: msg(
    "auto.features.agent.panel.components.toolscarousel.literal.10",
  ),
  request_user_inference: msg("auto.features.agent.panel.components.toolscarousel.literal.17"),
}));

interface ToolsCarouselProps {
  /**
   * Tool keys to show. Omit for the curated product tour (the hardcoded
   * {@link FEATURED_TOOLS}); pass a real roster (e.g. a turn's ``allowed_tools``)
   * to drive the same carousel from live data. Keys missing from
   * {@link TOOL_META} fall back to a prettified title, a neutral icon, and
   * ``info`` severity, so any tool name renders.
   */
  tools?: readonly string[];
  /**
   * Per-tool descriptions for *this* run, keyed by tool name — typically the
   * optimized descriptions from the candidate's own ReAct overlay. Used before
   * the static catalog so any optimized agent's tools render with their real
   * descriptions, not just the platform's catalogued ones. Missing keys fall
   * back to {@link TOUR_DESCRIPTIONS}/{@link TOOL_META}.
   */
  descriptions?: Record<string, string>;
  /** Overrides the header label (the tour's first-person default doesn't fit every context). */
  title?: string;
  /** Merged onto the root — pass ``w-full`` to fill an embedding container. */
  className?: string;
}

export function ToolsCarousel({
  tools: toolKeys = FEATURED_TOOLS,
  descriptions,
  title,
  className,
}: ToolsCarouselProps = {}) {
  const tools = React.useMemo<ToolEntry[]>(
    () =>
      toolKeys.map((key) => ({
        key,
        description:
          descriptions?.[key] ?? TOUR_DESCRIPTIONS[key] ?? TOOL_META[key]?.description?.() ?? "",
      })),
    [toolKeys, descriptions],
  );

  return (
    <Carousel
      items={tools}
      itemKey={(t) => t.key}
      renderItem={(t) => <ToolCard tool={t} />}
      title={title ?? msg("auto.features.agent.panel.components.toolscarousel.1")}
      ariaLabel={msg("auto.features.agent.panel.components.toolscarousel.literal.13")}
      bodyClassName="h-[132px]"
      className={cn("w-[min(300px,calc(100vw-2rem))] p-3", className)}
    />
  );
}

function ToolCard({ tool }: { tool: ToolEntry }) {
  return (
    <div className="h-full w-full p-3.5">
      <ToolHeader toolKey={tool.key} className="mb-2.5" />
      {tool.description ? (
        <p className="text-[0.75rem] leading-relaxed text-foreground/75 line-clamp-3">
          {tool.description}
        </p>
      ) : null}
    </div>
  );
}
