"use client";

import * as React from "react";
import { Wrench } from "@/shared/ui/icons";
import { msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";

import { cn } from "@/shared/lib/utils";

import { TOOL_META, getToolTitle, type ApprovalSeverity } from "../lib/tool-meta";

const SEVERITY: Record<ApprovalSeverity, { color: string; label: string | null }> = perLocale(
  () => ({
    destructive: {
      color: "#9B2C1F",
      label: msg("auto.features.agent.panel.components.toolscarousel.literal.11"),
    },
    warning: {
      color: "#A85A1A",
      label: msg("auto.features.agent.panel.components.toolscarousel.literal.16"),
    },
    info: {
      color: "#3D2E22",
      label: msg("auto.features.agent.panel.components.toolscarousel.literal.12"),
    },
  }),
);

interface ToolHeaderProps {
  /** Tool key — resolves the icon, friendly title and severity from metadata. */
  toolKey: string;
  /**
   * Severity provided by the run's own metadata (e.g. an optimized agent's tool
   * annotations). Used only as a fallback when the tool isn't in the curated
   * catalogue, which is more accurate. Unknown values are ignored, so an
   * uncatalogued tool with no provided severity stays badge-free.
   */
  severity?: string;
  /** Optional end-aligned slot (e.g. a change badge). */
  trailing?: React.ReactNode;
  /** Merged onto the identity row (e.g. ``mb-2.5`` or pinned-header padding). */
  className?: string;
}

// Neutral brand ink for the icon chip when severity is unknown — the same dark
// brown used elsewhere, carrying no safety connotation on its own.
const NEUTRAL_ACCENT = "#3D2E22";

/**
 * The shared tool identity row — a circular icon, the friendly title and (when
 * known) a severity label — resolved from {@link TOOL_META}. Keys absent from
 * the catalogue fall back to a wrench icon, a neutral chip and a prettified
 * title, and show *no* severity label: severity is only asserted when it is
 * real metadata, never fabricated, so an uncatalogued tool from any MCP is
 * never mislabelled "safe". A run-provided ``severity`` fills in for tools the
 * catalogue doesn't know (any optimized agent's tools), but is never invented.
 * Shared by the curated tool tour and the trajectory drawer's tool-description
 * pager so both wear one look.
 */
export function ToolHeader({ toolKey, severity, trailing, className }: ToolHeaderProps) {
  const meta = TOOL_META[toolKey];
  const Icon = meta?.icon ?? Wrench;
  const provided = severity && severity in SEVERITY ? (severity as ApprovalSeverity) : undefined;
  const resolvedSeverity = meta?.severity ?? provided;
  const sev = resolvedSeverity ? SEVERITY[resolvedSeverity] : null;
  const accent = sev?.color ?? NEUTRAL_ACCENT;
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <span
        className="inline-flex size-9 items-center justify-center rounded-full shrink-0"
        style={{
          backgroundColor: `${accent}14`,
          color: accent,
        }}
      >
        <Icon className="size-4" aria-hidden="true" />
      </span>
      <div className="min-w-0 flex-1">
        <div className="text-[0.8125rem] font-medium leading-tight truncate">
          {getToolTitle(toolKey)}
        </div>
        {sev?.label && (
          <div className="mt-0.5 flex items-center gap-1.5">
            <span
              className="inline-block size-1 rounded-full shrink-0"
              style={{ backgroundColor: sev.color, opacity: 0.55 }}
              aria-hidden="true"
            />
            <span className="text-[0.625rem] text-muted-foreground/75">{sev.label}</span>
          </div>
        )}
      </div>
      {trailing}
    </div>
  );
}
