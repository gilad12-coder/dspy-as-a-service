"use client";

import * as React from "react";
import { ArrowDown, ArrowUp } from "lucide-react";

import { cn } from "@/shared/lib/utils";
import { TERMS } from "@/shared/lib/terms";
import { Badge } from "@/shared/ui/primitives/badge";

/**
 * Small shared primitives for the agent's read-tool result cards. These render
 * inside a ``ToolCallRow`` body (via ``customBody``), so they follow the same
 * neutral, theme-token palette the row uses — the warm card frame belongs to
 * the row, not to these atoms.
 */

export type Gain = { text: string; kind: "positive" | "negative" | "neutral" };

/**
 * Baseline→optimized delta on the canonical 0–100 scale. Mirrors the explore
 * slice's ``formatGain`` (kept local so the agent panel doesn't reach across a
 * feature boundary): sub-0.05 swings read as neutral, everything else keeps its
 * sign. Returns null when either endpoint is missing.
 */
export function computeGain(
  baseline: number | null | undefined,
  optimized: number | null | undefined,
): Gain | null {
  if (baseline == null || optimized == null) return null;
  if (!Number.isFinite(baseline) || !Number.isFinite(optimized)) return null;
  const gain = optimized - baseline;
  if (Math.abs(gain) < 0.05) return { text: "0.0%", kind: "neutral" };
  if (gain > 0) return { text: `+${gain.toFixed(1)}%`, kind: "positive" };
  return { text: `${gain.toFixed(1)}%`, kind: "negative" };
}

/** One label/value cell in a metric grid; falls back to an em-dash on nullish. */
export function StatTile({
  label,
  value,
  valueDir,
}: {
  label: string;
  value: React.ReactNode;
  valueDir?: "ltr" | "rtl" | "auto";
}) {
  const empty = value == null || value === "";
  return (
    <div className="min-w-0">
      <div className="truncate text-[0.625rem] uppercase tracking-wide text-muted-foreground/60">
        {label}
      </div>
      <div
        dir={valueDir}
        className={cn(
          "truncate text-[0.75rem] tabular-nums",
          empty ? "text-muted-foreground/50" : "text-foreground/85",
        )}
      >
        {empty ? "—" : value}
      </div>
    </div>
  );
}

/** A pass/fail marker: green filled dot on pass, dimmed red dot on fail. */
export function PassDot({ pass }: { pass: boolean }) {
  return (
    <span
      aria-hidden="true"
      className="inline-block size-2 shrink-0 rounded-full"
      style={{
        backgroundColor: pass ? "var(--success)" : "var(--danger)",
        opacity: pass ? 1 : 0.55,
      }}
    />
  );
}

/** Run vs grid-search badge, localized via the shared TERMS vocabulary. */
export function TypeBadge({ type }: { type: string }) {
  if (type === "grid_search") {
    return (
      <Badge variant="outline" className="border-primary/30 text-primary">
        {TERMS.optimizationTypeGrid}
      </Badge>
    );
  }
  return <Badge variant="secondary">{TERMS.optimizationTypeRun}</Badge>;
}

/**
 * Colored delta pill for a baseline→optimized metric pair (0–100 scale). Renders
 * nothing when either side is missing, so callers can drop it in unconditionally.
 */
export function GainPill({
  baseline,
  optimized,
}: {
  baseline: number | null | undefined;
  optimized: number | null | undefined;
}) {
  const gain = computeGain(baseline, optimized);
  if (!gain) return null;
  if (gain.kind === "neutral") {
    return (
      <span dir="ltr" className="font-mono text-[0.6875rem] tabular-nums text-muted-foreground">
        {gain.text}
      </span>
    );
  }
  const positive = gain.kind === "positive";
  const Icon = positive ? ArrowUp : ArrowDown;
  return (
    <span
      dir="ltr"
      className="inline-flex items-center gap-0.5 rounded-full px-1.5 py-0.5 font-mono text-[0.6875rem] font-medium leading-none tabular-nums"
      style={{
        backgroundColor: positive ? "var(--success-dim)" : "var(--danger-dim)",
        color: positive ? "var(--success)" : "var(--danger)",
      }}
    >
      <Icon className="size-2.5" strokeWidth={2.5} aria-hidden="true" />
      {gain.text}
    </span>
  );
}
