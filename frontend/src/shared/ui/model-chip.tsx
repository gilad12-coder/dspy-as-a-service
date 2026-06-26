"use client";

import * as React from "react";
import { Settings, Copy, Trash2, Plus, Thermometer, Coins, Eye, Brain } from "lucide-react";
import { cn } from "@/shared/lib/utils";
import type { CatalogModel, ModelConfig } from "@/shared/types/api";
import { msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { TooltipButton } from "@/shared/ui/tooltip-button";

interface ModelChipProps {
  config: ModelConfig;
  roleLabel?: string;
  onClick: () => void;
  onClone?: () => void;
  onRemove?: () => void;
  /** If true, shows a subtle "required" style */
  required?: boolean;
  /** Shows a visible "copy from X" button when the chip is empty */
  copyFromLabel?: string;
  onCopyFrom?: () => void;
  /** Catalog used to resolve a model's vision capability for the badge. */
  catalogModels?: CatalogModel[];
  className?: string;
}

const REASONING_EFFORT_LABELS: Record<string, string> = {
  minimal: "Minimal",
  low: "Low",
  medium: "Medium",
  high: "High",
};

function reasoningEffortLabel(value: string | null | undefined): string | null {
  if (!value) return null;
  return REASONING_EFFORT_LABELS[value.toLowerCase()] ?? value;
}

function ReasoningPill({ value }: { value: string | null | undefined }) {
  const label = reasoningEffortLabel(value);
  if (!label) return null;
  return (
    <span
      className="shrink-0 inline-flex items-center gap-0.5 rounded bg-muted/50 px-1 py-0.5 text-[9px] font-semibold text-muted-foreground/80"
      title={`Reasoning effort: ${label}`}
    >
      <Brain className="size-2.5" />
      {label}
    </span>
  );
}

export function ModelChip({
  config,
  roleLabel,
  onClick,
  onClone,
  onRemove,
  required,
  copyFromLabel,
  onCopyFrom,
  catalogModels,
  className,
}: ModelChipProps) {
  const effort = config.extra?.reasoning_effort as string | undefined;
  const name =
    config.name ||
    (required ? msg("shared.model_chip.choose_model") : msg("shared.model_chip.not_configured"));
  const isEmpty = !config.name;
  const supportsVision = !!catalogModels?.find((m) => m.value === config.name)?.supports_vision;

  return (
    <div
      className={cn(
        "group relative flex items-center gap-2.5 rounded-lg border px-3 py-2 cursor-pointer",
        "transition-[border-color,box-shadow,background-color] duration-150",
        isEmpty
          ? "border-dashed border-border/60 bg-muted/20 hover:border-primary/40 hover:bg-muted/40"
          : "border-border/50 bg-card/80 hover:border-primary/40 hover:shadow-sm",
        className,
      )}
      onClick={onClick}
    >
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        {roleLabel && (
          <span className="text-[0.625rem] font-medium uppercase tracking-wide text-muted-foreground">
            {roleLabel}
          </span>
        )}
        <span
          className={cn(
            "truncate text-sm",
            isEmpty ? "text-muted-foreground" : "text-foreground font-mono font-medium",
          )}
          // Placeholder text is localized, so it follows the active direction;
          // a concrete model id is always Latin and stays LTR.
          dir={isEmpty ? getActiveDir() : "ltr"}
        >
          {isEmpty ? name : (name.split("/").pop() ?? name)}
        </span>
        {!isEmpty && (
          <div
            className="flex items-center gap-2.5 text-[0.625rem] text-muted-foreground"
            dir="ltr"
          >
            <span className="inline-flex items-center gap-0.5">
              <Thermometer className="size-2.5" />
              {config.temperature?.toFixed(1) ?? "0.7"}
            </span>
            {config.max_tokens && (
              <span className="inline-flex items-center gap-0.5">
                <Coins className="size-2.5" />
                {config.max_tokens}
              </span>
            )}
            {effort && <ReasoningPill value={effort} />}
            {supportsVision && (
              <span
                className="inline-flex items-center gap-0.5 rounded-sm bg-primary/10 px-1 py-px text-primary"
                title={msg("shared.model_chip.vision_badge")}
              >
                <Eye className="size-2.5" />
              </span>
            )}
          </div>
        )}
      </div>

      {isEmpty && copyFromLabel && onCopyFrom && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onCopyFrom();
          }}
          className="flex shrink-0 items-center gap-1 rounded-md border border-dashed border-primary/30 px-2 py-1 text-[0.625rem] font-medium text-primary/80 hover:bg-primary/5 hover:border-primary/50 transition-all cursor-pointer"
        >
          <Copy className="size-2.5" />
          {copyFromLabel}
        </button>
      )}

      <div className="flex shrink-0 items-center gap-1">
        {onClone && !isEmpty && (
          <TooltipButton tooltip={msg("shared.model_chip.clone")} side="top">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onClone();
              }}
              className="rounded-md p-1 text-muted-foreground opacity-0 group-hover:opacity-100 hover:bg-accent hover:text-foreground transition-all cursor-pointer"
            >
              <Copy className="size-3" />
            </button>
          </TooltipButton>
        )}
        {onRemove && !isEmpty && (
          <TooltipButton tooltip={msg("shared.model_chip.remove")} side="top">
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                onRemove();
              }}
              className="rounded-md p-1 text-muted-foreground opacity-0 group-hover:opacity-100 hover:bg-destructive/10 hover:text-destructive transition-all cursor-pointer"
            >
              <Trash2 className="size-3" />
            </button>
          </TooltipButton>
        )}
        <Settings className="size-3.5 text-muted-foreground/60 group-hover:text-foreground/70 transition-colors" />
      </div>
    </div>
  );
}

interface AddModelButtonProps {
  label?: string;
  onClick: () => void;
  className?: string;
}

export function AddModelButton({
  label = msg("shared.model_chip.add_model"),
  onClick,
  className,
}: AddModelButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 rounded-lg border border-dashed border-border/50 px-3 py-2",
        "text-sm text-muted-foreground hover:border-primary/40 hover:text-foreground hover:bg-muted/30",
        "transition-all duration-150 cursor-pointer",
        className,
      )}
    >
      <Plus className="size-3.5" />
      {label}
    </button>
  );
}
