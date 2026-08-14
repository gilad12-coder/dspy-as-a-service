"use client";

import * as React from "react";
import { Check, CaretDown, CaretRight } from "@/shared/ui/icons";

import { cachedCatalog, getModelCatalog } from "@/shared/lib/model-catalog";
import { effortLabel, effortsFor } from "@/shared/lib/model-efforts";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { ProviderLogo } from "@/shared/ui/provider-logo";
import type { ModelCatalogResponse } from "@/shared/types/api";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "@/shared/ui/primitives/dropdown-menu";

interface ComposerModelMenuProps {
  /** LiteLLM id of the chosen model; ``null`` runs the server default. */
  value: string | null;
  onChange: (model: string | null) => void;
  /** Reasoning-effort level for the chosen model; ``null`` runs its default. */
  effort: string | null;
  onEffortChange: (effort: string | null) => void;
  disabled?: boolean;
}

/** Short display name for a LiteLLM id ("openai/gpt-4o-mini" → "gpt-4o-mini"). */
function shortName(id: string): string {
  return id.split("/").pop() || id;
}

// Sentinel sent instead of a catalog id: the backend's per-turn router in
// its frontier-quality tier (Cursor-style "Intelligence"). Plain null stays
// the balanced router tier.
export const AUTO_INTELLIGENT_MODEL = "auto:intelligent";

/** Menu/chip label for the current choice, including the auto tiers. */
function displayName(value: string | null): string {
  if (!value) return msg("agent.model_menu.auto");
  if (value === AUTO_INTELLIGENT_MODEL) return msg("agent.model_menu.auto_intelligent");
  return shortName(value);
}

/** Company slug for a LiteLLM id — the OpenRouter prefix is transport, not brand. */
function providerSlug(id: string): string {
  const parts = id.split("/");
  const slug = (parts[0] === "openrouter" && parts.length > 2 ? parts[1] : parts[0]) ?? id;
  return slug === "x-ai" ? "xai" : slug;
}

/** Leading icon for the current choice: both auto tiers run OpenRouter's
 * Auto Router, so they wear the OpenRouter mark; explicit picks wear the
 * model company's mark. */
function choiceIcon(value: string | null, size: number): React.ReactNode {
  if (!value || value === AUTO_INTELLIGENT_MODEL) {
    return <ProviderLogo slug="openrouter" size={size} />;
  }
  return <ProviderLogo slug={providerSlug(value)} size={size} />;
}

// Display curation only. The composer picks a conversation partner, not a
// training target — the full gateway catalog (500+ ids, embeddings and TTS
// included) belongs to the submit wizard, not here. Ids that leave the
// catalog drop out silently. Kept deliberately short: the auto-router rows
// cover "just pick well for me", so the list is a few current picks per
// major lab.
const FEATURED_MODELS = [
  "openrouter/openai/gpt-5.6-sol",
  "openrouter/openai/gpt-5.6-terra",
  "openrouter/openai/gpt-5.6-luna",
  "openrouter/anthropic/claude-fable-5",
  "openrouter/anthropic/claude-opus-4.8",
  "openrouter/anthropic/claude-sonnet-5",
  "openrouter/anthropic/claude-haiku-4.5",
  "openrouter/google/gemini-3.1-pro-preview",
  "openrouter/google/gemini-3.6-flash",
  "openrouter/meta/muse-spark-1.1",
  "openrouter/x-ai/grok-4.5",
  "openrouter/deepseek/deepseek-v4-pro",
  "openrouter/moonshotai/kimi-k3",
  "openrouter/minimax/minimax-m3",
];

// Caps the fallback list when none of the featured ids exist in a deployment.
const FALLBACK_LIST_CAP = 50;

// One-liners distilled from each provider's own launch copy, kept short
// enough to never truncate in the submenu row. Non-featured ids get none.
function modelDescription(id: string): string | undefined {
  switch (id) {
    case "openrouter/openai/gpt-5.6-sol":
      return msg("agent.model_menu.desc_gpt_5_6_sol");
    case "openrouter/openai/gpt-5.6-terra":
      return msg("agent.model_menu.desc_gpt_5_6_terra");
    case "openrouter/openai/gpt-5.6-luna":
      return msg("agent.model_menu.desc_gpt_5_6_luna");
    case "openrouter/anthropic/claude-fable-5":
      return msg("agent.model_menu.desc_claude_fable_5");
    case "openrouter/anthropic/claude-opus-4.8":
      return msg("agent.model_menu.desc_claude_opus_4_8");
    case "openrouter/anthropic/claude-sonnet-5":
      return msg("agent.model_menu.desc_claude_sonnet_5");
    case "openrouter/anthropic/claude-haiku-4.5":
      return msg("agent.model_menu.desc_claude_haiku_4_5");
    case "openrouter/google/gemini-3.1-pro-preview":
      return msg("agent.model_menu.desc_gemini_3_1_pro");
    case "openrouter/google/gemini-3.6-flash":
      return msg("agent.model_menu.desc_gemini_3_6_flash");
    case "openrouter/meta/muse-spark-1.1":
      return msg("agent.model_menu.desc_muse_spark_1_1");
    case "openrouter/moonshotai/kimi-k3":
      return msg("agent.model_menu.desc_kimi_k3");
    case "openrouter/x-ai/grok-4.5":
      return msg("agent.model_menu.desc_grok_4_5");
    case "openrouter/deepseek/deepseek-v4-pro":
      return msg("agent.model_menu.desc_deepseek_v4_pro");
    case "openrouter/minimax/minimax-m3":
      return msg("agent.model_menu.desc_minimax_m3");
    default:
      return undefined;
  }
}

function effortHint(level: string | null): string {
  switch (level) {
    case "none":
      return msg("agent.model_menu.effort_none_hint");
    case "minimal":
      return msg("agent.model_menu.effort_minimal_hint");
    case "low":
      return msg("agent.model_menu.effort_low_hint");
    case "medium":
      return msg("agent.model_menu.effort_medium_hint");
    case "high":
      return msg("agent.model_menu.effort_high_hint");
    case "xhigh":
      return msg("agent.model_menu.effort_xhigh_hint");
    case "max":
      return msg("agent.model_menu.effort_max_hint");
    default:
      return msg("agent.model_menu.effort_default_hint");
  }
}

/**
 * The composer's model menu, structured like Codex's: a quiet chip naming the
 * current choice ("gpt-5 High") opens a compact two-row menu — Model and
 * Thinking level, each showing its current value — and each row fans out a
 * side submenu with the checkmarked options. Picking anything closes the
 * whole menu; the choice applies from the next turn of the surrounding
 * conversation. The thinking row is visible but inert on models without
 * reasoning support. The model submenu shows only the curated featured
 * shortlist.
 */
export function ComposerModelMenu({
  value,
  onChange,
  effort,
  onEffortChange,
  disabled,
}: ComposerModelMenuProps) {
  const [open, setOpen] = React.useState(false);
  const [catalog, setCatalog] = React.useState<ModelCatalogResponse | null>(
    cachedCatalog() ?? null,
  );
  // The synchronous cache may be stale (served without a TTL so the menu is
  // never empty) — always adopt the revalidated catalog when it lands.
  React.useEffect(() => {
    let cancelled = false;
    getModelCatalog()
      .then((c) => {
        if (!cancelled) setCatalog(c);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const available = React.useMemo(
    () => (catalog?.models ?? []).filter((m) => m.available),
    [catalog],
  );

  // Small deployments (an on-prem gateway with a handful of models) may miss
  // every featured id — fall back to the whole list rather than an empty menu.
  const featured = React.useMemo(() => {
    const byId = new Map(available.map((m) => [m.value, m]));
    const rows = FEATURED_MODELS.flatMap((id) => byId.get(id) ?? []);
    return rows.length ? rows : available.slice(0, FALLBACK_LIST_CAP);
  }, [available]);
  // A previously chosen model outside the shortlist keeps its checkmarked
  // row next time the menu opens.
  const currentExtra =
    value && !featured.some((m) => m.value === value)
      ? (available.find((m) => m.value === value) ?? null)
      : null;

  const canThink = !!available.find((m) => m.value === value)?.supports_thinking;
  const efforts = effortsFor(value);

  const pick = (model: string | null) => {
    onChange(model);
    // Effort only means something on a reasoning-capable model, and each
    // provider speaks its own vocabulary — carrying a level the new model
    // doesn't support would send a dead or rejected parameter.
    if (
      !model ||
      !available.find((m) => m.value === model)?.supports_thinking ||
      (effort !== null && !effortsFor(model).includes(effort))
    ) {
      onEffortChange(null);
    }
  };

  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          aria-label={msg("agent.model_menu.label")}
          className={cn(
            "flex h-[44px] max-w-[110px] min-w-0 items-center gap-1.5 rounded-full px-3 text-xs text-foreground sm:h-9 sm:max-w-none [@media(hover:none)_and_(pointer:coarse)]:h-[44px]",
            "cursor-pointer transition-colors hover:bg-accent/60",
            "focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40",
            "disabled:pointer-events-none disabled:opacity-50",
            open && "bg-accent/60",
          )}
        >
          {choiceIcon(value, 16)}
          <span className="min-w-0 max-w-20 truncate font-medium sm:max-w-40" dir="ltr">
            {displayName(value)}
          </span>
          {/* Codex-style chip: the effort reads as a lighter suffix after the
              model name ("gpt-5 High"), not a separated fragment. */}
          {value && effort && (
            <span className="shrink-0 text-muted-foreground">{effortLabel(effort)}</span>
          )}
          <CaretDown className="size-3 shrink-0 text-muted-foreground" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" sideOffset={6} className="w-64 py-1.5">
        <DropdownMenuSub>
          <DropdownMenuSubTrigger className="py-2.5">
            <span className="shrink-0">{msg("agent.model_menu.model")}</span>
            <span className="ms-auto truncate text-muted-foreground" dir="ltr">
              {displayName(value)}
            </span>
            <CaretRight className="size-3.5 shrink-0 text-muted-foreground rtl:rotate-180" />
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="w-72 overflow-hidden p-0">
            <div className="max-h-72 overflow-y-auto py-1">
              <MenuItem
                selected={value === null}
                label={msg("agent.model_menu.auto")}
                description={msg("agent.model_menu.auto_hint")}
                icon={<ProviderLogo slug="openrouter" size={18} />}
                onSelect={() => pick(null)}
              />
              <MenuItem
                selected={value === AUTO_INTELLIGENT_MODEL}
                label={msg("agent.model_menu.auto_intelligent")}
                description={msg("agent.model_menu.auto_intelligent_hint")}
                icon={<ProviderLogo slug="openrouter" size={18} />}
                onSelect={() => pick(AUTO_INTELLIGENT_MODEL)}
              />
              {currentExtra && (
                <MenuItem
                  selected
                  label={shortName(currentExtra.value)}
                  description={modelDescription(currentExtra.value)}
                  icon={<ProviderLogo slug={providerSlug(currentExtra.value)} size={18} />}
                  onSelect={() => pick(currentExtra.value)}
                />
              )}
              {featured.map((m) => (
                <MenuItem
                  key={m.value}
                  selected={value === m.value}
                  label={shortName(m.value)}
                  description={modelDescription(m.value)}
                  icon={<ProviderLogo slug={providerSlug(m.value)} size={18} />}
                  onSelect={() => pick(m.value)}
                />
              ))}
            </div>
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        <DropdownMenuSub>
          <DropdownMenuSubTrigger disabled={!canThink || efforts.length === 0} className="py-2.5">
            <span className="shrink-0">{msg("agent.model_menu.effort_label")}</span>
            <span className="ms-auto truncate text-muted-foreground">
              {effort ? effortLabel(effort) : msg("agent.model_menu.effort_default")}
            </span>
            <CaretRight className="size-3.5 shrink-0 text-muted-foreground rtl:rotate-180" />
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="w-60 py-1">
            {[null, ...efforts].map((level) => (
              <MenuItem
                key={level ?? "default"}
                selected={effort === level}
                label={level ? effortLabel(level) : msg("agent.model_menu.effort_default")}
                description={effortHint(level)}
                dir="auto"
                onSelect={() => onEffortChange(level)}
              />
            ))}
          </DropdownMenuSubContent>
        </DropdownMenuSub>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

function MenuItem({
  selected,
  label,
  description,
  icon,
  onSelect,
  dir = "ltr",
}: {
  selected: boolean;
  label: string;
  description?: string;
  icon?: React.ReactNode;
  onSelect: () => void;
  /** Model ids are latin so rows default LTR; localized rows pass ``auto``. */
  dir?: "ltr" | "auto";
}) {
  return (
    <DropdownMenuItem onSelect={onSelect}>
      {icon}
      <span className="flex min-w-0 flex-1 flex-col">
        <span
          className={cn("truncate text-sm text-foreground", selected && "font-medium")}
          dir={dir}
        >
          {label}
        </span>
        {/* Descriptions are localized even on LTR model-id rows — let the
            text pick its own direction so Hebrew copy orders correctly. */}
        {description && (
          <span className="truncate text-xs text-muted-foreground" dir="auto">
            {description}
          </span>
        )}
      </span>
      <Check className={cn("size-4 shrink-0 text-primary", !selected && "invisible")} />
    </DropdownMenuItem>
  );
}
