"use client";

import * as React from "react";
import { Check, ChevronDown, ChevronRight } from "lucide-react";

import { cachedCatalog, getModelCatalog } from "@/shared/lib/model-catalog";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import type { ModelCatalogResponse } from "@/shared/types/api";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
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

const EFFORT_LEVELS = ["low", "medium", "high"] as const;

function effortLabel(level: string): string {
  switch (level) {
    case "low":
      return msg("agent.model_menu.effort_low");
    case "medium":
      return msg("agent.model_menu.effort_medium");
    case "high":
      return msg("agent.model_menu.effort_high");
    default:
      return level;
  }
}

function effortHint(level: string | null): string {
  switch (level) {
    case "low":
      return msg("agent.model_menu.effort_low_hint");
    case "medium":
      return msg("agent.model_menu.effort_medium_hint");
    case "high":
      return msg("agent.model_menu.effort_high_hint");
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
 * reasoning support.
 */
export function ComposerModelMenu({
  value,
  onChange,
  effort,
  onEffortChange,
  disabled,
}: ComposerModelMenuProps) {
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState("");
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
  const q = query.trim().toLowerCase();
  const filtered = q
    ? available.filter(
        (m) => m.value.toLowerCase().includes(q) || m.label.toLowerCase().includes(q),
      )
    : available;
  const grouped = React.useMemo(() => {
    const groups = new Map<string, typeof filtered>();
    for (const m of filtered) {
      const arr = groups.get(m.provider) ?? [];
      arr.push(m);
      groups.set(m.provider, arr);
    }
    return groups;
  }, [filtered]);

  const providerLabel = (slug: string) =>
    catalog?.providers.find((p) => p.slug === slug)?.label ?? slug;

  const canThink = !!available.find((m) => m.value === value)?.supports_thinking;

  const pick = (model: string | null) => {
    onChange(model);
    // Effort only means something on a reasoning-capable model; carrying it
    // across to one that isn't would silently send a dead parameter.
    if (!model || !available.find((m) => m.value === model)?.supports_thinking) {
      onEffortChange(null);
    }
  };

  return (
    <DropdownMenu
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setQuery("");
      }}
    >
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          aria-label={msg("agent.model_menu.label")}
          className={cn(
            "flex items-center gap-1 rounded-full px-2 py-1 text-xs text-muted-foreground",
            "cursor-pointer transition-colors hover:bg-accent hover:text-foreground",
            "focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40",
            "disabled:pointer-events-none disabled:opacity-50",
            open && "bg-accent text-foreground",
          )}
        >
          <span className="max-w-40 truncate" dir="ltr">
            {value ? shortName(value) : msg("agent.model_menu.auto")}
          </span>
          {/* Codex-style chip: the effort reads as a lighter suffix after the
              model name ("gpt-5 High"), not a separated fragment. */}
          {value && effort && <span className="shrink-0 opacity-60">{effortLabel(effort)}</span>}
          <ChevronDown className="size-3 shrink-0" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" sideOffset={6} className="w-60 py-1">
        <DropdownMenuSub>
          <DropdownMenuSubTrigger>
            <span className="shrink-0">{msg("agent.model_menu.model")}</span>
            <span className="ms-auto truncate text-muted-foreground" dir="ltr">
              {value ? shortName(value) : msg("agent.model_menu.auto")}
            </span>
            <ChevronRight className="size-3.5 shrink-0 text-muted-foreground rtl:rotate-180" />
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="w-72 overflow-hidden p-0">
            {available.length > 8 && (
              <div className="border-b border-border/40 p-2">
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  // Typed characters must stay in the input — the menu's
                  // typeahead would otherwise swallow them to jump rows.
                  onKeyDown={(e) => e.stopPropagation()}
                  placeholder={msg("agent.model_menu.search")}
                  aria-label={msg("agent.model_menu.search")}
                  className={cn(
                    "w-full rounded-md border border-input bg-background px-2.5 py-1.5 text-sm",
                    "outline-none placeholder:text-muted-foreground/60 focus-visible:border-ring",
                  )}
                  dir="ltr"
                />
              </div>
            )}
            <div className="max-h-72 overflow-y-auto py-1">
              {!q && (
                <MenuItem
                  selected={value === null}
                  label={msg("agent.model_menu.auto")}
                  description={msg("agent.model_menu.auto_hint")}
                  onSelect={() => pick(null)}
                />
              )}
              {[...grouped.entries()].map(([provider, group]) => (
                <div key={provider}>
                  <DropdownMenuLabel>{providerLabel(provider)}</DropdownMenuLabel>
                  {group.map((m) => (
                    <MenuItem
                      key={m.value}
                      selected={value === m.value}
                      label={m.label}
                      onSelect={() => pick(m.value)}
                    />
                  ))}
                </div>
              ))}
              {filtered.length === 0 && (
                <p className="px-3 py-2 text-sm text-muted-foreground">
                  {msg("agent.model_menu.empty")}
                </p>
              )}
            </div>
          </DropdownMenuSubContent>
        </DropdownMenuSub>
        <DropdownMenuSub>
          <DropdownMenuSubTrigger disabled={!canThink}>
            <span className="shrink-0">{msg("agent.model_menu.effort_label")}</span>
            <span className="ms-auto truncate text-muted-foreground">
              {effort ? effortLabel(effort) : msg("agent.model_menu.effort_default")}
            </span>
            <ChevronRight className="size-3.5 shrink-0 text-muted-foreground rtl:rotate-180" />
          </DropdownMenuSubTrigger>
          <DropdownMenuSubContent className="w-60 py-1">
            {[null, ...EFFORT_LEVELS].map((level) => (
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
  onSelect,
  dir = "ltr",
}: {
  selected: boolean;
  label: string;
  description?: string;
  onSelect: () => void;
  /** Model ids are latin so rows default LTR; localized rows pass ``auto``. */
  dir?: "ltr" | "auto";
}) {
  return (
    <DropdownMenuItem onSelect={onSelect}>
      <span className="flex min-w-0 flex-1 flex-col">
        <span
          className={cn("truncate text-sm text-foreground", selected && "font-medium")}
          dir={dir}
        >
          {label}
        </span>
        {description && (
          <span className="truncate text-xs text-muted-foreground" dir={dir}>
            {description}
          </span>
        )}
      </span>
      <Check className={cn("size-4 shrink-0 text-primary", !selected && "invisible")} />
    </DropdownMenuItem>
  );
}
