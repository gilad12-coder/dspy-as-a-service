"use client";

import * as React from "react";
import { Check, ChevronDown } from "lucide-react";

import { cachedCatalog, getModelCatalog } from "@/shared/lib/model-catalog";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import type { ModelCatalogResponse } from "@/shared/types/api";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/ui/primitives/popover";

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

/**
 * The composer's model menu — the ChatGPT-style picker: a quiet pill naming
 * the current choice that opens a portaled menu of catalog models, the
 * default ("Auto") row first and a checkmark on the active row. When the
 * chosen model supports reasoning, a Codex-style thinking-level control
 * (Default/Low/Medium/High) docks under the list. The choice applies from
 * the next turn of the surrounding conversation.
 *
 * ``modal``: rendered inside a dialog, the parent's scroll lock would eat
 * wheel events on the portaled list — same lesson as the submit wizard's
 * ModelPicker.
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
  React.useEffect(() => {
    if (catalog) return;
    let cancelled = false;
    getModelCatalog()
      .then((c) => {
        if (!cancelled) setCatalog(c);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [catalog]);

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
    setOpen(false);
    setQuery("");
  };

  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setQuery("");
      }}
      modal
    >
      <PopoverTrigger asChild>
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
          {value && effort && <span className="shrink-0">· {effortLabel(effort)}</span>}
          <ChevronDown className="size-3 shrink-0" />
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" sideOffset={6} className="w-72 overflow-hidden p-0">
        {available.length > 8 && (
          <div className="border-b border-border/40 p-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
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
        <div
          role="listbox"
          aria-label={msg("agent.model_menu.label")}
          className="max-h-72 overflow-y-auto py-1"
        >
          {!q && (
            <MenuRow
              selected={value === null}
              label={msg("agent.model_menu.auto")}
              description={msg("agent.model_menu.auto_hint")}
              onClick={() => pick(null)}
            />
          )}
          {[...grouped.entries()].map(([provider, group]) => (
            <div key={provider}>
              <p className="px-3 pb-1 pt-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70">
                {providerLabel(provider)}
              </p>
              {group.map((m) => (
                <MenuRow
                  key={m.value}
                  selected={value === m.value}
                  label={m.label}
                  onClick={() => pick(m.value)}
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
        {canThink && (
          <div className="border-t border-border/40 p-2">
            <p className="px-1 pb-1.5 text-[11px] font-medium uppercase tracking-wide text-muted-foreground/70">
              {msg("agent.model_menu.effort_label")}
            </p>
            <div className="flex rounded-lg bg-muted p-0.5">
              {[null, ...EFFORT_LEVELS].map((level) => (
                <button
                  key={level ?? "default"}
                  type="button"
                  onClick={() => onEffortChange(level)}
                  className={cn(
                    "flex-1 cursor-pointer rounded-md px-2 py-1 text-center text-xs font-medium transition-colors",
                    "focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40",
                    effort === level
                      ? "bg-background text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {level ? effortLabel(level) : msg("agent.model_menu.effort_default")}
                </button>
              ))}
            </div>
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}

function MenuRow({
  selected,
  label,
  description,
  onClick,
}: {
  selected: boolean;
  label: string;
  description?: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      role="option"
      aria-selected={selected}
      onClick={onClick}
      className={cn(
        "flex w-full cursor-pointer items-center gap-2 px-3 py-2 text-start",
        "hover:bg-accent/60 focus-visible:bg-accent/60 focus-visible:outline-none",
      )}
    >
      <span className="flex min-w-0 flex-1 flex-col">
        <span
          className={cn("truncate text-sm text-foreground", selected && "font-medium")}
          dir="ltr"
        >
          {label}
        </span>
        {description && (
          <span className="truncate text-xs text-muted-foreground" dir="ltr">
            {description}
          </span>
        )}
      </span>
      <Check className={cn("size-4 shrink-0 text-primary", !selected && "invisible")} />
    </button>
  );
}
