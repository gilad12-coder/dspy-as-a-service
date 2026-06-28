"use client";

import * as React from "react";
import { Check, ChevronDown, Eye, Lock, Search, Loader2, RefreshCw } from "lucide-react";
import { formatMsg, msg } from "@/shared/lib/messages";

import { cn } from "@/shared/lib/utils";
import { getModelCatalog, cachedCatalog, discoverModels } from "@/shared/lib/model-catalog";
import type { CatalogModel, CatalogProvider } from "@/shared/types/api";

interface ModelPickerProps {
  value: string;
  onChange: (next: string) => void;
  id?: string;
  placeholder?: string;
  /** If set, also fetch models from this base URL's /v1/models endpoint. */
  discoverUrl?: string;
  discoverApiKey?: string;
  disabled?: boolean;
  className?: string;
  /** Constrain picks to this provider slug (e.g. "openai"). */
  providerFilter?: string;
  /** Returns true for a model the current token mode/balance can't run; renders it locked. */
  isLocked?: (modelValue: string) => boolean;
}

interface EnrichedModel extends CatalogModel {
  fromDiscovery?: boolean;
}

function formatCtx(tokens?: number): string {
  if (!tokens) return "";
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  return `${Math.round(tokens / 1000)}K`;
}

/** Searchable combobox for DSPy model IDs. Curated static catalog + live discovery. */
export function ModelPicker({
  value,
  onChange,
  id,
  placeholder = msg("auto.features.submit.components.modelpicker.literal.1"),
  discoverUrl,
  discoverApiKey,
  disabled,
  className,
  providerFilter,
  isLocked,
}: ModelPickerProps) {
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState("");
  // Use cached catalog instantly (prefetched on module load); fallback to async
  const [catalog, setCatalog] = React.useState<{
    providers: CatalogProvider[];
    models: CatalogModel[];
  } | null>(cachedCatalog);
  const [discovered, setDiscovered] = React.useState<string[]>([]);
  const [discovering, setDiscovering] = React.useState(false);
  const [discoveryError, setDiscoveryError] = React.useState<string | null>(null);

  const rootRef = React.useRef<HTMLDivElement>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  // If cache wasn't ready at mount time, await it once
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

  const runDiscover = React.useCallback(
    async (signal?: AbortSignal) => {
      if (!discoverUrl) {
        setDiscovered([]);
        setDiscoveryError(null);
        return;
      }
      setDiscovering(true);
      setDiscoveryError(null);
      try {
        const res = await discoverModels(discoverUrl, discoverApiKey, signal);
        if (signal?.aborted) return;
        setDiscovered(res.models);
        if (res.error) setDiscoveryError(res.error);
      } catch (e) {
        if (signal?.aborted) return;
        setDiscoveryError(
          e instanceof Error
            ? e.message
            : msg("auto.features.submit.components.modelpicker.literal.2"),
        );
      } finally {
        if (!signal?.aborted) setDiscovering(false);
      }
    },
    [discoverUrl, discoverApiKey],
  );

  // Auto-run discovery when URL stabilizes. The AbortController fires on
  // unmount or URL change so an in-flight fetch can't clobber state of a
  // newer URL (or a freshly mounted instance).
  React.useEffect(() => {
    if (!discoverUrl) {
      setDiscovered([]);
      setDiscoveryError(null);
      return;
    }
    const controller = new AbortController();
    const t = setTimeout(() => {
      void runDiscover(controller.signal);
    }, 400);
    return () => {
      clearTimeout(t);
      controller.abort();
    };
  }, [discoverUrl, runDiscover]);

  React.useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [open]);

  React.useEffect(() => {
    if (open) setTimeout(() => inputRef.current?.focus(), 50);
  }, [open]);

  const allModels: EnrichedModel[] = React.useMemo(() => {
    const staticModels = catalog?.models ?? [];
    const filtered = providerFilter
      ? staticModels.filter((m) => m.provider === providerFilter)
      : staticModels;
    if (discovered.length === 0) return filtered;
    const existingValues = new Set(filtered.map((m) => m.value));
    const discoveredEntries: EnrichedModel[] = discovered
      .map((id) => {
        // A bare id from an OpenAI-compatible endpoint (Ollama, vLLM, LM Studio,
        // llama.cpp) needs the openai/ prefix so LiteLLM routes it to the custom
        // base_url; ids that already carry it pass through unchanged.
        const value = id.startsWith("openai/") ? id : `openai/${id}`;
        return {
          value,
          label: id,
          provider: "discovered",
          supports_thinking: false,
          supports_vision: false,
          available: true,
          fromDiscovery: true,
        };
      })
      .filter((m) => !existingValues.has(m.value));
    return [...discoveredEntries, ...filtered];
  }, [catalog, discovered, providerFilter]);

  const filtered = React.useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return allModels;
    return allModels.filter(
      (m) => m.value.toLowerCase().includes(q) || m.label.toLowerCase().includes(q),
    );
  }, [allModels, query]);

  // Group by provider *and* data center: a provider that fans out across
  // several endpoints (e.g. a public API plus an on-prem gateway) gets one
  // section per data center so the user can see which endpoint a model
  // resolves through. The NUL byte can't appear in a slug/DC label.
  const grouped = React.useMemo(() => {
    const groups = new Map<string, EnrichedModel[]>();
    for (const m of filtered) {
      const key = `${m.provider}\u0000${m.data_center ?? ""}`;
      const arr = groups.get(key) ?? [];
      arr.push(m);
      groups.set(key, arr);
    }
    return groups;
  }, [filtered]);

  const providerLabel = React.useCallback(
    (groupKey: string): string => {
      const [slug = "", dataCenter = ""] = groupKey.split("\u0000");
      if (slug === "discovered")
        return msg("auto.features.submit.components.modelpicker.template.1");
      const base = catalog?.providers.find((p) => p.slug === slug)?.label ?? slug;
      if (!dataCenter) return base;
      return formatMsg("auto.features.submit.components.modelpicker.template.2", {
        p1: base,
        p2: dataCenter,
      });
    },
    [catalog],
  );

  const selectedModel = allModels.find((m) => m.value === value);

  const commit = (next: string) => {
    onChange(next);
    setOpen(false);
    setQuery("");
  };

  return (
    <div ref={rootRef} className={cn("relative w-full", className)}>
      <button
        type="button"
        id={id}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        className={cn(
          "flex w-full items-center justify-between gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm",
          "shadow-xs cursor-pointer transition-[border-color,background-color,box-shadow] duration-120",
          "hover:border-foreground/20 hover:bg-accent/40",
          "focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
          "disabled:cursor-not-allowed disabled:opacity-50",
          open && "border-foreground/25 bg-accent/40",
        )}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        {value ? (
          <span className="flex min-w-0 flex-1 items-center gap-2" dir="ltr">
            <span className="truncate font-mono text-[0.8125rem]">
              {selectedModel?.label ?? value}
            </span>
          </span>
        ) : (
          <span className="flex min-w-0 flex-1 items-center gap-2 text-muted-foreground">
            {placeholder}
          </span>
        )}
        <ChevronDown
          className={cn(
            "size-4 shrink-0 text-muted-foreground transition-transform duration-150",
            open && "rotate-180",
          )}
        />
      </button>

      {open && (
        <div
          className={cn(
            "absolute z-50 mt-1 w-full overflow-hidden rounded-xl border border-border/70 bg-popover shadow-lg",
            "animate-in fade-in-0 zoom-in-95 slide-in-from-top-1",
          )}
          role="listbox"
        >
          <div className="flex items-center gap-2 border-b border-border/50 px-3 py-2">
            <Search className="size-3.5 shrink-0 text-muted-foreground" />
            <input
              ref={inputRef}
              dir="auto"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={msg("auto.features.submit.components.modelpicker.literal.4")}
              className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  setOpen(false);
                }
              }}
            />
            {discoverUrl && (
              <button
                type="button"
                onClick={() => void runDiscover()}
                disabled={discovering}
                className="flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-[0.6875rem] font-medium text-muted-foreground hover:bg-accent hover:text-foreground disabled:opacity-50"
                title={msg("auto.features.submit.components.modelpicker.literal.3")}
              >
                {discovering ? (
                  <Loader2 className="size-3 animate-spin" />
                ) : (
                  <RefreshCw className="size-3" />
                )}
                {msg("auto.features.submit.components.modelpicker.1")}
              </button>
            )}
          </div>

          <div className="max-h-[120px] overflow-y-auto py-1">
            {discoveryError && discoverUrl && (
              <div className="px-3 py-2 text-xs text-amber-700 dark:text-amber-400">
                {msg("auto.features.submit.components.modelpicker.2")}
                {discoverUrl}: {discoveryError}
              </div>
            )}
            {filtered.length === 0 && (
              <div className="px-3 py-8 text-center text-xs text-muted-foreground">
                {msg("auto.features.submit.components.modelpicker.3")}
              </div>
            )}
            {Array.from(grouped.entries()).map(([provider, items]) => (
              <div key={provider} className="py-1">
                <div
                  className="px-3 py-1 text-[0.625rem] font-semibold uppercase tracking-wider text-muted-foreground text-start"
                  dir="ltr"
                >
                  {providerLabel(provider)}
                </div>
                {items.map((m) => {
                  const locked = isLocked?.(m.value) ?? false;
                  return (
                    <button
                      key={m.value}
                      type="button"
                      disabled={locked}
                      onClick={() => commit(m.value)}
                      title={locked ? msg("billing.lock.frontier") : undefined}
                      className={cn(
                        "flex w-full items-center gap-2 px-3 py-1.5 text-start text-sm transition-colors",
                        locked ? "cursor-not-allowed opacity-50" : "hover:bg-accent/70",
                        value === m.value && "bg-accent/50",
                        !m.available && !locked && "opacity-60",
                      )}
                      role="option"
                      aria-selected={value === m.value}
                      aria-disabled={locked}
                    >
                      <span className="flex min-w-0 flex-1 items-center gap-1.5" dir="ltr">
                        <span className="truncate text-[0.8125rem]">{m.label}</span>
                        {m.max_input_tokens && (
                          <span className="shrink-0 text-[9px] tabular-nums text-muted-foreground">
                            {formatCtx(m.max_input_tokens)}
                          </span>
                        )}
                        {m.supports_vision && (
                          <span
                            className="inline-flex shrink-0 items-center gap-0.5 rounded-sm bg-primary/10 px-1 py-px text-[9px] text-primary"
                            title={msg("shared.model_chip.vision_badge")}
                          >
                            <Eye className="size-2.5" />
                          </span>
                        )}
                      </span>
                      {locked ? (
                        <Lock className="size-3.5 shrink-0 text-muted-foreground" />
                      ) : (
                        <Check
                          className={cn(
                            "size-3.5 shrink-0",
                            value === m.value ? "opacity-100" : "opacity-0",
                          )}
                        />
                      )}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Is this a model that supports reasoning_effort? Trusts the LiteLLM catalog flag exclusively. */
export function modelSupportsThinking(modelValue: string, models?: CatalogModel[]): boolean {
  if (!modelValue || !models) return false;
  const hit = models.find((m) => m.value === modelValue);
  return hit?.supports_thinking ?? false;
}
