"use client";

import * as React from "react";
import { Check, ChevronDown, Eye, Search, Loader2, RefreshCw } from "lucide-react";
import { formatMsg, msg, type MessageKey } from "@/shared/lib/messages";

import { cn } from "@/shared/lib/utils";
import {
  getModelCatalog,
  cachedCatalog,
  getByokModelCatalog,
  cachedByokCatalog,
  discoverModels,
} from "@/shared/lib/model-catalog";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/ui/primitives/popover";
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
  /**
   * In BYOK mode the picker lists the BYOK catalog (every offered provider's
   * models, independent of platform keys) instead of the managed catalog,
   * narrowed to `byokProviders` — the LiteLLM provider slugs the user has saved
   * a key for.
   */
  byokMode?: boolean;
  byokProviders?: string[];
}

interface EnrichedModel extends CatalogModel {
  fromDiscovery?: boolean;
}

type ModelPurpose = "all" | "vision" | "reasoning" | "multilingual" | "onprem";

// Model families with a first-class multilingual focus. The catalog is dynamic
// (whatever OpenRouter serves) and carries no per-model purpose metadata, so
// this id heuristic is the categorization lever — extend the list as needed.
const MULTILINGUAL_FAMILIES = ["qwen", "glm", "aya", "command", "gemma", "mistral"];

const PURPOSE_PREDICATES: Record<Exclude<ModelPurpose, "all">, (m: EnrichedModel) => boolean> = {
  vision: (m) => Boolean(m.supports_vision),
  reasoning: (m) => Boolean(m.supports_thinking),
  multilingual: (m) => MULTILINGUAL_FAMILIES.some((f) => m.value.toLowerCase().includes(f)),
  // The backend labels internal-gateway endpoints "On-prem gateway" in
  // data_center — the only signal that a model is served in-house.
  onprem: (m) => (m.data_center ?? "").toLowerCase().includes("on-prem"),
};

const PURPOSE_LABEL_KEYS: Record<ModelPurpose, MessageKey> = {
  all: "submit.modelpicker.purpose.all",
  vision: "submit.modelpicker.purpose.vision",
  reasoning: "submit.modelpicker.purpose.reasoning",
  multilingual: "submit.modelpicker.purpose.multilingual",
  onprem: "submit.modelpicker.purpose.onprem",
};

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
  byokMode = false,
  byokProviders,
}: ModelPickerProps) {
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState("");
  // Use cached catalog instantly (prefetched on module load); fallback to async
  const [catalog, setCatalog] = React.useState<{
    providers: CatalogProvider[];
    models: CatalogModel[];
  } | null>(cachedCatalog);
  // BYOK catalog is fetched lazily the first time BYOK mode needs it.
  const [byokCatalog, setByokCatalog] = React.useState<{
    providers: CatalogProvider[];
    models: CatalogModel[];
  } | null>(cachedByokCatalog());
  const [discovered, setDiscovered] = React.useState<string[]>([]);
  const [discovering, setDiscovering] = React.useState(false);
  const [discoveryError, setDiscoveryError] = React.useState<string | null>(null);

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

  // Load the BYOK catalog the first time the picker is opened in BYOK mode.
  React.useEffect(() => {
    if (!byokMode || byokCatalog) return;
    let cancelled = false;
    getByokModelCatalog()
      .then((c) => {
        if (!cancelled) setByokCatalog(c);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [byokMode, byokCatalog]);

  const activeCatalog = byokMode ? byokCatalog : catalog;
  const byokProviderSet = React.useMemo(
    () => new Set(byokProviders ?? []),
    [byokProviders],
  );

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

  const allModels: EnrichedModel[] = React.useMemo(() => {
    let staticModels = activeCatalog?.models ?? [];
    // In BYOK mode, only surface models for providers the user has connected.
    if (byokMode) {
      staticModels = staticModels.filter((m) => byokProviderSet.has(m.provider));
    }
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
  }, [activeCatalog, byokMode, byokProviderSet, discovered, providerFilter]);

  // Purpose chips render only for categories the current catalog actually
  // has — a deploy with no on-prem gateway never shows an empty "On-prem".
  const [purpose, setPurpose] = React.useState<ModelPurpose>("all");
  const purposeOptions = React.useMemo(() => {
    const present = (Object.keys(PURPOSE_PREDICATES) as Array<Exclude<ModelPurpose, "all">>).filter(
      (p) => allModels.some(PURPOSE_PREDICATES[p]),
    );
    return present.length > 0 ? (["all", ...present] as ModelPurpose[]) : [];
  }, [allModels]);

  const filtered = React.useMemo(() => {
    const byPurpose =
      purpose === "all" ? allModels : allModels.filter(PURPOSE_PREDICATES[purpose]);
    const q = query.trim().toLowerCase();
    if (!q) return byPurpose;
    return byPurpose.filter(
      (m) => m.value.toLowerCase().includes(q) || m.label.toLowerCase().includes(q),
    );
  }, [allModels, purpose, query]);

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
      const base = activeCatalog?.providers.find((p) => p.slug === slug)?.label ?? slug;
      if (!dataCenter) return base;
      return formatMsg("auto.features.submit.components.modelpicker.template.2", {
        p1: base,
        p2: dataCenter,
      });
    },
    [activeCatalog],
  );

  const selectedModel = allModels.find((m) => m.value === value);

  const commit = (next: string) => {
    onChange(next);
    setOpen(false);
    setQuery("");
  };

  return (
    // The list is a portaled popover, not an absolutely-positioned child:
    // inside a scrollable dialog body an absolute dropdown still counts
    // toward the container's scrollable overflow, so opening it grew a
    // scrollbar on the whole panel. Radix also brings dismissal, Escape
    // layering under a parent dialog, and edge-collision flipping.
    // ``modal``: the parent dialog's scroll lock blocks wheel events on
    // everything outside its subtree — including this portaled content — so
    // the popover must own a scroll-lock layer that whitelists its list.
    <Popover open={open} onOpenChange={setOpen} modal>
      <PopoverTrigger asChild>
        <button
          type="button"
          id={id}
          disabled={disabled}
          className={cn(
            "flex w-full items-center justify-between gap-2 rounded-md border border-input bg-background px-3 py-2 text-sm",
            "shadow-xs cursor-pointer transition-[border-color,background-color,box-shadow] duration-120",
            "hover:border-foreground/20 hover:bg-accent/40",
            "focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 focus-visible:outline-none",
            "disabled:cursor-not-allowed disabled:opacity-50",
            open && "border-foreground/25 bg-accent/40",
            className,
          )}
          aria-haspopup="listbox"
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
      </PopoverTrigger>

      <PopoverContent
        align="start"
        sideOffset={4}
        role="listbox"
        className="w-(--radix-popover-trigger-width) overflow-hidden p-0"
        onOpenAutoFocus={(e) => {
          e.preventDefault();
          inputRef.current?.focus();
        }}
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

        {purposeOptions.length > 1 && (
          <div
            role="group"
            aria-label={msg("submit.modelpicker.purpose.aria")}
            className="flex flex-wrap gap-1 border-b border-border/50 px-3 py-2"
          >
            {purposeOptions.map((p) => (
              <button
                key={p}
                type="button"
                role="radio"
                aria-checked={purpose === p}
                onClick={() => setPurpose(p)}
                className={cn(
                  "rounded-full border px-2 py-0.5 text-[0.6875rem] font-medium transition-colors duration-150 cursor-pointer",
                  purpose === p
                    ? "border-foreground/25 bg-accent text-foreground"
                    : "border-border/50 text-muted-foreground hover:text-foreground",
                )}
              >
                {msg(PURPOSE_LABEL_KEYS[p])}
              </button>
            ))}
          </div>
        )}

        <div className="max-h-60 overflow-y-auto py-1">
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
              {items.map((m) => (
                <button
                  key={m.value}
                  type="button"
                  onClick={() => commit(m.value)}
                  className={cn(
                    "flex w-full items-center gap-2 px-3 py-1.5 text-start text-sm transition-colors hover:bg-accent/70",
                    value === m.value && "bg-accent/50",
                    !m.available && "opacity-60",
                  )}
                  role="option"
                  aria-selected={value === m.value}
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
                  <Check
                    className={cn(
                      "size-3.5 shrink-0",
                      value === m.value ? "opacity-100" : "opacity-0",
                    )}
                  />
                </button>
              ))}
            </div>
          ))}
        </div>
      </PopoverContent>
    </Popover>
  );
}

/** Is this a model that supports reasoning_effort? Trusts the LiteLLM catalog flag exclusively. */
export function modelSupportsThinking(modelValue: string, models?: CatalogModel[]): boolean {
  if (!modelValue || !models) return false;
  const hit = models.find((m) => m.value === modelValue);
  return hit?.supports_thinking ?? false;
}
