"use client";

import * as React from "react";
import { Clock, TrendingUp } from "lucide-react";
import { msg } from "@/shared/lib/messages";

interface SearchSuggestionsProps {
  recent: string[];
  /** Trending public queries, ranked server-side. Plain text, Google-style — no counts. */
  suggestions: string[];
  onSelect: (value: string) => void;
  onClearRecent: () => void;
}

/**
 * Dropdown shown when the search input is focused and empty. Surfaces the
 * user's own recent queries (persisted locally) plus the trending public
 * queries, so a blank field isn't a dead end. Selecting one commits it as the
 * query.
 *
 * The panel uses onMouseDown/preventDefault so picking an item doesn't blur
 * the input — which would otherwise close the panel before the click lands.
 */
export function SearchSuggestions({
  recent,
  suggestions,
  onSelect,
  onClearRecent,
}: SearchSuggestionsProps) {
  const hasRecent = recent.length > 0;
  const hasSuggestions = suggestions.length > 0;
  if (!hasRecent && !hasSuggestions) return null;

  return (
    <div
      onMouseDown={(e) => e.preventDefault()}
      className="absolute inset-x-0 top-[calc(100%+0.4rem)] z-20 max-h-[min(70vh,420px)] overflow-y-auto overscroll-contain rounded-2xl border border-border bg-background p-2 shadow-[0_8px_40px_-12px_oklch(0.25_0.04_45/.22)]"
    >
      {hasRecent && (
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center justify-between px-2 py-1">
            <span className="text-[11px] font-medium tracking-wide text-foreground/40">
              {msg("explore.suggest.recent")}
            </span>
            <button
              type="button"
              onClick={onClearRecent}
              className="rounded px-1.5 py-0.5 text-[11px] text-foreground/45 transition-colors cursor-pointer hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              {msg("explore.suggest.clear")}
            </button>
          </div>
          {recent.map((q) => (
            <button
              key={q}
              type="button"
              onClick={() => onSelect(q)}
              className="flex items-center gap-2.5 rounded-lg px-2 py-1.5 text-start text-[13.5px] text-foreground/80 transition-colors cursor-pointer hover:bg-accent focus-visible:outline-none focus-visible:bg-accent"
            >
              <Clock className="size-3.5 shrink-0 text-foreground/35" aria-hidden="true" />
              <span className="min-w-0 truncate">{q}</span>
            </button>
          ))}
        </div>
      )}

      {hasSuggestions && (
        <div className="flex flex-col gap-1 pt-1">
          {hasRecent && <div className="mx-2 mb-1 border-t border-border/60" aria-hidden="true" />}
          <span className="px-2 py-1 text-[11px] font-medium tracking-wide text-foreground/40">
            {msg("explore.suggest.popular")}
          </span>
          <div className="flex flex-wrap gap-1.5 px-1 pb-1">
            {suggestions.map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => onSelect(s)}
                className="inline-flex max-w-full items-center gap-1.5 rounded-full border border-border bg-background px-2.5 py-1 text-[12.5px] text-foreground/80 transition-colors cursor-pointer hover:border-foreground/30 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
              >
                <TrendingUp className="size-3 shrink-0 text-foreground/35" aria-hidden="true" />
                <span dir="auto" className="min-w-0 truncate">{s}</span>
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
