"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { ArrowUpDown } from "lucide-react";
import { msg, formatMsg } from "@/shared/lib/messages";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import type { SearchSort } from "@/shared/lib/api";

// Matches the corpus toggle's pill so both segmented controls animate alike.
const PILL_TRANSITION = { type: "tween", duration: 0.18, ease: [0.22, 1, 0.36, 1] } as const;

interface ResultsToolbarProps {
  total: number;
  sort: SearchSort;
  onSortChange: (sort: SearchSort) => void;
  /** Relevance is only offered when there's a query to rank against. */
  hasQuery: boolean;
}

/**
 * Thin bar above the results list: a live result count anchored to the
 * leading edge and the sort control on the trailing edge.
 */
export function ResultsToolbar({ total, sort, onSortChange, hasQuery }: ResultsToolbarProps) {
  const countLabel =
    total === 1
      ? msg("explore.results.count.one")
      : formatMsg("explore.results.count.many", { n: total });
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 px-1 pb-2">
      <span className="min-w-0 text-[12.5px] text-foreground/55 tabular-nums">{countLabel}</span>
      <SortControl sort={sort} onChange={onSortChange} hasQuery={hasQuery} />
    </div>
  );
}

const SORT_OPTIONS = [
  {
    value: "relevance" as const,
    label: () => msg("explore.sort.relevance"),
    tip: () => msg("explore.sort.relevance.tip"),
    queryOnly: true,
  },
  {
    value: "recent" as const,
    label: () => msg("explore.sort.recent"),
    tip: () => msg("explore.sort.recent.tip"),
    queryOnly: false,
  },
  {
    value: "gain" as const,
    label: () => msg("explore.sort.gain"),
    tip: () => msg("explore.sort.gain.tip"),
    queryOnly: false,
  },
];

function SortControl({
  sort,
  onChange,
  hasQuery,
}: {
  sort: SearchSort;
  onChange: (sort: SearchSort) => void;
  hasQuery: boolean;
}) {
  const options = SORT_OPTIONS.filter((o) => !o.queryOnly || hasQuery);
  return (
    <div
      role="group"
      aria-label={msg("explore.sort.aria")}
      className="inline-flex items-center gap-0.5 rounded-lg border border-border/70 bg-muted/30 p-0.5"
    >
      <ArrowUpDown className="mx-1 size-3 text-foreground/35" aria-hidden="true" />
      {options.map((o) => {
        const active = o.value === sort;
        return (
          <TooltipButton key={o.value} tooltip={o.tip()} side="bottom">
            <button
              type="button"
              aria-pressed={active}
              onClick={() => {
                if (!active) onChange(o.value);
              }}
              className={`relative rounded-md px-2.5 py-1 text-[12px] font-medium transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 ${
                active
                  ? "text-foreground"
                  : "cursor-pointer text-foreground/55 hover:text-foreground"
              }`}
            >
              {active && (
                <motion.span
                  layoutId="explore-sort-pill"
                  className="absolute inset-0 rounded-md bg-background shadow-[0_1px_2px_oklch(0.25_0.04_45/.12)]"
                  transition={PILL_TRANSITION}
                  aria-hidden="true"
                />
              )}
              <span className="relative z-10">{o.label()}</span>
            </button>
          </TooltipButton>
        );
      })}
    </div>
  );
}
