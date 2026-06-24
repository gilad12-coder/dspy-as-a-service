"use client";

import * as React from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { msg, formatMsg } from "@/shared/lib/messages";

interface PaginationProps {
  page: number;
  size: number;
  total: number;
  onPageChange: (next: number) => void;
}

/**
 * Bottom-of-results pager: numbered pages + prev/next, centered.
 * Hidden entirely when all results fit on a single page.
 */
export function Pagination({
  page,
  size,
  total,
  onPageChange,
}: PaginationProps) {
  const totalPages = Math.max(1, Math.ceil(total / size));
  const pages = React.useMemo(() => pageList(page, totalPages), [page, totalPages]);

  if (totalPages <= 1) return null;

  return (
    <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
      <nav
        aria-label={msg("explore.page.indicator")}
        className="flex flex-wrap items-center justify-center gap-1"
      >
        <PageNavButton
          disabled={page <= 1}
          onClick={() => onPageChange(page - 1)}
          aria="prev"
        >
          <ChevronRight className="size-4" aria-hidden="true" />
          <span>{msg("explore.page.prev")}</span>
        </PageNavButton>
        {pages.map((entry, idx) =>
          entry === "ellipsis" ? (
            <span
              key={`gap-${idx}`}
              className="px-1 text-foreground/40 tabular-nums"
              aria-hidden="true"
            >
              …
            </span>
          ) : (
            <PageNumber
              key={entry}
              value={entry}
              active={entry === page}
              onClick={() => onPageChange(entry)}
            />
          ),
        )}
        <PageNavButton
          disabled={page >= totalPages}
          onClick={() => onPageChange(page + 1)}
          aria="next"
        >
          <span>{msg("explore.page.next")}</span>
          <ChevronLeft className="size-4" aria-hidden="true" />
        </PageNavButton>
      </nav>
    </div>
  );
}

function PageNumber({
  value,
  active,
  onClick,
}: {
  value: number;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={formatMsg("explore.page.jump", { page: value })}
      aria-current={active ? "page" : undefined}
      className={`inline-flex h-8 min-w-8 items-center justify-center rounded-lg px-2 text-[12.5px] tabular-nums transition-[background-color,color] cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 ${
        active
          ? "bg-foreground text-background"
          : "text-foreground/65 hover:bg-accent hover:text-foreground"
      }`}
    >
      {value}
    </button>
  );
}

function PageNavButton({
  children,
  disabled,
  onClick,
  aria,
}: {
  children: React.ReactNode;
  disabled: boolean;
  onClick: () => void;
  aria: "prev" | "next";
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={msg(aria === "prev" ? "explore.page.prev" : "explore.page.next")}
      className="inline-flex h-8 items-center gap-1 rounded-lg px-2.5 text-[12.5px] text-foreground/65 transition-[background-color,color,opacity] hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-foreground/65 enabled:cursor-pointer"
    >
      {children}
    </button>
  );
}

/**
 * Compact page-number list with ellipses around the current page.
 *
 * For ≤ 7 pages we render everything. Beyond that we always show the
 * first and last page, plus a tight window of ±1 around the current page,
 * collapsing the rest into ellipses.
 */
function pageList(current: number, total: number): Array<number | "ellipsis"> {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i + 1);
  const set = new Set<number>([1, total, current, current - 1, current + 1]);
  if (current <= 4) {
    set.add(2);
    set.add(3);
    set.add(4);
    set.add(5);
  }
  if (current >= total - 3) {
    set.add(total - 4);
    set.add(total - 3);
    set.add(total - 2);
    set.add(total - 1);
  }
  const sorted = Array.from(set)
    .filter((n) => n >= 1 && n <= total)
    .sort((a, b) => a - b);
  const out: Array<number | "ellipsis"> = [];
  let prev = 0;
  for (const n of sorted) {
    if (n - prev > 1) out.push("ellipsis");
    out.push(n);
    prev = n;
  }
  return out;
}
