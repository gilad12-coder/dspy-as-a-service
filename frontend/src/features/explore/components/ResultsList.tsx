"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowUp, ArrowDown } from "lucide-react";
import { msg, formatMsg } from "@/shared/lib/messages";
import type { SearchResult } from "@/shared/lib/api";
import type { SearchType } from "../hooks/use-semantic-search";
import { formatGain, formatMetric, formatRelativeDate } from "../lib/format";

interface ResultsListProps {
  results: SearchResult[];
  /** Highlight any token from the query inside the title/snippet. Empty = no highlight. */
  highlight: string;
  /** Which backend branch served the query — drives the per-row source badge. */
  searchType: SearchType | null;
  /** Keyboard-highlighted row index, or -1. Driven by the search input's ↑/↓. */
  activeIndex: number;
  /** Fired when a row is opened — the explicit-commit signal for query trending. */
  onResultOpen: () => void;
}

/**
 * Vertically-rhythmic list of search hits. Each row is a single-tap card:
 * title with an inline score+delta tag, two-line summary, and a thin meta
 * strip carrying a relative timestamp at the end — plus a relevance badge at
 * the start on semantic searches.
 *
 * Hover lifts the title to full-opacity; the row itself is the open affordance.
 */
export function ResultsList({
  results,
  highlight,
  searchType,
  activeIndex,
  onResultOpen,
}: ResultsListProps) {
  const tokens = React.useMemo(() => tokenize(highlight), [highlight]);
  return (
    <ul id="explore-results" className="divide-y divide-border/55">
      {results.map((row, index) => (
        <li key={row.optimization_id}>
          <ResultRow
            row={row}
            index={index}
            active={index === activeIndex}
            tokens={tokens}
            searchType={searchType}
            onOpen={onResultOpen}
          />
        </li>
      ))}
    </ul>
  );
}

function ResultRow({
  row,
  index,
  active,
  tokens,
  searchType,
  onOpen,
}: {
  row: SearchResult;
  index: number;
  active: boolean;
  tokens: string[];
  searchType: SearchType | null;
  onOpen: () => void;
}) {
  const title = row.task_name?.trim() || msg("explore.row.no_summary");
  const gain = formatGain(row.baseline_metric, row.optimized_metric);
  const dateText = formatRelativeDate(row.created_at);
  const summary = row.summary_text?.trim();
  const ref = React.useRef<HTMLAnchorElement | null>(null);

  React.useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest" });
  }, [active]);

  return (
    <Link
      ref={ref}
      id={`explore-result-${index}`}
      href={`/optimizations/${row.optimization_id}`}
      onClick={onOpen}
      aria-label={formatMsg("explore.row.open_aria", { name: title })}
      data-active={active || undefined}
      className={`group relative flex flex-col gap-2 rounded-lg px-3 py-4 transition-[background-color,transform] duration-150 ease-out cursor-pointer hover:bg-accent/30 focus-visible:outline-none focus-visible:bg-accent/40 focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-[#C8A882]/45 ${
        active ? "bg-accent/40 ring-2 ring-inset ring-[#C8A882]/45" : ""
      }`}
    >
      <div className="flex items-baseline justify-between gap-4">
        <h3 className="min-w-0 flex-1 text-start text-[15.5px] font-medium leading-snug tracking-tight text-foreground/90 transition-colors group-hover:text-foreground">
          <Highlighted text={title} tokens={tokens} />
        </h3>
        {row.optimized_metric != null && (
          <ScoreTag score={row.optimized_metric} gain={gain} />
        )}
      </div>

      {summary && (
        <p className="line-clamp-2 max-w-[72ch] text-start text-[13.5px] leading-relaxed text-foreground/55">
          <Highlighted text={summary} tokens={tokens} />
        </p>
      )}

      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[11.5px] text-foreground/50">
        {searchType === "semantic" && row.relevance != null && (
          <RelevanceBadge relevance={row.relevance} />
        )}
        <time
          dateTime={row.created_at ?? undefined}
          title={row.created_at ?? undefined}
          className="ms-auto text-foreground/45 tabular-nums"
        >
          {dateText}
        </time>
      </div>
    </Link>
  );
}

function RelevanceBadge({ relevance }: { relevance: number }) {
  // Cosine similarity is in [-1, 1] but in practice falls in [0, 1] for
  // sentence embeddings. Clamp defensively and render as a 0–100 score so
  // users have an intuitive ranking signal next to each row.
  const pct = Math.max(0, Math.min(1, relevance)) * 100;
  const label = formatMsg("explore.row.relevance", { pct: pct.toFixed(0) });
  return (
    <span
      dir="ltr"
      className="inline-flex items-baseline gap-1 rounded-full bg-[oklch(0.94_0.03_82)] px-2 py-0.5 font-mono text-[10.5px] font-medium leading-none text-[oklch(0.42_0.10_82)] tabular-nums"
      title={msg("explore.row.relevance.title")}
    >
      <span>{label}</span>
    </span>
  );
}

function ScoreTag({
  score,
  gain,
}: {
  score: number | null | undefined;
  gain: { text: string; kind: "positive" | "negative" | "neutral" } | null;
}) {
  // A zero-change run carries no direction, so the neutral badge drops the
  // ± arrow and recedes to muted text — only real movement earns an icon. A
  // run without a baseline has no gain at all; the score still stands alone.
  const Icon =
    gain?.kind === "positive" ? ArrowUp : gain?.kind === "negative" ? ArrowDown : null;
  return (
    <span
      dir="ltr"
      className="inline-flex shrink-0 items-baseline gap-2 font-mono text-[12.5px] tabular-nums"
    >
      <span className="font-semibold text-foreground">{formatMetric(score)}</span>
      {gain && (
        <span
          className={`inline-flex items-baseline gap-0.5 text-[11px] ${GAIN_TONE[gain.kind]}`}
        >
          {Icon && <Icon className="size-2.5 self-center" aria-hidden="true" />}
          <span>{gain.text}</span>
        </span>
      )}
    </span>
  );
}

// A no-change gain recedes to bare muted text while ±gains stay filled
// pills, so colour is reserved for runs that actually moved the metric —
// the neutral case is the least interesting outcome and shouldn't compete.
const GAIN_TONE: Record<"positive" | "negative" | "neutral", string> = {
  positive: "rounded-full px-1.5 py-0.5 bg-[var(--success-dim)] text-[var(--success)]",
  negative: "rounded-full px-1.5 py-0.5 bg-[var(--danger-dim)] text-[var(--danger)]",
  neutral: "text-muted-foreground",
};

function tokenize(query: string): string[] {
  return query
    .trim()
    .split(/\s+/)
    .filter((t) => t.length >= 2)
    .map((t) => t.toLocaleLowerCase());
}

function Highlighted({ text, tokens }: { text: string; tokens: string[] }) {
  if (tokens.length === 0) return <>{text}</>;
  const segments = highlightSegments(text, tokens);
  return (
    <>
      {segments.map((seg, i) =>
        seg.match ? (
          <mark
            key={i}
            className="bg-transparent font-semibold text-foreground underline decoration-[#C8A882] decoration-[1.5px] underline-offset-[3px]"
          >
            {seg.text}
          </mark>
        ) : (
          <React.Fragment key={i}>{seg.text}</React.Fragment>
        ),
      )}
    </>
  );
}

type Segment = { text: string; match: boolean };

function highlightSegments(text: string, tokens: string[]): Segment[] {
  const lower = text.toLocaleLowerCase();
  const ranges: Array<{ start: number; end: number }> = [];
  for (const token of tokens) {
    let from = 0;
    while (from <= lower.length - token.length) {
      const idx = lower.indexOf(token, from);
      if (idx === -1) break;
      ranges.push({ start: idx, end: idx + token.length });
      from = idx + token.length;
    }
  }
  if (ranges.length === 0) return [{ text, match: false }];
  ranges.sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: Array<{ start: number; end: number }> = [];
  for (const r of ranges) {
    const last = merged[merged.length - 1];
    if (last && r.start <= last.end) {
      last.end = Math.max(last.end, r.end);
    } else {
      merged.push({ ...r });
    }
  }
  const segs: Segment[] = [];
  let cursor = 0;
  for (const r of merged) {
    if (r.start > cursor) segs.push({ text: text.slice(cursor, r.start), match: false });
    segs.push({ text: text.slice(r.start, r.end), match: true });
    cursor = r.end;
  }
  if (cursor < text.length) segs.push({ text: text.slice(cursor), match: false });
  return segs;
}
