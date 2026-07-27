"use client";

import * as React from "react";
import {
  ArrowDownLeft,
  BarChart3,
  Coins,
  Download,
  Gift,
  Plus,
  RefreshCw,
  Sparkles,
  type LucideIcon,
} from "lucide-react";
import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { toast } from "react-toastify";
import { ChartTooltip, ChartEmptyState } from "@/shared/charts/chart-utils";
import { ChartTable } from "@/shared/charts/chart-table";
import { useLiteMode } from "@/features/settings";
import { EmptyState } from "@/shared/ui/empty-state";
import { Button } from "@/shared/ui/primitives/button";
import {
  Tooltip as UITooltip,
  TooltipTrigger,
  TooltipContent,
} from "@/shared/ui/primitives/tooltip";
import { Popover as PopoverPrimitive } from "radix-ui";
import { formatMsg, msg, type MessageKey } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { getUsage, type BillingUsageEntry, type BillingUsageResponse } from "@/shared/lib/api";
import { SkynetDatePicker, toISODate } from "@/shared/ui/skynet-date-picker";
import { useCredits } from "../providers/credit-provider";
import { creditsToUsd, formatCredits, formatResetDate, formatUsd, type UsageEntry } from "../lib/credit";

/** A fixed look-back preset. `all` drops the lower bound. */
type PresetRange = "7d" | "30d" | "90d" | "all";
/** The selected window: a preset, or a user-picked `custom` from/to range. */
type RangeKey = PresetRange | "custom";
/** Time bucket for the spend-over-time chart. */
type GroupBy = "day" | "week";
/** Which slice of the ledger the activity list shows. */
type LedgerFilter = "all" | "costs";

const RANGE_DAYS: Record<PresetRange, number | null> = { "7d": 7, "30d": 30, "90d": 90, all: null };

const RANGE_LABEL: Record<RangeKey, MessageKey> = {
  "7d": "usage.range.7d",
  "30d": "usage.range.30d",
  "90d": "usage.range.90d",
  all: "usage.range.all",
  custom: "usage.range.custom",
};

const GROUP_LABEL: Record<GroupBy, MessageKey> = {
  day: "usage.group.day",
  week: "usage.group.week",
};

const LEDGER_FILTERS: ReadonlyArray<{ value: LedgerFilter; labelKey: MessageKey }> = [
  { value: "all", labelKey: "billing.wallet.filter_all" },
  { value: "costs", labelKey: "billing.wallet.filter_costs" },
];

// Run rows lead with a spark; top-ups/grants with their own glyph. Keyed loosely
// so an unrecognized backend kind still resolves to a sensible default.
const KIND_ICON: Record<string, React.ComponentType<{ className?: string }>> = {
  run: Sparkles,
  topup: Plus,
  grant: Gift,
};

// Warm monochrome ramp (matches --chart-1..5): billed spend anchors to the
// darkest step so the series stays legible without reaching for color outside
// the palette.
const MODEL_RAMP = [
  "var(--color-chart-1)",
  "var(--color-chart-2)",
  "var(--color-chart-3)",
  "var(--color-chart-4)",
  "var(--color-chart-5)",
];
const BILLED_FILL = "var(--color-chart-1)";

// Mirrors the wallet ledger pill so every segmented control in billing slides alike.
const PILL_TRANSITION = { type: "tween", duration: 0.16, ease: [0.22, 1, 0.36, 1] } as const;

const ENTRY_CAP = 200;

/** A spend-over-time bucket: billed credits under one axis label. */
interface Bucket {
  key: string;
  label: string;
  billed: number;
}

/** Resolve a range key to its ISO bounds (and ms, for the client-side fallback). */
function rangeBounds(
  range: RangeKey,
  customFrom: string | null,
  customTo: string | null,
): {
  startIso: string;
  endIso: string;
  startMs: number;
  endMs: number;
} {
  if (range === "custom") {
    // Picked YYYY-MM-DD dates are read as local-day bounds (inclusive of the
    // whole `to` day), then normalized to ISO instants for the API.
    const end = customTo ? new Date(`${customTo}T23:59:59.999`) : new Date();
    const start = customFrom
      ? new Date(`${customFrom}T00:00:00.000`)
      : new Date(end.getTime() - 30 * 86_400_000);
    return { startIso: start.toISOString(), endIso: end.toISOString(), startMs: start.getTime(), endMs: end.getTime() };
  }
  const end = new Date();
  const days = RANGE_DAYS[range];
  const start = days == null ? new Date(0) : new Date(end.getTime() - days * 86_400_000);
  return { startIso: start.toISOString(), endIso: end.toISOString(), startMs: start.getTime(), endMs: end.getTime() };
}

/**
 * Aggregate raw ledger rows into the dashboard shape — the same rollup the
 * backend `GET /billing/usage` performs. Used as the offline fallback so the tab
 * still renders (e.g. signed-out demo on the stub wallet) when the fetch fails.
 */
function deriveUsage(
  entries: UsageEntry[],
  startMs: number,
  endMs: number,
  startIso: string,
  endIso: string,
): BillingUsageResponse {
  const inRange = entries.filter((entry) => {
    const at = new Date(entry.at).getTime();
    return at >= startMs && at <= endMs;
  });
  let billed = 0;
  let runs = 0;
  const perDay = new Map<string, number>();
  const perModel = new Map<string | null, { credits: number; runs: number }>();
  for (const entry of inRange) {
    if (entry.kind !== "run" || entry.credits >= 0) continue;
    const spent = -entry.credits;
    billed += spent;
    runs += 1;
    const day = entry.at.slice(0, 10);
    perDay.set(day, (perDay.get(day) ?? 0) + spent);
    const model = perModel.get(entry.model) ?? { credits: 0, runs: 0 };
    model.credits += spent;
    model.runs += 1;
    perModel.set(entry.model, model);
  }
  return {
    start: startIso,
    end: endIso,
    billed_credits: billed,
    runs,
    by_day: [...perDay.entries()]
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, value]) => ({ date, billed_credits: value })),
    by_model: [...perModel.entries()]
      .sort(([, a], [, b]) => b.credits - a.credits)
      .map(([model, value]) => ({ model, credits: value.credits, runs: value.runs })),
    entries: inRange.slice(0, ENTRY_CAP),
  };
}

/** Locale-aware short date (e.g. `Jun 26`) for a `YYYY-MM-DD` bucket key. */
function formatDay(iso: string, locale: string): string {
  return new Intl.DateTimeFormat(locale, { month: "short", day: "numeric" }).format(
    new Date(`${iso}T00:00:00Z`),
  );
}

/** Monday-anchored ISO week key (`YYYY-MM-DD`) for a calendar day. */
function weekKey(iso: string): string {
  const date = new Date(`${iso}T00:00:00Z`);
  const offset = (date.getUTCDay() + 6) % 7;
  date.setUTCDate(date.getUTCDate() - offset);
  return date.toISOString().slice(0, 10);
}

/** Re-bucket the per-day series to the chosen granularity, ascending by time. */
function bucketDays(
  byDay: BillingUsageResponse["by_day"],
  mode: GroupBy,
  locale: string,
): Bucket[] {
  const map = new Map<string, number>();
  for (const day of byDay) {
    const key = mode === "week" ? weekKey(day.date) : day.date;
    map.set(key, (map.get(key) ?? 0) + day.billed_credits);
  }
  return [...map.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, billed]) => ({ key, label: formatDay(key, locale), billed }));
}

/** Quote a CSV cell only when it carries a comma, quote, or newline. */
function csvCell(value: string): string {
  return /[",\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

/** Download the in-range ledger as a CSV; toast the outcome. */
function exportCsv(entries: BillingUsageEntry[]): void {
  if (entries.length === 0) {
    toast.error(msg("usage.export.empty"));
    return;
  }
  const header = ["date", "label", "model", "kind", "credits"];
  const lines = [header, ...entries.map((e) => [e.at, e.label, e.model ?? "", e.kind, String(e.credits)])];
  const csv = lines.map((row) => row.map(csvCell).join(",")).join("\n");
  triggerDownload(csv, "skynet-usage.csv", "text/csv;charset=utf-8;");
  toast.success(formatMsg("usage.export.done", { p1: String(entries.length) }));
}

/** Download the in-range ledger as JSON; toast the outcome. Mirrors the CSV columns. */
function exportJson(entries: BillingUsageEntry[]): void {
  if (entries.length === 0) {
    toast.error(msg("usage.export.empty"));
    return;
  }
  const rows = entries.map((e) => ({
    date: e.at,
    label: e.label,
    model: e.model ?? null,
    kind: e.kind,
    credits: e.credits,
  }));
  triggerDownload(JSON.stringify(rows, null, 2), "skynet-usage.json", "application/json");
  toast.success(formatMsg("usage.export.done", { p1: String(entries.length) }));
}

/** Stream a blob to the browser as a file download. */
function triggerDownload(content: string, filename: string, mimeType: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mimeType }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function PanelHeading({ children }: { children: React.ReactNode }) {
  return (
    <p className="text-[0.6875rem] font-semibold uppercase tracking-widest text-muted-foreground">
      {children}
    </p>
  );
}

/** A sliding segmented control, matching the wallet ledger filter. */
function Segmented<T extends string>({
  options,
  value,
  onChange,
  ariaLabel,
  layoutId,
}: {
  options: ReadonlyArray<{ value: T; label: string }>;
  value: T;
  onChange: (next: T) => void;
  ariaLabel: string;
  layoutId: string;
}) {
  return (
    <div
      role="group"
      aria-label={ariaLabel}
      className="flex items-center gap-0.5 rounded-lg bg-muted/60 p-0.5"
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(option.value)}
            className={cn(
              "relative cursor-pointer rounded-md px-2.5 py-1 text-xs font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45",
              active ? "text-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {active && (
              <motion.span
                layoutId={layoutId}
                className="absolute inset-0 rounded-md bg-background shadow-[0_1px_2px_oklch(0.25_0.04_45/.12)]"
                transition={PILL_TRANSITION}
                aria-hidden="true"
              />
            )}
            <span className="relative z-10">{option.label}</span>
          </button>
        );
      })}
    </div>
  );
}

/** A compact icon button for the toolbar (refresh / export). */
function ToolbarButton({
  icon: Icon,
  label,
  onClick,
  spinning,
}: {
  icon: LucideIcon;
  label: string;
  onClick: () => void;
  spinning?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className="inline-flex size-8 cursor-pointer items-center justify-center rounded-md border border-border/60 text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
    >
      <Icon className={cn("size-3.5", spinning && "animate-spin")} aria-hidden="true" />
    </button>
  );
}

/** Export the in-range ledger from a compact icon dropdown: CSV or JSON. */
function ExportDropdown({ entries }: { entries: BillingUsageEntry[] }) {
  const formats = [
    { label: "CSV", run: () => exportCsv(entries) },
    { label: "JSON", run: () => exportJson(entries) },
  ];
  return (
    <PopoverPrimitive.Root>
      <UITooltip>
        <TooltipTrigger asChild>
          <PopoverPrimitive.Trigger asChild>
            <Button variant="ghost" size="icon-sm" aria-label={msg("usage.action.export")}>
              <Download className="size-4" />
            </Button>
          </PopoverPrimitive.Trigger>
        </TooltipTrigger>
        <TooltipContent>{msg("usage.action.export")}</TooltipContent>
      </UITooltip>
      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Content
          side="bottom"
          align="end"
          sideOffset={8}
          className="z-50 w-40 rounded-lg border bg-background p-1 shadow-lg animate-in fade-in-0 zoom-in-95"
        >
          {formats.map(({ label, run }) => (
            <PopoverPrimitive.Close key={label} asChild>
              <button
                type="button"
                onClick={run}
                className="flex w-full items-center rounded-md px-3 py-1.5 text-xs font-medium text-foreground cursor-pointer transition-colors hover:bg-accent"
              >
                {label}
              </button>
            </PopoverPrimitive.Close>
          ))}
        </PopoverPrimitive.Content>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
}

function StatCard({
  icon: Icon,
  label,
  value,
  secondary,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  secondary?: string;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-xl border border-border/60 bg-muted/20 px-4 py-3.5">
      <span className="flex items-center gap-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-muted-foreground">
        <Icon className="size-3.5" aria-hidden="true" />
        {label}
      </span>
      <span dir="ltr" className="text-2xl font-semibold tabular-nums text-foreground">
        {value}
      </span>
      <span dir="ltr" className="min-h-4 text-xs text-muted-foreground">
        {secondary}
      </span>
    </div>
  );
}

/** Billed credits over time. Lite mode falls back to a table. */
function SpendChart({ buckets }: { buckets: Bucket[] }) {
  const lite = useLiteMode();
  if (buckets.length === 0) return <ChartEmptyState message={msg("usage.empty.title")} />;
  if (lite) {
    return (
      <ChartTable
        rows={buckets}
        columns={[
          { key: "label", label: msg("usage.col.day") },
          { key: "billed", label: msg("usage.series.billed"), align: "end" },
        ]}
      />
    );
  }
  return (
    <div className="h-[220px] min-w-0">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={buckets} margin={{ left: 4, right: 4, top: 8, bottom: 4 }} barGap={2}>
          <CartesianGrid vertical={false} strokeDasharray="3 3" className="stroke-muted" />
          <XAxis
            dataKey="label"
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 10 }}
            className="fill-muted-foreground"
            interval="preserveStartEnd"
            minTickGap={18}
          />
          <YAxis
            tickLine={false}
            axisLine={false}
            tick={{ fontSize: 10 }}
            className="fill-muted-foreground"
            allowDecimals={false}
            width={36}
          />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--color-chart-5)", opacity: 0.35 }} />
          <Bar
            dataKey="billed"
            name={msg("usage.series.billed")}
            fill={BILLED_FILL}
            radius={[3, 3, 0, 0]}
            maxBarSize={26}
            animationDuration={300}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Per-model spend as a ranked bar list. Lite mode falls back to a table. */
/** Compact token count for the per-model rows ("1.2M"). */
function formatTokens(count: number, locale: string): string {
  return new Intl.NumberFormat(locale, { notation: "compact", maximumFractionDigits: 1 }).format(
    count,
  );
}

/** Total measured tokens behind a model row, or 0 when untracked (fallback rows). */
function rowTokens(row: BillingUsageResponse["by_model"][number]): number {
  return (row.input_tokens ?? 0) + (row.output_tokens ?? 0);
}

function ModelBreakdown({
  rows,
  locale,
}: {
  rows: BillingUsageResponse["by_model"];
  locale: string;
}) {
  const lite = useLiteMode();
  if (rows.length === 0) {
    return (
      <p className="py-6 text-center text-xs text-muted-foreground">{msg("usage.empty.title")}</p>
    );
  }
  if (lite) {
    return (
      <ChartTable
        rows={rows.map((row) => ({
          model: row.model ?? msg("usage.model.unknown"),
          credits: row.credits,
          runs: row.runs,
          tokens: rowTokens(row) > 0 ? formatTokens(rowTokens(row), locale) : "—",
        }))}
        columns={[
          { key: "model", label: msg("usage.col.model") },
          { key: "credits", label: msg("usage.col.credits"), align: "end" },
          { key: "runs", label: msg("usage.col.runs"), align: "end" },
          { key: "tokens", label: msg("usage.col.tokens"), align: "end" },
        ]}
      />
    );
  }
  const max = Math.max(...rows.map((row) => row.credits), 1);
  return (
    <ul className="flex flex-col gap-2.5">
      {rows.slice(0, 6).map((row, index) => {
        const label = row.model ?? msg("usage.model.unknown");
        const tokens = rowTokens(row);
        return (
          <li key={label} className="flex flex-col gap-1">
            <div className="flex items-baseline justify-between gap-3">
              <span dir="ltr" className="min-w-0 truncate text-xs text-foreground" title={label}>
                {label}
              </span>
              <span className="flex shrink-0 items-baseline gap-2">
                {tokens > 0 && (
                  <span
                    dir="ltr"
                    className="text-[11px] tabular-nums text-muted-foreground"
                    title={msg("usage.tokens.split", {
                      input: (row.input_tokens ?? 0).toLocaleString(locale),
                      output: (row.output_tokens ?? 0).toLocaleString(locale),
                    })}
                  >
                    {msg("usage.tokens.count", { count: formatTokens(tokens, locale) })}
                  </span>
                )}
                <span dir="ltr" className="text-xs font-medium tabular-nums text-foreground">
                  {formatCredits(row.credits, locale)}
                </span>
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.round((row.credits / max) * 100)}%`,
                  backgroundColor: MODEL_RAMP[index % MODEL_RAMP.length],
                }}
              />
            </div>
          </li>
        );
      })}
    </ul>
  );
}

/** The window's costliest runs, biggest spend first. */
function RunBreakdown({ entries, locale }: { entries: BillingUsageEntry[]; locale: string }) {
  const runs = entries
    .filter((entry) => entry.kind === "run" && entry.credits < 0)
    .sort((a, b) => a.credits - b.credits)
    .slice(0, 6);
  if (runs.length === 0) {
    return (
      <p className="py-6 text-center text-xs text-muted-foreground">{msg("usage.empty.title")}</p>
    );
  }
  return (
    <ul className="flex flex-col">
      {runs.map((run) => (
        <li
          key={run.id}
          className="flex items-center gap-3 border-b border-border/30 py-2 last:border-b-0"
        >
          <span className="grid size-7 shrink-0 place-items-center rounded-full bg-muted text-muted-foreground">
            <Sparkles className="size-3.5" aria-hidden="true" />
          </span>
          <span className="flex min-w-0 flex-1 flex-col">
            <span dir="auto" className="truncate text-sm text-foreground">
              {run.label}
            </span>
            {run.model && (
              <span dir="ltr" className="truncate text-[0.6875rem] text-muted-foreground">
                {run.model}
              </span>
            )}
          </span>
          <span dir="ltr" className="shrink-0 text-sm font-medium tabular-nums text-foreground">
            −{formatCredits(-run.credits, locale)}
          </span>
        </li>
      ))}
    </ul>
  );
}

/** One activity-list row. Numerals and model ids stay LTR-islanded. */
function LedgerRow({ entry, locale }: { entry: BillingUsageEntry; locale: string }) {
  const Icon = KIND_ICON[entry.kind] ?? Sparkles;
  const credited = entry.credits > 0;
  const free = entry.credits === 0;
  return (
    <li className="flex items-center gap-3 border-b border-border/30 py-2.5 last:border-b-0">
      <span className="grid size-7 shrink-0 place-items-center rounded-full bg-muted text-muted-foreground">
        {credited ? (
          <Icon className="size-3.5" aria-hidden="true" />
        ) : (
          <ArrowDownLeft className="size-3.5 rtl:-scale-x-100" aria-hidden="true" />
        )}
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <span dir="auto" className="truncate text-sm text-foreground">
          {entry.label}
        </span>
        {entry.model && (
          <span dir="ltr" className="truncate text-[0.6875rem] text-muted-foreground">
            {entry.model}
          </span>
        )}
      </span>
      <span className="flex shrink-0 flex-col items-end">
        {free ? (
          <span className="text-xs font-medium text-muted-foreground">
            {msg("billing.history.byok_tag")}
          </span>
        ) : (
          <span
            dir="ltr"
            className={cn(
              "text-sm font-medium tabular-nums",
              credited ? "text-foreground" : "text-muted-foreground",
            )}
          >
            {credited ? "+" : "−"}
            {formatCredits(Math.abs(entry.credits), locale)}
          </span>
        )}
        <span dir="ltr" className="text-[0.6875rem] text-muted-foreground/70">
          {formatResetDate(entry.at, locale)}
        </span>
      </span>
    </li>
  );
}

/**
 * Usage — the `usage` settings tab.
 *
 * A full personal spend dashboard over the managed-credit ledger: a date-ranged
 * window, headline stats (spent · runs), a billed-spend time series, per-model
 * and per-run breakdowns, and the raw activity list. Data
 * comes from the backend `GET /billing/usage` rollup; if that read fails the tab
 * derives the same shape client-side from the wallet's recent ledger so the
 * surface still renders.
 */
export function UsageTab() {
  const { wallet } = useCredits();
  const { locale } = useLocale();
  const [range, setRange] = React.useState<RangeKey>("30d");
  const [customFrom, setCustomFrom] = React.useState<string | null>(null);
  const [customTo, setCustomTo] = React.useState<string | null>(null);
  const [groupBy, setGroupBy] = React.useState<GroupBy>("day");
  const [data, setData] = React.useState<BillingUsageResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [usageFilter, setUsageFilter] = React.useState<LedgerFilter>("all");
  const reqId = React.useRef(0);

  const todayIso = React.useMemo(() => toISODate(new Date()), []);

  // Switching to Custom seeds a sensible 30-day window so the pickers never open
  // empty; an existing custom selection is preserved across toggles.
  const onRangeChange = React.useCallback(
    (next: RangeKey) => {
      if (next === "custom" && customFrom == null && customTo == null) {
        setCustomFrom(toISODate(new Date(Date.now() - 30 * 86_400_000)));
        setCustomTo(toISODate(new Date()));
      }
      setRange(next);
    },
    [customFrom, customTo],
  );

  const load = React.useCallback(() => {
    const { startIso, endIso, startMs, endMs } = rangeBounds(range, customFrom, customTo);
    const id = ++reqId.current;
    setLoading(true);
    getUsage(startIso, endIso)
      .then((resp) => {
        if (id === reqId.current) setData(resp);
      })
      .catch(() => {
        if (id === reqId.current) setData(deriveUsage(wallet.usage, startMs, endMs, startIso, endIso));
      })
      .finally(() => {
        if (id === reqId.current) setLoading(false);
      });
  }, [range, customFrom, customTo, wallet.usage]);

  React.useEffect(() => {
    load();
  }, [load]);

  const buckets = React.useMemo(
    () => bucketDays(data?.by_day ?? [], groupBy, locale),
    [data?.by_day, groupBy, locale],
  );

  const entries = data?.entries ?? [];
  const visibleEntries = React.useMemo(() => {
    if (usageFilter === "costs") return entries.filter((entry) => entry.credits < 0);
    return entries;
  }, [entries, usageFilter]);

  const rangeOptions = (Object.keys(RANGE_DAYS) as RangeKey[]).map((value) => ({
    value,
    label: msg(RANGE_LABEL[value]),
  }));
  const groupOptions = (Object.keys(GROUP_LABEL) as GroupBy[]).map((value) => ({
    value,
    label: msg(GROUP_LABEL[value]),
  }));
  const filterOptions = LEDGER_FILTERS.map(({ value, labelKey }) => ({ value, label: msg(labelKey) }));

  const toolbar = (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented
          options={rangeOptions}
          value={range}
          onChange={onRangeChange}
          ariaLabel={msg("usage.range.label")}
          layoutId="usage-range-pill"
        />
        <div className="flex items-center gap-2">
          <Segmented
            options={groupOptions}
            value={groupBy}
            onChange={setGroupBy}
            ariaLabel={msg("usage.group.label")}
            layoutId="usage-group-pill"
          />
          <ToolbarButton
            icon={RefreshCw}
            label={msg("usage.action.refresh")}
            onClick={load}
            spinning={loading}
          />
          <ExportDropdown entries={entries} />
        </div>
      </div>
      {range === "custom" && (
        <div className="flex flex-wrap items-center gap-2">
          <div className="w-[150px]">
            <SkynetDatePicker
              value={customFrom}
              onChange={setCustomFrom}
              max={customTo ?? todayIso}
              ariaLabel={msg("usage.range.from")}
              placeholder={msg("usage.range.from")}
            />
          </div>
          <span className="text-xs text-muted-foreground" aria-hidden="true">
            –
          </span>
          <div className="w-[150px]">
            <SkynetDatePicker
              value={customTo}
              onChange={setCustomTo}
              min={customFrom}
              max={todayIso}
              ariaLabel={msg("usage.range.to")}
              placeholder={msg("usage.range.to")}
            />
          </div>
        </div>
      )}
    </div>
  );

  if (data == null) {
    return (
      <div className="flex flex-col gap-5">
        {toolbar}
        <div className="flex h-64 items-center justify-center" aria-busy="true">
          <RefreshCw className="size-5 animate-spin text-muted-foreground" aria-hidden="true" />
        </div>
      </div>
    );
  }

  const totallyEmpty = !loading && entries.length === 0 && data.billed_credits === 0;

  if (totallyEmpty) {
    return (
      <div className="flex flex-col gap-5">
        {toolbar}
        <EmptyState
          variant="list"
          icon={BarChart3}
          title={msg("usage.empty.title")}
          description={msg("usage.empty.desc")}
        />
      </div>
    );
  }

  const avgPerRun =
    data.runs > 0
      ? formatMsg("usage.stat.per_run", {
          p1: formatCredits(Math.round(data.billed_credits / data.runs), locale),
        })
      : undefined;

  return (
    <div className="flex flex-col gap-6">
      {toolbar}

      <div className="grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))" }}>
        <StatCard
          icon={Coins}
          label={msg("usage.stat.spent")}
          value={formatCredits(data.billed_credits, locale)}
          secondary={formatUsd(creditsToUsd(data.billed_credits), locale)}
        />
        <StatCard
          icon={Sparkles}
          label={msg("usage.stat.runs")}
          value={formatCredits(data.runs, locale)}
          secondary={avgPerRun}
        />
      </div>

      <div className="flex flex-col gap-3">
        <PanelHeading>{msg("usage.panel.over_time")}</PanelHeading>
        <SpendChart buckets={buckets} />
      </div>

      <div
        className="grid gap-x-8 gap-y-6"
        style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(240px, 100%), 1fr))" }}
      >
        <div className="flex flex-col gap-3">
          <PanelHeading>{msg("usage.panel.by_model")}</PanelHeading>
          <ModelBreakdown rows={data.by_model} locale={locale} />
        </div>
        <div className="flex flex-col gap-3">
          <PanelHeading>{msg("usage.panel.by_run")}</PanelHeading>
          <RunBreakdown entries={entries} locale={locale} />
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-3">
          <PanelHeading>{msg("usage.panel.recent")}</PanelHeading>
          {entries.length > 0 && (
            <Segmented
              options={filterOptions}
              value={usageFilter}
              onChange={setUsageFilter}
              ariaLabel={msg("usage.panel.recent")}
              layoutId="usage-ledger-filter-pill"
            />
          )}
        </div>
        {visibleEntries.length === 0 ? (
          <p className="px-1 py-6 text-center text-xs text-muted-foreground">
            {msg("billing.wallet.filter_empty")}
          </p>
        ) : (
          <ul className="flex max-h-72 flex-col overflow-y-auto pe-1">
            {visibleEntries.map((entry) => (
              <LedgerRow key={entry.id} entry={entry} locale={locale} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
