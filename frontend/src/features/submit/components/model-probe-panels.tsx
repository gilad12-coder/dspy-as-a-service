import * as React from "react";
import { motion } from "framer-motion";
import {
  Activity,
  ArrowRight,
  Check,
  Circle,
  Crown,
  Loader2,
  Medal,
  Timer,
  TrendingUp,
  XCircle,
} from "lucide-react";
import { toast } from "react-toastify";

import { Badge } from "@/shared/ui/primitives/badge";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent } from "@/shared/ui/primitives/card";
import { Separator } from "@/shared/ui/primitives/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/primitives/tabs";
import { Table, TableBody, TableCell, TableHeader, TableRow } from "@/shared/ui/primitives/table";
import { formatLogTimestamp, logTimeBucket } from "@/shared/lib";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import {
  ColumnHeader,
  useColumnFilters,
  useColumnResize,
  type SortDir,
} from "@/shared/ui/excel-filter";
import { AnimatedNumber, FadeIn, TiltCard } from "@/shared/ui/motion";

import {
  rowAsymptote,
  rowColor,
  type ModelRow,
  type ProbeLogEntry,
  type RowStatus,
} from "./model-probe-model";
import { TrajectoryDetailChart, TrajectorySparkline } from "./model-probe-charts";

export { TrajectoryCompareChart } from "./model-probe-charts";

// `navigator.clipboard` is only defined in secure contexts (HTTPS / localhost).
// Plain-HTTP previews would otherwise throw a TypeError when the user clicks
// a log cell to copy. Falls back to a hidden textarea + execCommand("copy"),
// which works in any context but is deprecated — try the modern API first.
async function copyTextToClipboard(text: string): Promise<boolean> {
  if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // permission denied or insecure context — fall through
    }
  }
  if (typeof document === "undefined") return false;
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }
  document.body.removeChild(ta);
  return ok;
}

export function ModelDetailPanel({
  row,
  now,
  tabCls,
}: {
  row: ModelRow;
  now: number;
  tabCls: string;
}) {
  const elapsed =
    row.status === "running" && row.startedAt !== null
      ? now - row.startedAt
      : (row.durationMs ?? 0);

  const logScrollRef = React.useRef<HTMLDivElement>(null);
  const logFilters = useColumnFilters();
  const logResize = useColumnResize();
  const [logSortKey, setLogSortKey] = React.useState<string>("timestamp");
  const [logSortDir, setLogSortDir] = React.useState<SortDir>("desc");
  const toggleLogSort = (key: string) => {
    if (logSortKey === key) setLogSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setLogSortKey(key);
      setLogSortDir("asc");
    }
  };

  const filteredLogs = React.useMemo(() => {
    let result = row.logs.filter((l) => {
      for (const [col, allowed] of Object.entries(logFilters.filters)) {
        if (allowed.size === 0) continue;
        const val =
          col === "timestamp"
            ? logTimeBucket(l.timestamp)
            : String((l as unknown as Record<string, unknown>)[col] ?? "");
        if (!allowed.has(val)) return false;
      }
      return true;
    });
    result = [...result].sort((a, b) => {
      const av = String((a as unknown as Record<string, unknown>)[logSortKey] ?? "");
      const bv = String((b as unknown as Record<string, unknown>)[logSortKey] ?? "");
      const cmp = av.localeCompare(bv, "he", { numeric: true });
      return logSortDir === "asc" ? cmp : -cmp;
    });
    return result;
  }, [row.logs, logFilters.filters, logSortKey, logSortDir]);

  const logFilterOptions = React.useMemo(() => {
    const unique = (key: keyof ProbeLogEntry) =>
      [...new Set(row.logs.map((l) => l[key]))]
        .filter(Boolean)
        .sort()
        .map((v) => ({ value: v, label: v }));
    const timestampBuckets = [...new Set(row.logs.map((l) => logTimeBucket(l.timestamp)))]
      .filter(Boolean)
      .sort()
      .map((v) => ({ value: v, label: v }));
    return { timestamp: timestampBuckets, level: unique("level"), logger: unique("logger") };
  }, [row.logs]);

  const liveAsymptote =
    (row.scaling?.signal === "strong" || row.scaling?.signal === "observed") &&
    row.scaling?.asymptote != null
      ? row.scaling.asymptote
      : null;
  const liveLast = row.trajectory.length ? row.trajectory[row.trajectory.length - 1]!.score : null;
  const livePrediction = liveAsymptote ?? liveLast;

  const signalLabel =
    row.scaling?.signal === "strong"
      ? msg("submit.probe.asymptote_label")
      : row.scaling?.signal === "observed"
        ? msg("submit.probe.observed_label")
        : msg("submit.probe.signal_weak");

  React.useEffect(() => {
    const el = logScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [row.logs.length]);

  return (
    <Tabs defaultValue="overview" className="gap-3">
      <TabsList variant="line" className="w-full border-b border-border/50 pb-0 gap-0">
        <TabsTrigger value="overview" className={cn(tabCls, "flex-1")}>
          {msg("submit.probe.details.title")}
        </TabsTrigger>
        <TabsTrigger value="logs" className={cn(tabCls, "flex-1")}>
          {msg("submit.probe.details.logs")}
        </TabsTrigger>
      </TabsList>

      <TabsContent value="overview" className="space-y-4 mt-4">
        <FadeIn>
          <div className="flex flex-wrap gap-2.5">
            <ProbeInfoCard
              icon={<TrendingUp className="size-3.5" />}
              label={signalLabel}
              value={
                livePrediction !== null ? (
                  <span className="font-mono tabular-nums" dir="ltr">
                    {livePrediction.toFixed(1)}
                  </span>
                ) : (
                  "—"
                )
              }
            />
            <ProbeInfoCard
              icon={<Activity className="size-3.5" />}
              label={msg("submit.probe.details.points_count")}
              value={row.trajectory.length.toString()}
            />
            <ProbeInfoCard
              icon={<Timer className="size-3.5" />}
              label={msg("auto.features.submit.components.modelprobedialog.literal.8")}
              value={
                <span className="font-mono tabular-nums" dir="ltr">
                  {(elapsed / 1000).toFixed(1)}
                  {msg("auto.features.submit.components.modelprobedialog.24")}
                </span>
              }
            />
          </div>
          {row.trajectory.length > 0 ? (
            <>
              <TrajectoryDetailChart points={row.trajectory} asymptote={liveAsymptote} />
            </>
          ) : null}
        </FadeIn>
      </TabsContent>

      <TabsContent value="logs" className="mt-4">
        <FadeIn>
          {row.logs.length > 0 ? (
            <Card>
              <CardContent className="p-0">
                <div ref={logScrollRef} dir="ltr" className="max-h-[400px] overflow-auto">
                  <Table className="table-fixed w-full">
                    <colgroup>
                      <col style={{ width: logResize.widths["timestamp"] ?? "15%" }} />
                      <col style={{ width: logResize.widths["level"] ?? "12%" }} />
                      <col style={{ width: logResize.widths["logger"] ?? "17%" }} />
                      <col />
                    </colgroup>
                    <TableHeader>
                      <TableRow>
                        <ColumnHeader
                          label={msg("auto.features.submit.components.modelprobedialog.literal.9")}
                          sortKey="timestamp"
                          currentSort={logSortKey}
                          sortDir={logSortDir}
                          onSort={toggleLogSort}
                          filterCol="timestamp"
                          filterOptions={logFilterOptions.timestamp}
                          filters={logFilters.filters}
                          onFilter={logFilters.setColumnFilter}
                          openFilter={logFilters.openFilter}
                          setOpenFilter={logFilters.setOpenFilter}
                          width={logResize.widths["timestamp"]}
                          onResize={logResize.setColumnWidth}
                        />
                        <ColumnHeader
                          label={msg("auto.features.submit.components.modelprobedialog.literal.10")}
                          sortKey="level"
                          currentSort={logSortKey}
                          sortDir={logSortDir}
                          onSort={toggleLogSort}
                          filterCol="level"
                          filterOptions={logFilterOptions.level}
                          filters={logFilters.filters}
                          onFilter={logFilters.setColumnFilter}
                          openFilter={logFilters.openFilter}
                          setOpenFilter={logFilters.setOpenFilter}
                          width={logResize.widths["level"]}
                          onResize={logResize.setColumnWidth}
                        />
                        <ColumnHeader
                          label={msg("auto.features.submit.components.modelprobedialog.literal.11")}
                          sortKey="logger"
                          currentSort={logSortKey}
                          sortDir={logSortDir}
                          onSort={toggleLogSort}
                          filterCol="logger"
                          filterOptions={logFilterOptions.logger}
                          filters={logFilters.filters}
                          onFilter={logFilters.setColumnFilter}
                          openFilter={logFilters.openFilter}
                          setOpenFilter={logFilters.setOpenFilter}
                          width={logResize.widths["logger"]}
                          onResize={logResize.setColumnWidth}
                        />
                        <ColumnHeader
                          label={msg("auto.features.submit.components.modelprobedialog.literal.12")}
                          sortKey="message"
                          currentSort={logSortKey}
                          sortDir={logSortDir}
                          onSort={toggleLogSort}
                          width={logResize.widths["message"]}
                          onResize={logResize.setColumnWidth}
                        />
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {filteredLogs.slice(0, 300).map((log, i) => (
                        <TableRow
                          key={i}
                          className="cursor-pointer"
                          onClick={(e) => {
                            const td = (e.target as HTMLElement).closest("td");
                            if (!td) return;
                            const text = td.textContent?.trim();
                            if (text) {
                              void copyTextToClipboard(text).then((ok) => {
                                if (ok) toast.success(msg("clipboard.copied"));
                              });
                            }
                          }}
                        >
                          <TableCell
                            className="text-xs font-mono text-muted-foreground truncate overflow-hidden"
                            style={
                              logResize.widths["timestamp"]
                                ? {
                                    width: logResize.widths["timestamp"],
                                    maxWidth: logResize.widths["timestamp"],
                                  }
                                : undefined
                            }
                            dir="ltr"
                          >
                            {formatLogTimestamp(log.timestamp)}
                          </TableCell>
                          <TableCell
                            className="truncate overflow-hidden"
                            style={
                              logResize.widths["level"]
                                ? {
                                    width: logResize.widths["level"],
                                    maxWidth: logResize.widths["level"],
                                  }
                                : undefined
                            }
                          >
                            <Badge
                              variant={
                                log.level === "ERROR"
                                  ? "destructive"
                                  : log.level === "WARNING"
                                    ? "outline"
                                    : "secondary"
                              }
                              className="text-[0.625rem] font-mono"
                            >
                              {log.level}
                            </Badge>
                          </TableCell>
                          <TableCell
                            className="text-xs font-mono text-muted-foreground truncate overflow-hidden"
                            style={
                              logResize.widths["logger"]
                                ? {
                                    width: logResize.widths["logger"],
                                    maxWidth: logResize.widths["logger"],
                                  }
                                : undefined
                            }
                            title={log.logger}
                          >
                            {log.logger}
                          </TableCell>
                          <TableCell
                            className="text-xs font-mono whitespace-pre-wrap break-all overflow-hidden hover:underline"
                            style={
                              logResize.widths["message"]
                                ? {
                                    width: logResize.widths["message"],
                                    maxWidth: logResize.widths["message"],
                                  }
                                : undefined
                            }
                            title={log.message}
                          >
                            {log.message}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                    {filteredLogs.length > 300 && (
                      <tfoot>
                        <tr>
                          <td
                            colSpan={4}
                            className="text-center py-3 text-[0.625rem] text-muted-foreground"
                          >
                            {msg("auto.features.submit.components.modelprobedialog.25")}
                            {filteredLogs.length}
                            {msg("auto.features.submit.components.modelprobedialog.26")}
                          </td>
                        </tr>
                      </tfoot>
                    )}
                  </Table>
                </div>
              </CardContent>
            </Card>
          ) : (
            <div className="flex h-[80px] items-center justify-center rounded-md border border-dashed border-border/40 text-[0.6875rem] text-muted-foreground">
              {msg("auto.features.submit.components.modelprobedialog.27")}
            </div>
          )}
        </FadeIn>
      </TabsContent>
    </Tabs>
  );
}

export function RaceRow({
  row,
  now,
  onNavigate,
  onSelect,
}: {
  row: ModelRow;
  now: number;
  onNavigate: () => void;
  onSelect: (value: string) => void;
}) {
  const elapsed =
    row.status === "running" && row.startedAt !== null
      ? now - row.startedAt
      : (row.durationMs ?? 0);

  const color = rowColor(row);
  const isNavigable =
    (row.status === "running" || row.status === "done") &&
    row.trajectory.length + row.logs.length > 0;

  const liveAsymptote =
    (row.scaling?.signal === "strong" || row.scaling?.signal === "observed") &&
    row.scaling?.asymptote != null
      ? row.scaling.asymptote
      : null;
  const liveLast = row.trajectory.length ? row.trajectory[row.trajectory.length - 1]!.score : null;
  const livePrediction = liveAsymptote ?? liveLast;

  return (
    <motion.li
      layout
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.25, ease: [0.2, 0.8, 0.2, 1] }}
      className={cn(
        "rounded-lg border bg-card/60 ps-3 pe-3 py-2.5 transition-colors",
        row.status === "running" && "border-primary/40 shadow-sm",
        row.status === "done" && "border-border/60",
        row.status === "error" && "border-destructive/30 bg-destructive/5",
        row.status === "pending" && "border-border/40 opacity-70",
        isNavigable && "cursor-pointer hover:bg-accent/40",
      )}
      style={
        row.status === "done"
          ? { borderInlineStartColor: color, borderInlineStartWidth: 3 }
          : undefined
      }
      onClick={isNavigable ? onNavigate : undefined}
      role={isNavigable ? "button" : undefined}
      tabIndex={isNavigable ? 0 : undefined}
      onKeyDown={
        isNavigable
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onNavigate();
              }
            }
          : undefined
      }
    >
      <div className="flex items-center gap-3">
        <StatusIcon status={row.status} />
        <div className="min-w-0 flex-1 space-y-1.5">
          <div className="flex items-center gap-3">
            <div className="min-w-0 flex-1 flex items-center gap-2">
              <span className="truncate font-mono text-xs font-semibold" dir="ltr">
                {row.label}
              </span>
              {row.status === "error" && row.errorMessage && (
                <span className="truncate text-[0.6875rem] text-destructive" dir="ltr">
                  {row.errorMessage}
                </span>
              )}
            </div>
            <div className="flex shrink-0 items-center gap-3 text-xs">
              {(row.status === "running" || row.status === "done") && row.trajectory.length > 0 && (
                <div className={cn(row.status === "running" && "opacity-80")}>
                  <TrajectorySparkline
                    points={row.trajectory}
                    asymptote={liveAsymptote}
                    width={72}
                    height={22}
                    color={color}
                  />
                </div>
              )}
              {row.status === "done" && <RowScoreDisplay row={row} />}
              {row.status === "running" && livePrediction !== null && (
                <span
                  className="font-mono text-[0.6875rem] tabular-nums text-foreground/80"
                  dir="ltr"
                >
                  ≈{livePrediction.toFixed(1)}
                </span>
              )}
              {(row.status === "running" || row.status === "done") && (
                <span
                  className={cn(
                    "font-mono text-[0.6875rem] tabular-nums",
                    row.status === "running" ? "text-primary" : "text-muted-foreground",
                  )}
                  dir="ltr"
                >
                  {(elapsed / 1000).toFixed(1)}
                  {msg("auto.features.submit.components.modelprobedialog.28")}
                </span>
              )}
              {row.status === "done" && (
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-[0.6875rem]"
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelect(row.model);
                  }}
                >
                  {msg("auto.features.submit.components.modelprobedialog.29")}
                </Button>
              )}
              {row.status === "error" && (
                <span className="font-mono text-[0.6875rem] text-destructive">
                  {msg("auto.features.submit.components.modelprobedialog.30")}
                </span>
              )}
              {row.status === "pending" && (
                <span className="font-mono text-[0.6875rem] text-muted-foreground">
                  {msg("auto.features.submit.components.modelprobedialog.31")}
                </span>
              )}
              {isNavigable && (
                <ArrowRight
                  className="size-3.5 shrink-0 text-muted-foreground/60 rtl:rotate-180"
                  aria-hidden="true"
                />
              )}
            </div>
          </div>
          {row.status === "running" && (
            <div className="h-1.5 overflow-hidden rounded-full bg-muted">
              <motion.div
                className="h-full bg-gradient-to-l from-primary/60 to-primary"
                initial={{ width: 0 }}
                animate={{
                  width:
                    row.trajectory.length > 0
                      ? `${Math.min((row.trajectory.length / 10) * 100, 95)}%`
                      : "15%",
                }}
                transition={{ duration: 0.4, ease: [0.2, 0.8, 0.2, 1] }}
              />
            </div>
          )}
        </div>
      </div>
    </motion.li>
  );
}

function ProbeInfoCard({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="group min-w-0 flex-[1_1_7.5rem] rounded-lg border border-[#E3DCD0] bg-[#FBF9F4] px-3.5 py-3 transition-[border-color,box-shadow] duration-200 hover:border-[#C8A882]/55 hover:shadow-[0_2px_8px_-2px_rgba(124,99,80,0.1)]">
      <div className="flex items-center gap-1.5 mb-1.5">
        <span
          className="shrink-0 inline-flex items-center justify-center size-3.5 text-[#A89680] transition-colors duration-200 group-hover:text-[#7C6350]"
          aria-hidden="true"
        >
          {icon}
        </span>
        <p className="text-[0.625rem] font-semibold tracking-[0.08em] uppercase text-[#A89680] truncate">
          {label}
        </p>
      </div>
      <p className="text-sm font-semibold text-[#1C1612] truncate">
        {value ?? <span className="text-[#BFB3A3] font-normal">—</span>}
      </p>
    </div>
  );
}

function RowScoreDisplay({ row }: { row: ModelRow }) {
  const value = rowAsymptote(row) ?? row.score;
  if (value === null)
    return <span className="font-mono font-bold tabular-nums text-foreground">—</span>;
  return (
    <span className="font-mono font-bold tabular-nums text-foreground">
      <AnimatedNumber value={value} decimals={1} duration={0.5} />
    </span>
  );
}

function PodiumScore({ row }: { row: ModelRow }) {
  const displayValue = rowAsymptote(row) ?? row.score;
  return (
    <span className="font-mono text-lg font-bold tabular-nums text-foreground">
      {displayValue !== null ? (
        <AnimatedNumber value={displayValue} decimals={1} duration={0.8} />
      ) : (
        "—"
      )}
    </span>
  );
}

export function StatusIcon({ status }: { status: RowStatus }) {
  if (status === "running")
    return (
      <span className="inline-flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/10 text-primary">
        <Loader2 className="size-3.5 animate-spin" />
      </span>
    );
  if (status === "done")
    return (
      <span className="inline-flex size-7 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary">
        <Check className="size-3.5" />
      </span>
    );
  if (status === "error")
    return (
      <span className="inline-flex size-7 shrink-0 items-center justify-center rounded-full bg-destructive/15 text-destructive">
        <XCircle className="size-3.5" />
      </span>
    );
  return (
    <span className="inline-flex size-7 shrink-0 items-center justify-center rounded-full bg-muted text-muted-foreground">
      <Circle className="size-3.5" />
    </span>
  );
}

export function PodiumCard({
  row,
  rank,
  onSelect,
}: {
  row: ModelRow;
  rank: number;
  onSelect: (value: string) => void;
}) {
  const palette: Record<number, { badge: string; icon: React.ReactNode }> = {
    1: { badge: "bg-amber-500 text-white", icon: <Crown className="size-3" /> },
    2: { badge: "bg-slate-400 text-white", icon: <Medal className="size-3" /> },
    3: { badge: "bg-orange-400 text-white", icon: <Medal className="size-3" /> },
  };
  const p = palette[rank] ?? palette[3]!;

  return (
    <TiltCard className="rounded-xl border border-border/60 bg-card/50 p-3">
      <div className="flex items-start justify-between gap-2">
        <span
          className={cn(
            "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[0.625rem] font-bold",
            p.badge,
          )}
        >
          {p.icon}
          <span dir="ltr">#{rank}</span>
        </span>
      </div>
      <div className="mt-2 truncate font-mono text-xs font-semibold" dir="ltr">
        {row.label}
      </div>
      {row.trajectory.length > 0 && (
        <div className="mt-1.5 flex justify-center opacity-70">
          <TrajectorySparkline
            points={row.trajectory}
            asymptote={rowAsymptote(row)}
            width={100}
            height={24}
            color={rowColor(row)}
          />
        </div>
      )}
      <div className="mt-1.5 flex items-baseline justify-between">
        <PodiumScore row={row} />
        {row.durationMs !== null && (
          <span className="font-mono text-[0.625rem] tabular-nums text-muted-foreground">
            {(row.durationMs / 1000).toFixed(1)}
            {msg("auto.features.submit.components.modelprobedialog.32")}
          </span>
        )}
      </div>
      <Separator className="my-2" />
      <Button
        size="sm"
        variant="outline"
        className="h-7 w-full text-[0.6875rem]"
        onClick={() => onSelect(row.model)}
      >
        {msg("auto.features.submit.components.modelprobedialog.33")}
      </Button>
    </TiltCard>
  );
}
