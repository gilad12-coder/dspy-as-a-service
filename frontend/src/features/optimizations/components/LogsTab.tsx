"use client";

import { useMemo, useState } from "react";
import { toast } from "react-toastify";
import { motion } from "framer-motion";
import { Gauge, ScrollText } from "lucide-react";
import { Card, CardContent } from "@/shared/ui/primitives/card";
import { EmptyState } from "@/shared/ui/empty-state";
import { Badge } from "@/shared/ui/primitives/badge";
import { Table, TableBody, TableCell, TableHeader, TableRow } from "@/shared/ui/primitives/table";
import {
  ColumnHeader,
  useColumnFilters,
  useColumnResize,
  ResetColumnsButton,
  type SortDir,
} from "@/shared/ui/excel-filter";
import { ExportTableMenu } from "@/shared/ui/export-table-menu";
import { FadeIn } from "@/shared/ui/motion";
import { formatMsg, msg } from "@/shared/lib/messages";
import type { OptimizationLogEntry } from "@/shared/types/api";
import { formatLogTimestamp, logTimeBucket } from "@/shared/lib";

type Verbosity = "quiet" | "normal" | "verbose";

// "verbose" carries no preset — it clears the level filter so every captured
// level (incl. DEBUG) shows. Quiet/Normal map to explicit level sets that drive
// the shared `level` column filter, so the segment and that filter remain one
// source of truth.
const VERBOSITY_LEVELS: Record<Exclude<Verbosity, "verbose">, readonly string[]> = {
  quiet: ["WARNING", "ERROR", "CRITICAL"],
  normal: ["INFO", "WARNING", "ERROR", "CRITICAL"],
};

const VERBOSITY_OPTIONS: ReadonlyArray<{ value: Verbosity; label: () => string }> = [
  { value: "quiet", label: () => msg("optimizations.logs.verbosity.quiet") },
  { value: "normal", label: () => msg("optimizations.logs.verbosity.normal") },
  { value: "verbose", label: () => msg("optimizations.logs.verbosity.verbose") },
];

// Matches the explore sort pill so both segmented controls animate alike.
const PILL_TRANSITION = { type: "tween", duration: 0.16, ease: [0.22, 1, 0.36, 1] } as const;

/** Map the live `level` column-filter set back to the verbosity it represents. */
function verbosityFromLevelFilter(levelSet: Set<string> | undefined): Verbosity | null {
  if (!levelSet || levelSet.size === 0) return "verbose";
  const matches = (levels: readonly string[]) =>
    levels.length === levelSet.size && levels.every((l) => levelSet.has(l));
  if (matches(VERBOSITY_LEVELS.quiet)) return "quiet";
  if (matches(VERBOSITY_LEVELS.normal)) return "normal";
  return null;
}

function VerbosityControl({
  active,
  onChange,
}: {
  active: Verbosity | null;
  onChange: (verbosity: Verbosity) => void;
}) {
  return (
    <div
      role="group"
      aria-label={msg("optimizations.logs.verbosity.aria")}
      className="inline-flex items-center gap-0.5 rounded-lg border border-border/70 bg-muted/30 p-0.5"
    >
      <Gauge className="mx-1 size-3 text-foreground/35" aria-hidden="true" />
      {VERBOSITY_OPTIONS.map((o) => {
        const isActive = o.value === active;
        return (
          <button
            key={o.value}
            type="button"
            aria-pressed={isActive}
            onClick={() => {
              if (!isActive) onChange(o.value);
            }}
            className={`relative rounded-md px-2.5 py-1 text-[12px] font-medium transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 ${
              isActive ? "text-foreground" : "cursor-pointer text-foreground/55 hover:text-foreground"
            }`}
          >
            {isActive && (
              <motion.span
                layoutId="logs-verbosity-pill"
                className="absolute inset-0 rounded-md bg-background shadow-[0_1px_2px_oklch(0.25_0.04_45/.12)]"
                transition={PILL_TRANSITION}
                aria-hidden="true"
              />
            )}
            <span className="relative z-10">{o.label()}</span>
          </button>
        );
      })}
    </div>
  );
}

export function LogsTab({
  logs,
  pairNames,
}: {
  logs: OptimizationLogEntry[];
  pairNames?: Record<number, string>;
}) {
  const showPairCol = !!pairNames && Object.keys(pairNames).length > 0;
  // Open at the Normal verbosity (INFO+) — DEBUG is captured but hidden until
  // the operator opts into "verbose". Seeding here (vs. an effect) keeps the
  // first paint already filtered, and resets to Normal on every mount.
  const logFilters = useColumnFilters({ level: new Set(VERBOSITY_LEVELS.normal) });
  const logResize = useColumnResize();
  const activeVerbosity = useMemo(
    () => verbosityFromLevelFilter(logFilters.filters.level),
    [logFilters.filters.level],
  );
  const setVerbosity = (verbosity: Verbosity) => {
    logFilters.setColumnFilter(
      "level",
      verbosity === "verbose" ? new Set() : new Set(VERBOSITY_LEVELS[verbosity]),
    );
  };
  const [sortKey, setSortKey] = useState<string>("timestamp");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const toggleSort = (key: string) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const filtered = useMemo(() => {
    let result = logs.filter((l) => {
      for (const [col, allowed] of Object.entries(logFilters.filters)) {
        if (allowed.size === 0) continue;
        const val =
          col === "timestamp"
            ? logTimeBucket(l.timestamp)
            : col === "pair_index"
              ? l.pair_index != null
                ? String(l.pair_index)
                : "—"
              : String((l as unknown as Record<string, unknown>)[col] ?? "");
        if (!allowed.has(val)) return false;
      }
      return true;
    });
    result = [...result].sort((a, b) => {
      let cmp: number;
      if (sortKey === "pair_index") {
        // Numeric column — a direct subtraction beats spinning up the Intl
        // collator on every comparison across a long log table.
        cmp = (a.pair_index ?? -Infinity) - (b.pair_index ?? -Infinity);
      } else if (sortKey === "timestamp") {
        // ISO-8601 timestamps order correctly under plain string comparison,
        // so skip the collator on this hot path too.
        const av = String(a.timestamp ?? "");
        const bv = String(b.timestamp ?? "");
        cmp = av < bv ? -1 : av > bv ? 1 : 0;
      } else {
        // Textual columns (level/logger/message) may hold Hebrew — locale-aware
        // collation is reserved for these.
        const av = String((a as unknown as Record<string, unknown>)[sortKey] ?? "");
        const bv = String((b as unknown as Record<string, unknown>)[sortKey] ?? "");
        cmp = av.localeCompare(bv, "he", { numeric: true });
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return result;
  }, [logs, logFilters.filters, sortKey, sortDir]);

  const filterOptions = useMemo(() => {
    const unique = (key: string) => {
      const vals = [
        ...new Set(logs.map((l) => String((l as unknown as Record<string, unknown>)[key] ?? ""))),
      ]
        .filter(Boolean)
        .sort();
      return vals.map((v) => ({ value: v, label: v }));
    };
    const timestampBuckets = [...new Set(logs.map((l) => logTimeBucket(l.timestamp)))]
      .filter(Boolean)
      .sort()
      .map((v) => ({ value: v, label: v }));
    const pairOpts = showPairCol
      ? [...new Set(logs.map((l) => (l.pair_index != null ? String(l.pair_index) : "—")))]
          .sort()
          .map((v) => ({
            value: v,
            label:
              v === "—"
                ? msg("auto.features.optimizations.components.logstab.literal.1")
                : (pairNames?.[parseInt(v)] ??
                  formatMsg("auto.features.optimizations.components.logstab.template.1", {
                    p1: parseInt(v) + 1,
                  })),
          }))
      : [];
    return {
      level: unique("level"),
      logger: unique("logger"),
      timestamp: timestampBuckets,
      pair_index: pairOpts,
    };
  }, [logs, showPairCol, pairNames]);

  return (
    <div className="mt-4">
      <FadeIn>
        <div
          className="flex items-center justify-between gap-3 mb-4"
          data-tutorial="live-logs"
        >
          <div className="flex items-center gap-3">
            <VerbosityControl active={activeVerbosity} onChange={setVerbosity} />
            <ResetColumnsButton resize={logResize} />
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <span className="text-xs text-muted-foreground">
              {filtered.length}
              {msg("auto.features.optimizations.components.logstab.1")}
            </span>
            <ExportTableMenu
              iconOnly
              disabled={filtered.length === 0}
              getData={() => {
                const columns = [
                  ...(showPairCol ? ["pair_index"] : []),
                  "timestamp",
                  "level",
                  "logger",
                  "message",
                ];
                const rows = filtered.map((log) => {
                  const rec: Record<string, unknown> = {};
                  if (showPairCol) {
                    rec.pair_index =
                      log.pair_index != null
                        ? (pairNames?.[log.pair_index] ??
                          formatMsg("auto.features.optimizations.components.logstab.template.3", {
                            p1: log.pair_index + 1,
                          }))
                        : "—";
                  }
                  rec.timestamp = formatLogTimestamp(log.timestamp);
                  rec.level = log.level;
                  rec.logger = log.logger;
                  rec.message = log.message;
                  return rec;
                });
                return { columns, rows, filename: "logs" };
              }}
            />
          </div>
        </div>
      </FadeIn>
      {filtered.length === 0 ? (
        <EmptyState
          variant="list"
          icon={ScrollText}
          title={
            logs.length === 0
              ? msg("auto.features.optimizations.components.logstab.2")
              : activeVerbosity === "quiet" &&
                  Object.keys(logFilters.filters).length === 1 &&
                  !!logFilters.filters.level
                ? msg("optimizations.logs.verbosity.empty_quiet")
                : msg("optimizations.logs.verbosity.empty_filtered")
          }
        />
      ) : (
        <Card>
          <CardContent className="p-0">
            <div className="max-h-[600px] overflow-auto">
              <Table className={"table-fixed w-full"}>
                <colgroup>
                  {showPairCol && (
                    <col style={{ width: logResize.widths["pair_index"] ?? "12%" }} />
                  )}
                  <col
                    style={{
                      width: logResize.widths["timestamp"] ?? (showPairCol ? "13%" : "15%"),
                    }}
                  />
                  <col
                    style={{ width: logResize.widths["level"] ?? (showPairCol ? "10%" : "12%") }}
                  />
                  <col
                    style={{ width: logResize.widths["logger"] ?? (showPairCol ? "14%" : "17%") }}
                  />
                  <col />
                </colgroup>
                <TableHeader>
                  <TableRow>
                    {showPairCol && (
                      <ColumnHeader
                        label={msg("auto.features.optimizations.components.logstab.literal.2")}
                        sortKey="pair_index"
                        currentSort={sortKey}
                        sortDir={sortDir}
                        onSort={toggleSort}
                        filterCol="pair_index"
                        filterOptions={filterOptions.pair_index}
                        filters={logFilters.filters}
                        onFilter={logFilters.setColumnFilter}
                        openFilter={logFilters.openFilter}
                        setOpenFilter={logFilters.setOpenFilter}
                        width={logResize.widths["pair_index"]}
                        onResize={logResize.setColumnWidth}
                      />
                    )}
                    <ColumnHeader
                      label={msg("auto.features.optimizations.components.logstab.literal.3")}
                      sortKey="timestamp"
                      currentSort={sortKey}
                      sortDir={sortDir}
                      onSort={toggleSort}
                      filterCol="timestamp"
                      filterOptions={filterOptions.timestamp}
                      filters={logFilters.filters}
                      onFilter={logFilters.setColumnFilter}
                      openFilter={logFilters.openFilter}
                      setOpenFilter={logFilters.setOpenFilter}
                      width={logResize.widths["timestamp"]}
                      onResize={logResize.setColumnWidth}
                    />
                    <ColumnHeader
                      label={msg("auto.features.optimizations.components.logstab.literal.4")}
                      sortKey="level"
                      currentSort={sortKey}
                      sortDir={sortDir}
                      onSort={toggleSort}
                      filterCol="level"
                      filterOptions={filterOptions.level}
                      filters={logFilters.filters}
                      onFilter={logFilters.setColumnFilter}
                      openFilter={logFilters.openFilter}
                      setOpenFilter={logFilters.setOpenFilter}
                      width={logResize.widths["level"]}
                      onResize={logResize.setColumnWidth}
                    />
                    <ColumnHeader
                      label={msg("auto.features.optimizations.components.logstab.literal.5")}
                      sortKey="logger"
                      currentSort={sortKey}
                      sortDir={sortDir}
                      onSort={toggleSort}
                      filterCol="logger"
                      filterOptions={filterOptions.logger}
                      filters={logFilters.filters}
                      onFilter={logFilters.setColumnFilter}
                      openFilter={logFilters.openFilter}
                      setOpenFilter={logFilters.setOpenFilter}
                      width={logResize.widths["logger"]}
                      onResize={logResize.setColumnWidth}
                    />
                    <ColumnHeader
                      label={msg("auto.features.optimizations.components.logstab.literal.6")}
                      sortKey="message"
                      currentSort={sortKey}
                      sortDir={sortDir}
                      onSort={toggleSort}
                      width={logResize.widths["message"]}
                      onResize={logResize.setColumnWidth}
                    />
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filtered.slice(0, 300).map((log, i) => (
                    <TableRow
                      key={i}
                      className="cursor-pointer"
                      onClick={(e) => {
                        const td = (e.target as HTMLElement).closest("td");
                        if (!td) return;
                        const text = td.textContent?.trim();
                        if (text) {
                          void navigator.clipboard.writeText(text);
                          toast.success(msg("clipboard.copied"));
                        }
                      }}
                    >
                      {showPairCol && (
                        <TableCell
                          className="text-xs font-mono truncate overflow-hidden"
                          style={
                            logResize.widths["pair_index"]
                              ? {
                                  width: logResize.widths["pair_index"],
                                  maxWidth: logResize.widths["pair_index"],
                                }
                              : undefined
                          }
                        >
                          {log.pair_index != null ? (
                            <Badge variant="secondary" className="text-[9px] font-mono">
                              {pairNames?.[log.pair_index] ??
                                formatMsg(
                                  "auto.features.optimizations.components.logstab.template.3",
                                  { p1: log.pair_index + 1 },
                                )}
                            </Badge>
                          ) : (
                            <span className="text-muted-foreground/40">—</span>
                          )}
                        </TableCell>
                      )}
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
                {filtered.length > 300 && (
                  <tfoot>
                    <tr>
                      <td
                        colSpan={showPairCol ? 5 : 4}
                        className="text-center py-3 text-[0.625rem] text-muted-foreground"
                      >
                        {msg("auto.features.optimizations.components.logstab.3")}
                        {filtered.length}
                        {msg("auto.features.optimizations.components.logstab.4")}
                      </td>
                    </tr>
                  </tfoot>
                )}
              </Table>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
