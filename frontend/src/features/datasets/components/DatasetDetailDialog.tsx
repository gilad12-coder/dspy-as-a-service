"use client";

import * as React from "react";
import Link from "next/link";
import { toast } from "react-toastify";
import { ArrowUpRight, Inbox, Loader2, Sparkles, Table2 } from "lucide-react";
import { motion } from "framer-motion";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Table, TableBody, TableCell, TableHeader, TableRow } from "@/shared/ui/primitives/table";
import {
  ColumnHeader,
  ResetColumnsButton,
  useColumnFilters,
  useColumnResize,
  type SortDir,
} from "@/shared/ui/excel-filter";
import { StatusBadge } from "@/shared/ui/status-badge";
import { EmptyState } from "@/shared/ui/empty-state";
import { FadeIn } from "@/shared/ui/motion";
import {
  getDatasetRows,
  listDatasetOptimizations,
  type DatasetOptimizationRef,
  type DatasetRowsResponse,
  type DatasetSummary,
} from "@/shared/lib/api";
import { formatMsg, msg } from "@/shared/lib/messages";
import { formatRelativeTime } from "@/shared/lib/formatters";

// The grid sorts/filters the full row set in memory, but caps the DOM at this
// many rows so a large dataset never renders tens of thousands of <tr>s.
const RENDER_ROW_CAP = 200;

// Mirrors the Explore corpus toggle so the sliding pill feels identical app-wide.
const PILL_TRANSITION = { type: "tween", duration: 0.18, ease: [0.22, 1, 0.36, 1] } as const;

type DetailTab = "rows" | "usage";

/** Render any cell value as a short, single-line string for the preview grid. */
function cellText(value: unknown): string {
  if (value == null) return "";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/**
 * Read-only detail sheet for one library dataset, split by a sliding segmented
 * toggle into two views: an interactive row grid (sort / per-column filter /
 * resize / click-to-copy, the same excel-filter toolkit the optimization Data
 * tab uses) and the reverse link — every optimization the caller can see that
 * was submitted from this dataset. Driven open by the parent (a card click or
 * the ``?open=`` deep-link from an optimization's source link).
 */
export function DatasetDetailDialog({
  dataset,
  onClose,
}: {
  dataset: DatasetSummary | null;
  onClose: () => void;
}) {
  const [rows, setRows] = React.useState<DatasetRowsResponse | null>(null);
  const [optimizations, setOptimizations] = React.useState<DatasetOptimizationRef[] | null>(null);
  const [tab, setTab] = React.useState<DetailTab>("rows");
  const datasetId = dataset?.id ?? null;

  const colFilters = useColumnFilters();
  const colResize = useColumnResize();
  const [sortKey, setSortKey] = React.useState("");
  const [sortDir, setSortDir] = React.useState<SortDir>("asc");
  const { clearAll: clearFilters } = colFilters;
  const { resetAll: resetWidths } = colResize;

  React.useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;
    setRows(null);
    setOptimizations(null);
    setTab("rows");
    setSortKey("");
    setSortDir("asc");
    clearFilters();
    resetWidths();
    getDatasetRows(datasetId)
      .then((res) => !cancelled && setRows(res))
      .catch(
        () =>
          !cancelled &&
          setRows({ id: datasetId, columns: [], rows: [], row_count: 0, column_schema: {} }),
      );
    listDatasetOptimizations(datasetId)
      .then((res) => !cancelled && setOptimizations(res.optimizations))
      .catch(() => !cancelled && setOptimizations([]));
    return () => {
      cancelled = true;
    };
  }, [datasetId, clearFilters, resetWidths]);

  const columns = rows?.columns ?? [];
  const allRows = React.useMemo(() => rows?.rows ?? [], [rows]);

  const toggleSort = (key: string) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("asc");
    }
  };

  const filtered = React.useMemo(() => {
    let result = allRows.filter((r) => {
      for (const [col, allowed] of Object.entries(colFilters.filters)) {
        if (allowed.size === 0) continue;
        if (!allowed.has(cellText(r[col]))) return false;
      }
      return true;
    });
    if (sortKey) {
      result = [...result].sort((a, b) => {
        const cmp = cellText(a[sortKey]).localeCompare(cellText(b[sortKey]), "he", { numeric: true });
        return sortDir === "asc" ? cmp : -cmp;
      });
    }
    return result;
  }, [allRows, colFilters.filters, sortKey, sortDir]);

  const filterOptions = React.useMemo(() => {
    const opts: Record<string, Array<{ value: string; label: string }>> = {};
    for (const col of columns) {
      const vals = [...new Set(allRows.map((r) => cellText(r[col])))].filter(Boolean).sort();
      opts[col] = vals.map((v) => ({ value: v, label: v.length > 40 ? `${v.slice(0, 40)}…` : v }));
    }
    return opts;
  }, [allRows, columns]);

  const copyCell = React.useCallback((e: React.MouseEvent) => {
    const td = (e.target as HTMLElement).closest("td");
    if (!td) return;
    const text = td.textContent?.trim();
    if (!text) return;
    navigator.clipboard
      .writeText(text)
      .then(() => toast.success(msg("clipboard.copied")))
      .catch(() => toast.error(msg("clipboard.copy_failed")));
  }, []);

  const usageCount = optimizations?.length ?? 0;

  const segments: ReadonlyArray<{ value: DetailTab; label: string; icon: typeof Table2 }> = [
    { value: "rows", label: msg("datasets.detail.rows_title"), icon: Table2 },
    { value: "usage", label: msg("datasets.detail.tab.usage"), icon: Sparkles },
  ];

  return (
    <Dialog open={dataset !== null} onOpenChange={(next) => !next && onClose()}>
      <DialogContent
        className="w-[min(56rem,94vw)] max-w-[min(56rem,94vw)] overflow-hidden p-0"
        aria-describedby={undefined}
      >
        <div className="flex max-h-[85vh] flex-col">
          <DialogHeader className="shrink-0 px-6 pt-6 pb-4 text-start">
            <DialogTitle className="truncate">{dataset?.name}</DialogTitle>
            {dataset && (
              <DialogDescription>
                {formatMsg("datasets.count.rows", { count: dataset.row_count })}
                {" · "}
                {formatMsg("datasets.count.columns", { count: dataset.column_count })}
              </DialogDescription>
            )}
          </DialogHeader>

          <div className="flex min-h-0 flex-1 flex-col">
            <div className="flex shrink-0 justify-center border-b border-border/40 px-6 pb-4">
              <div
                role="radiogroup"
                aria-label={msg("datasets.detail.view_aria")}
                className="relative inline-flex items-center rounded-full border border-border/80 bg-muted/40 p-0.5"
              >
                {segments.map((seg) => {
                  const active = seg.value === tab;
                  const Icon = seg.icon;
                  return (
                    <button
                      key={seg.value}
                      type="button"
                      role="radio"
                      aria-checked={active}
                      onClick={() => !active && setTab(seg.value)}
                      className={`relative inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-[12.5px] font-medium transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 ${
                        active
                          ? "text-foreground"
                          : "cursor-pointer text-foreground/60 hover:text-foreground"
                      }`}
                    >
                      {active && (
                        <motion.span
                          layoutId="dataset-detail-tab-pill"
                          className="absolute inset-0 rounded-full bg-background shadow-[0_1px_2px_oklch(0.25_0.04_45/.12)]"
                          transition={PILL_TRANSITION}
                          aria-hidden="true"
                        />
                      )}
                      <span className="relative z-10 inline-flex items-center gap-1.5">
                        <Icon className="size-3.5" aria-hidden="true" />
                        <span>{seg.label}</span>
                        {seg.value === "usage" && usageCount > 0 && (
                          <span className="rounded-full bg-foreground/10 px-1.5 text-[0.6875rem] font-bold tabular-nums">
                            {usageCount}
                          </span>
                        )}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {tab === "rows" ? (
              <div className="flex min-h-0 flex-1 flex-col overflow-hidden px-6 py-4">
                {rows === null ? (
                  <div className="flex items-center gap-2 py-10 text-sm text-muted-foreground">
                    <Loader2 className="size-4 animate-spin" />
                    {msg("datasets.detail.loading")}
                  </div>
                ) : columns.length === 0 || allRows.length === 0 ? (
                  <div className="py-8">
                    <EmptyState variant="list" icon={Inbox} title={msg("datasets.detail.rows_empty")} />
                  </div>
                ) : (
                  <FadeIn className="flex min-h-0 flex-1 flex-col">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <span className="text-xs tabular-nums text-muted-foreground">
                        {formatMsg("datasets.detail.rows_count", { count: filtered.length })}
                      </span>
                      <ResetColumnsButton resize={colResize} />
                    </div>
                    {filtered.length === 0 ? (
                      <div className="rounded-lg border border-dashed border-border/60 py-8">
                        <EmptyState
                          variant="list"
                          icon={Inbox}
                          title={msg("datasets.detail.rows_empty")}
                        />
                      </div>
                    ) : (
                      <div className="min-h-0 flex-1 overflow-y-auto rounded-lg border border-border/50">
                        <Table className="table-fixed">
                          <TableHeader>
                            <TableRow>
                              {columns.map((col) => (
                                <ColumnHeader
                                  key={col}
                                  label={col}
                                  sortKey={col}
                                  currentSort={sortKey}
                                  sortDir={sortDir}
                                  onSort={toggleSort}
                                  filterCol={col}
                                  filterOptions={filterOptions[col] ?? []}
                                  filters={colFilters.filters}
                                  onFilter={colFilters.setColumnFilter}
                                  openFilter={colFilters.openFilter}
                                  setOpenFilter={colFilters.setOpenFilter}
                                  width={colResize.widths[col]}
                                  onResize={colResize.setColumnWidth}
                                />
                              ))}
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {filtered.slice(0, RENDER_ROW_CAP).map((row, i) => (
                              <TableRow
                                key={i}
                                className="cursor-pointer transition-colors hover:bg-muted/40"
                                onClick={copyCell}
                              >
                                {columns.map((col) => (
                                  <TableCell
                                    key={col}
                                    className="max-w-[280px] font-mono text-xs text-foreground/75"
                                    style={
                                      colResize.widths[col]
                                        ? {
                                            width: colResize.widths[col],
                                            maxWidth: colResize.widths[col],
                                          }
                                        : undefined
                                    }
                                    title={cellText(row[col])}
                                  >
                                    {cellText(row[col])}
                                  </TableCell>
                                ))}
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    )}
                    {filtered.length > RENDER_ROW_CAP && (
                      <p className="mt-2 text-center text-[0.625rem] text-muted-foreground">
                        {formatMsg("datasets.detail.rows_more", {
                          shown: RENDER_ROW_CAP,
                          total: filtered.length,
                        })}
                      </p>
                    )}
                  </FadeIn>
                )}
              </div>
            ) : (
              <div className="min-h-0 flex-1 overflow-y-auto px-6 py-4">
                {optimizations === null ? (
                  <div className="flex items-center gap-2 py-10 text-sm text-muted-foreground">
                    <Loader2 className="size-4 animate-spin" />
                    {msg("datasets.detail.loading")}
                  </div>
                ) : optimizations.length === 0 ? (
                  <div className="py-8">
                    <EmptyState
                      variant="list"
                      icon={Sparkles}
                      title={msg("datasets.detail.used_by_empty")}
                    />
                  </div>
                ) : (
                  <FadeIn>
                    <ul className="divide-y divide-border/40 rounded-lg border border-border/50">
                      {optimizations.map((opt) => (
                        <li key={opt.optimization_id}>
                          <Link
                            href={`/optimizations/${opt.optimization_id}`}
                            className="group/link flex items-center gap-3 px-3 py-2.5 transition-colors hover:bg-accent/40"
                          >
                            <span className="min-w-0 flex-1 truncate text-sm font-medium text-foreground">
                              {opt.name || opt.optimization_id}
                            </span>
                            {opt.status && <StatusBadge status={opt.status} />}
                            {opt.created_at && (
                              <span className="shrink-0 text-xs text-muted-foreground">
                                {formatRelativeTime(opt.created_at)}
                              </span>
                            )}
                            <ArrowUpRight className="size-4 shrink-0 text-muted-foreground/60 transition-colors group-hover/link:text-foreground" />
                          </Link>
                        </li>
                      ))}
                    </ul>
                  </FadeIn>
                )}
              </div>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
