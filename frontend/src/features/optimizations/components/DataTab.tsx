"use client";

import { useEffect, useMemo, useState } from "react";
import { toast } from "react-toastify";
import { Inbox, Loader2 } from "lucide-react";
import { Card, CardContent } from "@/shared/ui/primitives/card";
import { EmptyState } from "@/shared/ui/empty-state";
import { Table, TableBody, TableCell, TableHeader, TableRow } from "@/shared/ui/primitives/table";
import {
  ColumnHeader,
  useColumnFilters,
  useColumnResize,
  ResetColumnsButton,
  type SortDir,
} from "@/shared/ui/excel-filter";
import { DataTabSkeleton } from "./DataTabSkeleton";
import { FadeIn } from "@/shared/ui/motion";
import { HelpTip } from "@/shared/ui/help-tip";
import { msg } from "@/shared/lib/messages";
import { tip } from "@/shared/lib/tooltips";
import {
  getOptimizationDataset,
  getTestResults,
  getPairTestResults,
} from "@/shared/lib/api";
import type {
  OptimizationDatasetResponse,
  OptimizationStatusResponse,
  EvalExampleResult,
} from "@/shared/types/api";
import { DEMO_OPTIMIZATION_ID } from "@/features/tutorial";

type Split = "all" | "train" | "val" | "test";
type ProgramType = "optimized" | "baseline";

/**
 * Map a score (0–1) to a warm earth-tone color: terracotta → ochre → olive.
 *
 * The hue range (35°→130°) is shifted off pure red/green so the scale sits in
 * the same family as the cream/coffee/taupe chrome (#3D2E22, #8C7A6B, #E5DDD4)
 * instead of clashing with it. Constant OKLCH lightness keeps text contrast
 * stable across the cream backgrounds.
 */
function scoreColor(score: number): string {
  const t = Math.max(0, Math.min(1, score));
  const hue = 35 + t * 95;
  return `oklch(0.5 0.13 ${hue.toFixed(1)})`;
}

/**
 * Render a dataset cell value as a string the table can display.
 *
 * React datasets carry structured columns — `chat_history` is an array of
 * turn objects, `wizard_state` an object — that `String(value)` collapses to
 * the useless `[object Object]`. Stringify those to JSON so the cell shows the
 * real content (compact inline, indented for the hover title); scalars pass
 * through unchanged. `null`/`undefined` render as an empty string.
 */
function formatCellValue(value: unknown, pretty = false): string {
  if (value == null) return "";
  if (typeof value === "object") {
    try {
      return JSON.stringify(value, null, pretty ? 2 : undefined);
    } catch {
      return String(value);
    }
  }
  return String(value);
}

export function DataTab({
  job,
  pairIndex,
  sharedDataset,
  sharedTestResults,
}: {
  job: OptimizationStatusResponse;
  pairIndex?: number | null;
  /** Public share view: render this injected split instead of fetching (no per-example scores). */
  sharedDataset?: OptimizationDatasetResponse | null;
  /** Public share view: inject baseline/optimized eval results instead of fetching them. */
  sharedTestResults?: { baseline: EvalExampleResult[]; optimized: EvalExampleResult[] } | null;
}) {
  const [dataset, setDataset] = useState<OptimizationDatasetResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [split, setSplit] = useState<Split>("test");
  const [programType, setProgramType] = useState<ProgramType>("optimized");
  const [testResults, setTestResults] = useState<Record<string, Record<number, EvalExampleResult>>>(
    { optimized: {}, baseline: {} },
  );
  const [testResultsLoading, setTestResultsLoading] = useState(false);

  const colFilters = useColumnFilters();
  const [sortKey, setSortKey] = useState<string>("");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const toggleSort = (key: string) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir("asc");
    }
  };
  const colResize = useColumnResize();

  const isDemoMode = job.optimization_id === DEMO_OPTIMIZATION_ID;

  useEffect(() => {
    if (isDemoMode) {
      setDataset({
        total_rows: 12,
        splits: {
          train: [
            { index: 0, row: { email_text: "Click here to win $1000 now!", category: "spam" } },
            {
              index: 1,
              row: { email_text: "Meeting moved to 3pm tomorrow", category: "important" },
            },
            {
              index: 2,
              row: { email_text: "50% off all items this weekend only", category: "promotional" },
            },
            {
              index: 3,
              row: { email_text: "Your account has been compromised! Act now!", category: "spam" },
            },
            {
              index: 4,
              row: {
                email_text: "Q3 budget review attached for your approval",
                category: "important",
              },
            },
            {
              index: 5,
              row: { email_text: "Flash sale: 70% off electronics today", category: "promotional" },
            },
            {
              index: 6,
              row: {
                email_text: "Reminder: dentist appointment on Thursday",
                category: "important",
              },
            },
          ],
          val: [
            {
              index: 7,
              row: { email_text: "You've won a free iPhone! Claim here", category: "spam" },
            },
            {
              index: 8,
              row: {
                email_text: "New company policy update effective Monday",
                category: "important",
              },
            },
          ],
          test: [
            {
              index: 9,
              row: { email_text: "Limited time offer: buy 1 get 1 free", category: "promotional" },
            },
            {
              index: 10,
              row: { email_text: "Team standup notes from Monday", category: "important" },
            },
            {
              index: 11,
              row: {
                email_text: "Congratulations! You've been selected for a prize",
                category: "spam",
              },
            },
          ],
        },
        column_mapping: { inputs: { email_text: "email_text" }, outputs: { category: "category" } },
        split_counts: { train: 7, val: 2, test: 3 },
      });
      setTestResults({
        optimized: {
          9: { index: 9, outputs: { category: "promotional" }, score: 1.0, pass: true },
          10: { index: 10, outputs: { category: "important" }, score: 1.0, pass: true },
          11: { index: 11, outputs: { category: "spam" }, score: 1.0, pass: true },
        },
        baseline: {
          9: { index: 9, outputs: { category: "spam" }, score: 0.0, pass: false },
          10: { index: 10, outputs: { category: "important" }, score: 1.0, pass: true },
          11: { index: 11, outputs: { category: "promotional" }, score: 0.0, pass: false },
        },
      });
      setLoading(false);
      return;
    }
    if (sharedDataset) {
      setDataset(sharedDataset);
      setLoading(false);
      return;
    }
    getOptimizationDataset(job.optimization_id)
      .then(setDataset)
      .catch(() => setError(msg("auto.features.optimizations.components.datatab.literal.1")))
      .finally(() => setLoading(false));
  }, [job.optimization_id, isDemoMode, sharedDataset]);

  // Share view: index the injected baseline/optimized results by example index
  // (mirrors the fetch path's reshaping) instead of calling the authed endpoint.
  useEffect(() => {
    if (!sharedTestResults) return;
    const optimized: Record<number, EvalExampleResult> = {};
    const baseline: Record<number, EvalExampleResult> = {};
    for (const r of sharedTestResults.optimized ?? []) optimized[r.index] = r;
    for (const r of sharedTestResults.baseline ?? []) baseline[r.index] = r;
    setTestResults({ optimized, baseline });
  }, [sharedTestResults]);

  useEffect(() => {
    if (isDemoMode || sharedDataset || job.status !== "success") return;
    setTestResultsLoading(true);
    const fetchResults =
      pairIndex != null
        ? getPairTestResults(job.optimization_id, pairIndex)
        : getTestResults(job.optimization_id);
    fetchResults
      .then((res) => {
        const optimized: Record<number, EvalExampleResult> = {};
        const baseline: Record<number, EvalExampleResult> = {};
        for (const r of res.optimized ?? []) optimized[r.index] = r;
        for (const r of res.baseline ?? []) baseline[r.index] = r;
        setTestResults({ optimized, baseline });
      })
      .catch((err) => {
        // Test-results endpoint is non-critical for the data view; log so the
        // failure is visible in dev tools without breaking the dataset render.
        console.warn("test results fetch failed:", err);
      })
      .finally(() => setTestResultsLoading(false));
  }, [job.optimization_id, job.status, pairIndex, isDemoMode, sharedDataset]);

  const inputFields = useMemo(
    () => (dataset ? Object.values(dataset.column_mapping.inputs) : []),
    [dataset],
  );
  const outputFields = useMemo(
    () => (dataset ? Object.values(dataset.column_mapping.outputs) : []),
    [dataset],
  );
  const allColumns = useMemo(() => [...inputFields, ...outputFields], [inputFields, outputFields]);

  const currentResults = testResults[programType] ?? {};

  const rows = useMemo(() => {
    if (!dataset) return [];
    if (split === "all")
      return [...dataset.splits.train, ...dataset.splits.val, ...dataset.splits.test];
    return dataset.splits[split];
  }, [dataset, split]);

  const filtered = useMemo(() => {
    let result = rows.filter((r) => {
      for (const [col, allowed] of Object.entries(colFilters.filters)) {
        if (allowed.size === 0) continue;
        const val = formatCellValue(r.row[col]);
        if (!allowed.has(val)) return false;
      }
      return true;
    });
    if (sortKey) {
      result = [...result].sort((a, b) => {
        const av = formatCellValue(a.row[sortKey]);
        const bv = formatCellValue(b.row[sortKey]);
        const cmp = av.localeCompare(bv, "he", { numeric: true });
        return sortDir === "asc" ? cmp : -cmp;
      });
    }
    return result;
  }, [rows, colFilters.filters, sortKey, sortDir]);

  const filterOptions = useMemo(() => {
    const opts: Record<string, Array<{ value: string; label: string }>> = {};
    for (const col of allColumns) {
      const vals = [...new Set(rows.map((r) => formatCellValue(r.row[col])))]
        .filter(Boolean)
        .sort();
      opts[col] = vals.map((v) => ({
        value: v,
        label: v.length > 40 ? `${v.slice(0, 40)}...` : v,
      }));
    }
    return opts;
  }, [rows, allColumns]);

  const evalCount = Object.keys(currentResults).length;

  if (loading) return <DataTabSkeleton />;
  if (error || !dataset)
    return (
      <div className="text-sm text-destructive text-center py-16">
        {error ?? msg("auto.features.optimizations.components.datatab.literal.2")}
      </div>
    );

  return (
    <div className="space-y-4 mt-4">
      <FadeIn>
        <p className="text-sm text-muted-foreground">
          {msg("optimizations.datatab.description")}
        </p>
      </FadeIn>
      {/* Test evaluation bar — shows cached results */}
      {split === "test" && (
        <FadeIn delay={0.2}>
          <div
            className="rounded-2xl border border-[#E5DDD4] bg-gradient-to-l from-[#FAF8F5] to-[#F5F1EC] p-4 space-y-3"
            data-tutorial="eval-bar"
          >
            <div className="flex items-center gap-3">
              <div className="flex-1 min-w-0">
                <div className="text-sm font-semibold text-[#3D2E22]">
                  <HelpTip text={tip("code.predictions_table")}>
                    {msg("auto.features.optimizations.components.datatab.1")}
                  </HelpTip>
                </div>
              </div>
              <div className="relative inline-flex rounded-lg bg-[#F0EBE4] p-1 gap-1 text-[0.6875rem] shrink-0">
                <div
                  className="absolute top-1 bottom-1 rounded-md bg-[#3D2E22] shadow-sm transition-[inset-inline-start] duration-150 ease-out"
                  style={{
                    width: "calc(50% - 6px)",
                    insetInlineStart: programType === "baseline" ? 4 : "calc(50% + 2px)",
                  }}
                />
                <button
                  onClick={() => setProgramType("baseline")}
                  className={`relative z-10 flex items-center gap-1.5 rounded-md px-3 py-1.5 cursor-pointer transition-colors duration-150 ${programType === "baseline" ? "text-[#FAF8F5] font-semibold" : "text-[#8C7A6B] hover:text-[#3D2E22]"}`}
                >
                  {msg("auto.features.optimizations.components.datatab.2")}
                </button>
                <button
                  onClick={() => setProgramType("optimized")}
                  className={`relative z-10 flex items-center gap-1.5 rounded-md px-3 py-1.5 cursor-pointer transition-colors duration-150 ${programType === "optimized" ? "text-[#FAF8F5] font-semibold" : "text-[#8C7A6B] hover:text-[#3D2E22]"}`}
                >
                  {msg("auto.features.optimizations.components.datatab.3")}
                </button>
              </div>
              {testResultsLoading && (
                <Loader2 className="size-4 animate-spin text-[#8C7A6B] shrink-0" />
              )}
            </div>
          </div>
        </FadeIn>
      )}

      <FadeIn delay={0.3}>
        <div className="flex items-center gap-3 flex-wrap">
          {(() => {
            const splits: Array<[Split, string]> = [
              ["all", msg("auto.features.optimizations.components.datatab.literal.4")],
              ["train", msg("auto.features.optimizations.components.datatab.literal.5")],
              ["val", msg("auto.features.optimizations.components.datatab.literal.6")],
              ["test", msg("auto.features.optimizations.components.datatab.literal.7")],
            ];
            const idx = splits.findIndex(([s]) => s === split);
            const count = splits.length;
            return (
              <div
                className="relative flex w-full rounded-lg bg-muted p-1 gap-1 text-[0.6875rem]"
                data-tutorial="split-selector"
              >
                <div
                  className="absolute top-1 bottom-1 rounded-md bg-background shadow-sm transition-[inset-inline-start] duration-150 ease-out"
                  style={{
                    width: `calc(${100 / count}% - 6px)`,
                    insetInlineStart: `calc(${(idx / count) * 100}% + 4px)`,
                  }}
                />
                {splits.map(([s, label]) => (
                  <button
                    key={s}
                    onClick={() => setSplit(s)}
                    className={`relative z-10 flex-1 rounded-md px-3 py-1.5 cursor-pointer text-center transition-colors duration-150 ${split === s ? "text-foreground font-semibold" : "text-foreground/50 hover:text-foreground"}`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            );
          })()}
          <ResetColumnsButton resize={colResize} />
          <div className="text-[0.625rem] text-muted-foreground tabular-nums ms-auto">
            {filtered.length}
            {msg("auto.features.optimizations.components.datatab.4")}
          </div>
        </div>
      </FadeIn>

      <FadeIn delay={0.35}>
        {filtered.length === 0 ? (
          <EmptyState
            variant="list"
            icon={Inbox}
            title={msg("auto.features.optimizations.components.datatab.5")}
          />
        ) : (
          <Card data-tutorial="data-table">
            <CardContent className="p-0">
              <div className="table-scroll max-h-[520px] overflow-y-auto">
                <Table className="table-fixed">
                  <TableHeader>
                    <TableRow>
                      {split === "test" && evalCount > 0 && (
                        <ColumnHeader
                          label={msg("auto.features.optimizations.components.datatab.literal.8")}
                          sortKey="_score"
                          currentSort={sortKey}
                          sortDir={sortDir}
                          onSort={toggleSort}
                          width={colResize.widths["_score"]}
                          onResize={colResize.setColumnWidth}
                        />
                      )}
                      {inputFields.map((f) => (
                        <ColumnHeader
                          key={f}
                          label={f}
                          sortKey={f}
                          currentSort={sortKey}
                          sortDir={sortDir}
                          onSort={toggleSort}
                          filterCol={f}
                          filterOptions={filterOptions[f] ?? []}
                          filters={colFilters.filters}
                          onFilter={colFilters.setColumnFilter}
                          openFilter={colFilters.openFilter}
                          setOpenFilter={colFilters.setOpenFilter}
                          width={colResize.widths[f]}
                          onResize={colResize.setColumnWidth}
                        />
                      ))}
                      {outputFields.map((f) => (
                        <ColumnHeader
                          key={f}
                          label={f}
                          sortKey={f}
                          currentSort={sortKey}
                          sortDir={sortDir}
                          onSort={toggleSort}
                          filterCol={f}
                          filterOptions={filterOptions[f] ?? []}
                          filters={colFilters.filters}
                          onFilter={colFilters.setColumnFilter}
                          openFilter={colFilters.openFilter}
                          setOpenFilter={colFilters.setOpenFilter}
                          width={colResize.widths[f]}
                          onResize={colResize.setColumnWidth}
                        />
                      ))}
                      {split === "test" &&
                        evalCount > 0 &&
                        outputFields.map((f) => (
                          <ColumnHeader
                            key={`pred-${f}`}
                            label={`pred_${f}`}
                            sortKey={`_pred_${f}`}
                            currentSort={sortKey}
                            sortDir={sortDir}
                            onSort={toggleSort}
                            width={colResize.widths[`_pred_${f}`]}
                            onResize={colResize.setColumnWidth}
                          />
                        ))}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {filtered.slice(0, 200).map((row) => {
                      const ev = currentResults[row.index];
                      return (
                        <TableRow
                          key={row.index}
                          className="cursor-pointer"
                          onClick={(e) => {
                            const td = (e.target as HTMLElement).closest("td");
                            if (!td || td === td.parentElement?.lastElementChild) return;
                            const text = td.textContent?.trim();
                            if (!text) return;
                            navigator.clipboard
                              .writeText(text)
                              .then(() => toast.success(msg("clipboard.copied")))
                              .catch(() => toast.error(msg("clipboard.copy_failed")));
                          }}
                        >
                          {split === "test" && evalCount > 0 && (
                            <TableCell
                              className="!p-0 !px-1.5 !py-1"
                              style={
                                colResize.widths["_score"]
                                  ? { width: colResize.widths["_score"] }
                                  : { width: 72 }
                              }
                            >
                              {ev ? (
                                <div className="flex flex-col items-center gap-0.5">
                                  <span
                                    className="text-[0.625rem] font-mono tabular-nums font-medium"
                                    style={{ color: scoreColor(ev.score) }}
                                  >
                                    {ev.score.toFixed(2)}
                                  </span>
                                  <div className="w-full h-1.5 rounded-full overflow-hidden bg-muted">
                                    <div
                                      className="h-full rounded-full"
                                      style={{
                                        width: `${Math.max(0, Math.min(1, ev.score)) * 100}%`,
                                        background: scoreColor(ev.score),
                                      }}
                                    />
                                  </div>
                                </div>
                              ) : (
                                <span className="text-[0.625rem] text-[#E5DDD4] flex justify-center">
                                  —
                                </span>
                              )}
                            </TableCell>
                          )}
                          {inputFields.map((f) => (
                            <TableCell
                              key={f}
                              className="text-xs font-mono truncate overflow-hidden"
                              style={
                                colResize.widths[f]
                                  ? { width: colResize.widths[f], maxWidth: colResize.widths[f] }
                                  : undefined
                              }
                              title={formatCellValue(row.row[f], true)}
                            >
                              {formatCellValue(row.row[f])}
                            </TableCell>
                          ))}
                          {outputFields.map((f) => (
                            <TableCell
                              key={f}
                              className="text-xs font-mono truncate overflow-hidden"
                              style={
                                colResize.widths[f]
                                  ? { width: colResize.widths[f], maxWidth: colResize.widths[f] }
                                  : undefined
                              }
                              title={formatCellValue(row.row[f], true)}
                            >
                              {formatCellValue(row.row[f])}
                            </TableCell>
                          ))}
                          {split === "test" &&
                            evalCount > 0 &&
                            outputFields.map((f) => {
                              const sigField = Object.entries(dataset.column_mapping.outputs).find(
                                ([, col]) => col === f,
                              )?.[0];
                              const pred = ev?.outputs[sigField ?? ""];
                              const key = `_pred_${f}`;
                              return (
                                <TableCell
                                  key={key}
                                  className="text-xs font-mono truncate overflow-hidden"
                                  style={{
                                    ...(colResize.widths[key]
                                      ? {
                                          width: colResize.widths[key],
                                          maxWidth: colResize.widths[key],
                                        }
                                      : {}),
                                    color: ev ? scoreColor(ev.score) : undefined,
                                  }}
                                  title={formatCellValue(pred, true)}
                                >
                                  {formatCellValue(pred)}
                                </TableCell>
                              );
                            })}
                        </TableRow>
                      );
                    })}
                  </TableBody>
                  {filtered.length > 200 && (
                    <tfoot>
                      <tr>
                        <td
                          colSpan={99}
                          className="text-center py-3 text-[0.625rem] text-muted-foreground"
                        >
                          {msg("auto.features.optimizations.components.datatab.6")}
                          {filtered.length}
                          {msg("auto.features.optimizations.components.datatab.7")}
                        </td>
                      </tr>
                    </tfoot>
                  )}
                </Table>
              </div>
            </CardContent>
          </Card>
        )}
      </FadeIn>
    </div>
  );
}
