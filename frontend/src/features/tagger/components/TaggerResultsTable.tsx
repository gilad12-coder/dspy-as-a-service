"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { WarningCircle } from "@/shared/ui/icons";
import { Card, CardContent } from "@/shared/ui/primitives/card";
import { Badge } from "@/shared/ui/primitives/badge";
import { Table, TableBody, TableCell, TableHeader, TableRow } from "@/shared/ui/primitives/table";
import {
  ColumnHeader,
  ResetColumnsButton,
  ResetFiltersButton,
  useColumnFilters,
  useColumnResize,
  type SortDir,
} from "@/shared/ui/excel-filter";
import { ExportTableMenu } from "@/shared/ui/export-table-menu";
import { cn } from "@/shared/lib/utils";
import { msg, type MessageKey } from "@/shared/lib/messages";
import { flaggedRowIds } from "../lib/assist";
import { formatTaggerLabel } from "../lib/labels";
import type { Annotation, AssistState, DataRow, TaggerConfig } from "../lib/types";

type SortKey = "text" | "label" | "confidence" | "source";
type SortState = SortKey | "none";

/** One prepared table row: the data row plus everything its cells render. */
interface PreparedRow {
  index: number;
  id: string;
  text: string;
  label: string;
  confidence: number | null;
  provenance: string | null;
  flagged: boolean;
}

const PROVENANCE_KEYS: Record<string, MessageKey> = {
  human: "tagger.results.who.human",
  ai_confirmed: "tagger.results.who.ai_confirmed",
  ai_auto: "tagger.results.who.ai_auto",
};

/**
 * Review-at-a-glance table for a fully-labeled session — the primary surface
 * once every row carries a label. Built on the same shared table stack as the
 * optimization data views (ColumnHeader sorting, Excel-style column filters,
 * resizable columns); clicking a row opens the single-row focus view for edits.
 */
export function TaggerResultsTable({
  config,
  data,
  annotations,
  assist,
  onOpenRow,
}: {
  config: TaggerConfig;
  data: DataRow[];
  annotations: Record<string, Annotation>;
  assist: AssistState | null;
  onOpenRow: (index: number) => void;
}) {
  const [sortKey, setSortKey] = useState<SortState>("none");
  const [sortDir, setSortDir] = useState<SortDir>("asc");
  const colFilters = useColumnFilters();
  const colResize = useColumnResize();
  const [selected, setSelected] = useState(0);
  const selectedRef = useRef<HTMLTableRowElement | null>(null);

  const assisted = assist !== null;
  const flagged = useMemo(() => new Set(assist ? flaggedRowIds(assist) : []), [assist]);

  const rows = useMemo<PreparedRow[]>(
    () =>
      data.map((row, index) => {
        const id = String(row.id);
        const pred = assist?.predictions?.[id];
        return {
          index,
          id,
          text: row.text,
          label: formatTaggerLabel(config, annotations[id]),
          confidence: pred ? pred.confidence : null,
          provenance: assist?.provenance?.[id] ?? null,
          flagged: flagged.has(id),
        };
      }),
    [data, config, annotations, assist, flagged],
  );

  const labelOptions = useMemo(() => {
    // First-appearance order (by row index): the filter lists labels in the
    // same order they appear reading down the table, rather than an alphabetical
    // sort that wouldn't line up with what the user actually sees.
    const seen = new Set<string>();
    for (const row of rows) if (row.label) seen.add(row.label);
    return [...seen].map((value) => ({ value, label: value }));
  }, [rows]);

  const sourceOptions = useMemo(() => {
    const seen = new Set<string>();
    for (const row of rows) if (row.provenance) seen.add(row.provenance);
    return [...seen]
      .sort()
      .map((value) => ({ value, label: PROVENANCE_KEYS[value] ? msg(PROVENANCE_KEYS[value]) : value }));
  }, [rows]);

  const toggleSort = useCallback((key: SortState) => {
    if (key === "none") return;
    setSortKey((prev) => {
      if (prev === key) {
        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
        return prev;
      }
      setSortDir("asc");
      return key;
    });
  }, []);

  const visible = useMemo(() => {
    const labelFilter = colFilters.filters["label"];
    const sourceFilter = colFilters.filters["source"];
    let out = rows.filter(
      (row) =>
        (!labelFilter?.size || labelFilter.has(row.label)) &&
        (!sourceFilter?.size || sourceFilter.has(row.provenance ?? "")),
    );
    if (sortKey !== "none") {
      const dir = sortDir === "asc" ? 1 : -1;
      out = [...out].sort((a, b) => {
        if (sortKey === "confidence") {
          // Unscored rows always sink to the bottom, whatever the direction.
          if (a.confidence === null && b.confidence === null) return 0;
          if (a.confidence === null) return 1;
          if (b.confidence === null) return -1;
          return (a.confidence - b.confidence) * dir;
        }
        const av = sortKey === "source" ? (a.provenance ?? "") : a[sortKey];
        const bv = sortKey === "source" ? (b.provenance ?? "") : b[sortKey];
        return av.localeCompare(bv) * dir;
      });
    }
    return out;
  }, [rows, colFilters.filters, sortKey, sortDir]);

  // Selection follows filters and sorting — clamp instead of chasing the old row.
  useEffect(() => {
    setSelected((prev) => Math.min(prev, Math.max(0, visible.length - 1)));
  }, [visible.length]);

  useEffect(() => {
    selectedRef.current?.scrollIntoView({ block: "nearest" });
  }, [selected]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelected((prev) => Math.min(prev + 1, visible.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelected((prev) => Math.max(prev - 1, 0));
      } else if (e.key === "Enter") {
        const row = visible[selected];
        if (row) {
          e.preventDefault();
          onOpenRow(row.index);
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [visible, selected, onOpenRow]);

  return (
    <Card>
      <CardContent className="space-y-3 pt-6">
        <div className="flex items-center justify-end gap-2">
          <ResetFiltersButton filters={colFilters} />
          {colResize.hasResized && <ResetColumnsButton resize={colResize} />}
          <ExportTableMenu
            iconOnly
            disabled={visible.length === 0}
            getData={() => {
              const colText = msg("tagger.results.col.text");
              const colLabel = msg("tagger.results.col.label");
              const colConfidence = msg("tagger.results.col.confidence");
              const colSource = msg("tagger.results.col.source");
              const columns = assisted
                ? [colText, colLabel, colConfidence, colSource]
                : [colText, colLabel];
              return {
                columns,
                rows: visible.map((row) => {
                  const provKey = row.provenance ? PROVENANCE_KEYS[row.provenance] : undefined;
                  const out: Record<string, unknown> = {
                    [colText]: row.text,
                    [colLabel]: row.label,
                  };
                  if (assisted) {
                    out[colConfidence] =
                      row.confidence !== null ? `${Math.round(row.confidence * 100)}%` : "";
                    out[colSource] = provKey ? msg(provKey) : "";
                  }
                  return out;
                }),
                filename: "tagger_results",
              };
            }}
          />
        </div>

        {visible.length === 0 ? (
          <p className="py-10 text-center text-sm text-muted-foreground">
            {msg("tagger.results.empty")}
          </p>
        ) : (
          <div className="max-h-[calc(100dvh-var(--header-height,53px)-14rem)] overflow-auto rounded-lg border border-border/60">
            <Table className="table-fixed">
              <TableHeader className="bg-card">
                <TableRow>
                  <ColumnHeader
                    label={msg("tagger.results.col.text")}
                    sortKey="text"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    width={colResize.widths["text"]}
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("tagger.results.col.label")}
                    sortKey="label"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    filterCol="label"
                    filterOptions={labelOptions}
                    filters={colFilters.filters}
                    onFilter={colFilters.setColumnFilter}
                    openFilter={colFilters.openFilter}
                    setOpenFilter={colFilters.setOpenFilter}
                    width={colResize.widths["label"] ?? 200}
                    onResize={colResize.setColumnWidth}
                  />
                  {assisted && (
                    <>
                      <ColumnHeader
                        label={msg("tagger.results.col.confidence")}
                        sortKey="confidence"
                        currentSort={sortKey}
                        sortDir={sortDir}
                        onSort={toggleSort}
                        width={colResize.widths["confidence"] ?? 110}
                        onResize={colResize.setColumnWidth}
                      />
                      <ColumnHeader
                        label={msg("tagger.results.col.source")}
                        sortKey="source"
                        currentSort={sortKey}
                        sortDir={sortDir}
                        onSort={toggleSort}
                        filterCol="source"
                        filterOptions={sourceOptions}
                        filters={colFilters.filters}
                        onFilter={colFilters.setColumnFilter}
                        openFilter={colFilters.openFilter}
                        setOpenFilter={colFilters.setOpenFilter}
                        width={colResize.widths["source"] ?? 130}
                        onResize={colResize.setColumnWidth}
                      />
                    </>
                  )}
                </TableRow>
              </TableHeader>
              <TableBody>
                {visible.map((row, i) => {
                  const provKey = row.provenance ? PROVENANCE_KEYS[row.provenance] : undefined;
                  return (
                    <TableRow
                      key={row.id}
                      ref={i === selected ? selectedRef : undefined}
                      onClick={() => {
                        setSelected(i);
                        onOpenRow(row.index);
                      }}
                      className={cn(
                        "cursor-pointer transition-colors",
                        i === selected ? "bg-muted/70" : "hover:bg-muted/40",
                      )}
                    >
                      <TableCell className="max-w-0">
                        <span className="flex items-center gap-1.5">
                          {row.flagged && (
                            <WarningCircle className="size-3.5 shrink-0 text-amber-600 dark:text-amber-500" />
                          )}
                          <span className="truncate" dir="auto">
                            {row.text}
                          </span>
                        </span>
                      </TableCell>
                      <TableCell>
                        <span className="line-clamp-1 font-medium text-foreground" dir="auto">
                          {row.label}
                        </span>
                      </TableCell>
                      {assisted && (
                        <>
                          <TableCell className="tabular-nums text-muted-foreground">
                            {row.confidence !== null ? `${Math.round(row.confidence * 100)}%` : ""}
                          </TableCell>
                          <TableCell>
                            {provKey && (
                              <Badge variant="secondary" size="sm">
                                {msg(provKey)}
                              </Badge>
                            )}
                          </TableCell>
                        </>
                      )}
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
