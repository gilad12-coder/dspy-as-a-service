"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CircleAlert, Download } from "lucide-react";
import { Card, CardContent } from "@/shared/ui/primitives/card";
import { Badge } from "@/shared/ui/primitives/badge";
import { Button } from "@/shared/ui/primitives/button";
import { cn } from "@/shared/lib/utils";
import { msg, type MessageKey } from "@/shared/lib/messages";
import { exportAnnotations } from "../lib/export-csv";
import { FLAG_CONFIDENCE, flaggedRowIds } from "../lib/assist";
import type { Annotation, AssistState, DataRow, TaggerConfig } from "../lib/types";

type ResultsFilter = "all" | "flagged" | "low";

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

function displayLabel(config: TaggerConfig, ann: Annotation): string {
  if (ann === undefined || ann === null) return "";
  if (Array.isArray(ann)) {
    const cats = config.categories ?? [];
    return ann.map((id) => cats.find((c) => c.id === id)?.label ?? id).join(", ");
  }
  if (config.mode === "binary") {
    if (ann === "yes") return msg("tagger.assist.label.yes");
    if (ann === "no") return msg("tagger.assist.label.no");
  }
  return String(ann);
}

const PROVENANCE_KEYS: Record<string, MessageKey> = {
  human: "tagger.results.who.human",
  ai_confirmed: "tagger.results.who.ai_confirmed",
  ai_auto: "tagger.results.who.ai_auto",
};

/**
 * Review-at-a-glance table for a fully-labeled session — the primary surface
 * once every row carries a label. Each row shows the text, its label and (for
 * assisted sessions) the model confidence and who produced the label; clicking
 * a row opens the single-row focus view for edits.
 */
export function TaggerResultsTable({
  config,
  data,
  columns,
  annotations,
  assist,
  onOpenRow,
}: {
  config: TaggerConfig;
  data: DataRow[];
  columns: string[];
  annotations: Record<string, Annotation>;
  assist: AssistState | null;
  onOpenRow: (index: number) => void;
}) {
  const [filter, setFilter] = useState<ResultsFilter>("all");
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
          label: displayLabel(config, annotations[id]),
          confidence: pred ? pred.confidence : null,
          provenance: assist?.provenance?.[id] ?? null,
          flagged: flagged.has(id),
        };
      }),
    [data, config, annotations, assist, flagged],
  );

  const filtered = useMemo(() => {
    if (filter === "flagged") return rows.filter((r) => r.flagged);
    if (filter === "low") {
      return rows.filter((r) => r.confidence !== null && r.confidence < FLAG_CONFIDENCE);
    }
    return rows;
  }, [rows, filter]);

  const flaggedCount = useMemo(() => rows.filter((r) => r.flagged).length, [rows]);
  const lowCount = useMemo(
    () => rows.filter((r) => r.confidence !== null && r.confidence < FLAG_CONFIDENCE).length,
    [rows],
  );

  const handleExport = useCallback(() => {
    void exportAnnotations(data, columns, annotations, config, "csv", assist?.provenance);
  }, [data, columns, annotations, config, assist?.provenance]);

  // Selection follows the filter — clamp instead of chasing the old row.
  useEffect(() => {
    setSelected((prev) => Math.min(prev, Math.max(0, filtered.length - 1)));
  }, [filtered.length]);

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
        setSelected((prev) => Math.min(prev + 1, filtered.length - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelected((prev) => Math.max(prev - 1, 0));
      } else if (e.key === "Enter") {
        const row = filtered[selected];
        if (row) {
          e.preventDefault();
          onOpenRow(row.index);
        }
      } else if (e.key.toLowerCase() === "e") {
        e.preventDefault();
        handleExport();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [filtered, selected, onOpenRow, handleExport]);

  const filters: Array<{ id: ResultsFilter; label: string; count: number }> = [
    { id: "all", label: msg("tagger.results.filter.all"), count: rows.length },
    ...(assisted
      ? [
          {
            id: "flagged" as const,
            label: msg("tagger.results.filter.flagged"),
            count: flaggedCount,
          },
          { id: "low" as const, label: msg("tagger.results.filter.low"), count: lowCount },
        ]
      : []),
  ];

  return (
    <Card>
      <CardContent className="space-y-3 pt-6">
        <div className="flex flex-wrap items-center gap-2">
          {filters.map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => setFilter(f.id)}
              className={cn(
                "flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs transition-colors cursor-pointer",
                filter === f.id
                  ? "border-primary/40 bg-primary/10 font-medium text-primary"
                  : "border-border text-muted-foreground hover:bg-muted/60 hover:text-foreground",
              )}
            >
              {f.label}
              <span className="tabular-nums opacity-70">{f.count}</span>
            </button>
          ))}
          <Button
            variant="outline"
            size="sm"
            onClick={handleExport}
            className="ms-auto gap-1.5"
          >
            <Download className="size-3.5" />
            {msg("tagger.results.export")}
          </Button>
        </div>

        {filtered.length === 0 ? (
          <p className="py-10 text-center text-sm text-muted-foreground">
            {msg("tagger.results.empty")}
          </p>
        ) : (
          <div className="max-h-[calc(100dvh-var(--header-height,53px)-14rem)] overflow-auto rounded-lg border border-border/60">
            <table className="w-full text-sm">
              <thead className="sticky top-0 z-10 bg-card">
                <tr className="border-b border-border/60 text-xs text-muted-foreground">
                  <th className="px-3 py-2 text-start font-medium">
                    {msg("tagger.results.col.text")}
                  </th>
                  <th className="w-[22%] px-3 py-2 text-start font-medium">
                    {msg("tagger.results.col.label")}
                  </th>
                  {assisted && (
                    <>
                      <th className="w-24 px-3 py-2 text-start font-medium">
                        {msg("tagger.results.col.confidence")}
                      </th>
                      <th className="w-28 px-3 py-2 text-start font-medium">
                        {msg("tagger.results.col.source")}
                      </th>
                    </>
                  )}
                </tr>
              </thead>
              <tbody>
                {filtered.map((row, i) => {
                  const provKey = row.provenance ? PROVENANCE_KEYS[row.provenance] : undefined;
                  return (
                    <tr
                      key={row.id}
                      ref={i === selected ? selectedRef : undefined}
                      onClick={() => {
                        setSelected(i);
                        onOpenRow(row.index);
                      }}
                      className={cn(
                        "cursor-pointer border-b border-border/40 last:border-b-0 transition-colors",
                        i === selected ? "bg-muted/70" : "hover:bg-muted/40",
                      )}
                    >
                      <td className="max-w-0 px-3 py-2">
                        <span className="flex items-center gap-1.5">
                          {row.flagged && (
                            <CircleAlert className="size-3.5 shrink-0 text-amber-600 dark:text-amber-500" />
                          )}
                          <span className="truncate" dir="auto">
                            {row.text}
                          </span>
                        </span>
                      </td>
                      <td className="px-3 py-2">
                        <span className="line-clamp-1 font-medium text-foreground" dir="auto">
                          {row.label}
                        </span>
                      </td>
                      {assisted && (
                        <>
                          <td className="px-3 py-2 tabular-nums text-muted-foreground">
                            {row.confidence !== null ? `${Math.round(row.confidence * 100)}%` : ""}
                          </td>
                          <td className="px-3 py-2">
                            {provKey && (
                              <Badge variant="secondary" size="sm">
                                {msg(provKey)}
                              </Badge>
                            )}
                          </td>
                        </>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
