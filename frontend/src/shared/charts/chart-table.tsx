"use client";

import * as React from "react";
import { cn } from "@/shared/lib/utils";
import { getActiveDir } from "@/shared/lib/runtime-locale";

export interface ChartTableColumn<T> {
  key: Extract<keyof T, string>;
  label: string;
  align?: "start" | "end";
  format?: (value: unknown, row: T) => React.ReactNode;
}

function defaultFormat(value: unknown): React.ReactNode {
  if (value == null) return "—";
  if (typeof value === "number") return Number.isInteger(value) ? value : value.toFixed(2);
  return String(value);
}

/**
 * Static table shown in place of a recharts chart when Lite mode is on — same
 * data, none of the SVG/animation/layout cost that makes charts expensive on
 * weak hardware. Rows stay optionally clickable so a chart's drill-down
 * interaction survives the swap, and the body scrolls within the chart's box so
 * long series stay contained.
 */
export function ChartTable<T>({
  columns,
  rows,
  onRowClick,
  emptyLabel = "—",
}: {
  columns: Array<ChartTableColumn<T>>;
  rows: T[];
  onRowClick?: (row: T, index: number) => void;
  emptyLabel?: string;
}) {
  if (rows.length === 0) {
    return (
      <div className="flex h-full min-h-24 items-center justify-center text-sm text-muted-foreground">
        {emptyLabel}
      </div>
    );
  }
  const clickable = onRowClick != null;
  return (
    <div className="h-full max-h-full overflow-auto rounded-lg border border-border">
      <table className="w-full border-collapse text-sm" dir={getActiveDir()}>
        <thead className="sticky top-0 bg-muted text-xs">
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                className={cn(
                  "px-3 py-2 font-semibold text-muted-foreground",
                  c.align === "end" ? "text-end" : "text-start",
                )}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr
              key={i}
              onClick={clickable ? () => onRowClick(row, i) : undefined}
              className={cn(
                "border-t border-border/60",
                clickable && "cursor-pointer hover:bg-accent/60",
              )}
            >
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={cn(
                    "px-3 py-1.5 tabular-nums",
                    c.align === "end" ? "text-end" : "text-start",
                  )}
                >
                  {c.format ? c.format(row[c.key], row) : defaultFormat(row[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
