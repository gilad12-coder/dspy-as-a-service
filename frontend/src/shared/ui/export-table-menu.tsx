"use client";

import { Boxes, Braces, Database, Download, Feather, FileText, Sheet } from "lucide-react";
import { Popover as PopoverPrimitive } from "radix-ui";
import { toast } from "react-toastify";
import { Button } from "@/shared/ui/primitives/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";
import {
  COLUMNAR_FORMATS,
  EmptyTableExportError,
  exportTable,
  TABLE_EXPORT_FORMATS,
  type TableExportFormat,
  type TableExportOptions,
} from "@/shared/lib/export-table";

const FORMAT_META: Record<
  TableExportFormat,
  { label: string; ext: string; Icon: typeof FileText }
> = {
  csv: { label: "CSV", ext: ".csv", Icon: FileText },
  json: { label: "JSON", ext: ".json", Icon: Braces },
  xlsx: { label: "Excel", ext: ".xlsx", Icon: Sheet },
  parquet: { label: "Parquet", ext: ".parquet", Icon: Boxes },
  feather: { label: "Feather", ext: ".feather", Icon: Feather },
};

export interface ExportTableMenuProps {
  /**
   * Produce the export payload lazily, at click time, so the download always
   * reflects the table's *current* filtered/sorted rows rather than a snapshot
   * from render.
   */
  getData: () => TableExportOptions;
  /** Render just the download icon (for tight table toolbars) instead of a labelled button. */
  iconOnly?: boolean;
  /** Disable the trigger (e.g. while the table is still loading). */
  disabled?: boolean;
  /** Popover edge alignment; defaults to the menu's end (RTL-aware via Radix). */
  align?: "start" | "center" | "end";
  className?: string;
}

/**
 * Download control for any data table: a small trigger that opens a format
 * picker (CSV, JSON, Excel, plus columnar Parquet/Feather) and writes the file
 * client-side via {@link exportTable}. Drop it into a table's header and hand it
 * a `getData` thunk that returns the rows currently on screen.
 */
export function ExportTableMenu({
  getData,
  iconOnly = false,
  disabled = false,
  align = "end",
  className,
}: ExportTableMenuProps) {
  async function handle(format: TableExportFormat) {
    try {
      const data = getData();
      await exportTable(data, format);
      toast.success(formatMsg("export.table.done", { count: data.rows.length }));
    } catch (err) {
      if (err instanceof EmptyTableExportError) {
        toast.info(msg("export.table.empty"));
        return;
      }
      toast.error(msg("export.table.failed"));
    }
  }

  const itemCls =
    "flex w-full items-center gap-2.5 px-3.5 py-2 text-xs text-foreground hover:bg-muted/40 cursor-pointer transition-colors";
  const iconCls = "size-4 shrink-0 text-muted-foreground/60";
  const extCls = "ms-auto font-mono text-[0.625rem] text-muted-foreground/60";

  return (
    <PopoverPrimitive.Root>
      <Tooltip>
        <TooltipTrigger asChild>
          <PopoverPrimitive.Trigger asChild>
            {iconOnly ? (
              <Button
                variant="ghost"
                size="icon-sm"
                disabled={disabled}
                aria-label={msg("export.table.aria")}
                className={className}
              >
                <Download
                  className="size-[1.05rem] text-primary"
                  strokeWidth={2.25}
                  aria-hidden="true"
                />
              </Button>
            ) : (
              <Button
                variant="outline"
                size="sm"
                disabled={disabled}
                aria-label={msg("export.table.aria")}
                className={cn("gap-1.5", className)}
              >
                <Download
                  className="size-[1.05rem] text-primary"
                  strokeWidth={2.25}
                  aria-hidden="true"
                />
                {msg("export.table.button")}
              </Button>
            )}
          </PopoverPrimitive.Trigger>
        </TooltipTrigger>
        <TooltipContent>{msg("export.table.aria")}</TooltipContent>
      </Tooltip>
      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Content
          align={align}
          side="bottom"
          sideOffset={6}
          className="z-50 min-w-[190px] max-w-[min(240px,90vw)] rounded-2xl border border-border/40 bg-card py-1.5 shadow-[0_4px_24px_rgba(28,22,18,0.1)] animate-in fade-in-0 zoom-in-95"
        >
          {TABLE_EXPORT_FORMATS.map((fmt) => {
            const { label, ext, Icon } = FORMAT_META[fmt];
            // A labelled divider introduces the columnar group so the analytics
            // formats read as a distinct, purposeful choice.
            const startsColumnar = fmt === "parquet";
            return (
              <div key={fmt}>
                {startsColumnar && (
                  <div className="mx-3.5 mt-1.5 mb-1 border-t border-border/40 pt-1.5 text-[0.625rem] font-medium uppercase tracking-wide text-muted-foreground/50">
                    {msg("export.table.columnar")}
                  </div>
                )}
                <PopoverPrimitive.Close asChild>
                  <button type="button" onClick={() => handle(fmt)} className={itemCls}>
                    <Icon className={iconCls} />
                    <span className="flex-1 text-start">{label}</span>
                    <span className={extCls}>{ext}</span>
                  </button>
                </PopoverPrimitive.Close>
              </div>
            );
          })}
        </PopoverPrimitive.Content>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
}
