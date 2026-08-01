import { toast } from "react-toastify";
import type { OptimizationSummaryResponse } from "@/shared/types/api";
import { formatMsg, msg } from "@/shared/lib/messages";

export type JobsExportFormat = "csv" | "json" | "xlsx" | "xls";

type ExportValue = string | number | null;

type ExportColumn = {
  key: string;
  getValue: (job: OptimizationSummaryResponse) => ExportValue;
};

const BASE_COLUMNS: ExportColumn[] = [
  { key: "optimization_id", getValue: (job) => job.optimization_id },
  { key: "name", getValue: (job) => job.name ?? "" },
  { key: "optimization_type", getValue: (job) => job.optimization_type },
  { key: "status", getValue: (job) => job.status },
  { key: "module_name", getValue: (job) => job.module_name ?? "" },
  { key: "dataset_rows", getValue: (job) => job.dataset_rows ?? null },
  { key: "created_at", getValue: (job) => job.created_at },
  { key: "elapsed_seconds", getValue: (job) => job.elapsed_seconds ?? null },
  { key: "baseline_test_metric", getValue: (job) => job.baseline_test_metric ?? null },
  { key: "optimized_test_metric", getValue: (job) => job.optimized_test_metric ?? null },
  { key: "metric_improvement", getValue: (job) => job.metric_improvement ?? null },
];

const SHARED_COLUMNS: ExportColumn[] = [
  { key: "owner", getValue: (job) => job.username ?? "" },
  { key: "role", getValue: (job) => job.role ?? "owned" },
];

function columnsFor(showSharedColumns: boolean): ExportColumn[] {
  if (!showSharedColumns) return BASE_COLUMNS;
  return [...BASE_COLUMNS.slice(0, 2), ...SHARED_COLUMNS, ...BASE_COLUMNS.slice(2)];
}

/** Build stable, machine-readable rows for the visible jobs table. */
export function buildJobExportRows(
  jobs: OptimizationSummaryResponse[],
  showSharedColumns: boolean,
): { columns: string[]; rows: Array<Record<string, ExportValue>> } {
  const columns = columnsFor(showSharedColumns);
  return {
    columns: columns.map((column) => column.key),
    rows: jobs.map((job) =>
      Object.fromEntries(columns.map((column) => [column.key, column.getValue(job)])),
    ),
  };
}

function csvCell(value: ExportValue): string {
  const text = value == null ? "" : String(value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function triggerDownload(content: BlobPart, filename: string, mimeType: string): void {
  const url = URL.createObjectURL(new Blob([content], { type: mimeType }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function exportCsv(
  columns: string[],
  rows: Array<Record<string, ExportValue>>,
  filename: string,
): void {
  const csv = [columns, ...rows.map((row) => columns.map((column) => row[column] ?? null))]
    .map((row) => row.map(csvCell).join(","))
    .join("\n");
  triggerDownload(`\ufeff${csv}\n`, filename, "text/csv;charset=utf-8");
}

function exportJson(rows: Array<Record<string, ExportValue>>, filename: string): void {
  triggerDownload(JSON.stringify(rows, null, 2), filename, "application/json");
}

async function exportExcel(
  columns: string[],
  rows: Array<Record<string, ExportValue>>,
  filename: string,
  bookType: "xlsx" | "xlml",
): Promise<void> {
  const XLSX = await import("xlsx");
  const worksheet = XLSX.utils.json_to_sheet(rows, { header: columns });
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, "Optimizations");
  const buffer = XLSX.write(workbook, { type: "array", bookType });
  const mimeType =
    bookType === "xlsx"
      ? "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
      : "application/vnd.ms-excel";
  triggerDownload(buffer, filename, mimeType);
}

/** Export the visible jobs table in the selected file format. */
export async function exportJobs(
  jobs: OptimizationSummaryResponse[],
  showSharedColumns: boolean,
  format: JobsExportFormat,
): Promise<void> {
  if (jobs.length === 0) {
    toast.error(msg("usage.export.empty"));
    return;
  }

  const { columns, rows } = buildJobExportRows(jobs, showSharedColumns);
  const base = `skynet-optimizations-${new Date().toISOString().slice(0, 10)}`;

  switch (format) {
    case "csv":
      exportCsv(columns, rows, `${base}.csv`);
      break;
    case "json":
      exportJson(rows, `${base}.json`);
      break;
    case "xlsx":
      await exportExcel(columns, rows, `${base}.xlsx`, "xlsx");
      break;
    case "xls":
      await exportExcel(columns, rows, `${base}.xls`, "xlml");
      break;
  }

  toast.success(formatMsg("usage.export.done", { p1: String(rows.length) }));
}
