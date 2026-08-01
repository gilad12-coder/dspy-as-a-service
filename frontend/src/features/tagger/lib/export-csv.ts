import { exportTable } from "@/shared/lib/export-table";
import type { AnnotationProvenance, DataRow, Annotation, TaggerConfig } from "./types";

export type ExportFormat = "csv" | "json" | "xlsx" | "xls" | "feather" | "parquet";

/** Column the annotation is written under, keyed by tagging mode. */
function annotationColumn(config: TaggerConfig): string {
  return config.mode === "binary"
    ? "binary_label"
    : config.mode === "multiclass"
      ? "selected_categories"
      : "extracted_text";
}

const PROVENANCE_COL = "label_source";

function buildRows(
  data: DataRow[],
  columns: string[],
  annotations: Record<string, Annotation>,
  config: TaggerConfig,
  provenance?: Record<string, AnnotationProvenance>,
): { allCols: string[]; rows: Array<Record<string, string>> } {
  const annotCol = annotationColumn(config);

  const withProvenance = provenance !== undefined && Object.keys(provenance).length > 0;
  const allCols = withProvenance ? [...columns, annotCol, PROVENANCE_COL] : [...columns, annotCol];
  const rows: Array<Record<string, string>> = [];

  for (const item of data) {
    const row: Record<string, string> = {};
    for (const col of columns) {
      const v = item[col];
      row[col] = v !== undefined && v !== null ? String(v) : "";
    }

    const ann = annotations[String(item.id)];
    let annStr = "";
    if (config.mode === "multiclass" && Array.isArray(ann)) {
      const cats = config.categories ?? [];
      annStr = ann.map((catId) => cats.find((c) => c.id === catId)?.label ?? catId).join("; ");
    } else if (typeof ann === "string") {
      // Free-text annotations are stored raw while typing (a trim there would
      // eat the trailing space mid-word); tidy the edges only at export.
      annStr = ann.trim();
    }
    row[annotCol] = annStr;
    if (withProvenance) {
      row[PROVENANCE_COL] = annStr ? (provenance[String(item.id)] ?? "human") : "";
    }
    rows.push(row);
  }

  return { allCols, rows };
}

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function escapeCSV(val: string): string {
  if (val.includes(",") || val.includes('"') || val.includes("\n")) {
    return `"${val.replace(/"/g, '""')}"`;
  }
  return val;
}

function exportCSV(allCols: string[], rows: Array<Record<string, string>>, filename: string) {
  const BOM = "\ufeff";
  let csv = `${BOM + allCols.map(escapeCSV).join(",")}\n`;
  for (const row of rows) {
    csv += `${allCols.map((col) => escapeCSV(row[col] ?? "")).join(",")}\n`;
  }
  download(new Blob([csv], { type: "text/csv;charset=utf-8" }), filename);
}

function exportJSON(rows: Array<Record<string, string>>, filename: string) {
  const json = JSON.stringify(rows, null, 2);
  download(new Blob([json], { type: "application/json;charset=utf-8" }), filename);
}

async function exportExcel(
  allCols: string[],
  rows: Array<Record<string, string>>,
  filename: string,
  bookType: "xlsx" | "xlml",
) {
  // Lazy-load xlsx (~900KB) so it stays out of the initial tagger-route
  // chunk — only pulled in when an Excel export is actually requested.
  const XLSX = await import("xlsx");
  const ws = XLSX.utils.json_to_sheet(rows, { header: allCols });
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Annotations");
  const buf = XLSX.write(wb, { type: "array", bookType });
  const mime =
    bookType === "xlsx"
      ? "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
      : "application/vnd.ms-excel";
  download(new Blob([buf], { type: mime }), filename);
}

/**
 * Build annotated rows plus a saved column schema for the dataset library.
 * Input columns keep the "input" role and the annotation column is marked
 * "output", so picking the saved dataset in the submit wizard pre-fills the
 * annotation as the target column.
 */
export function buildLibraryRows(
  data: DataRow[],
  columns: string[],
  annotations: Record<string, Annotation>,
  config: TaggerConfig,
  provenance?: Record<string, AnnotationProvenance>,
): {
  rows: Array<Record<string, string>>;
  columnOrder: string[];
  columnRoles: Record<string, "input" | "output" | "ignore">;
} {
  const { allCols, rows } = buildRows(data, columns, annotations, config, provenance);
  const annotCol = annotationColumn(config);
  const columnRoles: Record<string, "input" | "output" | "ignore"> = {};
  for (const col of allCols) {
    columnRoles[col] =
      col === annotCol ? "output" : col === PROVENANCE_COL ? "ignore" : "input";
  }
  return { rows, columnOrder: allCols, columnRoles };
}

export async function exportAnnotations(
  data: DataRow[],
  columns: string[],
  annotations: Record<string, Annotation>,
  config: TaggerConfig,
  format: ExportFormat,
  provenance?: Record<string, AnnotationProvenance>,
) {
  const { allCols, rows } = buildRows(data, columns, annotations, config, provenance);
  const stem = `tagging_${config.mode}`;
  const base = `${stem}_${new Date().toISOString().slice(0, 10)}`;

  switch (format) {
    case "csv":
      exportCSV(allCols, rows, `${base}.csv`);
      break;
    case "json":
      exportJSON(rows, `${base}.json`);
      break;
    case "xlsx":
      await exportExcel(allCols, rows, `${base}.xlsx`, "xlsx");
      break;
    case "xls":
      await exportExcel(allCols, rows, `${base}.xls`, "xlml");
      break;
    case "feather":
    case "parquet":
      // Columnar formats reuse the shared writer; it stamps the date onto the
      // stem itself, so pass the undated stem to avoid a doubled date.
      await exportTable({ columns: allCols, rows, filename: stem }, format);
      break;
  }
}
