/**
 * Client-side table export — turn the rows any data table already holds into a
 * downloadable file, in the same formats the tagger offers (CSV/JSON/XLSX/XLS)
 * plus the columnar analytics formats a data platform is expected to speak:
 * Parquet and Feather (Arrow IPC).
 *
 * Everything runs in the browser off the rows currently on screen — no server
 * round-trip — mirroring the tagger's export path. The heavy writers
 * (`xlsx`, `apache-arrow`, `hyparquet-writer`) are pulled in with dynamic
 * `import()` so they never touch a route's initial chunk; only the format the
 * user actually picks is downloaded.
 */

import type { Vector } from "apache-arrow";
import type { BasicType } from "hyparquet-writer";

export type TableExportFormat = "csv" | "json" | "xlsx" | "xls" | "parquet" | "feather";

/** Display order for a format picker: spreadsheet-friendly first, columnar last. */
export const TABLE_EXPORT_FORMATS: readonly TableExportFormat[] = [
  "csv",
  "json",
  "xlsx",
  "xls",
  "parquet",
  "feather",
];

/** The columnar analytics formats — grouped apart in the menu, always last. */
export const COLUMNAR_FORMATS: ReadonlySet<TableExportFormat> = new Set(["parquet", "feather"]);

export interface TableExportData {
  /** Column order; also the header row. */
  columns: string[];
  /** One object per row, keyed by column name. Missing keys export as empty. */
  rows: Array<Record<string, unknown>>;
}

export interface TableExportOptions extends TableExportData {
  /** Filename stem (no extension, no date); the current date is appended. */
  filename: string;
}

/** Raised when there is nothing to write, so callers can show a distinct hint. */
export class EmptyTableExportError extends Error {
  constructor() {
    super("no rows to export");
    this.name = "EmptyTableExportError";
  }
}

/** A cell reduced to a primitive: objects/arrays become JSON, nullish becomes null. */
type Cell = string | number | boolean | null;
type ColumnKind = "number" | "boolean" | "string";

interface PreparedColumn {
  name: string;
  kind: ColumnKind;
  /** Row-aligned, homogeneous per `kind`; nulls preserved. */
  cells: Cell[];
}

function normalizeCell(value: unknown): Cell {
  if (value === null || value === undefined) return null;
  const t = typeof value;
  if (t === "number") return Number.isFinite(value as number) ? (value as number) : String(value);
  if (t === "boolean" || t === "string") return value as boolean | string;
  if (t === "bigint") return (value as bigint).toString();
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/**
 * Decide a column's type from its non-null cells: uniformly-numeric or
 * uniformly-boolean columns keep that type (so Parquet/Feather stay well-typed
 * and compressible); anything mixed or textual falls back to string.
 */
function columnKind(cells: Cell[]): ColumnKind {
  let sawNumber = false;
  let sawBoolean = false;
  let sawString = false;
  for (const c of cells) {
    if (c === null) continue;
    if (typeof c === "number") sawNumber = true;
    else if (typeof c === "boolean") sawBoolean = true;
    else sawString = true;
  }
  if (sawString || (sawNumber && sawBoolean)) return "string";
  if (sawNumber) return "number";
  if (sawBoolean) return "boolean";
  return "string";
}

/** Coerce a normalized cell to match its column's chosen kind (nulls pass through). */
function coerce(cell: Cell, kind: ColumnKind): Cell {
  if (cell === null) return null;
  if (kind === "string") return typeof cell === "string" ? cell : String(cell);
  return cell;
}

function prepare({ columns, rows }: TableExportData): PreparedColumn[] {
  return columns.map((name) => {
    const raw = rows.map((r) => normalizeCell(r[name]));
    const kind = columnKind(raw);
    return { name, kind, cells: raw.map((c) => coerce(c, kind)) };
  });
}

/** Rows as plain objects of typed cells — for JSON and the spreadsheet writer. */
function rowObjects(prepared: PreparedColumn[], rowCount: number): Array<Record<string, Cell>> {
  const out: Array<Record<string, Cell>> = [];
  for (let i = 0; i < rowCount; i++) {
    const row: Record<string, Cell> = {};
    for (const col of prepared) row[col.name] = col.cells[i] ?? null;
    out.push(row);
  }
  return out;
}

function triggerDownload(bytes: BlobPart, mime: string, filename: string): void {
  const url = URL.createObjectURL(new Blob([bytes], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Defer revoke so slow browsers have time to start the download.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function csvField(value: Cell): string {
  const s = value === null ? "" : String(value);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function exportCsv(prepared: PreparedColumn[], rowCount: number, filename: string): void {
  const lines = [prepared.map((c) => csvField(c.name)).join(",")];
  for (let i = 0; i < rowCount; i++) {
    lines.push(prepared.map((c) => csvField(c.cells[i] ?? null)).join(","));
  }
  // Leading BOM so Excel opens UTF-8 (incl. Hebrew) without mojibake.
  const body = `﻿${lines.join("\n")}\n`;
  triggerDownload(body, "text/csv;charset=utf-8", `${filename}.csv`);
}

function exportJson(prepared: PreparedColumn[], rowCount: number, filename: string): void {
  const json = JSON.stringify(rowObjects(prepared, rowCount), null, 2);
  triggerDownload(json, "application/json;charset=utf-8", `${filename}.json`);
}

async function exportExcel(
  prepared: PreparedColumn[],
  rowCount: number,
  filename: string,
  bookType: "xlsx" | "xlml",
): Promise<void> {
  const XLSX = await import("xlsx");
  const header = prepared.map((c) => c.name);
  const ws = XLSX.utils.json_to_sheet(rowObjects(prepared, rowCount), { header });
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, "Sheet1");
  const buf = XLSX.write(wb, { type: "array", bookType }) as ArrayBuffer;
  const mime =
    bookType === "xlsx"
      ? "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
      : "application/vnd.ms-excel";
  const ext = bookType === "xlsx" ? "xlsx" : "xls";
  triggerDownload(buf, mime, `${filename}.${ext}`);
}

async function exportFeather(prepared: PreparedColumn[], filename: string): Promise<void> {
  const { Table, vectorFromArray, tableToIPC } = await import("apache-arrow");
  const vectors: Record<string, Vector> = {};
  for (const col of prepared) {
    // vectorFromArray infers Utf8/Float64/Bool from the (homogeneous) cells and
    // carries nulls through the validity bitmap.
    vectors[col.name] = vectorFromArray(col.cells);
  }
  const bytes = tableToIPC(new Table(vectors), "file");
  // Copy into a fresh ArrayBuffer-backed view so it satisfies BlobPart.
  triggerDownload(new Uint8Array(bytes), "application/vnd.apache.arrow.file", `${filename}.feather`);
}

async function exportParquet(prepared: PreparedColumn[], filename: string): Promise<void> {
  const { parquetWriteBuffer } = await import("hyparquet-writer");
  const columnData = prepared.map((col) => {
    const type: BasicType =
      col.kind === "number" ? "DOUBLE" : col.kind === "boolean" ? "BOOLEAN" : "STRING";
    return { name: col.name, data: col.cells, type, nullable: true };
  });
  // codec defaults to SNAPPY — fast, compressible, the point of the format.
  const buf = parquetWriteBuffer({ columnData });
  triggerDownload(new Uint8Array(buf), "application/vnd.apache.parquet", `${filename}.parquet`);
}

/**
 * Write the given rows to a downloaded file in `format`.
 *
 * Args:
 *   opts: Column order, rows, and the filename stem (the current date is
 *     appended, so callers pass e.g. ``"runs"`` → ``runs_2026-08-01.csv``).
 *   format: One of ``TABLE_EXPORT_FORMATS``.
 *
 * Raises:
 *   EmptyTableExportError: If there are no rows, so the caller can distinguish
 *     "nothing to export" from a genuine writer failure.
 */
export async function exportTable(
  opts: TableExportOptions,
  format: TableExportFormat,
): Promise<void> {
  if (opts.rows.length === 0) throw new EmptyTableExportError();
  const prepared = prepare(opts);
  const rowCount = opts.rows.length;
  const stamp = new Date().toISOString().slice(0, 10);
  const filename = `${opts.filename}_${stamp}`;

  switch (format) {
    case "csv":
      return exportCsv(prepared, rowCount, filename);
    case "json":
      return exportJson(prepared, rowCount, filename);
    case "xlsx":
      return exportExcel(prepared, rowCount, filename, "xlsx");
    case "xls":
      return exportExcel(prepared, rowCount, filename, "xlml");
    case "feather":
      return exportFeather(prepared, filename);
    case "parquet":
      return exportParquet(prepared, filename);
  }
}
