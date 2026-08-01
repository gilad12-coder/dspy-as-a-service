"use client";

import * as React from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import {
  ArrowLeft,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Plus,
  Save,
  Tags,
  Trash2,
  X,
  XCircle,
} from "lucide-react";
import { toast } from "react-toastify";
import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { Badge } from "@/shared/ui/primitives/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { ExportTableMenu } from "@/shared/ui/export-table-menu";
import {
  editDatasetRows,
  getDatasetRows,
  setApiAuthToken,
  type DatasetColumnSchema,
} from "@/shared/lib/api";
import { formatMsg, msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";

const PAGE_SIZE = 100;

type Row = Record<string, unknown>;
type ColumnRole = "input" | "output" | "ignore";

/** Flatten any stored cell value into editable text. */
function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

type EditorState =
  | { mode: "loading" }
  | { mode: "notfound" }
  | {
      mode: "ready";
      columns: string[];
      rows: Row[];
      roles: Record<string, ColumnRole>;
      kinds: Record<string, "text" | "image">;
    };

/**
 * Spreadsheet-style editor for one library dataset: edit cells, add/delete
 * rows and columns, rename columns — then save in place (the dataset keeps its
 * identity, shares and links) or hand the rows to the tagger. Datasets can be
 * large, so only one page of rows renders at a time; all edits happen on the
 * in-memory copy and nothing touches the server until Save.
 */
export function DatasetEditorView() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { data: session, status } = useSession();
  const [state, setState] = React.useState<EditorState>({ mode: "loading" });
  const [page, setPage] = React.useState(0);
  const [dirty, setDirty] = React.useState(false);
  const [saving, setSaving] = React.useState(false);
  // Column renames commit on blur/Enter — remapping every row's keys per
  // keystroke would churn huge datasets for nothing.
  const [headerDrafts, setHeaderDrafts] = React.useState<Record<number, string>>({});
  const [newColumn, setNewColumn] = React.useState<string | null>(null);
  const name =
    typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("name")
      : null;

  React.useEffect(() => {
    if (status === "loading") return;
    if (session?.backendAccessToken) setApiAuthToken(session.backendAccessToken);
    let cancelled = false;
    getDatasetRows(id)
      .then((detail) => {
        if (cancelled) return;
        setState({
          mode: "ready",
          columns: detail.columns,
          rows: detail.rows as Row[],
          roles: (detail.column_schema?.column_roles ?? {}) as Record<string, ColumnRole>,
          kinds: (detail.column_schema?.column_kinds ?? {}) as Record<string, "text" | "image">,
        });
      })
      .catch(() => {
        if (!cancelled) setState({ mode: "notfound" });
      });
    return () => {
      cancelled = true;
    };
  }, [id, status, session?.backendAccessToken]);

  // A closed tab silently discards edits; warn while any are unsaved.
  React.useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  if (state.mode === "loading") {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Loader2 className="size-8 animate-spin text-primary" />
      </div>
    );
  }
  if (state.mode === "notfound") {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4">
        <XCircle className="size-12 text-destructive" />
        <p className="text-lg text-muted-foreground">{msg("datasets.editor.notfound")}</p>
        <Button asChild variant="outline">
          <Link href="/datasets">{msg("datasets.editor.back")}</Link>
        </Button>
      </div>
    );
  }

  const { columns, rows, roles, kinds } = state;
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pageRows = rows.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE);
  const from = rows.length === 0 ? 0 : safePage * PAGE_SIZE + 1;
  const to = Math.min(rows.length, (safePage + 1) * PAGE_SIZE);

  const patch = (next: Partial<Extract<EditorState, { mode: "ready" }>>) => {
    setState((prev) => (prev.mode === "ready" ? { ...prev, ...next } : prev));
    setDirty(true);
  };

  const setCell = (rowIdx: number, column: string, value: string) => {
    const next = [...rows];
    next[rowIdx] = { ...next[rowIdx], [column]: value };
    patch({ rows: next });
  };

  const addRow = () => {
    const empty: Row = Object.fromEntries(columns.map((c) => [c, ""]));
    patch({ rows: [...rows, empty] });
    setPage(Math.ceil((rows.length + 1) / PAGE_SIZE) - 1);
  };

  const deleteRow = (rowIdx: number) => {
    patch({ rows: rows.filter((_, i) => i !== rowIdx) });
  };

  const commitRename = (colIdx: number) => {
    const draft = headerDrafts[colIdx]?.trim();
    setHeaderDrafts((prev) => {
      const next = { ...prev };
      delete next[colIdx];
      return next;
    });
    const oldName = columns[colIdx]!;
    if (!draft || draft === oldName || columns.includes(draft)) return;
    const rename = (obj: Record<string, unknown>) => {
      const { [oldName]: value, ...rest } = obj;
      return { ...rest, [draft]: value };
    };
    patch({
      columns: columns.map((c, i) => (i === colIdx ? draft : c)),
      rows: rows.map((row) => rename(row)),
      roles: rename(roles) as Record<string, ColumnRole>,
      kinds: rename(kinds) as Record<string, "text" | "image">,
    });
  };

  const addColumn = () => {
    const draft = newColumn?.trim();
    setNewColumn(null);
    if (!draft || columns.includes(draft)) return;
    patch({
      columns: [...columns, draft],
      rows: rows.map((row) => ({ ...row, [draft]: "" })),
      roles: { ...roles, [draft]: "input" },
    });
  };

  const deleteColumn = (colIdx: number) => {
    if (columns.length <= 1) return;
    const gone = columns[colIdx]!;
    const strip = (obj: Record<string, unknown>) => {
      const { [gone]: _removed, ...rest } = obj;
      return rest;
    };
    patch({
      columns: columns.filter((_, i) => i !== colIdx),
      rows: rows.map((row) => strip(row)),
      roles: strip(roles) as Record<string, ColumnRole>,
      kinds: strip(kinds) as Record<string, "text" | "image">,
    });
  };

  const save = async () => {
    if (saving || rows.length === 0) return;
    setSaving(true);
    try {
      const schema: DatasetColumnSchema = {
        column_order: columns,
        column_roles: roles,
        column_kinds: kinds,
      };
      await editDatasetRows(id, rows, schema);
      setDirty(false);
      toast.success(msg("datasets.editor.saved"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("datasets.editor.save_failed"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 pb-8">
      <div className="flex flex-wrap items-center gap-3">
        <Button asChild variant="ghost" size="icon-sm" aria-label={msg("datasets.editor.back")}>
          <Link href="/datasets">
            <ArrowLeft className="size-4 rtl:rotate-180" />
          </Link>
        </Button>
        <div className="min-w-0 flex-1">
          <h2 className="truncate text-base font-semibold text-foreground" dir="auto">
            {name || msg("datasets.editor.title_fallback")}
          </h2>
          <p className="text-xs text-muted-foreground tabular-nums">
            {formatMsg("datasets.count.rows", { count: rows.length })}
            {" · "}
            {formatMsg("datasets.count.columns", { count: columns.length })}
          </p>
        </div>
        {dirty && (
          <Badge variant="secondary" size="sm">
            {msg("datasets.editor.unsaved")}
          </Badge>
        )}
        <ExportTableMenu
          iconOnly
          disabled={rows.length === 0}
          getData={() => ({
            columns,
            rows: rows.map((row) => Object.fromEntries(columns.map((col) => [col, row[col]]))),
            filename: name || "dataset",
          })}
        />
        <Tooltip>
          <TooltipTrigger asChild>
            <span>
              <Button
                variant="outline"
                onClick={() =>
                  router.push(`/tagger?dataset=${id}&name=${encodeURIComponent(name ?? "")}`)
                }
                disabled={dirty}
                className="gap-2"
              >
                <Tags className="size-4" />
                {msg("datasets.editor.tag")}
              </Button>
            </span>
          </TooltipTrigger>
          {dirty && <TooltipContent>{msg("datasets.editor.tag_dirty_hint")}</TooltipContent>}
        </Tooltip>
        <Button onClick={save} disabled={!dirty || saving || rows.length === 0} className="gap-2">
          {saving ? <Loader2 className="size-4 animate-spin" /> : <Save className="size-4" />}
          {msg("datasets.editor.save")}
        </Button>
      </div>

      <div className="overflow-x-auto rounded-xl border border-border/60 bg-card">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-border/60 bg-muted/40">
              <th className="w-12 px-2 py-2 text-start text-xs font-medium text-muted-foreground">
                #
              </th>
              {columns.map((column, colIdx) => (
                <th key={column} className="min-w-[10rem] px-1 py-1 text-start">
                  <div className="flex items-center gap-1">
                    <input
                      value={headerDrafts[colIdx] ?? column}
                      onChange={(e) =>
                        setHeaderDrafts((prev) => ({ ...prev, [colIdx]: e.target.value }))
                      }
                      onBlur={() => commitRename(colIdx)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                      }}
                      dir="ltr"
                      className={cn(
                        "w-full min-w-0 rounded-md border border-transparent bg-transparent px-2 py-1",
                        "font-mono text-xs font-semibold text-foreground outline-none",
                        "hover:border-input focus-visible:border-ring",
                      )}
                      aria-label={msg("datasets.editor.rename_column")}
                    />
                    {roles[column] === "output" && (
                      <Badge variant="ghost" size="sm" className="shrink-0 opacity-60">
                        {msg("datasets.editor.output_badge")}
                      </Badge>
                    )}
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      onClick={() => deleteColumn(colIdx)}
                      disabled={columns.length <= 1}
                      aria-label={msg("datasets.editor.delete_column")}
                      className="shrink-0 text-muted-foreground hover:text-destructive"
                    >
                      <X className="size-3.5" />
                    </Button>
                  </div>
                </th>
              ))}
              <th className="w-40 px-2 py-1 text-start">
                {newColumn === null ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setNewColumn("")}
                    className="gap-1.5 text-muted-foreground"
                  >
                    <Plus className="size-3.5" />
                    {msg("datasets.editor.add_column")}
                  </Button>
                ) : (
                  <input
                    value={newColumn}
                    onChange={(e) => setNewColumn(e.target.value)}
                    onBlur={addColumn}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") addColumn();
                      if (e.key === "Escape") setNewColumn(null);
                    }}
                    placeholder={msg("datasets.editor.column_placeholder")}
                    dir="ltr"
                    autoFocus
                    className="w-full rounded-md border border-input bg-background px-2 py-1 font-mono text-xs outline-none focus-visible:border-ring"
                  />
                )}
              </th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, i) => {
              const rowIdx = safePage * PAGE_SIZE + i;
              return (
                <tr key={rowIdx} className="group border-b border-border/30 last:border-b-0">
                  <td className="px-2 py-1 text-xs text-muted-foreground tabular-nums">
                    {rowIdx + 1}
                  </td>
                  {columns.map((column) => (
                    <td key={column} className="px-1 py-0.5 align-top">
                      <input
                        value={cellText(row[column])}
                        onChange={(e) => setCell(rowIdx, column, e.target.value)}
                        dir="auto"
                        className={cn(
                          "w-full min-w-0 rounded-md border border-transparent bg-transparent px-2 py-1",
                          "text-sm text-foreground outline-none",
                          "hover:border-input/60 focus-visible:border-ring focus-visible:bg-background",
                        )}
                      />
                    </td>
                  ))}
                  <td className="px-2 py-0.5">
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      onClick={() => deleteRow(rowIdx)}
                      aria-label={msg("datasets.editor.delete_row")}
                      className="text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 hover:text-destructive"
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </td>
                </tr>
              );
            })}
            {rows.length === 0 && (
              <tr>
                <td
                  colSpan={columns.length + 2}
                  className="px-4 py-10 text-center text-sm text-muted-foreground"
                >
                  {msg("datasets.editor.empty")}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between gap-3">
        <Button variant="outline" size="sm" onClick={addRow} className="flex-1 gap-1.5">
          <Plus className="size-3.5" />
          {msg("datasets.editor.add_row")}
        </Button>
        {pageCount > 1 && (
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setPage(Math.max(0, safePage - 1))}
              disabled={safePage === 0}
              aria-label={msg("datasets.editor.prev_page")}
            >
              <ChevronLeft className="size-4 rtl:rotate-180" />
            </Button>
            <span className="text-xs text-muted-foreground tabular-nums">
              {formatMsg("datasets.editor.rows_range", { from, to, total: rows.length })}
            </span>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => setPage(Math.min(pageCount - 1, safePage + 1))}
              disabled={safePage >= pageCount - 1}
              aria-label={msg("datasets.editor.next_page")}
            >
              <ChevronRight className="size-4 rtl:rotate-180" />
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
