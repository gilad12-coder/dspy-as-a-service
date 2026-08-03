"use client";

import * as React from "react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import {
  ArrowClockwise,
  ArrowCounterClockwise,
  ArrowDown,
  ArrowLeft,
  ArrowUp,
  CaretLeft,
  CaretRight,
  Check,
  CircleNotch,
  Copy,
  Plus,
  Tag,
  Trash,
  WarningCircle,
  X,
  XCircle,
} from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { Badge } from "@/shared/ui/primitives/badge";
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from "@/shared/ui/primitives/context-menu";
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
const SAVE_DEBOUNCE_MS = 800;
const SAVE_RETRY_MS = 10_000;
const HISTORY_LIMIT = 100;

type Row = Record<string, unknown>;
type ColumnRole = "input" | "output" | "ignore";
type SaveState = "saved" | "pending" | "saving" | "error";
type CellRef = { row: number; col: string };

/** Flatten any stored cell value into editable text. */
function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

type Snap = {
  columns: string[];
  rows: Row[];
  roles: Record<string, ColumnRole>;
  kinds: Record<string, "text" | "image">;
};

type EditorState = { mode: "loading" } | { mode: "notfound" } | ({ mode: "ready" } & Snap);

type HistEntry = { snap: Snap; anchor?: CellRef };

/**
 * Spreadsheet-style editor for one library dataset: edit cells, add/delete/
 * duplicate rows, add/delete/rename columns — or hand the rows to the tagger.
 * Editing follows the Google-Sheets grammar: every change auto-saves (debounced
 * whole-dataset PUT, since edits replace the row set in place the dataset keeps
 * its identity, shares and links), Cmd/Ctrl+Z undoes and Cmd/Ctrl+Shift+Z
 * redoes committed steps, Enter/arrows move between cells, Escape restores a
 * cell's pre-edit value, and right-clicking a row opens insert/duplicate/
 * delete. Datasets can be large, so only one page of rows renders at a time.
 */
export function DatasetEditorView() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { data: session, status } = useSession();
  const [state, setState] = React.useState<EditorState>({ mode: "loading" });
  const [page, setPage] = React.useState(0);
  const [saveState, setSaveState] = React.useState<SaveState>("saved");
  const [touched, setTouched] = React.useState(false);
  const [pendingFocus, setPendingFocus] = React.useState<CellRef | null>(null);
  // Column renames commit on blur/Enter — remapping every row's keys per
  // keystroke would churn huge datasets for nothing.
  const [headerDrafts, setHeaderDrafts] = React.useState<Record<number, string>>({});
  const [newColumn, setNewColumn] = React.useState<string | null>(null);
  const stateRef = React.useRef(state);
  stateRef.current = state;
  // dirtyRef tracks edits not yet handed to a PUT; savingRef serializes PUTs
  // so overlapping saves can't land out of order.
  const dirtyRef = React.useRef(false);
  const savingRef = React.useRef(false);
  const debounceTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  // Undo history: snapshots share row-object references with live state, so an
  // entry costs one pointer array, not a deep copy. burstKey coalesces the
  // keystroke-level onChange stream for one focused cell into a single entry.
  const history = React.useRef<{ past: HistEntry[]; future: HistEntry[]; burstKey: string | null }>(
    { past: [], future: [], burstKey: null },
  );
  const undoRef = React.useRef<() => void>(() => {});
  const redoRef = React.useRef<() => void>(() => {});
  const name =
    typeof window !== "undefined"
      ? new URLSearchParams(window.location.search).get("name")
      : null;
  const isMac =
    typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

  const runSave = React.useCallback(async () => {
    const current = stateRef.current;
    // The backend rejects an empty row set, so saving waits (with an inline
    // hint) until at least one row exists again. The dirty guard keeps a
    // stale retry timer from re-sending an already-persisted dataset.
    if (current.mode !== "ready" || savingRef.current || !dirtyRef.current) return;
    if (current.rows.length === 0) return;
    savingRef.current = true;
    dirtyRef.current = false;
    setSaveState("saving");
    const schema: DatasetColumnSchema = {
      column_order: current.columns,
      column_roles: current.roles,
      column_kinds: current.kinds,
    };
    try {
      await editDatasetRows(id, current.rows, schema);
      savingRef.current = false;
      if (dirtyRef.current) {
        void runSave();
      } else {
        if (retryTimer.current) clearTimeout(retryTimer.current);
        setSaveState("saved");
      }
    } catch {
      savingRef.current = false;
      dirtyRef.current = true;
      setSaveState("error");
      if (retryTimer.current) clearTimeout(retryTimer.current);
      retryTimer.current = setTimeout(() => void runSave(), SAVE_RETRY_MS);
    }
  }, [id]);

  const markDirty = React.useCallback(() => {
    dirtyRef.current = true;
    setTouched(true);
    setSaveState((prev) => (prev === "saving" ? prev : "pending"));
    if (debounceTimer.current) clearTimeout(debounceTimer.current);
    debounceTimer.current = setTimeout(() => void runSave(), SAVE_DEBOUNCE_MS);
  }, [runSave]);

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

  // A closed tab silently discards edits; warn until the last save lands.
  React.useEffect(() => {
    if (saveState === "saved") return;
    const warn = (e: BeforeUnloadEvent) => {
      e.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [saveState]);

  // Document-level undo/redo, like Sheets: Cmd/Ctrl+Z, Cmd/Ctrl+Shift+Z,
  // plus Ctrl+Y. Handlers live in refs because they close over render state.
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(e.metaKey || e.ctrlKey) || e.altKey) return;
      const key = e.key.toLowerCase();
      if (key === "z") {
        e.preventDefault();
        (e.shiftKey ? redoRef : undoRef).current();
      } else if (key === "y") {
        e.preventDefault();
        redoRef.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Cell focus lands after the page/rows re-render commits; rAF gives the DOM
  // one frame to mount the target input.
  React.useEffect(() => {
    if (!pendingFocus) return;
    const frame = requestAnimationFrame(() => {
      const el = document.querySelector<HTMLInputElement>(
        `[data-cell="${CSS.escape(`${pendingFocus.row}:${pendingFocus.col}`)}"]`,
      );
      el?.focus();
      el?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
    return () => cancelAnimationFrame(frame);
  }, [pendingFocus]);

  // beforeunload only covers tab closes — client-side navigation unmounts
  // without it, so flush any edit the debounce hasn't sent yet.
  React.useEffect(() => {
    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
      if (retryTimer.current) clearTimeout(retryTimer.current);
      const current = stateRef.current;
      if (dirtyRef.current && current.mode === "ready" && current.rows.length > 0) {
        void editDatasetRows(id, current.rows, {
          column_order: current.columns,
          column_roles: current.roles,
          column_kinds: current.kinds,
        }).catch(() => {});
      }
    };
  }, [id]);

  if (state.mode === "loading") {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <CircleNotch className="size-8 animate-spin text-primary" />
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

  const restoreSnap = (snap: Snap) => {
    setState((prev) => (prev.mode === "ready" ? { ...prev, ...snap } : prev));
    setHeaderDrafts({});
    markDirty();
  };

  const focusCell = (row: number, col: string) => {
    setPage(Math.floor(row / PAGE_SIZE));
    setPendingFocus({ row, col });
  };

  /**
   * Apply an edit as one undoable step. Consecutive edits carrying the same
   * `burstKey` (keystrokes in one focused cell) coalesce into the entry opened
   * by the first of them; anything else starts a new entry and, as in Sheets,
   * clears the redo stack.
   */
  const apply = (next: Partial<Snap>, opts: { burstKey?: string; anchor?: CellRef } = {}) => {
    const h = history.current;
    if (!opts.burstKey || h.burstKey !== opts.burstKey) {
      h.past.push({ snap: { columns, rows, roles, kinds }, anchor: opts.anchor });
      if (h.past.length > HISTORY_LIMIT) h.past.shift();
      h.future = [];
    }
    h.burstKey = opts.burstKey ?? null;
    setState((prev) => (prev.mode === "ready" ? { ...prev, ...next } : prev));
    markDirty();
  };

  const undo = () => {
    const h = history.current;
    const entry = h.past.pop();
    if (!entry) return;
    h.future.push({ snap: { columns, rows, roles, kinds }, anchor: entry.anchor });
    h.burstKey = null;
    restoreSnap(entry.snap);
    if (entry.anchor && entry.snap.columns.includes(entry.anchor.col) && entry.snap.rows.length) {
      focusCell(Math.min(entry.anchor.row, entry.snap.rows.length - 1), entry.anchor.col);
    }
  };

  const redo = () => {
    const h = history.current;
    const entry = h.future.pop();
    if (!entry) return;
    h.past.push({ snap: { columns, rows, roles, kinds }, anchor: entry.anchor });
    h.burstKey = null;
    restoreSnap(entry.snap);
    if (entry.anchor && entry.snap.columns.includes(entry.anchor.col) && entry.snap.rows.length) {
      focusCell(Math.min(entry.anchor.row, entry.snap.rows.length - 1), entry.anchor.col);
    }
  };

  undoRef.current = undo;
  redoRef.current = redo;

  const setCell = (rowIdx: number, column: string, value: string) => {
    const next = [...rows];
    next[rowIdx] = { ...next[rowIdx], [column]: value };
    apply(
      { rows: next },
      { burstKey: `cell:${rowIdx}:${column}`, anchor: { row: rowIdx, col: column } },
    );
  };

  /** Escape mid-edit: drop the current burst, restoring the pre-edit value. */
  const abortBurst = (rowIdx: number, column: string) => {
    const h = history.current;
    if (h.burstKey !== `cell:${rowIdx}:${column}`) return;
    const entry = h.past.pop();
    h.burstKey = null;
    if (entry) restoreSnap(entry.snap);
  };

  const handleCellKey = (
    e: React.KeyboardEvent<HTMLInputElement>,
    rowIdx: number,
    colIdx: number,
    column: string,
  ) => {
    if (e.nativeEvent.isComposing) return;
    if (e.key === "Enter" || e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      history.current.burstKey = null;
      const delta = e.key === "ArrowUp" || (e.key === "Enter" && e.shiftKey) ? -1 : 1;
      const next = rowIdx + delta;
      if (next >= 0 && next < rows.length) focusCell(next, column);
    } else if (e.key === "Tab") {
      const nextCol = colIdx + (e.shiftKey ? -1 : 1);
      // Row edges fall through to the native tab order so the row's delete
      // button and surrounding controls stay keyboard-reachable.
      if (nextCol >= 0 && nextCol < columns.length) {
        e.preventDefault();
        history.current.burstKey = null;
        focusCell(rowIdx, columns[nextCol]!);
      }
    } else if (e.key === "Escape") {
      e.preventDefault();
      abortBurst(rowIdx, column);
    }
  };

  const insertRow = (rowIdx: number) => {
    const empty: Row = Object.fromEntries(columns.map((c) => [c, ""]));
    apply(
      { rows: [...rows.slice(0, rowIdx), empty, ...rows.slice(rowIdx)] },
      { anchor: { row: rowIdx, col: columns[0]! } },
    );
    focusCell(rowIdx, columns[0]!);
  };

  const duplicateRow = (rowIdx: number) => {
    apply(
      { rows: [...rows.slice(0, rowIdx + 1), { ...rows[rowIdx] }, ...rows.slice(rowIdx + 1)] },
      { anchor: { row: rowIdx + 1, col: columns[0]! } },
    );
    focusCell(rowIdx + 1, columns[0]!);
  };

  const addRow = () => {
    insertRow(rows.length);
  };

  const deleteRow = (rowIdx: number) => {
    apply(
      { rows: rows.filter((_, i) => i !== rowIdx) },
      { anchor: { row: rowIdx, col: columns[0]! } },
    );
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
    apply({
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
    apply({
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
    apply({
      columns: columns.filter((_, i) => i !== colIdx),
      rows: rows.map((row) => strip(row)),
      roles: strip(roles) as Record<string, ColumnRole>,
      kinds: strip(kinds) as Record<string, "text" | "image">,
    });
  };

  const retrySave = () => {
    if (retryTimer.current) clearTimeout(retryTimer.current);
    void runSave();
  };

  const blocked = saveState !== "saved" && rows.length === 0;

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
        {touched && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground" aria-live="polite">
            {blocked ? (
              <>
                <WarningCircle className="size-3.5" />
                {msg("datasets.editor.autosave_empty")}
              </>
            ) : saveState === "error" ? (
              <button
                type="button"
                onClick={retrySave}
                className="flex cursor-pointer items-center gap-1.5 text-destructive hover:underline"
              >
                <WarningCircle className="size-3.5" />
                {msg("datasets.editor.autosave_error")}
              </button>
            ) : saveState === "saved" ? (
              <>
                <Check className="size-3.5" />
                {msg("datasets.editor.autosave_saved")}
              </>
            ) : (
              <>
                <CircleNotch className="size-3.5 animate-spin" />
                {msg("datasets.editor.autosave_saving")}
              </>
            )}
          </div>
        )}
        <div className="flex items-center gap-0.5">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={undo}
                disabled={history.current.past.length === 0}
                aria-label={msg("datasets.editor.undo")}
              >
                <ArrowCounterClockwise className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {msg("datasets.editor.undo")} · {isMac ? "⌘Z" : "Ctrl+Z"}
            </TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon-sm"
                onClick={redo}
                disabled={history.current.future.length === 0}
                aria-label={msg("datasets.editor.redo")}
              >
                <ArrowClockwise className="size-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {msg("datasets.editor.redo")} · {isMac ? "⌘⇧Z" : "Ctrl+Y"}
            </TooltipContent>
          </Tooltip>
        </div>
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
                disabled={saveState !== "saved"}
                className="gap-2"
              >
                <Tag className="size-4" />
                {msg("datasets.editor.tag")}
              </Button>
            </span>
          </TooltipTrigger>
          {saveState !== "saved" && (
            <TooltipContent>{msg("datasets.editor.tag_dirty_hint")}</TooltipContent>
          )}
        </Tooltip>
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
                <ContextMenu key={rowIdx}>
                  <ContextMenuTrigger asChild>
                    <tr className="group border-b border-border/30 last:border-b-0">
                      {/* The row number doubles as the delete affordance: hovering
                          the row swaps it for the trash button, keeping the action
                          at the start of the row instead of a trailing column that
                          drifts off-screen on wide datasets. */}
                      <td className="relative px-2 py-1 text-xs text-muted-foreground tabular-nums">
                        <span className="transition-opacity group-hover:opacity-0">
                          {rowIdx + 1}
                        </span>
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          onClick={() => deleteRow(rowIdx)}
                          aria-label={msg("datasets.editor.delete_row")}
                          className="absolute inset-y-0 start-1 my-auto text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100 focus-visible:bg-accent focus-visible:opacity-100 hover:text-destructive"
                        >
                          <Trash className="size-3.5" />
                        </Button>
                      </td>
                      {columns.map((column, colIdx) => (
                        <td key={column} className="px-1 py-0.5 align-top">
                          <input
                            value={cellText(row[column])}
                            data-cell={`${rowIdx}:${column}`}
                            onChange={(e) => setCell(rowIdx, column, e.target.value)}
                            onKeyDown={(e) => handleCellKey(e, rowIdx, colIdx, column)}
                            onBlur={() => {
                              // Leaving the cell commits the burst; re-entering
                              // later starts a fresh undo step, as in Sheets.
                              if (history.current.burstKey === `cell:${rowIdx}:${column}`) {
                                history.current.burstKey = null;
                              }
                            }}
                            dir="auto"
                            className={cn(
                              "w-full min-w-0 rounded-md border border-transparent bg-transparent px-2 py-1",
                              "text-sm text-foreground outline-none",
                              "hover:border-input/60 focus-visible:border-ring focus-visible:bg-background",
                            )}
                          />
                        </td>
                      ))}
                    </tr>
                  </ContextMenuTrigger>
                  <ContextMenuContent className="min-w-44 py-1">
                    <ContextMenuItem onSelect={() => insertRow(rowIdx)}>
                      <ArrowUp className="size-3.5 text-muted-foreground" />
                      {msg("datasets.editor.insert_row_above")}
                    </ContextMenuItem>
                    <ContextMenuItem onSelect={() => insertRow(rowIdx + 1)}>
                      <ArrowDown className="size-3.5 text-muted-foreground" />
                      {msg("datasets.editor.insert_row_below")}
                    </ContextMenuItem>
                    <ContextMenuItem onSelect={() => duplicateRow(rowIdx)}>
                      <Copy className="size-3.5 text-muted-foreground" />
                      {msg("datasets.editor.duplicate_row")}
                    </ContextMenuItem>
                    <ContextMenuSeparator />
                    <ContextMenuItem
                      onSelect={() => deleteRow(rowIdx)}
                      className="text-destructive data-[highlighted]:bg-destructive/10"
                    >
                      <Trash className="size-3.5" />
                      {msg("datasets.editor.delete_row")}
                    </ContextMenuItem>
                  </ContextMenuContent>
                </ContextMenu>
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
          <tfoot>
            <tr className="border-t border-border/40">
              <td colSpan={columns.length + 2} className="p-1">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={addRow}
                  className="w-full justify-start gap-1.5 text-muted-foreground"
                >
                  <Plus className="size-3.5" />
                  {msg("datasets.editor.add_row")}
                </Button>
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      {pageCount > 1 && (
        <div className="flex items-center justify-end gap-2">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setPage(Math.max(0, safePage - 1))}
            disabled={safePage === 0}
            aria-label={msg("datasets.editor.prev_page")}
          >
            <CaretLeft className="size-4 rtl:rotate-180" />
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
            <CaretRight className="size-4 rtl:rotate-180" />
          </Button>
        </div>
      )}
    </div>
  );
}
