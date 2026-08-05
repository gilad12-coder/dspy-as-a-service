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
type GridPos = { row: number; col: number };
type Selection = { anchor: GridPos; head: GridPos };
type Bounds = { top: number; bottom: number; left: number; right: number };

/** Flatten any stored cell value into editable text. */
function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** Quote a cell for TSV the way Sheets/Excel do, so round-trips survive. */
function escapeTsvCell(value: string): string {
  return /[\t\n\r"]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

/**
 * Parse clipboard text into a cell grid: tab-separated columns, newline-
 * separated rows, with Sheets/Excel-style quoted cells for embedded tabs,
 * newlines and quotes.
 */
function parseClipboardGrid(text: string): string[][] {
  const grid: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]!;
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          cell += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cell += ch;
      }
    } else if (ch === '"' && cell === "") {
      inQuotes = true;
    } else if (ch === "\t") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      grid.push(row);
      row = [];
      cell = "";
    } else {
      cell += ch;
    }
  }
  row.push(cell);
  grid.push(row);
  // Sheets terminates its clipboard payload with a newline; drop the
  // phantom empty row it would otherwise paste.
  if (grid.length > 1 && grid[grid.length - 1]!.length === 1 && grid[grid.length - 1]![0] === "") {
    grid.pop();
  }
  return grid;
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
 * redoes committed steps, and right-clicking a row opens insert/duplicate/
 * delete. Cells follow the two-mode grammar: click selects, drag/Shift/Cmd+A
 * grow a range, arrows and Tab navigate, typing or Enter/F2/double-click edit,
 * Escape restores a cell's pre-edit value, Delete clears the range, and
 * Cmd/Ctrl+C/X/V move TSV through the clipboard (Sheets/Excel-compatible).
 * Datasets can be large, so only one page of rows renders at a time.
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
  // Two-mode grammar, like Sheets: a cell (or range) is *selected* for
  // navigation and clipboard work, and only the anchor cell ever *edits*.
  const [sel, setSel] = React.useState<Selection | null>(null);
  const [editing, setEditing] = React.useState<GridPos | null>(null);
  const dragging = React.useRef(false);
  // Excel-style fill handle: grab the square at a selection's bottom-right and
  // drag vertically to flood the source value(s) down (or up) the column(s).
  // filling gates the drag; fillSrcRef holds the block dragged from; fillHead is
  // the row the pointer is currently over (state, so the preview re-renders).
  const filling = React.useRef(false);
  const fillSrcRef = React.useRef<Bounds | null>(null);
  const [fillHead, setFillHead] = React.useState<number | null>(null);
  const commitFillRef = React.useRef<() => void>(() => {});
  const gridRef = React.useRef<HTMLDivElement | null>(null);
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

  // A drag-select (or fill-handle drag) can end anywhere on the page, not just
  // over a cell — so the release, and the fill commit, live on the window.
  React.useEffect(() => {
    const end = () => {
      commitFillRef.current();
      dragging.current = false;
    };
    window.addEventListener("mouseup", end);
    return () => window.removeEventListener("mouseup", end);
  }, []);

  // Scrolling lands after the page/rows re-render commits; rAF gives the DOM
  // one frame to mount the target cell.
  React.useEffect(() => {
    if (!pendingFocus) return;
    const frame = requestAnimationFrame(() => {
      document
        .querySelector<HTMLInputElement>(
          `[data-cell="${CSS.escape(`${pendingFocus.row}:${pendingFocus.col}`)}"]`,
        )
        ?.scrollIntoView({ block: "nearest", inline: "nearest" });
    });
    return () => cancelAnimationFrame(frame);
  }, [pendingFocus]);

  // Entering edit mode focuses the cell's input with the caret at the end —
  // both for Enter/double-click (existing text) and type-to-replace (seed).
  React.useEffect(() => {
    if (!editing) return;
    const current = stateRef.current;
    if (current.mode !== "ready") return;
    const col = current.columns[editing.col];
    if (col === undefined) return;
    const frame = requestAnimationFrame(() => {
      const el = document.querySelector<HTMLInputElement>(
        `[data-cell="${CSS.escape(`${editing.row}:${col}`)}"]`,
      );
      if (el) {
        el.focus();
        el.setSelectionRange(el.value.length, el.value.length);
      }
    });
    return () => cancelAnimationFrame(frame);
  }, [editing]);

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

  const scrollToCell = (row: number, col: string) => {
    setPage(Math.floor(row / PAGE_SIZE));
    setPendingFocus({ row, col });
  };

  // Selection coordinates survive row/column removals by clamping, so a
  // stale range degrades to the nearest live cells instead of crashing.
  const clampPos = (p: GridPos): GridPos => ({
    row: Math.max(0, Math.min(p.row, rows.length - 1)),
    col: Math.max(0, Math.min(p.col, columns.length - 1)),
  });
  const selRange =
    sel && rows.length > 0 && columns.length > 0
      ? { anchor: clampPos(sel.anchor), head: clampPos(sel.head) }
      : null;
  const selBounds: Bounds | null = selRange
    ? {
        top: Math.min(selRange.anchor.row, selRange.head.row),
        bottom: Math.max(selRange.anchor.row, selRange.head.row),
        left: Math.min(selRange.anchor.col, selRange.head.col),
        right: Math.max(selRange.anchor.col, selRange.head.col),
      }
    : null;

  // The rows a live fill-handle drag would flood, previewed while dragging. The
  // fill is vertical only, so it keeps the source's columns and just extends the
  // row span past the source, either downward or upward.
  const fillSrc = fillSrcRef.current;
  const fillPreview: Bounds | null =
    fillHead !== null && fillSrc
      ? fillHead > fillSrc.bottom
        ? { top: fillSrc.bottom + 1, bottom: fillHead, left: fillSrc.left, right: fillSrc.right }
        : fillHead < fillSrc.top
          ? { top: fillHead, bottom: fillSrc.top - 1, left: fillSrc.left, right: fillSrc.right }
          : null
      : null;

  // Keyboard events flow through the grid container while a cell is merely
  // selected (no input focused), so selecting always refocuses it.
  const gridFocus = () => gridRef.current?.focus();

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

  /** Undo/redo re-select the affected cell so the user sees what reverted. */
  const showAnchor = (snap: Snap, anchor: CellRef | undefined) => {
    setEditing(null);
    if (!anchor || !snap.rows.length) return;
    const col = snap.columns.indexOf(anchor.col);
    if (col === -1) return;
    const pos = { row: Math.min(anchor.row, snap.rows.length - 1), col };
    setSel({ anchor: pos, head: pos });
    scrollToCell(pos.row, anchor.col);
  };

  const undo = () => {
    const h = history.current;
    const entry = h.past.pop();
    if (!entry) return;
    h.future.push({ snap: { columns, rows, roles, kinds }, anchor: entry.anchor });
    h.burstKey = null;
    restoreSnap(entry.snap);
    showAnchor(entry.snap, entry.anchor);
  };

  const redo = () => {
    const h = history.current;
    const entry = h.future.pop();
    if (!entry) return;
    h.past.push({ snap: { columns, rows, roles, kinds }, anchor: entry.anchor });
    h.burstKey = null;
    restoreSnap(entry.snap);
    showAnchor(entry.snap, entry.anchor);
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

  const selectOnly = (pos: GridPos) => {
    setEditing(null);
    history.current.burstKey = null;
    setSel({ anchor: pos, head: pos });
    scrollToCell(pos.row, columns[pos.col]!);
  };

  const startEdit = (pos: GridPos, seed?: string) => {
    setSel({ anchor: pos, head: pos });
    if (seed !== undefined) setCell(pos.row, columns[pos.col]!, seed);
    setEditing(pos);
  };

  const commitEdit = () => {
    setEditing(null);
    history.current.burstKey = null;
    gridFocus();
  };

  /**
   * Move the selection by a delta — the plain-arrow move collapses the range
   * to the new anchor, Shift extends the head while the anchor stays put.
   */
  const moveSel = (dRow: number, dCol: number, extend: boolean) => {
    if (!selRange) return;
    const base = extend ? selRange.head : selRange.anchor;
    const pos = clampPos({ row: base.row + dRow, col: base.col + dCol });
    setSel(extend ? { anchor: selRange.anchor, head: pos } : { anchor: pos, head: pos });
    scrollToCell(pos.row, columns[pos.col]!);
  };

  /** Tab order wraps at row edges the way Sheets does. */
  const tabSel = (backwards: boolean) => {
    if (!selRange) return;
    const total = rows.length * columns.length;
    const flat = selRange.anchor.row * columns.length + selRange.anchor.col;
    const next = Math.max(0, Math.min(total - 1, flat + (backwards ? -1 : 1)));
    selectOnly({ row: Math.floor(next / columns.length), col: next % columns.length });
  };

  const copyRange = () => {
    if (!selBounds) return;
    const tsv = rows
      .slice(selBounds.top, selBounds.bottom + 1)
      .map((row) =>
        columns
          .slice(selBounds.left, selBounds.right + 1)
          .map((col) => escapeTsvCell(cellText(row[col])))
          .join("\t"),
      )
      .join("\n");
    void navigator.clipboard.writeText(tsv).catch(() => {});
  };

  const clearRange = () => {
    if (!selBounds) return;
    const next = rows.map((row, r) => {
      if (r < selBounds.top || r > selBounds.bottom) return row;
      const cleared = { ...row };
      for (const col of columns.slice(selBounds.left, selBounds.right + 1)) cleared[col] = "";
      return cleared;
    });
    apply({ rows: next }, { anchor: { row: selBounds.top, col: columns[selBounds.left]! } });
  };

  const pasteGrid = (grid: string[][]) => {
    if (!selBounds || grid.length === 0) return;
    const base = { row: selBounds.top, col: selBounds.left };
    // A single copied cell fills the whole selected range, as in Sheets.
    const fillOne =
      grid.length === 1 &&
      grid[0]!.length === 1 &&
      (selBounds.bottom > selBounds.top || selBounds.right > selBounds.left);
    const height = fillOne ? selBounds.bottom - selBounds.top + 1 : grid.length;
    const next = [...rows];
    // Pasting past the last row grows the table; extra columns are clipped.
    while (next.length < base.row + height) {
      next.push(Object.fromEntries(columns.map((c) => [c, ""])));
    }
    let right = base.col;
    for (let r = 0; r < height; r++) {
      const source = fillOne ? grid[0]! : (grid[r] ?? []);
      const target = { ...next[base.row + r] };
      const width = fillOne ? selBounds.right - selBounds.left + 1 : source.length;
      for (let c = 0; c < width && base.col + c < columns.length; c++) {
        target[columns[base.col + c]!] = fillOne ? source[0]! : (source[c] ?? "");
        right = Math.max(right, base.col + c);
      }
      next[base.row + r] = target;
    }
    apply({ rows: next }, { anchor: { row: base.row, col: columns[base.col]! } });
    setSel({ anchor: base, head: { row: base.row + height - 1, col: right } });
    scrollToCell(base.row, columns[base.col]!);
  };

  const handleGridPaste = (e: React.ClipboardEvent) => {
    // Only when the container itself holds focus (selection mode): pastes
    // into cell inputs, header drafts etc. keep their native behavior.
    if (editing || e.target !== e.currentTarget) return;
    if (!selRange) return;
    const text = e.clipboardData.getData("text/plain");
    if (!text) return;
    e.preventDefault();
    pasteGrid(parseClipboardGrid(text));
  };

  /** Selection-mode keys, on the grid container: navigation and clipboard. */
  const handleGridKey = (e: React.KeyboardEvent) => {
    // Only when the container itself holds focus: keys inside cell inputs,
    // header drafts and row buttons keep their native behavior.
    if (editing || e.target !== e.currentTarget) return;
    if (!selRange) return;
    const rtl = gridRef.current ? getComputedStyle(gridRef.current).direction === "rtl" : false;
    const visual = (dCol: number) => (rtl ? -dCol : dCol);
    if (e.metaKey || e.ctrlKey) {
      const key = e.key.toLowerCase();
      if (key === "c" || key === "x") {
        e.preventDefault();
        copyRange();
        if (key === "x") clearRange();
      } else if (key === "a") {
        e.preventDefault();
        setSel({
          anchor: { row: 0, col: 0 },
          head: { row: rows.length - 1, col: columns.length - 1 },
        });
      }
      return;
    }
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      moveSel(e.key === "ArrowDown" ? 1 : -1, 0, e.shiftKey);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      moveSel(0, visual(e.key === "ArrowRight" ? 1 : -1), e.shiftKey);
    } else if (e.key === "Tab") {
      e.preventDefault();
      tabSel(e.shiftKey);
    } else if (e.key === "Enter" || e.key === "F2") {
      e.preventDefault();
      startEdit(selRange.anchor);
    } else if (e.key === "Escape") {
      setSel(null);
    } else if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      clearRange();
    } else if (e.key.length === 1 && !e.nativeEvent.isComposing) {
      // Typing replaces the cell's content and opens the editor, like Sheets.
      e.preventDefault();
      startEdit(selRange.anchor, e.key);
    }
  };

  const handleCellMouseDown = (e: React.MouseEvent, pos: GridPos) => {
    if (e.button !== 0) return;
    // Clicks inside the cell being edited keep their native caret behavior.
    if (editing && editing.row === pos.row && editing.col === pos.col) return;
    // Native focus would land in the readOnly input; selection owns the cell.
    e.preventDefault();
    if (editing) commitEdit();
    gridFocus();
    if (e.shiftKey && selRange) {
      setSel({ anchor: selRange.anchor, head: pos });
    } else {
      setSel({ anchor: pos, head: pos });
      dragging.current = true;
    }
  };

  const handleCellMouseEnter = (pos: GridPos) => {
    // A fill drag tracks the row under the pointer; a range drag tracks the cell.
    if (filling.current) {
      setFillHead(pos.row);
      return;
    }
    if (!dragging.current) return;
    setSel((prev) => (prev ? { anchor: prev.anchor, head: pos } : prev));
  };

  const startFill = (e: React.MouseEvent) => {
    if (e.button !== 0 || !selBounds) return;
    // Own the gesture outright: no text-select, and no range-drag from the cell
    // underneath (its mousedown would otherwise reset the selection).
    e.preventDefault();
    e.stopPropagation();
    if (editing) commitEdit();
    gridFocus();
    fillSrcRef.current = selBounds;
    filling.current = true;
    setFillHead(selBounds.bottom);
  };

  /**
   * Commit the fill-handle drag: flood the source block's value(s) into the rows
   * the handle was dragged over, as one undoable step. Vertical only — the
   * columns stay the source's; a single source cell simply repeats, and a taller
   * source tiles so it stays contiguous with itself in either direction.
   */
  const endFill = () => {
    if (!filling.current) return;
    filling.current = false;
    const src = fillSrcRef.current;
    const target = fillHead;
    fillSrcRef.current = null;
    setFillHead(null);
    if (!src || target === null || (target >= src.top && target <= src.bottom)) return;
    const height = src.bottom - src.top + 1;
    const next = [...rows];
    const fillRow = (r: number) => {
      const offset = (((r - src.top) % height) + height) % height;
      const source = rows[src.top + offset]!;
      const cell = { ...next[r] };
      for (let c = src.left; c <= src.right; c++) cell[columns[c]!] = cellText(source[columns[c]!]);
      next[r] = cell;
    };
    if (target > src.bottom) {
      for (let r = src.bottom + 1; r <= target && r < rows.length; r++) fillRow(r);
    } else {
      for (let r = target; r < src.top; r++) fillRow(r);
    }
    apply({ rows: next }, { anchor: { row: src.top, col: columns[src.left]! } });
    setSel({
      anchor: { row: Math.min(src.top, target), col: src.left },
      head: { row: Math.max(src.bottom, target), col: src.right },
    });
  };
  commitFillRef.current = endFill;

  /** Keys inside the actively edited cell input. */
  const handleCellKey = (e: React.KeyboardEvent<HTMLInputElement>, rowIdx: number, column: string) => {
    if (e.nativeEvent.isComposing) return;
    if (e.key === "Enter" || e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      commitEdit();
      const delta = e.key === "ArrowUp" || (e.key === "Enter" && e.shiftKey) ? -1 : 1;
      moveSel(delta, 0, false);
    } else if (e.key === "Tab") {
      e.preventDefault();
      commitEdit();
      tabSel(e.shiftKey);
    } else if (e.key === "Escape") {
      e.preventDefault();
      abortBurst(rowIdx, column);
      setEditing(null);
      gridFocus();
    }
  };

  const insertRow = (rowIdx: number) => {
    const empty: Row = Object.fromEntries(columns.map((c) => [c, ""]));
    apply(
      { rows: [...rows.slice(0, rowIdx), empty, ...rows.slice(rowIdx)] },
      { anchor: { row: rowIdx, col: columns[0]! } },
    );
    selectOnly({ row: rowIdx, col: 0 });
  };

  const duplicateRow = (rowIdx: number) => {
    apply(
      { rows: [...rows.slice(0, rowIdx + 1), { ...rows[rowIdx] }, ...rows.slice(rowIdx + 1)] },
      { anchor: { row: rowIdx + 1, col: columns[0]! } },
    );
    selectOnly({ row: rowIdx + 1, col: 0 });
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
        {/* Autosave stays silent on success — only the states that need the
            user's attention (empty, failed, in-flight) surface a chip. */}
        {touched && (blocked || saveState !== "saved") && (
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
                variant="ghost"
                size="icon-sm"
                onClick={() =>
                  router.push(`/tagger?dataset=${id}&name=${encodeURIComponent(name ?? "")}`)
                }
                disabled={saveState !== "saved"}
                aria-label={msg("datasets.editor.tag")}
              >
                <Tag className="size-4" />
              </Button>
            </span>
          </TooltipTrigger>
          {/* The label lives in the tooltip now that the button is icon-only;
              while it's disabled the dirty hint explains why instead. */}
          <TooltipContent>
            {saveState !== "saved"
              ? msg("datasets.editor.tag_dirty_hint")
              : msg("datasets.editor.tag")}
          </TooltipContent>
        </Tooltip>
      </div>

      {/* The container is the keyboard surface for selection mode: it holds
          DOM focus while cells are selected-but-not-editing, so arrows, Tab,
          typing and clipboard events all land here. */}
      <div
        ref={gridRef}
        tabIndex={0}
        onKeyDown={handleGridKey}
        onPaste={handleGridPaste}
        onFocus={(e) => {
          if (e.target !== e.currentTarget) return;
          if (!sel && rows.length > 0 && columns.length > 0) {
            setSel({
              anchor: { row: safePage * PAGE_SIZE, col: 0 },
              head: { row: safePage * PAGE_SIZE, col: 0 },
            });
          }
        }}
        className="overflow-x-auto rounded-xl border border-border/60 bg-card outline-none focus-visible:border-ring/60"
      >
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
                      {columns.map((column, colIdx) => {
                        const isEditing =
                          editing !== null && editing.row === rowIdx && editing.col === colIdx;
                        const inRange =
                          !isEditing &&
                          selBounds !== null &&
                          rowIdx >= selBounds.top &&
                          rowIdx <= selBounds.bottom &&
                          colIdx >= selBounds.left &&
                          colIdx <= selBounds.right;
                        const isAnchor =
                          selRange !== null &&
                          selRange.anchor.row === rowIdx &&
                          selRange.anchor.col === colIdx;
                        const inFill =
                          fillPreview !== null &&
                          rowIdx >= fillPreview.top &&
                          rowIdx <= fillPreview.bottom &&
                          colIdx >= fillPreview.left &&
                          colIdx <= fillPreview.right;
                        // The handle rides the selection's bottom-right cell.
                        const showHandle =
                          editing === null &&
                          selBounds !== null &&
                          rowIdx === selBounds.bottom &&
                          colIdx === selBounds.right;
                        return (
                          <td
                            key={column}
                            className="relative px-1 py-0.5 align-top"
                            onMouseDown={(e) =>
                              handleCellMouseDown(e, { row: rowIdx, col: colIdx })
                            }
                            onMouseEnter={() =>
                              handleCellMouseEnter({ row: rowIdx, col: colIdx })
                            }
                            onDoubleClick={() => startEdit({ row: rowIdx, col: colIdx })}
                          >
                            <input
                              value={cellText(row[column])}
                              data-cell={`${rowIdx}:${column}`}
                              readOnly={!isEditing}
                              tabIndex={-1}
                              onChange={(e) => setCell(rowIdx, column, e.target.value)}
                              onKeyDown={(e) => handleCellKey(e, rowIdx, column)}
                              onBlur={() => {
                                // Click-away commits the edit; the burst reset
                                // makes a later re-entry a fresh undo step.
                                if (isEditing) {
                                  setEditing(null);
                                  history.current.burstKey = null;
                                }
                              }}
                              dir="auto"
                              className={cn(
                                "w-full min-w-0 rounded-md border border-transparent bg-transparent px-2 py-1",
                                "text-sm text-foreground outline-none",
                                isEditing
                                  ? "border-ring bg-background"
                                  : "cursor-default hover:border-input/60",
                                inRange && "bg-primary/10",
                                inFill && "border-dashed border-primary/50 bg-primary/5",
                                isAnchor && !isEditing && "border-ring",
                              )}
                            />
                            {showHandle && (
                              <button
                                type="button"
                                tabIndex={-1}
                                aria-label={msg("datasets.editor.fill_handle")}
                                onMouseDown={startFill}
                                className="absolute bottom-0.5 end-1 z-20 size-2 cursor-ns-resize rounded-[2px] border border-background bg-primary"
                              />
                            )}
                          </td>
                        );
                      })}
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
