"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ArrowUp, ArrowDown, ArrowUpDown, Filter, Search, RotateCcw } from "lucide-react";
import { Button } from "@/shared/ui/primitives/button";
import { Input } from "@/shared/ui/primitives/input";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";

export type SortDir = "asc" | "desc";
export type Filters = Record<string, Set<string>>;

export function ColumnHeader<K extends string>({
  label,
  sortKey,
  currentSort,
  sortDir,
  onSort,
  filterCol,
  filterOptions,
  filters,
  onFilter,
  openFilter,
  setOpenFilter,
  width,
  onResize,
}: {
  label: string;
  sortKey: K;
  currentSort: K;
  sortDir: SortDir;
  onSort: (key: K) => void;
  filterCol?: string;
  filterOptions?: Array<{ value: string; label: string }>;
  filters?: Filters;
  onFilter?: (col: string, values: Set<string>) => void;
  openFilter?: string | null;
  setOpenFilter?: (col: string | null) => void;
  width?: number;
  onResize?: (key: K, width: number) => void;
}) {
  const sortActive = currentSort === sortKey;
  const hasFilter = filterCol && filterOptions && filters && onFilter && setOpenFilter;
  const filterActive = hasFilter && filters[filterCol] && filters[filterCol].size > 0;
  const isOpen = hasFilter && openFilter === filterCol;
  const filterIconRef = useRef<HTMLButtonElement>(null);
  const thRef = useRef<HTMLTableCellElement>(null);
  const resizeAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      resizeAbortRef.current?.abort();
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
  }, []);

  const handleResizeStart = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const th = thRef.current;
      if (!th) return;
      resizeAbortRef.current?.abort();
      const controller = new AbortController();
      resizeAbortRef.current = controller;
      const startX = e.clientX;
      const startW = th.offsetWidth;
      const onMove = (ev: MouseEvent) => {
        if (!thRef.current || controller.signal.aborted) return;
        const isRtl = getComputedStyle(thRef.current).direction === "rtl";
        const delta = isRtl ? startX - ev.clientX : ev.clientX - startX;
        const newW = Math.max(60, startW + delta);
        onResize?.(sortKey, newW);
      };
      const onUp = () => {
        controller.abort();
        resizeAbortRef.current = null;
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
      };
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      document.addEventListener("mousemove", onMove, { signal: controller.signal });
      document.addEventListener("mouseup", onUp, { signal: controller.signal });
    },
    [sortKey, onResize],
  );

  const handleResizeDblClick = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      // Auto-fit: measure the widest natural content in this column (Excel-style)
      // Temporarily remove constraints to get true content width, then restore.
      const th = thRef.current;
      if (!th) return;
      const table = th.closest("table");
      if (!table) return;
      const colIdx = Array.from(th.parentElement?.children ?? []).indexOf(th);
      const PADDING = 14; // ~7px each side, matching Excel
      const MIN_W = 60;
      const MAX_W = 600;

      const cells: HTMLElement[] = [th];
      table.querySelectorAll("tbody tr").forEach((row) => {
        const cell = row.children[colIdx] as HTMLElement | undefined;
        if (cell) cells.push(cell);
      });

      // Temporarily remove overflow/truncation so scrollWidth reflects true content
      const saved = cells.map((cell) => ({
        el: cell,
        maxW: cell.style.maxWidth,
        w: cell.style.width,
        overflow: cell.style.overflow,
      }));
      for (const cell of cells) {
        cell.style.maxWidth = "none";
        cell.style.width = "auto";
        cell.style.overflow = "visible";
      }
      let widest = 0;
      for (const cell of cells) {
        widest = Math.max(widest, cell.scrollWidth);
      }
      for (const s of saved) {
        s.el.style.maxWidth = s.maxW;
        s.el.style.width = s.w;
        s.el.style.overflow = s.overflow;
      }

      const fitW = Math.max(MIN_W, Math.min(widest + PADDING, MAX_W));
      onResize?.(sortKey, fitW);
    },
    [sortKey, onResize],
  );

  return (
    <th
      ref={thRef}
      className={`relative select-none ps-2 pe-4 py-3 text-center text-[0.75rem] font-semibold ${sortActive ? "text-foreground" : "text-muted-foreground"}`}
      style={width ? { width, minWidth: width, maxWidth: width } : undefined}
    >
      <div className="flex items-center justify-center gap-0.5 overflow-hidden">
        <button
          type="button"
          onClick={() => onSort(sortKey)}
          className="inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 hover:bg-accent/60 hover:text-foreground cursor-pointer"
          aria-label={formatMsg("shared.excel_filter.sort_by", { label })}
        >
          <span>{label}</span>
          {sortActive ? (
            sortDir === "asc" ? (
              <ArrowUp className="size-3" />
            ) : (
              <ArrowDown className="size-3" />
            )
          ) : (
            <ArrowUpDown className="size-3" />
          )}
        </button>
        {hasFilter ? (
          <button
            type="button"
            ref={filterIconRef}
            className={`rounded-full p-1 hover:bg-accent/60 hover:text-foreground cursor-pointer ${filterActive ? "text-primary" : "text-muted-foreground"}`}
            onClick={(e) => {
              e.stopPropagation();
              setOpenFilter(isOpen ? null : filterCol);
            }}
            aria-label={formatMsg("shared.excel_filter.filter_column", { label })}
          >
            <Filter className="size-3" />
          </button>
        ) : null}
      </div>
      {isOpen ? (
        <FilterDropdown
          options={filterOptions}
          selected={filters[filterCol] || new Set()}
          onApply={(values) => {
            onFilter(filterCol, values);
            setOpenFilter(null);
          }}
          onClose={() => setOpenFilter(null)}
          ignoreRef={filterIconRef}
          anchorRef={filterIconRef}
        />
      ) : null}
      {onResize && (
        <div
          className="absolute top-0 bottom-0 w-[5px] cursor-col-resize end-0 z-10 after:absolute after:inset-y-2 after:start-[2px] after:w-px after:bg-border/40 hover:after:bg-primary/50 active:after:bg-primary/70 after:transition-colors"
          onMouseDown={handleResizeStart}
          onDoubleClick={handleResizeDblClick}
        />
      )}
    </th>
  );
}

function FilterDropdown({
  options,
  selected,
  onApply,
  onClose,
  ignoreRef,
  anchorRef,
}: {
  options: Array<{ value: string; label: string }>;
  selected: Set<string>;
  onApply: (values: Set<string>) => void;
  onClose: () => void;
  ignoreRef?: React.RefObject<HTMLElement | null>;
  anchorRef?: React.RefObject<HTMLElement | null>;
}) {
  // selected.size === 0 means "no filter" (all shown). We convert to explicit set for local editing.
  const [localSelected, setLocalSelected] = useState<Set<string>>(() =>
    selected.size === 0 ? new Set(options.map((o) => o.value)) : new Set(selected),
  );
  const [search, setSearch] = useState("");
  const [pos, setPos] = useState<{ top: number; left: number; availH: number } | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const updatePos = useCallback(() => {
    if (!anchorRef?.current) return;
    const rect = anchorRef.current.getBoundingClientRect();
    const dropdownW = 260;
    const gap = 4;
    const top = rect.bottom + gap;
    const availH = Math.max(150, window.innerHeight - top - gap);
    // Center dropdown on the icon, then clamp within viewport
    const idealLeft = rect.left + rect.width / 2 - dropdownW / 2;
    const left = Math.max(4, Math.min(idealLeft, window.innerWidth - dropdownW - 4));
    setPos({ top, left, availH });
  }, [anchorRef]);

  useLayoutEffect(() => {
    updatePos();
  }, [updatePos]);

  useEffect(() => {
    const scrollParents: HTMLElement[] = [];
    let el = anchorRef?.current?.parentElement;
    while (el) {
      if (el.scrollHeight > el.clientHeight || el.scrollWidth > el.clientWidth)
        scrollParents.push(el);
      el = el.parentElement;
    }
    scrollParents.forEach((sp) => sp.addEventListener("scroll", onClose, { passive: true }));
    window.addEventListener("resize", onClose);
    return () => {
      scrollParents.forEach((sp) => sp.removeEventListener("scroll", onClose));
      window.removeEventListener("resize", onClose);
    };
  }, [anchorRef, onClose]);

  useEffect(() => {
    searchRef.current?.focus();
  }, []);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      const target = e.target as Node;
      if (
        ref.current &&
        !ref.current.contains(target) &&
        !(ignoreRef?.current && ignoreRef.current.contains(target))
      ) {
        onClose();
      }
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [onClose, ignoreRef]);

  const allValues = new Set(options.map((o) => o.value));
  const allSelected =
    allValues.size > 0 &&
    localSelected.size === allValues.size &&
    [...allValues].every((v) => localSelected.has(v));

  const visibleOptions = search.trim()
    ? options.filter((o) => o.label.toLowerCase().includes(search.trim().toLowerCase()))
    : options;

  function toggleValue(val: string) {
    setLocalSelected((prev) => {
      const next = new Set(prev);
      if (next.has(val)) next.delete(val);
      else next.add(val);
      return next;
    });
  }

  function toggleSelectAll() {
    if (allSelected) {
      setLocalSelected(new Set());
    } else {
      setLocalSelected(new Set(options.map((o) => o.value)));
    }
  }

  const dropdown = (
    <div
      ref={ref}
      className="fixed z-[9999] max-w-[min(90vw,320px)] w-full rounded-[22px] border border-border/70 bg-popover/95 p-2 shadow-[var(--shadow-card)] backdrop-blur-xl"
      style={
        pos
          ? { top: pos.top, left: pos.left, minWidth: "clamp(180px, 25vw, 260px)" }
          : { visibility: "hidden" as const }
      }
      onClick={(e) => e.stopPropagation()}
    >
      <div className="relative mb-1.5">
        <div className="absolute end-2 top-1/2 -translate-y-1/2 text-muted-foreground">
          <Search className="size-3" />
        </div>
        <Input
          ref={searchRef}
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="h-7 text-[0.6875rem] py-1.5 pe-7 ps-2 text-right"
          placeholder={msg("shared.excel_filter.search_placeholder")}
          dir={getActiveDir()}
        />
      </div>

      <label className="flex cursor-pointer items-center gap-2 rounded-xl px-2 py-1.5 text-[0.75rem] font-semibold text-muted-foreground hover:bg-muted/70">
        <input
          type="checkbox"
          className="accent-[#3D2E22]"
          checked={allSelected}
          onChange={toggleSelectAll}
        />
        {msg("shared.excel_filter.select_all")}
      </label>

      <div className="my-1 border-t border-border/70" />

      <div
        className="overflow-y-auto"
        style={{ maxHeight: Math.min(200, (pos?.availH ?? 200) - 120) }}
      >
        {visibleOptions.length === 0 ? (
          <p className="text-[0.6875rem] text-center py-2 text-muted-foreground">
            {msg("shared.excel_filter.no_results")}
          </p>
        ) : (
          visibleOptions.map((opt) => (
            <label
              key={opt.value}
              className="flex cursor-pointer items-center gap-2 rounded-xl px-2 py-1.5 text-[0.75rem] text-muted-foreground hover:bg-muted/70"
              title={opt.value}
            >
              <input
                type="checkbox"
                className="shrink-0 accent-[#3D2E22]"
                checked={localSelected.has(opt.value)}
                onChange={() => toggleValue(opt.value)}
              />
              <span className="truncate" dir={/[\u0590-\u05FF]/.test(opt.label) ? "rtl" : "ltr"}>
                {opt.label}
              </span>
            </label>
          ))
        )}
      </div>

      <div className="my-1 border-t border-border/70" />

      <div className="flex items-center gap-2 px-1">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="flex-1 text-[0.6875rem] font-semibold"
          onClick={() => {
            onApply(new Set());
            onClose();
          }}
        >
          {msg("shared.excel_filter.cancel")}
        </Button>
        <Button
          type="button"
          variant="default"
          size="sm"
          className="flex-1 text-[0.6875rem] font-semibold"
          onClick={() => {
            // If all options are selected, apply empty set (= no filter)
            const allVals = new Set(options.map((o) => o.value));
            const isAll =
              localSelected.size === allVals.size &&
              [...allVals].every((v) => localSelected.has(v));
            onApply(isAll ? new Set() : localSelected);
          }}
        >
          {msg("shared.excel_filter.apply")}
        </Button>
      </div>
    </div>
  );

  return createPortal(dropdown, document.body);
}

export function useColumnFilters(initial?: Filters) {
  // `initial` seeds the first render only (e.g. the logs view opens pre-filtered
  // to its default verbosity), so callers can avoid an unfiltered first paint.
  const [filters, setFilters] = useState<Filters>(initial ?? {});
  const [openFilter, setOpenFilter] = useState<string | null>(null);

  const setColumnFilter = useCallback((col: string, values: Set<string>) => {
    setFilters((prev) => {
      const next = { ...prev };
      if (values.size === 0) {
        delete next[col];
      } else {
        next[col] = values;
      }
      return next;
    });
  }, []);

  const clearAll = useCallback(() => setFilters({}), []);

  const activeCount = Object.values(filters).filter((s) => s.size > 0).length;

  return { filters, setColumnFilter, openFilter, setOpenFilter, clearAll, activeCount };
}

export function useColumnResize() {
  const [widths, setWidths] = useState<Record<string, number>>({});

  const setColumnWidth = useCallback((col: string, width: number) => {
    setWidths((prev) => ({ ...prev, [col]: width }));
  }, []);

  const resetAll = useCallback(() => setWidths({}), []);

  const hasResized = Object.keys(widths).length > 0;

  return { widths, setColumnWidth, resetAll, hasResized };
}

export function ResetColumnsButton({
  resize,
}: {
  resize: { hasResized: boolean; resetAll: () => void };
}) {
  if (!resize.hasResized) return null;
  return (
    <button
      type="button"
      onClick={resize.resetAll}
      className="inline-flex items-center gap-1 text-[0.625rem] text-muted-foreground/60 hover:text-foreground transition-colors cursor-pointer underline underline-offset-2 decoration-muted-foreground/30 hover:decoration-foreground/40"
      title={msg("shared.excel_filter.reset_width_title")}
    >
      <RotateCcw className="size-3" />
      <span>{msg("shared.excel_filter.reset_width_label")}</span>
    </button>
  );
}
