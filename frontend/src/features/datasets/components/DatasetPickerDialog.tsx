"use client";

import * as React from "react";
import { Database, Loader2, Search } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Input } from "@/shared/ui/primitives/input";
import { Badge } from "@/shared/ui/primitives/badge";
import type { DatasetSummary } from "@/shared/lib/api";
import { formatMsg, msg } from "@/shared/lib/messages";
import { formatBytes, formatRelativeTime } from "@/shared/lib/formatters";
import { useDatasets } from "../hooks/use-datasets";

/**
 * Submit-wizard consumer picker: a searchable list of the caller's library
 * datasets (owned + shared-in). Selecting one hands its summary back via
 * ``onPick`` — the wizard then loads its rows and saved column mapping by
 * reference. The library is fetched lazily, only while the dialog is open.
 */
export function DatasetPickerDialog({
  open,
  onOpenChange,
  onPick,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onPick: (dataset: DatasetSummary) => void;
}) {
  const { datasets, loading, error } = useDatasets(open);
  const [search, setSearch] = React.useState("");

  const query = search.trim().toLowerCase();
  const filtered = query
    ? datasets.filter((d) => d.name.toLowerCase().includes(query))
    : datasets;

  const handlePick = (dataset: DatasetSummary) => {
    onPick(dataset);
    onOpenChange(false);
  };

  // Fade the scroll edges so a long library reads as scrollable: the top fade
  // appears once scrolled away from the start, the bottom fade while more rows
  // remain below. Recomputed on scroll and whenever the rendered set changes.
  const listRef = React.useRef<HTMLDivElement>(null);
  const [edges, setEdges] = React.useState({ top: false, bottom: false });
  const updateEdges = React.useCallback(() => {
    const el = listRef.current;
    if (!el) return;
    const remaining = el.scrollHeight - el.clientHeight - el.scrollTop;
    setEdges({ top: el.scrollTop > 1, bottom: remaining > 1 });
  }, []);
  React.useEffect(() => {
    updateEdges();
  }, [updateEdges, filtered.length, loading, error]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[min(34rem,94vw)] max-w-[min(34rem,94vw)] sm:max-w-lg">
        <DialogHeader className="text-start">
          <DialogTitle>{msg("submit.dataset.library_picker_title")}</DialogTitle>
          <DialogDescription>{msg("submit.dataset.library_picker_subtitle")}</DialogDescription>
        </DialogHeader>

        <div className="relative">
          <Search className="pointer-events-none absolute end-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder={msg("submit.dataset.library_search")}
            className="pe-9"
            aria-label={msg("submit.dataset.library_search")}
          />
        </div>

        <div className="relative">
          <div
            ref={listRef}
            onScroll={updateEdges}
            className="max-h-[min(24rem,55vh)] space-y-1.5 overflow-y-auto px-0.5 py-1"
          >
            {loading ? (
              <div className="flex items-center justify-center py-10">
                <Loader2 className="size-6 animate-spin text-primary" />
              </div>
            ) : error ? (
              <p className="py-10 text-center text-sm text-muted-foreground">
                {msg("submit.dataset.library_error")}
              </p>
            ) : filtered.length === 0 ? (
              <p className="py-10 text-center text-sm text-muted-foreground">
                {query
                  ? msg("submit.dataset.library_search_empty")
                  : msg("submit.dataset.library_empty")}
              </p>
            ) : (
              filtered.map((dataset) => {
                return (
                  <button
                    key={dataset.id}
                    type="button"
                    onClick={() => handlePick(dataset)}
                    className="group flex w-full cursor-pointer items-center gap-3 rounded-lg border border-[#DDD4C8]/60 bg-gradient-to-b from-white/95 to-[#F8F4EF] px-3 py-2.5 text-start transition-[border-color,box-shadow] duration-200 hover:border-[#C8B9A8]/70 hover:shadow-[0_2px_10px_rgba(28,22,18,0.06)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
                  >
                    <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-[#3D2E22]/8 text-[#3D2E22]">
                      <Database className="size-4.5" />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <p className="truncate text-sm font-semibold text-foreground">{dataset.name}</p>
                        {dataset.role !== "owner" && (
                          <Badge variant="secondary" size="sm">
                            {msg("datasets.shared_badge")}
                          </Badge>
                        )}
                      </div>
                      <p className="mt-0.5 truncate text-xs text-muted-foreground">
                        {formatMsg("datasets.count.rows", { count: dataset.row_count })}
                        {" · "}
                        {formatMsg("datasets.count.columns", { count: dataset.column_count })}
                        {" · "}
                        {formatBytes(dataset.byte_size)}
                        {" · "}
                        {formatRelativeTime(dataset.updated_at)}
                      </p>
                    </div>
                  </button>
                );
              })
            )}
          </div>
          <div
            aria-hidden="true"
            className={`pointer-events-none absolute inset-x-0 top-0 h-6 bg-gradient-to-b from-background to-transparent transition-opacity duration-200 ${edges.top ? "opacity-100" : "opacity-0"}`}
          />
          <div
            aria-hidden="true"
            className={`pointer-events-none absolute inset-x-0 bottom-0 h-6 bg-gradient-to-t from-background to-transparent transition-opacity duration-200 ${edges.bottom ? "opacity-100" : "opacity-0"}`}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}
