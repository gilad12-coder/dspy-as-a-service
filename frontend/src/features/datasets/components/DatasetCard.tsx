"use client";

import * as React from "react";
import { CopyPlus, Database, Loader2, Pencil, Trash2 } from "lucide-react";
import { toast } from "react-toastify";
import { Badge } from "@/shared/ui/primitives/badge";
import { Button } from "@/shared/ui/primitives/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Input } from "@/shared/ui/primitives/input";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import {
  cloneDataset,
  deleteDataset,
  isStorageQuotaError,
  listDatasetOptimizations,
  renameDataset,
  type DatasetSummary,
} from "@/shared/lib/api";
import { formatMsg, msg } from "@/shared/lib/messages";
import { formatBytes, formatRelativeTime } from "@/shared/lib/formatters";
import { DatasetShareDialog } from "./DatasetShareDialog";

/**
 * One library dataset rendered as a clickable card: name, row/column
 * counts, size and last-updated, plus role-gated actions. Owners get share /
 * rename / delete; everyone else (shared-in) gets clone-to-my-library. Clicking
 * the card body opens the detail sheet via ``onOpen``.
 */
export function DatasetCard({
  dataset,
  onOpen,
  onChanged,
}: {
  dataset: DatasetSummary;
  onOpen: (dataset: DatasetSummary) => void;
  onChanged: () => void;
}) {
  const isOwner = dataset.role === "owner";
  const [renameOpen, setRenameOpen] = React.useState(false);
  const [renameValue, setRenameValue] = React.useState(dataset.name);
  const [renaming, setRenaming] = React.useState(false);
  const [deleteOpen, setDeleteOpen] = React.useState(false);
  const [deleting, setDeleting] = React.useState(false);
  const [cloning, setCloning] = React.useState(false);
  // How many optimizations were built from this dataset, fetched only when the
  // delete dialog opens. ``null`` while unknown/loading; a positive count warns
  // that those runs' back-link will dangle (the runs themselves keep working —
  // they own a copy of the rows, not a reference). Owner-only, mirroring delete.
  const [usedCount, setUsedCount] = React.useState<number | null>(null);

  React.useEffect(() => {
    if (!deleteOpen || !isOwner) {
      setUsedCount(null);
      return;
    }
    let cancelled = false;
    listDatasetOptimizations(dataset.id)
      .then((res) => !cancelled && setUsedCount(res.optimizations.length))
      .catch(() => !cancelled && setUsedCount(0));
    return () => {
      cancelled = true;
    };
  }, [deleteOpen, isOwner, dataset.id]);

  const handleRename = async () => {
    const name = renameValue.trim();
    if (!name || renaming) return;
    setRenaming(true);
    try {
      await renameDataset(dataset.id, name);
      toast.success(msg("datasets.toast.renamed"));
      setRenameOpen(false);
      onChanged();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("datasets.toast.rename_failed"));
    } finally {
      setRenaming(false);
    }
  };

  const handleDelete = async () => {
    if (deleting) return;
    setDeleting(true);
    try {
      await deleteDataset(dataset.id);
      toast.success(msg("datasets.toast.deleted"));
      setDeleteOpen(false);
      onChanged();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("datasets.toast.delete_failed"));
    } finally {
      setDeleting(false);
    }
  };

  const handleClone = async () => {
    if (cloning) return;
    setCloning(true);
    try {
      const res = await cloneDataset(dataset.id);
      toast.success(res.deduplicated ? msg("datasets.toast.deduplicated") : msg("datasets.toast.cloned"));
      onChanged();
    } catch (err) {
      if (!isStorageQuotaError(err)) {
        toast.error(err instanceof Error ? err.message : msg("datasets.toast.clone_failed"));
      }
    } finally {
      setCloning(false);
    }
  };

  // Buttons inside the clickable card stop propagation so their own handlers
  // fire without also opening the detail sheet.
  const stop = (e: React.MouseEvent) => e.stopPropagation();

  return (
    <>
      <div
        role="button"
        tabIndex={0}
        onClick={() => onOpen(dataset)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            onOpen(dataset);
          }
        }}
        className="group flex cursor-pointer items-center gap-4 rounded-xl border border-[#DDD4C8]/60 bg-gradient-to-b from-white/95 to-[#F8F4EF] px-4 py-3.5 text-start shadow-[0_1px_3px_rgba(28,22,18,0.03)] transition-[border-color,box-shadow] duration-200 hover:border-[#C8B9A8]/70 hover:shadow-[0_2px_10px_rgba(28,22,18,0.06)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
      >
        <span className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-[#3D2E22]/8 text-[#3D2E22]">
          <Database className="size-5" />
        </span>

        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <p className="truncate text-sm font-semibold text-foreground">{dataset.name}</p>
            {!isOwner && (
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

        <div className="flex shrink-0 items-center gap-1" onClick={stop}>
          {isOwner ? (
            <>
              <DatasetShareDialog datasetId={dataset.id} />
              <TooltipButton tooltip={msg("datasets.action.rename")}>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="text-muted-foreground hover:text-foreground"
                  onClick={() => {
                    setRenameValue(dataset.name);
                    setRenameOpen(true);
                  }}
                  aria-label={msg("datasets.action.rename")}
                >
                  <Pencil className="size-4" />
                </Button>
              </TooltipButton>
              <TooltipButton tooltip={msg("datasets.action.delete")}>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  className="text-muted-foreground hover:text-destructive"
                  onClick={() => setDeleteOpen(true)}
                  aria-label={msg("datasets.action.delete")}
                >
                  <Trash2 className="size-4" />
                </Button>
              </TooltipButton>
            </>
          ) : (
            <TooltipButton tooltip={msg("datasets.action.clone")}>
              <Button
                variant="ghost"
                size="icon-sm"
                className="text-muted-foreground hover:text-foreground"
                onClick={handleClone}
                disabled={cloning}
                aria-label={msg("datasets.action.clone")}
              >
                {cloning ? <Loader2 className="size-4 animate-spin" /> : <CopyPlus className="size-4" />}
              </Button>
            </TooltipButton>
          )}
        </div>
      </div>

      <Dialog open={renameOpen} onOpenChange={setRenameOpen}>
        <DialogContent className="w-[min(28rem,92vw)] max-w-[min(28rem,92vw)] sm:max-w-md">
          <DialogHeader className="text-start">
            <DialogTitle>{msg("datasets.rename.title")}</DialogTitle>
          </DialogHeader>
          <Input
            value={renameValue}
            onChange={(e) => setRenameValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                void handleRename();
              }
            }}
            aria-label={msg("datasets.rename.label")}
            autoFocus
          />
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setRenameOpen(false)}
              disabled={renaming}
              className="w-full justify-center"
            >
              {msg("datasets.rename.cancel")}
            </Button>
            <Button
              onClick={handleRename}
              disabled={renaming || renameValue.trim().length === 0}
              className="w-full justify-center shadow-xs"
            >
              {renaming ? <Loader2 className="size-4 animate-spin" /> : msg("datasets.rename.save")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={deleteOpen} onOpenChange={setDeleteOpen}>
        <DialogContent className="w-[min(28rem,92vw)] max-w-[min(28rem,92vw)] sm:max-w-md">
          <DialogHeader className="text-start">
            <DialogTitle>{msg("datasets.delete.title")}</DialogTitle>
          </DialogHeader>
          <p className="text-sm text-muted-foreground">{msg("datasets.delete.body")}</p>
          {usedCount !== null &&
            usedCount > 0 &&
            (() => {
              // Bold the affected-run count to match how every other delete dialog
              // emphasizes its key value; keep the amber colour (no text-foreground).
              const [warnBefore, warnAfter = ""] = msg("datasets.delete.used_warning").split("{count}");
              return (
                <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800">
                  {warnBefore}
                  <span className="font-semibold">{usedCount}</span>
                  {warnAfter}
                </p>
              );
            })()}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setDeleteOpen(false)}
              disabled={deleting}
              className="w-full justify-center"
            >
              {msg("datasets.delete.cancel")}
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={deleting}
              className="w-full justify-center shadow-xs"
            >
              {deleting ? <Loader2 className="size-4 animate-spin" /> : msg("datasets.delete.confirm")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
