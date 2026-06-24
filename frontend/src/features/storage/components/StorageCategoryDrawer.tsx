"use client";

import * as React from "react";
import { Check, HardDrive, Minus, X } from "lucide-react";
import { toast } from "react-toastify";
import {
  bulkDeleteStorageItems,
  deleteDataset,
  deleteJob,
  deleteStagedUpload,
  getStorageCategoryItems,
  STORAGE_CHANGED_EVENT,
  type StorageItem,
} from "@/shared/lib/api";
import { deleteConversation } from "@/features/agent-panel";
import { EmptyState } from "@/shared/ui/empty-state";
import { Button } from "@/shared/ui/primitives/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/shared/ui/primitives/sheet";
import { cn } from "@/shared/lib/utils";
import { formatStorageSize } from "@/shared/lib/formatters";
import { formatMsg, msg, type MessageKey } from "@/shared/lib/messages";
import { StorageItemRow } from "./StorageItemRow";

/** Per-category label keys, mirroring the backend ``STORAGE_CATEGORIES``. */
const CATEGORY_LABELS: Record<string, MessageKey> = {
  optimizations: "storage.category.optimizations",
  datasets: "storage.category.datasets",
  agent_chats: "storage.category.agent_chats",
  staged_uploads: "storage.category.staged_uploads",
};

/** One-line "what is this category" shown under the drawer title. */
const CATEGORY_DESCRIPTIONS: Record<string, MessageKey> = {
  optimizations: "storage.category.desc.optimizations",
  datasets: "storage.category.desc.datasets",
  agent_chats: "storage.category.desc.agent_chats",
  staged_uploads: "storage.category.desc.staged_uploads",
};

/** Each category's homogeneous item type, used to route the bulk-delete batch. */
const CATEGORY_ITEM_TYPE: Record<string, StorageItem["type"]> = {
  optimizations: "optimization",
  datasets: "dataset",
  agent_chats: "chat",
  staged_uploads: "staged_upload",
};

/** Ids per bulk-delete request. The progress dialog advances one chunk at a time,
 *  and batches no larger than this delete in a single request (no progress bar). */
const CHUNK = 25;

/** Route the per-type single delete to the matching API call. */
function deleteItem(item: StorageItem): Promise<unknown> {
  if (item.type === "optimization") return deleteJob(item.id);
  if (item.type === "dataset") return deleteDataset(item.id);
  if (item.type === "staged_upload") return deleteStagedUpload(item.id);
  return deleteConversation(item.id);
}

interface StorageCategoryDrawerProps {
  /** The open category, or ``null`` when the drawer is closed. */
  category: string | null;
  /** Close the drawer (clear the selected category). */
  onClose: () => void;
  /** Fired after a successful delete so the page can refresh its usage gauge. */
  onChanged: () => void;
}

/**
 * A slide-over (matching the admin drawer) listing every item in one storage
 * category. Each row carries a leading checkbox for multi-select; a select-all
 * header, a sticky action bar, a summary confirm, and a determinate progress
 * dialog drive bulk deletion. A per-row trash still offers a quick single delete.
 * Loads the full set — not a top-N — so the user can clear a whole category here.
 */
export function StorageCategoryDrawer({ category, onClose, onChanged }: StorageCategoryDrawerProps) {
  const [items, setItems] = React.useState<StorageItem[] | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [pending, setPending] = React.useState<StorageItem | null>(null);
  const [deleting, setDeleting] = React.useState(false);
  const [selected, setSelected] = React.useState<Set<string>>(new Set());
  const [anchorIndex, setAnchorIndex] = React.useState<number | null>(null);
  const [bulkConfirm, setBulkConfirm] = React.useState(false);
  const [progress, setProgress] = React.useState<{ done: number; total: number } | null>(null);

  React.useEffect(() => {
    setSelected(new Set());
    setAnchorIndex(null);
    if (!category) {
      setItems(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    getStorageCategoryItems(category)
      .then((res) => {
        if (!cancelled) setItems(res.items);
      })
      .catch(() => {
        if (cancelled) return;
        setItems([]);
        toast.error(msg("storage.items.error"));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [category]);

  const confirmDelete = React.useCallback(async () => {
    if (!pending) return;
    setDeleting(true);
    try {
      await deleteItem(pending);
      setItems((prev) => prev?.filter((it) => !(it.id === pending.id && it.type === pending.type)) ?? null);
      setSelected((prev) => {
        if (!prev.has(pending.id)) return prev;
        const next = new Set(prev);
        next.delete(pending.id);
        return next;
      });
      toast.success(msg("storage.delete.success"));
      onChanged();
      window.dispatchEvent(new Event(STORAGE_CHANGED_EVENT));
      setPending(null);
    } catch {
      toast.error(msg("storage.delete.failed"));
    } finally {
      setDeleting(false);
    }
  }, [pending, onChanged]);

  const toggleItem = React.useCallback(
    (item: StorageItem, shiftKey: boolean) => {
      const current = items ?? [];
      const index = current.findIndex((it) => it.id === item.id);
      setSelected((prev) => {
        const next = new Set(prev);
        const willSelect = !prev.has(item.id);
        if (shiftKey && anchorIndex !== null && index !== -1) {
          const [lo, hi] = anchorIndex < index ? [anchorIndex, index] : [index, anchorIndex];
          for (let i = lo; i <= hi; i++) {
            const row = current[i];
            if (!row) continue;
            if (willSelect) next.add(row.id);
            else next.delete(row.id);
          }
        } else if (willSelect) {
          next.add(item.id);
        } else {
          next.delete(item.id);
        }
        return next;
      });
      setAnchorIndex(index);
    },
    [items, anchorIndex],
  );

  const toggleAll = React.useCallback(() => {
    const current = items ?? [];
    setSelected((prev) =>
      prev.size === current.length ? new Set() : new Set(current.map((it) => it.id)),
    );
    setAnchorIndex(null);
  }, [items]);

  const clearSelection = React.useCallback(() => {
    setSelected(new Set());
    setAnchorIndex(null);
  }, []);

  const itemType = category ? CATEGORY_ITEM_TYPE[category] : undefined;

  const runBulkDelete = React.useCallback(async () => {
    const targets = (items ?? []).filter((it) => selected.has(it.id));
    if (targets.length === 0 || !itemType) return;
    const ids = targets.map((it) => it.id);
    setBulkConfirm(false);
    setProgress({ done: 0, total: ids.length });
    const removed = new Set<string>();
    let skippedCount = 0;
    for (let i = 0; i < ids.length; i += CHUNK) {
      const chunk = ids.slice(i, i + CHUNK);
      try {
        const res = await bulkDeleteStorageItems(itemType, chunk);
        res.deleted.forEach((id) => removed.add(id));
        skippedCount += res.skipped.length;
      } catch {
        skippedCount += chunk.length;
      }
      setProgress({ done: Math.min(i + CHUNK, ids.length), total: ids.length });
      setItems((prev) => prev?.filter((it) => !removed.has(it.id)) ?? null);
    }
    setSelected(new Set());
    setAnchorIndex(null);
    setProgress(null);
    onChanged();
    window.dispatchEvent(new Event(STORAGE_CHANGED_EVENT));
    const n = removed.size;
    if (skippedCount === 0) toast.success(formatMsg("storage.bulk.deleted", { n }));
    else if (n === 0) toast.error(msg("storage.bulk.failed"));
    else toast.warn(formatMsg("storage.bulk.partial", { n, skipped: skippedCount }));
  }, [items, selected, itemType, onChanged]);

  const list = items ?? [];
  const selectedBytes = list.reduce((sum, it) => (selected.has(it.id) ? sum + it.bytes : sum), 0);
  const allSelected = list.length > 0 && selected.size === list.length;
  const someSelected = selected.size > 0 && !allSelected;
  const running = progress !== null;
  const busy = deleting || running;

  const labelKey = category ? CATEGORY_LABELS[category] : undefined;
  const descKey = (category ? CATEGORY_DESCRIPTIONS[category] : undefined) ?? "storage.category.subtitle";

  // Bold the item name in the delete prompt, like the dashboard delete dialogs.
  // ``msg`` returns a plain string, so split the template on its placeholders and
  // render name + size in their own ``<bdi>`` spans — bolding the name and
  // keeping the directional isolation ``formatTemplate`` would otherwise inject.
  const [deleteBefore, deleteRest = ""] = msg("storage.delete.body").split("{name}");
  const [deleteMid, deleteTail = ""] = deleteRest.split("{size}");
  // Same split for the bulk confirm so the count and freed size get the bold
  // emphasis the single-delete name does, instead of inlining into a flat string.
  const [bulkBefore, bulkRest = ""] = msg("storage.bulk.confirm.body").split("{n}");
  const [bulkMid, bulkTail = ""] = bulkRest.split("{size}");

  return (
    <>
      <Sheet open={category !== null} onOpenChange={(open) => !open && !busy && onClose()}>
        <SheetContent side="left" aria-describedby={undefined} className="w-full gap-0 p-0 sm:max-w-lg">
          <SheetHeader className="shrink-0 border-b border-border/40 px-6 py-4">
            <div className="flex items-center gap-2">
              <HardDrive className="size-4 text-muted-foreground" aria-hidden="true" />
              <SheetTitle>{labelKey ? msg(labelKey) : ""}</SheetTitle>
            </div>
            <p className="text-xs text-muted-foreground">{msg(descKey)}</p>
          </SheetHeader>

          <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
            {loading ? (
              <p className="px-2 py-6 text-center text-sm text-muted-foreground">
                {msg("storage.quota.loading")}
              </p>
            ) : list.length === 0 ? (
              <div className="py-6">
                <EmptyState icon={HardDrive} title={msg("storage.items.empty")} />
              </div>
            ) : (
              <>
                <div className="flex items-center gap-3 px-2 pb-1">
                  <button
                    type="button"
                    role="checkbox"
                    aria-checked={allSelected ? true : someSelected ? "mixed" : false}
                    aria-label={msg("storage.select.all")}
                    onClick={toggleAll}
                    className={cn(
                      "grid size-5 shrink-0 cursor-pointer place-items-center rounded-md border transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45",
                      allSelected || someSelected
                        ? "border-transparent bg-foreground text-background"
                        : "border-border/70 bg-background hover:border-foreground/40",
                    )}
                  >
                    {allSelected ? (
                      <Check className="size-3.5" strokeWidth={3} aria-hidden="true" />
                    ) : someSelected ? (
                      <Minus className="size-3.5" strokeWidth={3} aria-hidden="true" />
                    ) : null}
                  </button>
                  <span className="text-xs font-medium text-muted-foreground">{msg("storage.select.all")}</span>
                </div>
                <ul className="flex flex-col">
                  {list.map((item) => (
                    <StorageItemRow
                      key={`${item.type}:${item.id}`}
                      item={item}
                      selected={selected.has(item.id)}
                      onToggle={toggleItem}
                      onDelete={setPending}
                      onNavigate={onClose}
                    />
                  ))}
                </ul>
              </>
            )}
          </div>

          {selected.size > 0 && (
            <div className="shrink-0 flex items-center justify-between gap-3 border-t border-border/40 bg-background/95 px-4 py-3 backdrop-blur supports-[backdrop-filter]:bg-background/80">
              <div className="flex min-w-0 items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={clearSelection}
                  disabled={busy}
                  aria-label={msg("storage.select.clear")}
                >
                  <X className="size-4" />
                </Button>
                <span className="truncate text-sm tabular-nums text-foreground">
                  {formatMsg("storage.bulk.bar.summary", {
                    n: selected.size,
                    size: formatStorageSize(selectedBytes),
                  })}
                </span>
              </div>
              <Button variant="destructive" onClick={() => setBulkConfirm(true)} disabled={busy}>
                {formatMsg("storage.bulk.delete", { n: selected.size })}
              </Button>
            </div>
          )}
        </SheetContent>
      </Sheet>

      <Dialog open={pending !== null} onOpenChange={(next) => !next && !deleting && setPending(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{msg("storage.delete.title")}</DialogTitle>
            <DialogDescription>
              {pending ? (
                <>
                  {deleteBefore}
                  <bdi className="font-semibold text-foreground">{pending.name}</bdi>
                  {deleteMid}
                  <bdi className="font-semibold text-foreground">{formatStorageSize(pending.bytes)}</bdi>
                  {deleteTail}
                </>
              ) : (
                ""
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setPending(null)} disabled={deleting}>
              {msg("storage.delete.cancel")}
            </Button>
            <Button variant="destructive" onClick={confirmDelete} disabled={deleting}>
              {msg("storage.delete.confirm")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={bulkConfirm} onOpenChange={(next) => !next && !running && setBulkConfirm(false)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{formatMsg("storage.bulk.confirm.title", { n: selected.size })}</DialogTitle>
            <DialogDescription>
              {bulkBefore}
              <bdi className="font-semibold text-foreground">{selected.size}</bdi>
              {bulkMid}
              <bdi className="font-semibold text-foreground">{formatStorageSize(selectedBytes)}</bdi>
              {bulkTail}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="ghost" onClick={() => setBulkConfirm(false)} disabled={running}>
              {msg("storage.delete.cancel")}
            </Button>
            <Button variant="destructive" onClick={runBulkDelete} disabled={running}>
              {msg("storage.delete.confirm")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={running && (progress?.total ?? 0) > CHUNK}>
        <DialogContent
          aria-describedby={undefined}
          showCloseButton={false}
          className="sm:max-w-sm"
          onEscapeKeyDown={(event) => event.preventDefault()}
          onPointerDownOutside={(event) => event.preventDefault()}
          onInteractOutside={(event) => event.preventDefault()}
        >
          <DialogHeader>
            <DialogTitle>{msg("storage.bulk.progress.title")}</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-2">
            <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
              <div
                role="progressbar"
                aria-valuenow={progress?.done ?? 0}
                aria-valuemin={0}
                aria-valuemax={progress?.total ?? 0}
                className="h-full origin-right rounded-full bg-foreground transition-transform duration-300 ease-out"
                style={{
                  transform: `scaleX(${progress && progress.total ? progress.done / progress.total : 0})`,
                }}
              />
            </div>
            <p className="text-center text-sm tabular-nums text-muted-foreground">
              {formatMsg("storage.bulk.progress.count", {
                done: progress?.done ?? 0,
                total: progress?.total ?? 0,
              })}
            </p>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
