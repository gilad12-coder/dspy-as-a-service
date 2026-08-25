"use client";

import * as React from "react";
import Link from "next/link";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Button } from "@/shared/ui/primitives/button";
import { formatMsg, msg, type MessageKey } from "@/shared/lib/messages";
import { formatStorageSize } from "@/shared/lib/formatters";
import type { StorageUsageResponse } from "@/shared/lib/api";

/** Per-category label keys, mirroring the backend ``STORAGE_CATEGORIES``. */
const CATEGORY_LABELS: Record<string, MessageKey> = {
  optimizations: "storage.category.optimizations",
  datasets: "storage.category.datasets",
  agent_chats: "storage.category.agent_chats",
  staged_uploads: "storage.category.staged_uploads",
};

/** Inputs for the presentational quota modal; data is fetched by the host. */
interface StorageQuotaModalProps {
  open: boolean;
  usage: StorageUsageResponse | null;
  loading: boolean;
  onClose: () => void;
}

/**
 * The account-wide storage budget modal: a usage meter, a per-category
 * breakdown of where the space went, and links to the two places a user frees
 * it. Presentational — the host owns the open state and the usage fetch.
 */
export function StorageQuotaModal({ open, usage, loading, onClose }: StorageQuotaModalProps) {
  const usedBytes = usage?.used_bytes ?? 0;
  const quotaBytes = usage?.quota_bytes ?? 0;
  const usagePct = quotaBytes > 0 ? Math.min(100, (usedBytes / quotaBytes) * 100) : 0;

  const rows = React.useMemo(() => {
    if (!usage) return [] as Array<[string, number]>;
    return Object.entries(usage.breakdown)
      .filter(([, bytes]) => bytes > 0)
      .sort((a, b) => b[1] - a[1]);
  }, [usage]);

  return (
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{msg("storage.quota.title")}</DialogTitle>
          <DialogDescription>{msg("storage.quota.body")}</DialogDescription>
        </DialogHeader>

        <div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#E5DDD4]">
            <div
              className="h-full rounded-full bg-[#3D2E22]/70 transition-[width] duration-500 ease-out"
              style={{ width: `${usagePct}%` }}
            />
          </div>
          <p className="mt-1.5 text-end text-xs text-muted-foreground tabular-nums">
            {formatMsg("storage.quota.usage", {
              used: formatStorageSize(usedBytes),
              total: formatStorageSize(quotaBytes),
            })}
          </p>
        </div>

        {(loading || rows.length > 0) && (
          <div>
            <h3 className="mb-2 text-sm font-medium text-foreground">
              {msg("storage.quota.breakdown_title")}
            </h3>
            {loading ? (
              <p className="text-sm text-muted-foreground">{msg("storage.quota.loading")}</p>
            ) : (
              <ul className="flex flex-col gap-2.5">
                {rows.map(([key, bytes]) => {
                  const labelKey = CATEGORY_LABELS[key];
                  const pct = usedBytes > 0 ? Math.max(2, (bytes / usedBytes) * 100) : 0;
                  return (
                    <li key={key}>
                      <div className="flex items-baseline justify-between text-sm">
                        <span className="text-foreground">{labelKey ? msg(labelKey) : key}</span>
                        <span className="text-muted-foreground tabular-nums">
                          {formatStorageSize(bytes)}
                        </span>
                      </div>
                      <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-[#E5DDD4]/60">
                        <div
                          className="h-full rounded-full bg-[#3D2E22]/30"
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}

        <Button asChild onClick={onClose} className="w-full">
          <Link href="/storage">{msg("storage.quota.cta.manage")}</Link>
        </Button>
      </DialogContent>
    </Dialog>
  );
}
