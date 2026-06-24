"use client";

import * as React from "react";
import { ChevronLeft, HardDrive } from "lucide-react";
import { getStorageUsage, type StorageUsageResponse } from "@/shared/lib/api";
import { EmptyState } from "@/shared/ui/empty-state";
import { formatStorageSize } from "@/shared/lib/formatters";
import { formatMsg, msg, type MessageKey } from "@/shared/lib/messages";
import { StorageCategoryDrawer } from "./StorageCategoryDrawer";
import { StorageSkeleton } from "./StorageSkeleton";

/** Per-category label keys, mirroring the backend ``STORAGE_CATEGORIES``. */
const CATEGORY_LABELS: Record<string, MessageKey> = {
  optimizations: "storage.category.optimizations",
  datasets: "storage.category.datasets",
  agent_chats: "storage.category.agent_chats",
  staged_uploads: "storage.category.staged_uploads",
};

/**
 * Top-level /storage page: the account-wide cleanup surface. A usage gauge over
 * the caller's budget, then a per-category breakdown where each category opens a
 * drawer listing all of its items for in-place deletion. Byproduct bytes (logs,
 * progress events, embeddings) are folded by the backend into the footprint of
 * the optimization or chat that owns them, so every row here is deletable.
 */
export function StorageView() {
  const [usage, setUsage] = React.useState<StorageUsageResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(false);
  const [openCategory, setOpenCategory] = React.useState<string | null>(null);

  const refreshUsage = React.useCallback(() => {
    getStorageUsage()
      .then(setUsage)
      .catch(() => {
        /* keep the last figure rather than blanking the gauge */
      });
  }, []);

  React.useEffect(() => {
    let cancelled = false;
    getStorageUsage()
      .then((res) => {
        if (!cancelled) setUsage(res);
      })
      .catch(() => {
        if (!cancelled) setError(true);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const breakdown = React.useMemo(() => {
    if (!usage) return [] as Array<[string, number]>;
    return Object.entries(usage.breakdown)
      .filter(([, bytes]) => bytes > 0)
      .sort((a, b) => b[1] - a[1]);
  }, [usage]);

  if (loading) return <StorageSkeleton />;

  if (error || !usage) {
    return (
      <div className="pb-16">
        <div className="mt-8">
          <EmptyState icon={HardDrive} title={msg("storage.page.error")} />
        </div>
      </div>
    );
  }

  const used = usage.used_bytes;
  const quota = usage.quota_bytes;
  const usagePct = quota > 0 ? Math.min(100, (used / quota) * 100) : 0;
  const free = Math.max(0, quota - used);

  return (
    <div className="pb-16">
      <section className="mt-8">
        <div className="flex items-baseline justify-between gap-3">
          <p className="text-foreground">
            <span className="text-2xl font-semibold tabular-nums">{formatStorageSize(used)}</span>
            <span className="ms-1.5 text-sm text-muted-foreground">
              {formatMsg("storage.page.of_total", { total: formatStorageSize(quota) })}
            </span>
          </p>
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
            {formatMsg("storage.page.free", { free: formatStorageSize(free) })}
          </span>
        </div>
        <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-[#E5DDD4]">
          <div
            className="h-full rounded-full bg-[#3D2E22]/70 transition-[width] duration-500 ease-out"
            style={{ width: `${usagePct}%` }}
          />
        </div>
        <p className="mt-2 text-xs tabular-nums text-muted-foreground">
          {formatMsg("storage.page.percent", {
            percent: used > 0 ? Math.max(1, Math.round(usagePct)) : 0,
          })}
        </p>
      </section>

      <section className="mt-10">
        <h2 className="text-sm font-semibold text-foreground">{msg("storage.breakdown.title")}</h2>
        {breakdown.length === 0 ? (
          <p className="mt-3 text-sm text-muted-foreground">{msg("storage.breakdown.empty")}</p>
        ) : (
          <ul className="mt-3 flex flex-col gap-1.5">
            {breakdown.map(([key, bytes]) => {
              const labelKey = CATEGORY_LABELS[key];
              const label = labelKey ? msg(labelKey) : key;
              const pct = used > 0 ? Math.max(2, (bytes / used) * 100) : 0;
              const bar = (
                <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-[#E5DDD4]/60">
                  <div className="h-full rounded-full bg-[#3D2E22]/30" style={{ width: `${pct}%` }} />
                </div>
              );

              return (
                <li key={key}>
                  <button
                    type="button"
                    onClick={() => setOpenCategory(key)}
                    aria-label={formatMsg("storage.category.open", { category: label })}
                    className="group w-full cursor-pointer rounded-lg px-2 py-2 text-start transition-colors duration-150 hover:bg-muted/40"
                  >
                    <div className="flex items-baseline justify-between gap-2 text-sm">
                      <span className="flex items-center gap-1.5 text-foreground">
                        {label}
                        <ChevronLeft
                          className="size-3.5 text-muted-foreground/60 transition-transform duration-150 group-hover:-translate-x-0.5"
                          aria-hidden="true"
                        />
                      </span>
                      <span className="tabular-nums text-muted-foreground">{formatStorageSize(bytes)}</span>
                    </div>
                    {bar}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <StorageCategoryDrawer
        category={openCategory}
        onClose={() => setOpenCategory(null)}
        onChanged={refreshUsage}
      />
    </div>
  );
}
