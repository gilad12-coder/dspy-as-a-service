"use client";

import * as React from "react";
import Link from "next/link";
import { HardDrive } from "@/shared/ui/icons";
import {
  getStorageUsage,
  STORAGE_CHANGED_EVENT,
  type StorageUsageResponse,
} from "@/shared/lib/api";
import { formatStorageSize } from "@/shared/lib/formatters";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";

/**
 * Compact account-wide storage gauge for the sidebar footer: a thin bar plus an
 * "X of Y used" caption against the caller's storage budget, linking to the
 * /storage cleanup page.
 *
 * Fetches the unified usage figure once on mount, again whenever the tab regains
 * focus, and whenever a {@link STORAGE_CHANGED_EVENT} fires (a delete on the
 * storage page), so the figure stays current without a reload. Renders nothing
 * until the first figure resolves (and on error), so it never flashes an empty
 * bar in the always-visible footer. When ``collapsed`` (the icon rail), it
 * shrinks to a lone drive icon whose tooltip carries the same usage figure.
 */
export function StorageMeter({ collapsed = false }: { collapsed?: boolean }) {
  const [usage, setUsage] = React.useState<StorageUsageResponse | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    const load = () => {
      getStorageUsage()
        .then((res) => {
          if (!cancelled) setUsage(res);
        })
        .catch(() => {
          /* ambient gauge — stay silent if the figure is briefly unavailable */
        });
    };
    load();
    const onVisible = () => {
      if (document.visibilityState === "visible") load();
    };
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener(STORAGE_CHANGED_EVENT, load);
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener(STORAGE_CHANGED_EVENT, load);
    };
  }, []);

  if (!usage || usage.quota_bytes <= 0) return null;

  const usagePct = Math.min(100, (usage.used_bytes / usage.quota_bytes) * 100);
  const usageLabel = formatMsg("storage.quota.usage", {
    used: formatStorageSize(usage.used_bytes),
    total: formatStorageSize(usage.quota_bytes),
  });

  if (collapsed) {
    const tooltipSide = getActiveDir() === "rtl" ? "left" : "right";
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Link
            href="/storage"
            aria-label={usageLabel}
            className="flex min-h-[44px] min-w-[44px] items-center justify-center px-2 py-2.5 transition-colors duration-150 hover:bg-sidebar-accent/40 lg:min-h-0 lg:min-w-0"
          >
            <HardDrive className="size-4 text-muted-foreground" />
          </Link>
        </TooltipTrigger>
        <TooltipContent side={tooltipSide}>{usageLabel}</TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Link
      href="/storage"
      aria-label={msg("storage.page.title")}
      className="block min-h-[44px] px-3 pt-3 pb-1 transition-colors duration-150 hover:bg-sidebar-accent/40 lg:min-h-0"
    >
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#E5DDD4]">
        <div
          className="h-full rounded-full bg-[#3D2E22]/70 transition-[width] duration-500 ease-out"
          style={{ width: `${usagePct}%` }}
        />
      </div>
      <p className="mt-1.5 text-start text-[0.6875rem] text-muted-foreground tabular-nums">
        {usageLabel}
      </p>
    </Link>
  );
}
