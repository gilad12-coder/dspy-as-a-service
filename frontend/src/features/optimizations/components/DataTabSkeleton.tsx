"use client";

import { Skeleton } from "@/shared/ui/skeleton";

export function DataTabSkeleton() {
  return (
    <div className="space-y-4 mt-4" aria-hidden="true">
      <Skeleton width="55%" height={14} />

      <div className="rounded-2xl border border-[#E5DDD4] bg-gradient-to-l from-[#FAF8F5] to-[#F5F1EC] p-4 space-y-3">
        <div className="flex items-center gap-3">
          <div className="flex-1 min-w-0">
            <Skeleton width={140} height={14} />
          </div>
          <Skeleton width={150} height={32} borderRadius={8} />
        </div>
      </div>

      <div className="flex items-center gap-3 flex-wrap">
        <div className="flex w-full gap-1 rounded-lg bg-muted p-1">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height={28} className="flex-1" />
          ))}
        </div>
        <Skeleton width={64} height={10} className="ms-auto" />
      </div>

      <div className="relative rounded-2xl border border-[#DDD4C8]/50 bg-gradient-to-b from-white/95 to-[#F8F4EF] py-5 overflow-hidden">
        <div className="flex h-12 items-center gap-4 border-b border-border/70 px-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height={14} containerClassName="flex-1" />
          ))}
        </div>
        {Array.from({ length: 9 }).map((_, i) => (
          <div
            key={i}
            className="flex h-10 items-center gap-4 border-b border-border/60 px-2 last:border-0"
          >
            <Skeleton height={12} containerClassName="flex-1" width="55%" />
            <Skeleton height={12} containerClassName="flex-1" width="80%" />
            <Skeleton height={12} containerClassName="flex-1" width="70%" />
            <Skeleton height={12} containerClassName="flex-1" width="45%" />
          </div>
        ))}
      </div>
    </div>
  );
}
