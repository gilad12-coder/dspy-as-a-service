"use client";

import { Skeleton } from "@/shared/ui/skeleton";

export function OptimizationDetailSkeleton() {
  return (
    <div className="space-y-6 pb-12" aria-hidden="true">
      <div className="rounded-xl border border-border/40 bg-gradient-to-br from-card to-card/80 p-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 flex-1 space-y-2">
            <div className="flex flex-wrap items-center gap-3">
              <Skeleton width={72} height={22} borderRadius={11} />
              <span className="block w-[55%] max-w-[320px]">
                <Skeleton height={26} />
              </span>
            </div>
            <span className="block w-[80%] max-w-[420px]">
              <Skeleton height={12} />
            </span>
            <span className="block w-[40%] max-w-[220px]">
              <Skeleton height={10} />
            </span>
            <div className="flex flex-wrap items-center gap-3 pt-1">
              <Skeleton width={60} height={20} borderRadius={6} />
              <Skeleton width={84} height={16} />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Skeleton width={32} height={32} borderRadius={8} />
            <Skeleton width={32} height={32} borderRadius={8} />
            <Skeleton width={32} height={32} borderRadius={8} />
          </div>
        </div>
      </div>

      <div className="flex items-center justify-end gap-5 border-b border-border/50 pb-2.5">
        {Array.from({ length: 4 }).map((_, i) => (
          <span key={i} className="flex items-center gap-1.5">
            <Skeleton width={14} height={14} />
            <Skeleton width={48} height={14} />
          </span>
        ))}
      </div>

      <div className="space-y-6">
        <span className="block w-[70%] max-w-[460px]">
          <Skeleton height={14} />
        </span>

        <div className="flex items-start justify-between">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="flex flex-col items-center gap-2">
              <Skeleton circle width={28} height={28} />
              <Skeleton width={56} height={11} />
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="rounded-xl border border-border/50 bg-card p-6 text-center space-y-2"
            >
              <Skeleton width={96} height={11} />
              <Skeleton width="50%" height={30} />
            </div>
          ))}
        </div>

        <div className="rounded-2xl border border-[#DDD4C8]/50 bg-gradient-to-b from-white/95 to-[#F8F4EF] py-5 space-y-3">
          <div className="flex items-center justify-between px-6">
            <span className="flex items-center gap-2">
              <Skeleton width={16} height={16} />
              <Skeleton width={120} height={16} />
            </span>
            <Skeleton width={72} height={14} />
          </div>
          <div className="px-6 space-y-3">
            <div className="rounded-xl border border-border/40 bg-background/70 px-4 pt-3 pb-4 space-y-3">
              <Skeleton width={88} height={11} />
              <Skeleton height={4} borderRadius={2} />
            </div>
            <Skeleton height={300} borderRadius={12} />
          </div>
        </div>
      </div>
    </div>
  );
}
