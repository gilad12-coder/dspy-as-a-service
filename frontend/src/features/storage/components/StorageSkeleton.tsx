"use client";

import { AppSkeletonTheme, Skeleton } from "@/shared/ui/skeleton";

/** Loading placeholder for the /storage route: header, gauge, breakdown, items. */
export function StorageSkeleton() {
  return (
    <AppSkeletonTheme>
      <div className="pb-16">
        <div className="w-1/2">
          <Skeleton height={28} width="40%" />
          <div className="mt-2">
            <Skeleton height={16} width="85%" />
          </div>
        </div>

        <div className="mt-8">
          <Skeleton height={22} width={180} />
          <div className="mt-3">
            <Skeleton height={8} borderRadius={9999} />
          </div>
          <div className="mt-2">
            <Skeleton height={14} width="55%" />
          </div>
        </div>

        <div className="mt-10 flex flex-col gap-3">
          <Skeleton height={18} width={140} />
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} height={20} />
          ))}
        </div>

        <div className="mt-10 flex flex-col gap-2.5">
          <Skeleton height={18} width={160} />
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} height={56} borderRadius={10} />
          ))}
        </div>
      </div>
    </AppSkeletonTheme>
  );
}
