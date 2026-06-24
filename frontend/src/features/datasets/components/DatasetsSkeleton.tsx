"use client";

import { AppSkeletonTheme, Skeleton } from "@/shared/ui/skeleton";

/** Loading placeholder for the /datasets route: header, toolbar, and card rows. */
export function DatasetsSkeleton() {
  return (
    <AppSkeletonTheme>
      <div className="pb-16">
        <div className="flex items-end justify-between">
          <div className="w-1/2">
            <Skeleton height={26} width="60%" />
            <div className="mt-2">
              <Skeleton height={16} width="80%" />
            </div>
          </div>
          <Skeleton height={28} width={180} />
        </div>

        <div className="mt-6 flex items-center gap-3">
          <div className="flex-1">
            <Skeleton height={38} />
          </div>
          <Skeleton height={38} width={150} />
        </div>

        <div className="mt-5 flex flex-col gap-2.5">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} height={68} borderRadius={12} />
          ))}
        </div>
      </div>
    </AppSkeletonTheme>
  );
}
