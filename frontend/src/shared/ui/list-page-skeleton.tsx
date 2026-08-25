"use client";

import { AppSkeletonTheme, Skeleton } from "@/shared/ui/skeleton";

/**
 * Loading silhouette shared by the Data hub list surfaces (dataset library and
 * labeling-session chooser): the search toolbar (h-11 field + action button)
 * and card rows at their loaded geometry, so the content swap shifts nothing.
 */
export function ListPageSkeleton() {
  return (
    <AppSkeletonTheme>
      <div className="flex items-center gap-2.5">
        <div className="flex-1">
          <Skeleton height={44} borderRadius={16} />
        </div>
        <Skeleton height={44} width={150} borderRadius={16} />
      </div>

      <div className="mt-5 flex flex-col gap-2.5 p-0.5">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} height={70} borderRadius={12} />
        ))}
      </div>
    </AppSkeletonTheme>
  );
}
