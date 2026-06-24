"use client";

import { Skeleton } from "@/shared/ui/skeleton";
import { ResultsSkeleton } from "./ResultsSkeleton";

export function ExploreSkeleton() {
  return (
    <div className="pb-16" aria-hidden="true">
      <div className="flex flex-col gap-1.5">
        <div className="flex flex-col gap-3">
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-2.5">
            <div className="flex items-center justify-center">
              <Skeleton width={168} height={34} borderRadius={9999} />
            </div>
            <Skeleton height={48} borderRadius={16} />
          </div>
        </div>
        <div className="border-t border-border/55">
          <ResultsSkeleton rows={4} />
        </div>
      </div>
    </div>
  );
}
