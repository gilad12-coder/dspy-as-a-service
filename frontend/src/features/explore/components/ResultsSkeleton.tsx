"use client";

import * as React from "react";
import { Skeleton } from "@/shared/ui/skeleton";

interface ResultsSkeletonProps {
  /** Number of placeholder rows to render. */
  rows?: number;
}

/**
 * Placeholder rows shown while a fresh query is in flight and there are no
 * prior results to keep on screen. Matches the rhythm of ResultsList so the
 * page doesn't jump when real rows replace it.
 */
export function ResultsSkeleton({ rows = 4 }: ResultsSkeletonProps) {
  return (
    <ul aria-hidden="true" className="divide-y divide-border/55">
      {Array.from({ length: rows }).map((_, i) => (
        <li key={i} className="flex flex-col gap-2 rounded-lg px-3 py-4">
          <div className="flex items-baseline justify-between gap-4">
            <div className="min-w-0 flex-1" style={{ maxWidth: `${50 + ((i * 13) % 30)}%` }}>
              <Skeleton height={16} borderRadius={4} />
            </div>
            <div className="w-16 shrink-0">
              <Skeleton height={14} borderRadius={9999} />
            </div>
          </div>
          <div className="max-w-[72ch]">
            <Skeleton height={13} count={2} borderRadius={4} />
          </div>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <div className="w-20">
              <Skeleton height={12} borderRadius={9999} />
            </div>
            <div className="w-24">
              <Skeleton height={12} borderRadius={4} />
            </div>
            <div className="w-16">
              <Skeleton height={12} borderRadius={4} />
            </div>
            <div className="ms-auto w-14">
              <Skeleton height={12} borderRadius={4} />
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}
