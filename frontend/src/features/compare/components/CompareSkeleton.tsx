"use client";

import { Skeleton } from "@/shared/ui/skeleton";
import { Card, CardContent, CardHeader } from "@/shared/ui/primitives/card";

function StatCard() {
  return (
    <div className="min-w-0 flex-[1_1_10rem] rounded-xl border border-border/50 bg-card px-4 py-3">
      <div className="flex items-center gap-1.5">
        <Skeleton width={12} height={12} borderRadius={6} />
        <Skeleton width={70} height={10} />
      </div>
      <div className="mt-1">
        <Skeleton width={90} height={20} />
      </div>
    </div>
  );
}

export function CompareSkeleton() {
  return (
    <div className="space-y-6 pb-16" aria-hidden="true" data-tutorial="compare-verdict">
      <div className="flex items-center gap-2">
        <Skeleton width={48} height={14} />
        <Skeleton width={12} height={12} />
        <Skeleton width={180} height={16} />
      </div>

      <div className="space-y-3">
        <div className="flex items-center gap-2.5 rounded-xl border border-border/50 bg-card px-4 sm:px-5 py-3">
          <Skeleton width={16} height={16} borderRadius={4} />
          <Skeleton width={48} height={10} />
          <Skeleton width={120} height={24} borderRadius={6} />
          <span className="ms-auto">
            <Skeleton width={32} height={32} borderRadius={8} />
          </span>
        </div>
        <div className="flex flex-wrap gap-3">
          <StatCard />
          <StatCard />
          <StatCard />
        </div>
      </div>

      <div className="flex w-full border-b border-border/50">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="flex flex-1 items-center justify-center gap-1.5 py-2.5">
            <Skeleton width={14} height={14} borderRadius={4} />
            <Skeleton width={56} height={14} />
          </div>
        ))}
      </div>

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Skeleton width={16} height={16} borderRadius={4} />
              <Skeleton width={140} height={16} />
            </div>
          </CardHeader>
          <CardContent>
            <div className="h-[280px]">
              <Skeleton height={280} />
            </div>
            <div className="flex flex-wrap justify-center gap-4 mt-3">
              <Skeleton width={120} height={14} borderRadius={4} />
              <Skeleton width={120} height={14} borderRadius={4} />
            </div>
          </CardContent>
        </Card>

        <Card className="overflow-hidden">
          <CardContent className="p-0">
            <div className="table-scroll">
              <table className="w-full text-sm border-separate border-spacing-0">
              <thead>
                <tr>
                  <th className="py-2.5 w-[120px] sm:w-[180px] border-b border-border/40 border-e border-border/30">
                    <div className="flex justify-center">
                      <Skeleton width={56} height={11} />
                    </div>
                  </th>
                  {Array.from({ length: 2 }).map((_, i) => (
                    <th key={i} className="py-2.5 px-2 border-b border-border/40">
                      <div className="flex justify-center">
                        <Skeleton width={64} height={22} borderRadius={6} />
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Array.from({ length: 5 }).map((_, rowIdx) => (
                  <tr key={rowIdx} className={rowIdx > 0 ? "border-t border-border/30" : ""}>
                    <td className="py-2.5 px-3 border-e border-border/30">
                      <Skeleton width={72} height={12} />
                    </td>
                    {Array.from({ length: 2 }).map((_, i) => (
                      <td key={i} className="py-2.5 px-2">
                        <div className="flex justify-center">
                          <Skeleton width={56} height={20} borderRadius={6} />
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
