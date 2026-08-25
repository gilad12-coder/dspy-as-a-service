"use client";

import { Skeleton } from "@/shared/ui/skeleton";

const SECTIONS = [
  { labelWidth: 70, rows: ["90%", "75%", "82%"] },
  { labelWidth: 56, rows: ["68%", "85%"] },
];

export function ConversationDrawerSkeleton() {
  return (
    <div aria-hidden="true">
      {SECTIONS.map((section, s) => (
        <div key={s} className="mt-3">
          <div className="px-2 pb-1">
            <Skeleton height={9} width={section.labelWidth} />
          </div>
          <ul className="space-y-0.5">
            {section.rows.map((width, r) => (
              <li key={r}>
                <div className="flex items-center gap-1.5 rounded-md px-2 py-1.5">
                  <div className="min-w-0 flex-1">
                    <Skeleton height={13} width={width} />
                    <Skeleton height={11} width="55%" />
                  </div>
                  <Skeleton width={14} height={14} borderRadius={4} />
                </div>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}
