"use client";

import { BarChart3 } from "lucide-react";

import { msg } from "@/shared/lib/messages";
import { EmptyState } from "@/shared/ui/empty-state";
import { getActiveDir } from "@/shared/lib/runtime-locale";

export function ChartTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number; name: string; color?: string }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-xl border border-border/60 bg-background/95 backdrop-blur-sm p-3 shadow-lg text-sm"
      dir={getActiveDir()}
    >
      {label && <p className="font-semibold mb-2 text-foreground">{label}</p>}
      <div className="space-y-1">
        {payload.map((p, i) => (
          <div key={i} className="flex items-center gap-2 text-muted-foreground">
            {p.color && (
              <span
                className="size-2.5 rounded-full shrink-0 ring-1 ring-black/5"
                style={{ backgroundColor: p.color }}
              />
            )}
            <span className="text-xs">{p.name}:</span>
            <span
              className="font-mono font-semibold text-foreground ms-auto tabular-nums"
              dir="ltr"
            >
              {p.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ChartEmptyState({ message }: { message?: string }) {
  return (
    <EmptyState
      variant="list"
      icon={BarChart3}
      title={message ?? msg("auto.shared.charts.chart.utils.literal.1")}
      className="h-[300px] justify-center"
    />
  );
}
