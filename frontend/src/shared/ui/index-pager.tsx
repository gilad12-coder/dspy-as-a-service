"use client";

import { CaretLeft, CaretRight } from "@/shared/ui/icons";
import { cn } from "@/shared/lib/utils";
import { TooltipButton } from "@/shared/ui/tooltip-button";

interface IndexPagerProps {
  currentIndex: number;
  total: number;
  onChange: (next: number) => void;
  prevLabel: string;
  nextLabel: string;
  className?: string;
}

/**
 * Compact pager pill: prev arrow · "N/total" · next arrow.
 *
 * Renders nothing when `total < 2`. Direction is forced to LTR so the
 * "1/3" digits stay legible in RTL surfaces.
 */
export function IndexPager({
  currentIndex,
  total,
  onChange,
  prevLabel,
  nextLabel,
  className,
}: IndexPagerProps) {
  if (total < 2) return null;
  const atFirst = currentIndex <= 0;
  const atLast = currentIndex >= total - 1;
  return (
    <div
      dir="ltr"
      className={cn(
        "inline-flex items-center gap-0.5 rounded-full border border-border/50 bg-background/60 p-0.5 text-[0.75rem]",
        className,
      )}
    >
      <TooltipButton tooltip={prevLabel} side="top">
        <button
          type="button"
          onClick={() => onChange(currentIndex - 1)}
          disabled={atFirst}
          className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer transition-colors"
          aria-label={prevLabel}
        >
          <CaretLeft className="size-4" />
        </button>
      </TooltipButton>
      <span className="font-mono tabular-nums px-1.5 min-w-[2.25rem] text-center text-foreground/80 select-none">
        {currentIndex + 1}/{total}
      </span>
      <TooltipButton tooltip={nextLabel} side="top">
        <button
          type="button"
          onClick={() => onChange(currentIndex + 1)}
          disabled={atLast}
          className="inline-flex size-7 items-center justify-center rounded-full text-muted-foreground hover:bg-muted hover:text-foreground disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer transition-colors"
          aria-label={nextLabel}
        >
          <CaretRight className="size-4" />
        </button>
      </TooltipButton>
    </div>
  );
}
