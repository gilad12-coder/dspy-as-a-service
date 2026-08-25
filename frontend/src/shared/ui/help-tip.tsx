"use client";

import { Tooltip, TooltipTrigger, TooltipContent } from "@/shared/ui/primitives/tooltip";
import { cn } from "@/shared/lib/utils";
import { getActiveDir } from "@/shared/lib/runtime-locale";

/**
 * Zero-visual-impact jargon tooltip.
 * Wraps children in a Tooltip that appears on hover.
 * No icons, underlines, or dotted borders — the label looks exactly as before.
 */
export function HelpTip({
  text,
  children,
  side = "top",
  className,
}: {
  text: string;
  children: React.ReactNode;
  side?: "top" | "bottom" | "left" | "right";
  className?: string;
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        {/* inline-flex w-fit so the trigger box hugs the text — without it,
            a block child (e.g. a section-title <div>) makes the trigger span
            the full row width and Radix centers the tooltip over the row
            instead of over the label itself. Pass className="w-full" when the
            child is meant to span its container (e.g. a full-width header bar). */}
        <span className={cn("inline-flex w-fit max-w-full cursor-default", className)}>
          {children}
        </span>
      </TooltipTrigger>
      <TooltipContent side={side} className="max-w-64 text-center leading-relaxed" dir={getActiveDir()}>
        {text}
      </TooltipContent>
    </Tooltip>
  );
}
