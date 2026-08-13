"use client";

import * as React from "react";

import { cn } from "@/shared/lib/utils";
import { ArrowCounterClockwise, CircleNotch, type Icon } from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import { TooltipButton } from "@/shared/ui/tooltip-button";

interface RetryIconButtonProps
  extends Omit<React.ComponentProps<typeof Button>, "aria-label" | "children" | "size"> {
  label: string;
  loading?: boolean;
  icon?: Icon;
  tooltipSide?: "top" | "right" | "bottom" | "left";
}

/** Render the app-standard retry action as an end-aligned icon with a tooltip. */
export function RetryIconButton({
  label,
  loading = false,
  icon: RetryIcon = ArrowCounterClockwise,
  tooltipSide = "top",
  className,
  disabled,
  variant = "outline",
  ...props
}: RetryIconButtonProps) {
  return (
    <TooltipButton tooltip={label} side={tooltipSide}>
      <Button
        {...props}
        variant={variant}
        size="icon-sm"
        className={cn("ms-auto", className)}
        disabled={disabled || loading}
        aria-label={label}
        aria-busy={loading || undefined}
      >
        {loading ? (
          <CircleNotch className="size-3.5 animate-spin" aria-hidden="true" />
        ) : (
          <RetryIcon className="size-3.5" aria-hidden="true" />
        )}
      </Button>
    </TooltipButton>
  );
}
