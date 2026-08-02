"use client";

import * as React from "react";
import { X, XCircle } from "@/shared/ui/icons";

import { cn } from "@/shared/lib/utils";

interface InlineErrorRowProps {
  /** Headline rendered in semibold above the message. Optional. */
  title?: React.ReactNode;
  /** The error body. Rendered with `dir="auto"` so user-facing strings flip RTL. */
  message: React.ReactNode;
  /** Show a dismiss `×` button when provided. */
  onDismiss?: () => void;
  /** Hide the leading icon (defaults to XCircle). */
  hideIcon?: boolean;
  /** ARIA label for the dismiss button. */
  dismissLabel?: string;
  className?: string;
}

/**
 * Bordered destructive-tinted row used for inline error feedback (load
 * failures, validation rejections, stream errors). Centralizes the spacing,
 * icon, and dismiss treatment so error UIs stay consistent.
 */
export function InlineErrorRow({
  title,
  message,
  onDismiss,
  hideIcon = false,
  dismissLabel = "Dismiss",
  className,
}: InlineErrorRowProps) {
  return (
    <div
      role="alert"
      className={cn(
        "flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-[0.75rem] text-destructive",
        className,
      )}
    >
      {!hideIcon && <XCircle className="size-4 shrink-0 mt-0.5" />}
      <div className="min-w-0 flex-1">
        {title && <p className="font-semibold">{title}</p>}
        <p className={cn("break-words", title && "mt-0.5")} dir="auto">
          {message}
        </p>
      </div>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label={dismissLabel}
          className="close-button shrink-0"
        >
          <X />
        </button>
      )}
    </div>
  );
}
