"use client";

import * as React from "react";
import { XCircle } from "@/shared/ui/icons";
import { RetryIconButton } from "@/shared/ui/retry-icon-button";
import { cn } from "@/shared/lib/utils";

interface ChatErrorBannerProps {
  message: React.ReactNode;
  retryLabel?: string;
  onRetry?: () => void;
  /** Extra content under the message, e.g. a "switch to manual" fallback link. */
  action?: React.ReactNode;
  className?: string;
}

// Shared warm-toned error card for agent chat streams (auth/model/rate-limit
// failures). Mirrors the icon-badge + text + action layout established by
// SubmitSummaryCard/ApprovalCard so error state reads as a first-class card
// instead of a cramped inline row.
export function ChatErrorBanner({
  message,
  retryLabel,
  onRetry,
  action,
  className,
}: ChatErrorBannerProps) {
  return (
    <div
      role="alert"
      className={cn("rounded-2xl border border-[#9B2C1F]/20 bg-[#FCEFEB]/60 px-4 py-3", className)}
    >
      <div className="flex items-start gap-2.5">
        <span className="flex size-7 shrink-0 items-center justify-center rounded-full bg-[#9B2C1F]/10 text-[#9B2C1F]">
          <XCircle className="size-3.5" aria-hidden="true" />
        </span>
        <div className="min-w-0 flex-1 space-y-1.5 pt-0.5">
          <p
            className="max-h-32 overflow-y-auto break-words text-xs leading-[1.45] text-[#7A1E13]"
            dir="auto"
          >
            {message}
          </p>
          {action}
        </div>
        {onRetry && retryLabel && (
          <RetryIconButton
            label={retryLabel}
            onClick={onRetry}
            className="size-[44px] shrink-0 border-[#9B2C1F]/25 bg-transparent text-[#7A1E13] shadow-none hover:bg-[#9B2C1F]/10 hover:text-[#7A1E13] md:size-7 [@media(hover:none)_and_(pointer:coarse)]:size-[44px]"
          />
        )}
      </div>
    </div>
  );
}
