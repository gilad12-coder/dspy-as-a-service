"use client";

import * as React from "react";
import Link from "next/link";
import type { Icon } from "@/shared/ui/icons";

import { Button } from "@/shared/ui/primitives/button";
import { RetryIconButton } from "@/shared/ui/retry-icon-button";
import { cn } from "@/shared/lib/utils";

interface EmptyStateProps {
  icon?: Icon;
  /**
   * How the icon is presented.
   * - "none" (default): plain icon at muted color, used for page-level empties.
   * - "tile": rounded-square brand-tinted tile, used inside chat panels.
   * - "circle": small rounded circle tile, used by the dock-pill agent.
   */
  iconWrap?: "none" | "tile" | "circle";
  title: string;
  description?: string;
  /**
   * Density. "page" is the large empty for top-level routes; "compact" is the
   * smaller variant used inside scroll panels (chats, side dialogs); "list" is
   * the small centered empty shown inside list/feed surfaces (sidebar runs,
   * explore, dashboard) — tiny icon, two muted lines, one optional subtle CTA.
   */
  variant?: "page" | "compact" | "list";
  action?: {
    label: string;
    onClick?: () => void;
    href?: string;
    /** Optional leading icon for the action (used by the "list" CTA). */
    icon?: Icon;
    /** Present simple recovery actions as the shared icon-and-tooltip control. */
    iconOnly?: boolean;
    loading?: boolean;
  };
  /** Extra content rendered after the action (e.g. demo cards). */
  children?: React.ReactNode;
  className?: string;
}

export function EmptyState({
  icon: Icon,
  iconWrap = "none",
  title,
  description,
  variant = "page",
  action,
  children,
  className,
}: EmptyStateProps) {
  const isCompact = variant === "compact";
  const isList = variant === "list";
  const ActionIcon = action?.icon;

  if (isList) {
    return (
      <div
        className={cn(
          "flex flex-col items-center gap-2.5 px-4 pt-9 pb-6 text-center",
          className,
        )}
      >
        {Icon && (
          <Icon
            className="size-6 text-muted-foreground/25"
            aria-hidden="true"
          />
        )}
        {action?.iconOnly && action.onClick ? (
          <div className="flex w-full max-w-sm items-center gap-3 text-start">
            <div className="min-w-0 flex-1">
              <p className="text-[0.8125rem] font-medium text-muted-foreground/75">{title}</p>
              {description && (
                <p className="mt-1 max-w-[18rem] text-[0.6875rem] leading-relaxed text-muted-foreground/45">
                  {description}
                </p>
              )}
            </div>
            <RetryIconButton
              label={action.label}
              onClick={action.onClick}
              loading={action.loading}
              icon={ActionIcon}
            />
          </div>
        ) : (
          <>
            <p className="text-[0.8125rem] font-medium text-muted-foreground/75">{title}</p>
            {description && (
              <p className="max-w-[11rem] text-[0.6875rem] leading-relaxed text-muted-foreground/45">
                {description}
              </p>
            )}
          </>
        )}
        {action && !action.iconOnly && (
          <Button
            variant="outline"
            size="sm"
            onClick={action.onClick}
            className="mt-2"
            {...(action.href ? { asChild: true } : {})}
          >
            {action.href ? (
              <Link href={action.href}>
                {ActionIcon && <ActionIcon className="size-3.5" aria-hidden="true" />}
                {action.label}
              </Link>
            ) : (
              <>
                {ActionIcon && <ActionIcon className="size-3.5" aria-hidden="true" />}
                {action.label}
              </>
            )}
          </Button>
        )}
        {children}
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex flex-col items-center text-center",
        isCompact ? "gap-4 py-12 px-6" : "gap-3 py-16",
        className,
      )}
    >
      {Icon && iconWrap === "none" && (
        <Icon className="size-12 text-muted-foreground/30" />
      )}
      {Icon && iconWrap === "tile" && (
        <div className="size-12 rounded-2xl bg-[#3D2E22]/8 flex items-center justify-center">
          <Icon className="size-5 text-[#3D2E22]/40" />
        </div>
      )}
      {Icon && iconWrap === "circle" && (
        <span className="inline-flex size-10 items-center justify-center rounded-full bg-[#3D2E22]/10 text-[#3D2E22]">
          <Icon className="size-5" />
        </span>
      )}

      <div
        className={cn(
          isCompact && description && "max-w-[260px]",
          action?.iconOnly && "flex w-full max-w-sm items-center gap-3 text-start",
        )}
      >
        <div
          className={cn(
            isCompact && description && "space-y-1.5",
            action?.iconOnly && "min-w-0 flex-1",
          )}
        >
          <p
            className={cn(
              isCompact ? "text-sm font-medium text-foreground/70" : "text-base font-medium",
            )}
          >
            {title}
          </p>
          {description && (
            <p
              className={cn(
                isCompact
                  ? "text-xs text-muted-foreground/60 leading-relaxed"
                  : "text-sm text-muted-foreground",
                !action?.iconOnly && "max-w-xs",
              )}
            >
              {description}
            </p>
          )}
        </div>
        {action?.iconOnly && action.onClick && (
          <RetryIconButton
            label={action.label}
            onClick={action.onClick}
            loading={action.loading}
            icon={ActionIcon}
          />
        )}
      </div>

      {action && !action.iconOnly && (
        <Button
          variant="outline"
          size="sm"
          onClick={action.onClick}
          className="mt-2"
          {...(action.href ? { asChild: true } : {})}
        >
          {action.href ? <Link href={action.href}>{action.label}</Link> : action.label}
        </Button>
      )}

      {children}
    </div>
  );
}
