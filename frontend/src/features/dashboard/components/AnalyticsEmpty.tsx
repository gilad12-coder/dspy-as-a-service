"use client";

import { ChartBar, Database, WarningCircle } from "@/shared/ui/icons";
import type { Icon } from "@/shared/ui/icons";
import { EmptyState } from "@/shared/ui/empty-state";
import { FadeIn } from "@/shared/ui/motion";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";

interface AnalyticsEmptyProps {
  variant?: "no-data" | "no-results" | "loading-error";
  onClearFilters?: () => void;
  onRetry?: () => void;
}

export function AnalyticsEmpty({
  variant = "no-data",
  onClearFilters,
  onRetry,
}: AnalyticsEmptyProps) {
  const configs: Record<
    NonNullable<AnalyticsEmptyProps["variant"]>,
    {
      icon: Icon;
      title: string;
      description?: string;
      action?: {
        label: string;
        onClick?: () => void;
        href?: string;
        icon?: Icon;
        iconOnly?: boolean;
      };
    }
  > = {
    "no-data": {
      icon: Database,
      title: msg("auto.features.dashboard.components.analyticsempty.literal.1"),
    },
    "no-results": {
      icon: ChartBar,
      title: msg("auto.features.dashboard.components.analyticsempty.literal.2"),
      description: formatMsg("auto.features.dashboard.components.analyticsempty.template.2", {
        p1: TERMS.optimizationPlural,
      }),
      action: onClearFilters
        ? {
            label: msg("auto.features.dashboard.components.analyticsempty.1"),
            onClick: onClearFilters,
          }
        : undefined,
    },
    "loading-error": {
      icon: WarningCircle,
      title: msg("auto.features.dashboard.components.analyticsempty.literal.3"),
      description: msg("auto.features.dashboard.components.analyticsempty.literal.4"),
      action: onRetry
        ? {
            label: msg("auto.features.dashboard.components.analyticsempty.2"),
            onClick: onRetry,
            iconOnly: true,
          }
        : undefined,
    },
  };

  const config = configs[variant];
  const isNoData = variant === "no-data";

  return (
    <FadeIn>
      <EmptyState
        variant={isNoData ? "page" : "list"}
        icon={config.icon}
        iconWrap={isNoData ? "tile" : "none"}
        title={config.title}
        description={config.description}
        action={config.action}
        className={isNoData ? undefined : "min-h-[40vh] justify-center"}
      />
    </FadeIn>
  );
}
