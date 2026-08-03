import type { ReactNode } from "react";
import { Users } from "@/shared/ui/icons";
import { AnimatedNumber } from "@/shared/ui/motion";
import { Card } from "@/shared/ui/primitives/card";
import type { DashboardStats } from "../lib/get-dashboard-stats";
import { msg } from "@/shared/lib/messages";

type DashboardHeaderProps = {
  stats: DashboardStats;
};

type StatCardProps = {
  label: string;
  value: number;
  accent?: "default" | "warning" | "success" | "danger";
  pulse?: boolean;
  icon?: ReactNode;
};

const ACCENT_TEXT: Record<NonNullable<StatCardProps["accent"]>, string> = {
  default: "text-foreground",
  warning: "text-[var(--warning)]",
  success: "text-emerald-600",
  danger: "text-red-600",
};

const ACCENT_DOT: Record<NonNullable<StatCardProps["accent"]>, string> = {
  default: "bg-foreground/25",
  warning: "bg-[var(--warning)]",
  success: "bg-emerald-500",
  danger: "bg-red-500",
};

function StatCard({ label, value, accent = "default", pulse = false, icon }: StatCardProps) {
  return (
    <Card className="group/stat min-w-0 flex-[1_1_11rem] gap-3.5 p-5 xl:flex-[1_1_9rem]">
      <div className="flex items-center gap-2">
        {icon ?? (
          <span
            className={`size-1.5 rounded-full ${ACCENT_DOT[accent]} ${pulse ? "animate-pulse" : ""}`}
            aria-hidden
          />
        )}
        <p className="text-[0.625rem] font-semibold uppercase tracking-[0.14em] text-muted-foreground/70">
          {label}
        </p>
      </div>
      <p
        className={`text-3xl sm:text-[2rem] font-bold leading-none tracking-tight tabular-nums ${ACCENT_TEXT[accent]}`}
      >
        <AnimatedNumber value={value} />
      </p>
    </Card>
  );
}

export function DashboardHeader({ stats }: DashboardHeaderProps) {
  return (
    <>
      {stats && (
        <div
          className="flex flex-wrap gap-3 sm:gap-4"
          data-tutorial="dashboard-kpis"
        >
          <StatCard
            label={msg("auto.features.dashboard.components.dashboardheader.3")}
            value={stats.total}
          />
          <StatCard
            label={msg("auto.features.dashboard.components.dashboardheader.4")}
            value={stats.running}
            accent={stats.running > 0 ? "warning" : "default"}
            pulse={stats.running > 0}
          />
          <StatCard
            label={msg("auto.features.dashboard.components.dashboardheader.6")}
            value={stats.success}
            accent={stats.success > 0 ? "success" : "default"}
          />
          <StatCard
            label={msg("auto.features.dashboard.components.dashboardheader.7")}
            value={stats.failed}
            accent={stats.failed > 0 ? "danger" : "default"}
          />
          {stats.shared > 0 && (
            <StatCard
              label={msg("dashboard.stat.shared")}
              value={stats.shared}
              icon={<Users className="size-3.5 text-muted-foreground/60" aria-hidden />}
            />
          )}
        </div>
      )}
    </>
  );
}
