import * as React from "react";
import { X } from "lucide-react";
import { getStatusLabel } from "@/shared/constants/job-status";
import { msg } from "@/shared/lib/messages";
import type { UseAnalyticsFiltersReturn } from "../hooks/use-analytics-filters";

export function AnalyticsFilterChips({
  filters,
  sessionUser,
}: {
  filters: Pick<
    UseAnalyticsFiltersReturn,
    | "model"
    | "status"
    | "jobId"
    | "date"
    | "owner"
    | "access"
    | "setModel"
    | "setStatus"
    | "setJobId"
    | "setDate"
    | "setOwner"
    | "setAccess"
  >;
  sessionUser: string;
}) {
  const {
    model,
    status,
    jobId,
    date,
    owner,
    access,
    setModel,
    setStatus,
    setJobId,
    setDate,
    setOwner,
    setAccess,
  } = filters;
  const hasFilters = jobId || date || owner || access || model !== "all" || status !== "all";
  if (!hasFilters) return null;

  const clearAllFilters = () => {
    setJobId(null);
    setDate(null);
    setOwner(null);
    setAccess(null);
    setModel("all");
    setStatus("all");
  };

  const ownerIsMe = Boolean(owner) && owner!.toLowerCase() === sessionUser.toLowerCase();
  const accessLabels: Record<string, string> = {
    mine: msg("dashboard.role.mine"),
    owner: msg("dashboard.role_short.owner"),
    editor: msg("dashboard.role_short.editor"),
    viewer: msg("dashboard.role_short.viewer"),
  };

  return (
    <div className="flex items-center gap-2 flex-wrap">
      {access && (
        <FilterChip
          label={accessLabels[access] ?? access}
          ariaLabel={msg("dashboard.analytics.access_filter_clear")}
          onClear={() => setAccess(null)}
        />
      )}
      {owner && (
        <FilterChip
          dir={ownerIsMe ? undefined : "ltr"}
          label={ownerIsMe ? msg("dashboard.owner.me") : owner}
          title={owner}
          truncate={!ownerIsMe}
          ariaLabel={msg("dashboard.analytics.owner_filter_clear")}
          onClear={() => setOwner(null)}
        />
      )}
      {jobId && (
        <FilterChip
          dir="ltr"
          label={`${jobId.slice(0, 8)}...`}
          ariaLabel={msg("auto.features.dashboard.components.analyticstab.literal.1")}
          onClear={() => setJobId(null)}
        />
      )}
      {date && (
        <FilterChip
          label={new Date(date).toLocaleDateString("he-IL", {
            day: "numeric",
            month: "short",
            year: "numeric",
          })}
          ariaLabel={msg("auto.features.dashboard.components.analyticstab.literal.2")}
          onClear={() => setDate(null)}
        />
      )}
      {model !== "all" && (
        <FilterChip
          dir="ltr"
          label={model}
          title={model}
          truncate
          ariaLabel={msg("auto.features.dashboard.components.analyticstab.literal.4")}
          onClear={() => setModel("all")}
        />
      )}
      {status !== "all" && (
        <FilterChip
          label={getStatusLabel(status)}
          ariaLabel={msg("auto.features.dashboard.components.analyticstab.literal.5")}
          onClear={() => setStatus("all")}
        />
      )}
      <button
        onClick={clearAllFilters}
        className="text-[0.625rem] text-[#3D2E22]/40 hover:text-[#3D2E22]/70 transition-colors cursor-pointer ms-0.5"
      >
        {msg("auto.features.dashboard.components.analyticstab.3")}
      </button>
    </div>
  );
}

function FilterChip({
  label,
  ariaLabel,
  onClear,
  dir,
  title,
  truncate,
}: {
  label: string;
  ariaLabel: string;
  onClear: () => void;
  dir?: "ltr" | "rtl";
  title?: string;
  truncate?: boolean;
}) {
  return (
    <span className="group inline-flex items-center gap-1.5 rounded-lg bg-[#3D2E22]/[0.06] border border-[#3D2E22]/10 pe-1 ps-2.5 py-1 transition-all duration-150 hover:bg-[#3D2E22]/[0.1] hover:border-[#3D2E22]/20">
      <span
        className={`text-[0.6875rem] font-medium text-[#3D2E22]/80 ${truncate ? "font-mono truncate max-w-[140px]" : ""}`}
        dir={dir}
        title={title}
      >
        {label}
      </span>
      <button
        onClick={onClear}
        className="close-button"
        style={
          {
            "--close-btn-size": "20px",
            "--close-btn-radius": "6px",
            "--close-btn-icon": "12px",
          } as React.CSSProperties
        }
        aria-label={ariaLabel}
      >
        <X />
      </button>
    </span>
  );
}
