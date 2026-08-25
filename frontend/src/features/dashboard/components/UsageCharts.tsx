"use client";

import type { KeyboardEvent, ReactNode } from "react";
import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";
import { ChartTooltip } from "@/shared/charts/chart-utils";
import { ChartTable } from "@/shared/charts/chart-table";
import { useLiteMode } from "@/features/settings";
import { msg } from "@/shared/lib/messages";

type NameCount = { name: string; count: number };

// Warm monochrome ramp (matches --chart-2..5). The caller is anchored to the
// darkest step (--chart-1) so "me" reads as the primary slice; collaborators
// cycle the lighter steps. Access tiers map to fixed steps so the segmented
// bar and the by-access legend stay in lockstep with the table's Role column.
const OWNER_RAMP = [
  "var(--color-chart-2)",
  "var(--color-chart-3)",
  "var(--color-chart-4)",
  "var(--color-chart-5)",
];

const ACCESS_COLOR: Record<string, string> = {
  mine: "var(--color-chart-1)",
  owner: "var(--color-chart-2)",
  editor: "var(--color-chart-3)",
  viewer: "var(--color-chart-4)",
};

function accessLabel(tier: string): string {
  switch (tier) {
    case "mine":
      return msg("dashboard.role.mine");
    case "owner":
      return msg("dashboard.role_short.owner");
    case "editor":
      return msg("dashboard.role_short.editor");
    case "viewer":
      return msg("dashboard.role_short.viewer");
    default:
      return tier;
  }
}

function onActivate(e: KeyboardEvent, fn: () => void) {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fn();
  }
}

function PanelHeading({ children }: { children: ReactNode }) {
  return (
    <p className="mb-4 text-[0.6875rem] font-semibold uppercase tracking-widest text-muted-foreground">
      {children}
    </p>
  );
}

function LegendRow({
  color,
  label,
  pct,
  count,
  dir,
  title,
  onSelect,
}: {
  color: string;
  label: string;
  pct: number;
  count: number;
  dir?: "ltr" | "rtl";
  title?: string;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="flex w-full items-center gap-2 rounded-md py-0.5 text-sm transition-opacity hover:opacity-70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
    >
      <span
        className="size-2.5 shrink-0 rounded-full ring-1 ring-black/5"
        style={{ backgroundColor: color }}
        aria-hidden
      />
      <span className="min-w-0 flex-1 truncate text-start" dir={dir} title={title}>
        {label}
      </span>
      <span className="shrink-0 tabular-nums text-muted-foreground">{pct}%</span>
      <span className="w-6 shrink-0 text-end tabular-nums font-medium">{count}</span>
    </button>
  );
}

function OwnerDonut({
  owners,
  sessionUser,
  onSelect,
}: {
  owners: NameCount[];
  sessionUser: string;
  onSelect: (name: string) => void;
}) {
  const lite = useLiteMode();
  const total = owners.reduce((sum, o) => sum + o.count, 0);
  const data = owners.map((o, i) => {
    const isMe = o.name.toLowerCase() === sessionUser.toLowerCase();
    return {
      ...o,
      isMe,
      label: isMe ? msg("dashboard.owner.me") : o.name,
      fill: isMe
        ? "var(--color-chart-1)"
        : (OWNER_RAMP[i % OWNER_RAMP.length] ?? "var(--color-chart-5)"),
    };
  });

  if (lite) {
    return (
      <div>
        <PanelHeading>{msg("dashboard.analytics.by_owner")}</PanelHeading>
        <ChartTable
          rows={data}
          onRowClick={(row) => onSelect(row.name)}
          columns={[
            { key: "label", label: msg("dashboard.analytics.by_owner") },
            { key: "count", label: msg("dashboard.analytics.runs"), align: "end" },
          ]}
        />
      </div>
    );
  }

  return (
    <div>
      <PanelHeading>{msg("dashboard.analytics.by_owner")}</PanelHeading>
      <div className="relative mx-auto h-[150px] w-full max-w-[190px]">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              dataKey="count"
              nameKey="label"
              innerRadius="64%"
              outerRadius="92%"
              paddingAngle={data.length > 1 ? 2 : 0}
              stroke="none"
              cursor="pointer"
              animationDuration={400}
              onClick={(_, index) => {
                const slice = data[index];
                if (slice) onSelect(slice.name);
              }}
            >
              {data.map((d) => (
                <Cell key={d.name} fill={d.fill} />
              ))}
            </Pie>
            <Tooltip content={<ChartTooltip />} />
          </PieChart>
        </ResponsiveContainer>
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold leading-none tabular-nums">{total}</span>
          <span className="mt-1 text-[0.625rem] text-muted-foreground">
            {msg("dashboard.analytics.runs")}
          </span>
        </div>
      </div>
      <ul className="mt-5 space-y-1">
        {data.map((d) => (
          <li key={d.name}>
            <LegendRow
              color={d.fill}
              label={d.label}
              pct={total > 0 ? Math.round((d.count / total) * 100) : 0}
              count={d.count}
              dir={d.isMe ? undefined : "ltr"}
              title={d.name}
              onSelect={() => onSelect(d.name)}
            />
          </li>
        ))}
      </ul>
    </div>
  );
}

function AccessSegments({
  access,
  onSelect,
}: {
  access: NameCount[];
  onSelect: (name: string) => void;
}) {
  const total = access.reduce((sum, a) => sum + a.count, 0);

  return (
    <div>
      <PanelHeading>{msg("dashboard.analytics.by_access")}</PanelHeading>
      <div className="flex h-3.5 w-full gap-0.5 overflow-hidden rounded-full">
        {access.map((a) => {
          const pct = total > 0 ? (a.count / total) * 100 : 0;
          return (
            <button
              key={a.name}
              type="button"
              title={`${accessLabel(a.name)} · ${a.count}`}
              aria-label={accessLabel(a.name)}
              onClick={() => onSelect(a.name)}
              onKeyDown={(e) => onActivate(e, () => onSelect(a.name))}
              className="h-full min-w-[3px] transition-opacity duration-300 hover:opacity-75 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
              style={{ width: `${pct}%`, backgroundColor: ACCESS_COLOR[a.name] ?? "var(--color-chart-5)" }}
            />
          );
        })}
      </div>
      <ul className="mt-5 space-y-1">
        {access.map((a) => (
          <li key={a.name}>
            <LegendRow
              color={ACCESS_COLOR[a.name] ?? "var(--color-chart-5)"}
              label={accessLabel(a.name)}
              pct={total > 0 ? Math.round((a.count / total) * 100) : 0}
              count={a.count}
              onSelect={() => onSelect(a.name)}
            />
          </li>
        ))}
      </ul>
    </div>
  );
}

type SharingBreakdownProps = {
  owners: NameCount[];
  access: NameCount[];
  showOwners: boolean;
  showAccess: boolean;
  sessionUser: string;
  onOwnerSelect: (name: string) => void;
  onAccessSelect: (name: string) => void;
};

// Two composition charts side-by-side: a donut for ownership share and a 100%
// segmented bar for access tiers. Self-collapsing grid — a lone panel stretches
// full width; both panels split. Replaces the old trio of identical bar lists.
export default function SharingBreakdown({
  owners,
  access,
  showOwners,
  showAccess,
  sessionUser,
  onOwnerSelect,
  onAccessSelect,
}: SharingBreakdownProps) {
  return (
    <div
      className="grid gap-x-10 gap-y-8"
      style={{ gridTemplateColumns: "repeat(auto-fit, minmax(min(220px, 100%), 1fr))" }}
    >
      {showOwners && <OwnerDonut owners={owners} sessionUser={sessionUser} onSelect={onOwnerSelect} />}
      {showAccess && <AccessSegments access={access} onSelect={onAccessSelect} />}
    </div>
  );
}
