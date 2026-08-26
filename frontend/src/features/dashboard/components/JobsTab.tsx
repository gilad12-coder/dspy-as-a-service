import {
  CaretLeft,
  CaretRight,
  Crown,
  Eye,
  PaperPlaneTilt,
  PencilSimple,
  Plus,
  ShieldCheck,
} from "@/shared/ui/icons";
import { Badge } from "@/shared/ui/primitives/badge";
import { Button } from "@/shared/ui/primitives/button";
import { Card, CardContent } from "@/shared/ui/primitives/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/shared/ui/primitives/table";
import { EmptyState } from "@/shared/ui/empty-state";
import { InlineErrorRow } from "@/shared/ui/inline-error-row";
import { PingDot } from "@/shared/ui/ping-dot";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { ExportTableMenu } from "@/shared/ui/export-table-menu";
import type { useColumnResize } from "@/shared/ui/excel-filter";
import {
  ColumnHeader,
  ResetColumnsButton,
  ResetFiltersButton,
  type SortDir,
} from "@/shared/ui/excel-filter";
import { formatDate, formatId, formatRelativeTime, moduleLabel } from "@/shared/lib";
import { ACTIVE_STATUSES } from "@/shared/constants/job-status";
import { LiveElapsed } from "./LiveElapsed";
import { PhoneJobList } from "./PhoneJobList";
import type { OptimizationSummaryResponse, PaginatedJobsResponse } from "@/shared/types/api";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { TERMS } from "@/shared/lib/terms";
import { useIsPhone } from "@/shared/hooks/use-device-class";
import { FETCH_PAGE_SIZE } from "../constants";
import { formatScore, typeBadge } from "../lib/status-badges";
import { StatusBadge } from "@/shared/ui/status-badge";

// Default widths sized so all columns fit on screen at once (no horizontal
// scroll) while each header still shows its full Hebrew label. This works in
// tandem with the compact density overrides on the <Table> below (smaller
// header font, tighter padding, smaller sort/filter icons) — without those
// the labels would clip at these widths. Users can still resize individually.
const DEFAULT_COL_WIDTHS: Record<string, number> = {
  optimization_id: 86,
  name: 104,
  username: 94,
  role: 92,
  optimization_type: 80,
  status: 94,
  module_name: 94,
  dataset_rows: 72,
  created_at: 94,
  elapsed_seconds: 66,
  optimized_test_metric: 94,
};

type ColResize = ReturnType<typeof useColumnResize>;

type ShareRole = "viewer" | "editor" | "owner";

// Icon + tooltip for the caller's tier on a run. Icons denote state only (per
// the design system); the tooltip names the tier in plain Hebrew and doubles
// as the accessible label. A null role means the caller's own run — the union
// only ever yields owned or granted rows — so it earns the ownership crown
// rather than a blank dash.
function RoleBadge({ role }: { role?: ShareRole | null }) {
  const tier = role ?? "owned";
  const Icon =
    tier === "viewer"
      ? Eye
      : tier === "editor"
        ? PencilSimple
        : tier === "owner"
          ? ShieldCheck
          : Crown;
  const tip =
    tier === "viewer"
      ? msg("dashboard.role.viewer")
      : tier === "editor"
        ? msg("dashboard.role.editor")
        : tier === "owner"
          ? msg("dashboard.role.owner")
          : msg("dashboard.role.owned");
  return (
    <TooltipButton tooltip={tip}>
      <span
        tabIndex={0}
        aria-label={tip}
        className="inline-flex rounded text-muted-foreground/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
      >
        <Icon className="size-3.5" />
      </span>
    </TooltipButton>
  );
}

type JobsTabProps = {
  data: PaginatedJobsResponse | null;
  loading: boolean;
  error: string | null;
  filteredItems: OptimizationSummaryResponse[];
  activeCount: number;
  clearAllFilters: () => void;
  colResize: ColResize;
  sortKey: string;
  sortDir: SortDir;
  toggleSort: (key: string) => void;
  filters: Record<string, Set<string>>;
  setColumnFilter: (col: string, values: Set<string>) => void;
  openFilter: string | null;
  setOpenFilter: (col: string | null) => void;
  filterOptions: Record<string, Array<{ value: string; label: string }>>;
  showSharedColumns: boolean;
  sessionUser: string;
  selectedIds: Set<string>;
  toggleRowSelected: (id: string) => void;
  selectablePageIds: string[];
  pageAllSelected: boolean;
  pageSomeSelected: boolean;
  togglePageSelection: () => void;
  pageOffset: number;
  setPageOffset: React.Dispatch<React.SetStateAction<number>>;
  onOpenJob: (id: string) => void;
};

export function JobsTab({
  data,
  loading,
  error,
  filteredItems,
  activeCount,
  clearAllFilters,
  colResize,
  sortKey,
  sortDir,
  toggleSort,
  filters,
  setColumnFilter,
  openFilter,
  setOpenFilter,
  filterOptions,
  showSharedColumns,
  sessionUser,
  selectedIds,
  toggleRowSelected,
  selectablePageIds,
  pageAllSelected,
  pageSomeSelected,
  togglePageSelection,
  pageOffset,
  setPageOffset,
  onOpenJob,
}: JobsTabProps) {
  const rtl = getActiveDir() === "rtl";
  const isPhone = useIsPhone();
  const PrevIcon = rtl ? CaretRight : CaretLeft;
  const NextIcon = rtl ? CaretLeft : CaretRight;

  return (
    <Card className="overflow-hidden border-border/60">
      <CardContent className="px-3 pt-4 sm:px-6 sm:pt-5">
        <div className="mb-3 flex min-h-[44px] items-center gap-2 max-lg:[&_button]:size-[44px] lg:min-h-0">
          {filteredItems.length > 0 && (
            <span className="text-[0.6875rem] text-muted-foreground tabular-nums">
              {filteredItems.length}
              {msg("auto.features.dashboard.components.jobstab.3")}
            </span>
          )}
          {activeCount > 0 && (
            <Badge variant="secondary" className="text-xs">
              {activeCount}
              {msg("auto.features.dashboard.components.jobstab.1")}
            </Badge>
          )}
          <ResetFiltersButton filters={{ activeCount, clearAll: clearAllFilters }} />
          <ResetColumnsButton resize={colResize} />
          {!isPhone && filteredItems.length > 0 && (
            <ExportTableMenu
              iconOnly
              className="ms-auto"
              getData={() => {
                const cols = [
                  "optimization_id",
                  "name",
                  ...(showSharedColumns ? ["username", "role"] : []),
                  "optimization_type",
                  "status",
                  "module_name",
                  "dataset_rows",
                  "created_at",
                  "elapsed_seconds",
                  "optimized_test_metric",
                ];
                return {
                  columns: cols,
                  rows: filteredItems.map((job) => {
                    const rec = job as unknown as Record<string, unknown>;
                    return Object.fromEntries(cols.map((c) => [c, rec[c]]));
                  }),
                  filename: "optimizations",
                };
              }}
            />
          )}
        </div>

        {error && <InlineErrorRow message={error} className="mb-4" />}

        {!loading && data && filteredItems.length === 0 && data.total === 0 && (
          <EmptyState
            icon={PaperPlaneTilt}
            iconWrap="tile"
            title={`${msg("auto.features.dashboard.components.jobstab.4")}${TERMS.optimizationPlural}`}
            description={msg("auto.features.dashboard.components.jobstab.5")}
            action={
              isPhone ? undefined : { label: TERMS.notificationNewOpt, href: "/submit", icon: Plus }
            }
          />
        )}

        {filteredItems.length > 0 && isPhone && (
          <PhoneJobList
            items={filteredItems}
            showOwner={showSharedColumns}
            sessionUser={sessionUser}
            onOpenJob={onOpenJob}
          />
        )}

        {filteredItems.length > 0 && !isPhone && (
          <div
            className="overflow-x-auto rounded-2xl border border-border/40 bg-card/60"
            data-tutorial="dashboard-table"
          >
            <Table
              style={{ minWidth: "640px" }}
              className="table-stack no-copy-underline [&_thead_th]:ps-1 [&_thead_th]:pe-2 [&_thead_th]:py-2 [&_thead_th]:text-[0.6875rem] [&_thead_th_button]:px-1 [&_thead_svg]:size-2.5 [&_tbody_td]:px-1.5"
            >
              <TableHeader className="bg-muted/20 [&_tr]:border-b-border/40">
                <TableRow>
                  <TableHead className="w-12 px-0 text-center">
                    <label className="inline-grid size-[44px] cursor-pointer place-items-center has-[:disabled]:cursor-not-allowed has-[:disabled]:opacity-50">
                      <input
                        type="checkbox"
                        aria-label={msg("auto.features.dashboard.components.jobstab.literal.1")}
                        className="size-4 cursor-pointer accent-primary"
                        checked={pageAllSelected}
                        ref={(el) => {
                          if (el) el.indeterminate = pageSomeSelected;
                        }}
                        disabled={selectablePageIds.length === 0}
                        onChange={togglePageSelection}
                        onClick={(e) => e.stopPropagation()}
                      />
                    </label>
                  </TableHead>
                  <ColumnHeader
                    label={msg("auto.features.dashboard.components.jobstab.template.1")}
                    sortKey="optimization_id"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    filterCol="optimization_id"
                    filterOptions={filterOptions.optimization_id}
                    filters={filters}
                    onFilter={setColumnFilter}
                    openFilter={openFilter}
                    setOpenFilter={setOpenFilter}
                    width={
                      colResize.widths["optimization_id"] ?? DEFAULT_COL_WIDTHS.optimization_id
                    }
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("auto.features.dashboard.components.jobstab.literal.9")}
                    sortKey="name"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    filterCol="name"
                    filterOptions={filterOptions.name}
                    filters={filters}
                    onFilter={setColumnFilter}
                    openFilter={openFilter}
                    setOpenFilter={setOpenFilter}
                    width={colResize.widths["name"] ?? DEFAULT_COL_WIDTHS.name}
                    onResize={colResize.setColumnWidth}
                  />
                  {showSharedColumns && (
                    <>
                      <ColumnHeader
                        label={msg("dashboard.col.owner")}
                        sortKey="username"
                        currentSort={sortKey}
                        sortDir={sortDir}
                        onSort={toggleSort}
                        filterCol="username"
                        filterOptions={filterOptions.username}
                        filters={filters}
                        onFilter={setColumnFilter}
                        openFilter={openFilter}
                        setOpenFilter={setOpenFilter}
                        width={colResize.widths["username"] ?? DEFAULT_COL_WIDTHS.username}
                        onResize={colResize.setColumnWidth}
                      />
                      <ColumnHeader
                        label={msg("dashboard.col.role")}
                        sortKey="role"
                        currentSort={sortKey}
                        sortDir={sortDir}
                        onSort={toggleSort}
                        filterCol="role"
                        filterOptions={filterOptions.role}
                        filters={filters}
                        onFilter={setColumnFilter}
                        openFilter={openFilter}
                        setOpenFilter={setOpenFilter}
                        width={colResize.widths["role"] ?? DEFAULT_COL_WIDTHS.role}
                        onResize={colResize.setColumnWidth}
                      />
                    </>
                  )}
                  <ColumnHeader
                    label={msg("auto.features.dashboard.components.jobstab.literal.2")}
                    sortKey="optimization_type"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    filterCol="optimization_type"
                    filterOptions={filterOptions.optimization_type}
                    filters={filters}
                    onFilter={setColumnFilter}
                    openFilter={openFilter}
                    setOpenFilter={setOpenFilter}
                    width={
                      colResize.widths["optimization_type"] ?? DEFAULT_COL_WIDTHS.optimization_type
                    }
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("auto.features.dashboard.components.jobstab.literal.3")}
                    sortKey="status"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    filterCol="status"
                    filterOptions={filterOptions.status}
                    filters={filters}
                    onFilter={setColumnFilter}
                    openFilter={openFilter}
                    setOpenFilter={setOpenFilter}
                    width={colResize.widths["status"] ?? DEFAULT_COL_WIDTHS.status}
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("auto.features.dashboard.components.jobstab.literal.4")}
                    sortKey="module_name"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    filterCol="module_name"
                    filterOptions={filterOptions.module_name}
                    filters={filters}
                    onFilter={setColumnFilter}
                    openFilter={openFilter}
                    setOpenFilter={setOpenFilter}
                    width={colResize.widths["module_name"] ?? DEFAULT_COL_WIDTHS.module_name}
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("auto.features.dashboard.components.jobstab.literal.5")}
                    sortKey="dataset_rows"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    width={colResize.widths["dataset_rows"] ?? DEFAULT_COL_WIDTHS.dataset_rows}
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("auto.features.dashboard.components.jobstab.literal.6")}
                    sortKey="created_at"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    width={colResize.widths["created_at"] ?? DEFAULT_COL_WIDTHS.created_at}
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("auto.features.dashboard.components.jobstab.literal.7")}
                    sortKey="elapsed_seconds"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    width={
                      colResize.widths["elapsed_seconds"] ?? DEFAULT_COL_WIDTHS.elapsed_seconds
                    }
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("auto.features.dashboard.components.jobstab.literal.8")}
                    sortKey="optimized_test_metric"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    width={
                      colResize.widths["optimized_test_metric"] ??
                      DEFAULT_COL_WIDTHS.optimized_test_metric
                    }
                    onResize={colResize.setColumnWidth}
                  />
                  {/* Spacer over the per-row open chevron. */}
                  <TableHead aria-hidden="true" className="w-8" />
                </TableRow>
              </TableHeader>
              <TableBody className="transition-opacity duration-200">
                {filteredItems.map((job, idx) => {
                  const isSelected = selectedIds.has(job.optimization_id);
                  return (
                    <TableRow
                      key={job.optimization_id}
                      data-selected={isSelected}
                      className="group border-border/30 transition-colors duration-150 hover:bg-muted/50 data-[selected=true]:bg-primary/[0.08] data-[selected=true]:hover:bg-primary/[0.12] cursor-pointer [&_td:first-child]:cursor-default"
                      style={{
                        animation: `fadeSlideIn 0.25s ease-out ${idx * 0.03}s both`,
                      }}
                      onClick={(e) => {
                        // Convenience target only — the ID button remains the
                        // accessible/keyboard path. The checkbox cell is
                        // excluded so a missed checkbox click can't navigate.
                        const target = e.target as HTMLElement;
                        if (target.closest("button, a, input")) return;
                        const td = target.closest("td");
                        if (!td || td === td.parentElement?.firstElementChild) return;
                        onOpenJob(job.optimization_id);
                      }}
                    >
                      <TableCell className="w-12 px-0 text-center">
                        <label className="inline-grid size-[44px] cursor-pointer place-items-center">
                          <input
                            type="checkbox"
                            aria-label={formatMsg(
                              "auto.features.dashboard.components.jobstab.template.2",
                              { p1: TERMS.optimization, p2: job.optimization_id },
                            )}
                            className="size-4 cursor-pointer accent-primary"
                            checked={isSelected}
                            onClick={(e) => e.stopPropagation()}
                            onChange={() => toggleRowSelected(job.optimization_id)}
                          />
                        </label>
                      </TableCell>
                      <TableCell
                        className="px-2 max-w-[100px]"
                        data-label={msg("auto.features.dashboard.components.jobstab.template.1")}
                      >
                        {/* ``min-w-0`` lets the flex item shrink so the
                            cell's overflow-hidden + text-ellipsis can
                            actually fire on the button below; without it
                            the flex child claims its content's intrinsic
                            width and overflows past the cell boundary. */}
                        <div className="flex items-center gap-1.5 min-w-0">
                          {ACTIVE_STATUSES.has(job.status) && <PingDot className="shrink-0" />}
                          {/* ``dir="ltr"`` keeps hex IDs that start with a
                              digit from bidi-reordering in RTL locales. */}
                          <button
                            type="button"
                            dir="ltr"
                            onClick={() => onOpenJob(job.optimization_id)}
                            className="font-mono text-xs text-primary truncate min-w-0 rounded-md cursor-pointer underline-offset-2 group-hover:underline focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
                            aria-label={formatMsg(
                              "auto.features.dashboard.components.jobstab.template.3",
                              { p1: TERMS.optimization },
                            )}
                          >
                            {formatId(job.optimization_id)}
                          </button>
                        </div>
                      </TableCell>
                      <TableCell
                        className="px-2 max-w-[140px] text-sm truncate overflow-hidden"
                        title={job.name ?? ""}
                        dir="auto"
                        data-label={msg("auto.features.dashboard.components.jobstab.literal.9")}
                      >
                        {job.name ? (
                          <span className="text-foreground">{job.name}</span>
                        ) : (
                          <span className="text-muted-foreground/60">—</span>
                        )}
                      </TableCell>
                      {showSharedColumns && (
                        <>
                          <TableCell
                            className="px-2 max-w-[120px] text-sm truncate overflow-hidden"
                            title={job.username ?? ""}
                            data-label={msg("dashboard.col.owner")}
                          >
                            {job.username ? (
                              job.username.toLowerCase() === sessionUser.toLowerCase() ? (
                                <span className="font-semibold text-foreground">
                                  {msg("dashboard.owner.me")}
                                </span>
                              ) : (
                                <span className="font-semibold text-foreground" dir="ltr">
                                  {job.username}
                                </span>
                              )
                            ) : (
                              <span className="text-muted-foreground/40">—</span>
                            )}
                          </TableCell>
                          <TableCell className="px-2" data-label={msg("dashboard.col.role")}>
                            <RoleBadge role={job.role} />
                          </TableCell>
                        </>
                      )}
                      <TableCell
                        className="px-2 truncate overflow-hidden"
                        data-label={msg("auto.features.dashboard.components.jobstab.literal.2")}
                      >
                        {typeBadge(job.optimization_type)}
                      </TableCell>
                      <TableCell
                        className="px-2 truncate overflow-hidden"
                        data-label={msg("auto.features.dashboard.components.jobstab.literal.3")}
                      >
                        <StatusBadge status={job.status} compact />
                      </TableCell>
                      <TableCell
                        className="px-2 max-w-[120px] text-sm truncate overflow-hidden"
                        title={job.module_name ?? ""}
                        data-label={msg("auto.features.dashboard.components.jobstab.literal.4")}
                      >
                        {moduleLabel(job.module_name)}
                      </TableCell>
                      <TableCell
                        className="px-2 text-sm tabular-nums truncate overflow-hidden"
                        title={String(job.dataset_rows ?? "")}
                        data-label={msg("auto.features.dashboard.components.jobstab.literal.5")}
                      >
                        {job.dataset_rows ?? "-"}
                      </TableCell>
                      <TableCell
                        className="px-2 text-xs text-muted-foreground truncate overflow-hidden whitespace-nowrap"
                        title={formatDate(job.created_at)}
                        data-label={msg("auto.features.dashboard.components.jobstab.literal.6")}
                      >
                        {formatRelativeTime(job.created_at)}
                      </TableCell>
                      <TableCell
                        className="px-2 text-xs tabular-nums truncate overflow-hidden whitespace-nowrap"
                        data-label={msg("auto.features.dashboard.components.jobstab.literal.7")}
                      >
                        <LiveElapsed
                          startedAt={job.started_at}
                          createdAt={job.created_at}
                          elapsedSeconds={job.elapsed_seconds}
                          isActive={ACTIVE_STATUSES.has(job.status)}
                        />
                      </TableCell>
                      <TableCell
                        className="px-2 truncate overflow-hidden"
                        data-label={msg("auto.features.dashboard.components.jobstab.literal.8")}
                      >
                        {formatScore(job)}
                      </TableCell>
                      {/* Open-affordance chevron: always faintly visible so the
                          row reads as navigable, brightening and nudging on
                          hover. Hidden in the stacked mobile layout, where the
                          !important beats table-stack's display:block on td. */}
                      <TableCell className="w-8 pe-3 max-[900px]:!hidden">
                        <CaretRight
                          aria-hidden="true"
                          className="size-3.5 text-muted-foreground/40 transition-[color,transform] duration-150 group-hover:translate-x-0.5 group-hover:text-primary rtl:rotate-180 rtl:group-hover:-translate-x-0.5"
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        )}

        {data && data.total > FETCH_PAGE_SIZE && (
          <div className="flex items-center justify-center gap-3 pt-5 border-t border-border/50 mt-4">
            <Button
              variant="outline"
              size="sm"
              disabled={pageOffset === 0 || loading}
              onClick={() => setPageOffset(Math.max(0, pageOffset - FETCH_PAGE_SIZE))}
              className="min-h-[44px] gap-1 lg:min-h-0"
            >
              <PrevIcon className="size-3.5" />
              {msg("auto.features.dashboard.components.jobstab.11")}
            </Button>
            <span className="text-sm text-muted-foreground tabular-nums px-3 py-1 rounded-md bg-muted/50">
              {Math.floor(pageOffset / FETCH_PAGE_SIZE) + 1} /{" "}
              {Math.max(1, Math.ceil(data.total / FETCH_PAGE_SIZE))}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={pageOffset + FETCH_PAGE_SIZE >= data.total || loading}
              onClick={() => setPageOffset(pageOffset + FETCH_PAGE_SIZE)}
              className="min-h-[44px] gap-1 lg:min-h-0"
            >
              {msg("auto.features.dashboard.components.jobstab.12")}
              <NextIcon className="size-3.5" />
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
