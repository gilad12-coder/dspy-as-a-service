import { toast } from "react-toastify";
import {
  ChevronLeft,
  ChevronRight,
  Crown,
  Eye,
  ExternalLink,
  Pencil,
  Plus,
  Send,
  ShieldCheck,
} from "lucide-react";
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
import type { useColumnResize } from "@/shared/ui/excel-filter";
import { ColumnHeader, ResetColumnsButton, type SortDir } from "@/shared/ui/excel-filter";
import { formatDate, formatId, formatRelativeTime, moduleLabel } from "@/shared/lib";
import { ACTIVE_STATUSES } from "@/shared/constants/job-status";
import { LiveElapsed } from "./LiveElapsed";
import type { OptimizationSummaryResponse, PaginatedJobsResponse } from "@/shared/types/api";
import { formatMsg, msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";
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
        ? Pencil
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
  return (
    <Card className="border-border/60">
      <CardContent className="pt-5">
        <div className="flex items-center gap-2 mb-3">
          {activeCount > 0 && (
            <>
              <Badge variant="secondary" className="text-xs">
                {activeCount}
                {msg("auto.features.dashboard.components.jobstab.1")}
              </Badge>
              <button
                type="button"
                onClick={clearAllFilters}
                className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
              >
                {msg("auto.features.dashboard.components.jobstab.2")}
              </button>
            </>
          )}
          <ResetColumnsButton resize={colResize} />
          {filteredItems.length > 0 && (
            <span className="text-[0.6875rem] text-muted-foreground tabular-nums ms-auto">
              {filteredItems.length}
              {msg("auto.features.dashboard.components.jobstab.3")}
            </span>
          )}
        </div>

        {error && <InlineErrorRow message={error} className="mb-4" />}

        {!loading && data && filteredItems.length === 0 && data.total === 0 && (
          <EmptyState
            variant="list"
            icon={Send}
            title={`${msg("auto.features.dashboard.components.jobstab.4")}${TERMS.optimizationPlural}`}
            description={msg("auto.features.dashboard.components.jobstab.5")}
            action={{ label: TERMS.notificationNewOpt, href: "/submit", icon: Plus }}
          />
        )}

        {filteredItems.length > 0 && (
          <div
            className="overflow-x-auto rounded-2xl border border-border/40 bg-card/60"
            data-tutorial="dashboard-table"
          >
            <Table
              style={{ minWidth: "640px" }}
              className="table-stack [&_thead_th]:ps-1 [&_thead_th]:pe-2 [&_thead_th]:py-2 [&_thead_th]:text-[0.6875rem] [&_thead_th_button]:px-1 [&_thead_svg]:size-2.5 [&_tbody_td]:px-1.5"
            >
              <TableHeader className="bg-muted/20 [&_tr]:border-b-border/40">
                <TableRow>
                  <TableHead className="w-10 px-3">
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
                    width={colResize.widths["optimization_id"] ?? DEFAULT_COL_WIDTHS.optimization_id}
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
                    width={colResize.widths["optimization_type"] ?? DEFAULT_COL_WIDTHS.optimization_type}
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
                    width={colResize.widths["elapsed_seconds"] ?? DEFAULT_COL_WIDTHS.elapsed_seconds}
                    onResize={colResize.setColumnWidth}
                  />
                  <ColumnHeader
                    label={msg("auto.features.dashboard.components.jobstab.literal.8")}
                    sortKey="optimized_test_metric"
                    currentSort={sortKey}
                    sortDir={sortDir}
                    onSort={toggleSort}
                    width={colResize.widths["optimized_test_metric"] ?? DEFAULT_COL_WIDTHS.optimized_test_metric}
                    onResize={colResize.setColumnWidth}
                  />
                  <TableHead className="w-10" />
                </TableRow>
              </TableHeader>
              <TableBody className="transition-opacity duration-200">
                {filteredItems.map((job, idx) => {
                  const isSelected = selectedIds.has(job.optimization_id);
                  return (
                    <TableRow
                      key={job.optimization_id}
                      data-selected={isSelected}
                      className="group border-border/30 transition-colors duration-150 data-[selected=true]:bg-primary/[0.08] hover:bg-foreground/[0.025] data-[selected=true]:hover:bg-primary/[0.12] cursor-pointer [&_td:first-child]:cursor-default [&_td:last-child]:cursor-default focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
                      style={{
                        animation: `fadeSlideIn 0.25s ease-out ${idx * 0.03}s both`,
                      }}
                      onClick={(e) => {
                        const target = e.target as HTMLElement;
                        if (target.closest("button, a, input")) return;
                        const td = target.closest("td");
                        if (!td) return;
                        const parent = td.parentElement;
                        if (td === parent?.lastElementChild) return;
                        if (td === parent?.firstElementChild) return;
                        const text = td.textContent?.trim();
                        if (!text) return;
                        navigator.clipboard
                          .writeText(text)
                          .then(() => toast.success(msg("clipboard.copied")))
                          .catch(() => {});
                      }}
                      data-tutorial="job-link"
                    >
                      <TableCell className="w-10 px-3">
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
                      </TableCell>
                      <TableCell
                        className="px-2 max-w-[100px]"
                        title={job.optimization_id}
                        data-label={msg("auto.features.dashboard.components.jobstab.template.1")}
                      >
                        {/* ``min-w-0`` lets the flex item shrink so the
                            cell's overflow-hidden + text-ellipsis can
                            actually fire on the span below; without it
                            the flex child claims its content's intrinsic
                            width and overflows past the cell boundary. */}
                        <div className="flex items-center gap-1.5 min-w-0">
                          {ACTIVE_STATUSES.has(job.status) && <PingDot className="shrink-0" />}
                          <span className="font-mono text-xs text-primary truncate">
                            {formatId(job.optimization_id)}
                          </span>
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
                                <span className="font-semibold text-foreground">{msg("dashboard.owner.me")}</span>
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
                      <TableCell className="px-2">
                        <div className="flex items-center gap-0.5">
                          <TooltipButton
                            tooltip={msg("auto.features.dashboard.components.jobstab.9")}
                          >
                            <button
                              type="button"
                              onClick={() => onOpenJob(job.optimization_id)}
                              className="p-1 rounded hover:bg-accent/60 text-muted-foreground hover:text-foreground transition-all cursor-pointer"
                              aria-label={formatMsg(
                                "auto.features.dashboard.components.jobstab.template.3",
                                { p1: TERMS.optimization },
                              )}
                            >
                              <ExternalLink className="size-3.5" />
                            </button>
                          </TooltipButton>
                        </div>
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
              className="gap-1"
            >
              <ChevronRight className="size-3.5" />
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
              className="gap-1"
            >
              {msg("auto.features.dashboard.components.jobstab.12")}
              <ChevronLeft className="size-3.5" />
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
