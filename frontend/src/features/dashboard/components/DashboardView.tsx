"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "next-auth/react";
import { motion, useReducedMotion } from "framer-motion";
import { BarChart3, TableIcon } from "lucide-react";
import { DashboardSkeleton } from "./DashboardSkeleton";
import { toast } from "react-toastify";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/primitives/tabs";
import { FadeIn } from "@/shared/ui/motion";
import { msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";
import { useColumnFilters, useColumnResize, type SortDir } from "@/shared/ui/excel-filter";
import { getJobTypeLabel, getStatusLabel } from "@/shared/constants/job-status";
import type { OptimizationSummaryResponse, PaginatedJobsResponse } from "@/shared/types/api";
import type { DashboardAnalytics } from "@/shared/lib/api";
import { registerTutorialHook, registerTutorialQuery } from "@/features/tutorial";
import { transformChartData } from "../lib/transform-chart-data";
import { useQueueStatus } from "../hooks/use-queue-status";
import { getDashboardStats } from "../lib/get-dashboard-stats";
import { useAnalyticsFilters } from "../hooks/use-analytics-filters";
import { useJobsList } from "../hooks/use-jobs-list";
import { useDashboardAnalytics } from "../hooks/use-dashboard-analytics";
import { useJobsRealtime } from "../hooks/use-jobs-realtime";
import { useBulkDelete } from "../hooks/use-bulk-delete";
import { COMPARE_MAX } from "../constants";
import { DashboardHeader } from "./DashboardHeader";
import { QueueStatusAlert } from "./QueueStatusAlert";
import { BulkActionBar } from "./BulkActionBar";
import { DeleteDialogs } from "./DeleteDialogs";
import { JobsTab } from "./JobsTab";
import { AnalyticsTab } from "./AnalyticsTab";

// The active-tab background is a single shared pill that slides between
// triggers via Framer's layoutId (see DashboardView). The button itself stays
// transparent and only fades text color + reacts to the press transform.
const DASHBOARD_TAB_CLASS =
  "relative z-10 min-h-10 rounded-md px-3 py-2 text-sm font-semibold cursor-pointer border-none bg-transparent text-foreground/65 shadow-none transition-[color,transform] data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:border-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 sm:px-4";

function getJobField(job: OptimizationSummaryResponse, key: string): unknown {
  return (job as unknown as Record<string, unknown>)[key];
}

function compareJobValues(av: unknown, bv: unknown): number {
  const aMissing = av == null || av === "";
  const bMissing = bv == null || bv === "";
  if (aMissing && bMissing) return 0;
  if (aMissing) return -1;
  if (bMissing) return 1;
  if (typeof av === "number" && typeof bv === "number") return av - bv;
  return String(av).localeCompare(String(bv), "he", { numeric: true });
}

export function DashboardView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session } = useSession();
  const sessionUser = session?.user?.name ?? "";
  const isAdmin = session?.user?.role === "admin";

  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const prefersReducedMotion = useReducedMotion();

  const [activeTab, setActiveTab] = useState("jobs");
  // Sync from URL on mount / when ?tab= changes (supports deep-linking from
  // UserFieldPreview and bookmarked tab URLs).
  const urlTab = searchParams.get("tab");
  useEffect(() => {
    if (urlTab === "jobs" || urlTab === "analytics") {
      setActiveTab(urlTab);
    }
  }, [urlTab]);
  // Mirror the active tab back into ?tab= so reload / share-link / back-button
  // round-trip the user to the same tab they were on.
  const handleTabChange = useCallback(
    (next: string) => {
      setActiveTab(next);
      const params = new URLSearchParams(searchParams.toString());
      params.set("tab", next);
      router.replace(`?${params.toString()}`, { scroll: false });
    },
    [router, searchParams],
  );
  // Expose for tutorial beforeShow
  useEffect(() => registerTutorialHook("setTab", setActiveTab), []);

  const analyticsFilters = useAnalyticsFilters();

  const {
    data,
    setData,
    loading,
    initialLoad,
    error,
    pageOffset,
    setPageOffset,
    counts,
    fetchJobs,
  } = useJobsList({ sessionUser, isAdmin });

  const { analyticsData, setAnalyticsData, analyticsLoading } = useDashboardAnalytics({
    sessionUser,
    isAdmin,
    activeTab,
    model: analyticsFilters.model,
    status: analyticsFilters.status,
    jobId: analyticsFilters.jobId,
    date: analyticsFilters.date,
    owner: analyticsFilters.owner,
    access: analyticsFilters.access,
  });

  // Demo overlay state — tutorial injects fake data here so background
  // fetches (realtime polling, initial load) can never overwrite it.
  const [demoJobs, setDemoJobs] = useState<PaginatedJobsResponse | null>(null);
  const [demoAnalytics, setDemoAnalytics] = useState<DashboardAnalytics | null>(null);
  const effectiveData = demoJobs ?? data;
  const effectiveAnalytics = demoAnalytics ?? analyticsData;

  // Tutorial hooks — registered after data hooks are available
  useEffect(
    () => registerTutorialQuery("hasDashboardData", () => (data?.items.length ?? 0) > 0),
    [data],
  );
  useEffect(() => registerTutorialHook("setDemoJobs", setDemoJobs), []);
  useEffect(() => registerTutorialHook("setDemoAnalytics", setDemoAnalytics), []);

  // Clear demo overlay when tutorial exits so real data shows through
  useEffect(() => {
    const onExit = () => {
      setDemoJobs(null);
      setDemoAnalytics(null);
    };
    window.addEventListener("tutorial-exited", onExit);
    return () => window.removeEventListener("tutorial-exited", onExit);
  }, []);

  const queueStatus = useQueueStatus();

  // Cross-component sync (sidebar delete/rename, etc.): refetch the
  // current jobs page and invalidate the analytics cache so the next
  // analytics-tab visit re-pulls.
  useEffect(() => {
    const onJobsChanged = () => {
      void fetchJobs();
      setAnalyticsData(null);
    };
    window.addEventListener("optimizations-changed", onJobsChanged);
    return () => window.removeEventListener("optimizations-changed", onJobsChanged);
  }, [fetchJobs, setAnalyticsData]);

  useJobsRealtime({ data, fetchJobs });

  const { filters, setColumnFilter, openFilter, setOpenFilter, clearAll, activeCount } =
    useColumnFilters();
  const colResize = useColumnResize();
  const [sortKey, setSortKey] = useState<string>("created_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const toggleSort = useCallback(
    (key: string) => {
      if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      else {
        setSortKey(key);
        setSortDir("asc");
      }
    },
    [sortKey],
  );

  const {
    selectedIds,
    setSelectedIds,
    toggleRowSelected,
    clearSelection,
    bulkDeleteOpen,
    setBulkDeleteOpen,
    bulkDeleting,
    confirmBulkDelete,
  } = useBulkDelete({ data, setData, setPageOffset, fetchJobs, visibleData: effectiveData });

  useEffect(
    () => registerTutorialHook("setSelectedJobIds", (ids) => setSelectedIds(new Set(ids))),
    [setSelectedIds],
  );

  // Compare-eligible subset of the current selection. Exact metric/test-set
  // compatibility is validated on the compare page after loading payloads.
  const compareEligibleIds = useMemo(() => {
    if (!effectiveData || selectedIds.size === 0) return [];
    const selectedItems = effectiveData.items.filter(
      (j) => selectedIds.has(j.optimization_id) && j.status === "success",
    );
    return selectedItems.map((j) => j.optimization_id);
  }, [effectiveData, selectedIds]);
  const canCompare = compareEligibleIds.length >= 2;

  // Admins delete anything; everyone else may bulk-delete only runs they own
  // outright (role null/absent) or co-own (role "owner"). Shared viewer/editor
  // grants can't delete. Mirrors the detail page's canDeleteRun (effective_role
  // == null) and the backend's owner-role check in /optimizations/bulk-delete,
  // which skips non-owned ids — gating here avoids a misleading "deleted" toast
  // when the server actually skipped them.
  const canBulkDelete = useMemo(() => {
    if (isAdmin) return true;
    if (!effectiveData || selectedIds.size === 0) return false;
    const selected = effectiveData.items.filter((j) => selectedIds.has(j.optimization_id));
    if (selected.length !== selectedIds.size) return false;
    return selected.every((j) => j.role == null || j.role === "owner");
  }, [isAdmin, effectiveData, selectedIds]);

  const onCompare = useCallback(() => {
    if (compareEligibleIds.length < 2) return;
    if (compareEligibleIds.length > COMPARE_MAX) {
      toast.error(msg("compare.cap_reached"));
      return;
    }
    router.push(`/compare?jobs=${compareEligibleIds.join(",")}`);
  }, [compareEligibleIds, router]);

  const filteredItems = useMemo(() => {
    if (!effectiveData) return [];
    const items = effectiveData.items.filter((job) => {
      for (const [col, allowed] of Object.entries(filters)) {
        if (allowed.size === 0) continue;
        const val = String(getJobField(job, col) ?? "");
        if (!allowed.has(val)) return false;
      }
      return true;
    });
    items.sort((a, b) => {
      const cmp = compareJobValues(getJobField(a, sortKey), getJobField(b, sortKey));
      return sortDir === "asc" ? cmp : -cmp;
    });
    return items;
  }, [effectiveData, filters, sortKey, sortDir]);

  // Every currently-visible (filtered + loaded) row is selectable. The
  // bulk-delete handler cancels active jobs before deleting, mirroring
  // the single-row delete button's behavior.
  const selectablePageIds = useMemo(
    () => filteredItems.map((j) => j.optimization_id),
    [filteredItems],
  );
  const selectedOnPage = selectablePageIds.filter((id) => selectedIds.has(id)).length;
  const pageAllSelected =
    selectablePageIds.length > 0 && selectedOnPage === selectablePageIds.length;
  const pageSomeSelected = selectedOnPage > 0 && !pageAllSelected;
  const togglePageSelection = () => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (pageAllSelected) {
        for (const id of selectablePageIds) next.delete(id);
      } else {
        for (const id of selectablePageIds) next.add(id);
      }
      return next;
    });
  };

  const filterOptions = useMemo(() => {
    if (!effectiveData) return {} as Record<string, Array<{ value: string; label: string }>>;
    const items = effectiveData.items;
    const unique = (key: string, labelFn?: (v: string) => string) => {
      const vals = [...new Set(items.map((j) => String(getJobField(j, key) ?? "")))]
        .filter(Boolean)
        .sort();
      return vals.map((v) => ({ value: v, label: labelFn ? labelFn(v) : v }));
    };
    // Role / access filter: owned runs carry no grant role, so they surface
    // under the empty-string bucket labeled "שלי". Multi-select then covers
    // both shared-vs-mine and viewer/editor-vs-owner framings.
    const roleLabels: Record<string, string> = {
      "": msg("dashboard.role.mine"),
      owner: msg("dashboard.role_short.owner"),
      editor: msg("dashboard.role_short.editor"),
      viewer: msg("dashboard.role_short.viewer"),
    };
    const presentRoles = new Set(items.map((j) => String(getJobField(j, "role") ?? "")));
    const role = ["", "owner", "editor", "viewer"]
      .filter((v) => presentRoles.has(v))
      .map((v) => ({ value: v, label: roleLabels[v] ?? v }));

    return {
      optimization_id: unique("optimization_id"),
      name: unique("name"),
      username: unique("username", (v) =>
        v.toLowerCase() === sessionUser.toLowerCase() ? msg("dashboard.owner.me") : v,
      ),
      role,
      status: unique("status", getStatusLabel),
      optimization_type: unique("optimization_type", getJobTypeLabel),
      module_name: unique("module_name"),
      optimizer_name: unique("optimizer_name"),
    };
  }, [effectiveData, sessionUser]);

  const chartData = useMemo(() => transformChartData(effectiveAnalytics), [effectiveAnalytics]);

  const stats = useMemo(
    () =>
      getDashboardStats({
        data: effectiveData,
        filteredItems,
        counts,
        analyticsData: effectiveAnalytics,
        activeTab,
      }),
    [effectiveData, filteredItems, counts, effectiveAnalytics, activeTab],
  );

  // Owner/Role columns and the shared stat card only appear once the caller
  // actually collaborates — a solo user's control panel stays unchanged.
  const hasShared =
    (counts?.shared ?? 0) > 0 || (effectiveData?.items.some((j) => Boolean(j.role)) ?? false);

  if (initialLoad && !effectiveData) {
    return <DashboardSkeleton />;
  }

  const tabPillTransition = prefersReducedMotion
    ? { duration: 0 }
    : { type: "tween" as const, duration: 0.2, ease: [0.2, 0.8, 0.2, 1] as const };

  return (
    <>
      <div className="flex flex-col gap-8 -mt-2 md:-mt-4">
        <DashboardHeader stats={stats} />
        <QueueStatusAlert queueStatus={queueStatus} />

        <FadeIn delay={0.2}>
          {mounted && (
            <Tabs value={activeTab} onValueChange={handleTabChange}>
              <TabsList className="inline-flex h-auto w-full gap-1 rounded-lg border border-border/60 bg-muted/50 p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.5)]">
                <TabsTrigger
                  value="jobs"
                  className={DASHBOARD_TAB_CLASS}
                >
                  {activeTab === "jobs" && (
                    <motion.span
                      layoutId="dashboardTabPill"
                      transition={tabPillTransition}
                      className="absolute inset-0 z-0 rounded-md bg-background shadow-sm"
                      aria-hidden="true"
                    />
                  )}
                  <span className="relative z-10 inline-flex items-center gap-1.5">
                    <TableIcon className="size-3.5" />
                    {TERMS.optimizationPlural}
                  </span>
                </TabsTrigger>
                <TabsTrigger
                  value="analytics"
                  data-tutorial="analytics-tab"
                  className={DASHBOARD_TAB_CLASS}
                >
                  {activeTab === "analytics" && (
                    <motion.span
                      layoutId="dashboardTabPill"
                      transition={tabPillTransition}
                      className="absolute inset-0 z-0 rounded-md bg-background shadow-sm"
                      aria-hidden="true"
                    />
                  )}
                  <span className="relative z-10 inline-flex items-center gap-1.5">
                    <BarChart3 className="size-3.5" />
                    {msg("auto.features.dashboard.components.dashboardview.1")}
                  </span>
                </TabsTrigger>
              </TabsList>

              <TabsContent value="jobs">
                <JobsTab
                  data={effectiveData}
                  loading={loading}
                  error={error}
                  filteredItems={filteredItems}
                  activeCount={activeCount}
                  clearAllFilters={clearAll}
                  colResize={colResize}
                  sortKey={sortKey}
                  sortDir={sortDir}
                  toggleSort={toggleSort}
                  filters={filters}
                  setColumnFilter={setColumnFilter}
                  openFilter={openFilter}
                  setOpenFilter={setOpenFilter}
                  filterOptions={filterOptions}
                  showSharedColumns={hasShared}
                  sessionUser={sessionUser}
                  selectedIds={selectedIds}
                  toggleRowSelected={toggleRowSelected}
                  selectablePageIds={selectablePageIds}
                  pageAllSelected={pageAllSelected}
                  pageSomeSelected={pageSomeSelected}
                  togglePageSelection={togglePageSelection}
                  pageOffset={pageOffset}
                  setPageOffset={setPageOffset}
                  onOpenJob={(id) => router.push(`/optimizations/${id}`)}
                />
              </TabsContent>

              <TabsContent value="analytics">
                <AnalyticsTab
                  analyticsLoading={analyticsLoading}
                  analyticsData={effectiveAnalytics}
                  chartData={chartData}
                  filters={analyticsFilters}
                  sessionUser={sessionUser}
                />
              </TabsContent>

            </Tabs>
          )}
        </FadeIn>

        <BulkActionBar
          canDelete={canBulkDelete}
          selectedCount={selectedIds.size}
          compareEligibleCount={compareEligibleIds.length}
          canCompare={canCompare}
          onClear={clearSelection}
          onCompare={onCompare}
          onRequestBulkDelete={() => setBulkDeleteOpen(true)}
        />
        <DeleteDialogs
          bulkDeleteOpen={bulkDeleteOpen}
          setBulkDeleteOpen={setBulkDeleteOpen}
          bulkDeleting={bulkDeleting}
          confirmBulkDelete={confirmBulkDelete}
          selectedCount={selectedIds.size}
        />
      </div>
    </>
  );
}
