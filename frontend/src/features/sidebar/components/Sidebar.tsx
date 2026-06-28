"use client";

import * as React from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard,
  Send,
  Tags,
  Trash2,
  MoreHorizontal,
  Share,
  Pencil,
  Pin,
  Loader2,
  Grid2x2,
  ChevronLeft,
  Compass,
  CopyPlus,
  Database,
  RotateCcw,
  Play,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import { SidebarMoreSkeleton } from "./SidebarMoreSkeleton";
import { cn } from "@/shared/lib/utils";
import { Button } from "@/shared/ui/primitives/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import {
  listJobsSidebar,
  listJobsSharedWithMe,
  deleteJob,
  renameOptimization,
  togglePinOptimization,
  restartJob,
  resumeJob,
} from "@/shared/lib/api";
import type { SidebarJobItem } from "@/shared/lib/api";
import { isActiveStatus } from "@/shared/constants/job-status";
import { useJobsStream } from "@/shared/hooks/use-jobs-stream";
import { toast } from "react-toastify";
import { useSession } from "next-auth/react";
import { groupJobsByRecency } from "@/features/sidebar";
import { useUserPrefs } from "@/features/settings";
import { AccountMenu } from "@/shared/layout/account-menu";
import { AnimatedWordmark } from "@/shared/ui/animated-wordmark";
import { ShareDialog } from "@/features/optimizations";
import { StorageMeter } from "@/features/storage";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { sessionIdentity } from "@/shared/lib/session-identity";
import { TERMS } from "@/shared/lib/terms";
import { EmptyState } from "@/shared/ui/empty-state";

const NAV_ITEMS = [
  {
    href: "/",
    label: msg("auto.features.sidebar.components.sidebar.literal.1"),
    icon: LayoutDashboard,
  },
  { href: "/tagger", label: msg("auto.features.sidebar.components.sidebar.literal.2"), icon: Tags },
  { href: "/submit", label: TERMS.notificationNewOpt, icon: Send },
  { href: "/explore", label: msg("sidebar.nav.explore"), icon: Compass },
  { href: "/datasets", label: msg("sidebar.nav.datasets"), icon: Database },
] as const;

const PAGE_SIZE = 20;

const SIDEBAR_WIDTH_EXPANDED = 240;
const SIDEBAR_WIDTH_COLLAPSED = 64;
const SIDEBAR_COLLAPSED_STORAGE_KEY = "skynet.sidebar.collapsed";
const DESKTOP_MQ = "(min-width: 768px)";

export function Sidebar() {
  const pathname = usePathname();
  const router = useRouter();
  const searchParams = useSearchParams();
  const activePairParam = searchParams.get("pair");
  const parsedPair = activePairParam != null ? parseInt(activePairParam, 10) : NaN;
  const activePairIndex = Number.isFinite(parsedPair) ? parsedPair : null;
  const { data: session } = useSession();
  const sessionUser = sessionIdentity(session);
  const isAdmin = (session?.user as { role?: string } | undefined)?.role === "admin";
  const isRtl = getActiveDir() === "rtl";
  // Collapsed-rail tooltips read toward the content area: right in LTR, left in RTL.
  const tooltipSide = isRtl ? "left" : "right";

  const [tab, setTab] = React.useState<"mine" | "shared">("mine");
  // ``tab`` is the *requested* tab; ``renderedTab`` trails it and only flips
  // once that tab's rows have actually loaded. Keeping the two separate lets us
  // hold the previous list on screen during the in-flight fetch and crossfade
  // straight to the new one — no blank frame, so no flicker on toggle.
  const [renderedTab, setRenderedTab] = React.useState<"mine" | "shared">("mine");
  const tabRef = React.useRef<"mine" | "shared">("mine");
  const switchingRef = React.useRef(false);
  const [jobs, setJobs] = React.useState<SidebarJobItem[]>([]);
  const [activeCount, setActiveCount] = React.useState(0);
  const [loadedAll, setLoadedAll] = React.useState(false);
  const [loadingMore, setLoadingMore] = React.useState(false);
  const [collapsed, setCollapsed] = React.useState(false);
  const [isDesktop, setIsDesktop] = React.useState(true);

  // Collapse is a desktop affordance — the mobile drawer always shows the full
  // sidebar. Persist the choice and hydrate it client-side (SSR can't read
  // localStorage without risking a mismatch), and track the desktop breakpoint
  // so a persisted-collapsed state never narrows the mobile drawer.
  React.useEffect(() => {
    try {
      setCollapsed(window.localStorage.getItem(SIDEBAR_COLLAPSED_STORAGE_KEY) === "1");
    } catch {
      /* localStorage unavailable */
    }
    const mq = window.matchMedia(DESKTOP_MQ);
    const update = () => setIsDesktop(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  const toggleCollapsed = React.useCallback(() => {
    setCollapsed((c) => {
      const next = !c;
      try {
        window.localStorage.setItem(SIDEBAR_COLLAPSED_STORAGE_KEY, next ? "1" : "0");
      } catch {
        /* noop */
      }
      return next;
    });
  }, []);

  const compact = collapsed && isDesktop;

  // Sidebar infinite scroll: fetchData (polling + external invalidation)
  // re-requests ``max(PAGE_SIZE, loadedItemsRef.current)`` rows so a
  // background 30s refresh doesn't truncate the user's scrolled position
  // back to page 1. The ref lives outside React state so fetchData doesn't
  // need to be re-created when the loaded count changes.
  const loadedItemsRef = React.useRef(0);
  // Synchronous in-flight guard: setLoadingMore is async, so the observer
  // can fire twice before React re-renders. The ref blocks the second call.
  const loadingMoreRef = React.useRef(false);
  const listRef = React.useRef<HTMLDivElement>(null);
  const sentinelRef = React.useRef<HTMLDivElement | null>(null);

  const fetchData = React.useCallback(async () => {
    try {
      const limit = Math.min(200, Math.max(PAGE_SIZE, loadedItemsRef.current));
      const res =
        tab === "shared"
          ? await listJobsSharedWithMe({ limit, offset: 0 })
          : await listJobsSidebar({
              username: isAdmin ? undefined : sessionUser || undefined,
              limit,
              offset: 0,
            });
      // Drop the response if the user has since toggled away — applying it
      // would clobber the newer tab's list with stale rows.
      if (tabRef.current !== tab) return;
      setJobs(res.items);
      setActiveCount(res.items.filter((j) => isActiveStatus(j.status)).length);
      setLoadedAll(res.items.length >= res.total);
      loadedItemsRef.current = res.items.length;
      switchingRef.current = false;
      setRenderedTab(tab);
    } catch (err) {
      console.warn("sidebar fetch failed:", err);
    }
  }, [sessionUser, isAdmin, tab]);

  React.useEffect(() => {
    // Dep change (login / admin toggle) — reset depth so the next fetch
    // starts fresh at one page.
    loadedItemsRef.current = 0;
    void fetchData();
    const tick = () => {
      if (document.visibilityState === "visible") void fetchData();
    };
    const interval = setInterval(tick, 30000);
    // Catch up immediately when the tab becomes visible after being hidden.
    const onVisibility = () => {
      if (document.visibilityState === "visible") void fetchData();
    };
    document.addEventListener("visibilitychange", onVisibility);
    const onJobsChanged = () => fetchData();
    window.addEventListener("optimizations-changed", onJobsChanged);
    return () => {
      clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("optimizations-changed", onJobsChanged);
    };
  }, [fetchData]);

  // Live updates while jobs are active: subscribe to the shared dashboard SSE
  // stream (3s server cadence) so the running badge and per-job status pills
  // refresh within seconds of a job finishing instead of waiting up to 30s for
  // the background poll above. The 30s poll stays the baseline for the idle
  // case — it also catches jobs created in other sessions and re-opens the
  // shared stream by flipping ``activeCount`` below.
  useJobsStream({ active: activeCount > 0, onTick: () => void fetchData() });

  const loadMore = React.useCallback(async () => {
    // ``switchingRef`` blocks pagination while a tab change is in flight — the
    // stale ``jobs.length`` offset would otherwise fetch the wrong page.
    if (loadingMoreRef.current || loadedAll || switchingRef.current) return;
    loadingMoreRef.current = true;
    setLoadingMore(true);
    try {
      const res =
        tab === "shared"
          ? await listJobsSharedWithMe({ limit: PAGE_SIZE, offset: jobs.length })
          : await listJobsSidebar({
              username: isAdmin ? undefined : sessionUser || undefined,
              limit: PAGE_SIZE,
              offset: jobs.length,
            });
      setJobs((prev) => {
        // Dedupe by optimization_id in case a new job was inserted above
        // the offset between the previous fetch and this one.
        const existing = new Set(prev.map((j) => j.optimization_id));
        const appended = res.items.filter((j) => !existing.has(j.optimization_id));
        const merged = [...prev, ...appended];
        loadedItemsRef.current = merged.length;
        setLoadedAll(merged.length >= res.total);
        return merged;
      });
    } catch (err) {
      console.warn("sidebar loadMore failed:", err);
    } finally {
      loadingMoreRef.current = false;
      setLoadingMore(false);
    }
  }, [loadedAll, isAdmin, sessionUser, jobs.length, tab]);

  // Infinite-scroll sentinel. The sidebar scrolls in its own container
  // (``listRef``), so the observer's root must point at that element — not
  // the default viewport — otherwise the sentinel would appear "in view"
  // based on page scroll rather than sidebar scroll and fire incorrectly.
  React.useEffect(() => {
    const node = sentinelRef.current;
    const root = listRef.current;
    if (!node || !root) return;
    if (loadedAll) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) void loadMore();
      },
      { root, rootMargin: "120px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [loadedAll, loadMore, jobs.length]);

  const [deleteConfirm, setDeleteConfirm] = React.useState<string | null>(null);
  const [deleteLoading, setDeleteLoading] = React.useState(false);

  const handleDelete = React.useCallback((e: React.MouseEvent, optimizationId: string) => {
    e.preventDefault();
    e.stopPropagation();
    setDeleteConfirm(optimizationId);
  }, []);

  const confirmDelete = async () => {
    const optimizationId = deleteConfirm;
    if (!optimizationId) return;
    setDeleteLoading(true);
    setJobs((prev) => prev.filter((j) => j.optimization_id !== optimizationId));
    try {
      await deleteJob(optimizationId);
      toast.success(msg("sidebar.delete.success"));
      window.dispatchEvent(new Event("optimizations-changed"));
      if (pathname === `/optimizations/${optimizationId}`) router.push("/");
    } catch {
      toast.error(msg("sidebar.delete.failed"));
      void fetchData();
    } finally {
      setDeleteLoading(false);
      setDeleteConfirm(null);
    }
  };

  const handleTabChange = React.useCallback(
    (next: "mine" | "shared") => {
      if (next === tab) return;
      // Keep the current rows on screen; the new tab's list crossfades in once
      // its fetch resolves (see ``renderedTab``). ``switchingRef`` parks
      // pagination until then.
      switchingRef.current = true;
      tabRef.current = next;
      setTab(next);
    },
    [tab],
  );

  const groupedJobs = React.useMemo(() => groupJobsByRecency(jobs), [jobs]);

  const deleteJobInfo = React.useMemo(() => {
    if (!deleteConfirm) return null;
    const job = jobs.find((j) => j.optimization_id === deleteConfirm);
    if (!job) return { name: deleteConfirm, id: deleteConfirm };
    const name =
      job.name ||
      [job.module_name, job.optimizer_name].filter(Boolean).join(" · ") ||
      job.optimization_id.slice(0, 8);
    return { name, id: job.optimization_id };
  }, [deleteConfirm, jobs]);

  // The sidebar is position:fixed on desktop (a locked rail), so it's out of the
  // shell's flow and can't reserve its own width. Publish the effective width as
  // a CSS var the shell's <main> consumes for its inline margin — one source of
  // truth, updated when the rail collapses so content and rail never overlap.
  const widthExpr = compact
    ? `${SIDEBAR_WIDTH_COLLAPSED}px`
    : `min(${SIDEBAR_WIDTH_EXPANDED}px, 40vw, 92vw)`;
  React.useEffect(() => {
    document.documentElement.style.setProperty("--app-sidebar-width", widthExpr);
  }, [widthExpr]);

  return (
    <aside
      className="relative flex h-full shrink-0 flex-col border-e border-sidebar-border/60 bg-sidebar/80 backdrop-blur-xl overflow-hidden"
      style={{ width: widthExpr }}
      data-tutorial="sidebar-full"
    >
      <div className="flex flex-col h-full">
        <div
          className={cn(
            "hidden items-center px-3 pt-3 md:flex",
            compact ? "justify-center" : "justify-between",
          )}
        >
          {/* The morphing SKYNET wordmark anchors the rail (md+ only). It's dropped
              in the collapsed 64px state, where there's no room beside the toggle. */}
          {!compact && <AnimatedWordmark size={16} autoMorph morphSpeed={250} />}
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={toggleCollapsed}
                aria-label={compact ? msg("sidebar.expand") : msg("sidebar.collapse")}
                aria-expanded={!compact}
                className="group rounded-lg p-1.5 text-sidebar-foreground/60 transition-[color,background-color,transform] duration-200 hover:bg-sidebar-accent/40 hover:text-sidebar-foreground active:scale-90 cursor-pointer"
              >
                {/* Directional icon swap. Both icons stay mounted (symmetric +
                    interruptible — no AnimatePresence exit to drop mid-reflow) and
                    slide PAST each other on the start↔end axis: the outgoing icon
                    leaves toward the start while the incoming arrives from the end, so
                    they never sit superimposed. A plain opacity crossfade ghosted the
                    two near-identical panels into a doubled chevron at the midpoint;
                    the offset reads instead as the panel itself sliding open/closed.
                    Slide direction is mirrored in RTL. Expo ease keeps it snappy. */}
                <span className="relative inline-flex size-4 items-center justify-center">
                  <motion.span
                    className="absolute inset-0 inline-flex items-center justify-center"
                    initial={false}
                    animate={{
                      opacity: compact ? 0 : 1,
                      x: compact ? (isRtl ? 6 : -6) : 0,
                      scale: compact ? 0.6 : 1,
                    }}
                    transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                  >
                    <PanelLeftClose className="size-4 rtl:-scale-x-100" />
                  </motion.span>
                  <motion.span
                    className="absolute inset-0 inline-flex items-center justify-center"
                    initial={false}
                    animate={{
                      opacity: compact ? 1 : 0,
                      x: compact ? 0 : isRtl ? -6 : 6,
                      scale: compact ? 1 : 0.6,
                    }}
                    transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
                  >
                    <PanelLeftOpen className="size-4 rtl:-scale-x-100" />
                  </motion.span>
                </span>
              </button>
            </TooltipTrigger>
            <TooltipContent side={tooltipSide}>
              {compact ? msg("sidebar.expand") : msg("sidebar.collapse")}
            </TooltipContent>
          </Tooltip>
        </div>

        <nav
          className={cn("flex flex-col gap-1 px-3 pb-3 pt-2", compact && "items-center")}
          role="navigation"
          aria-label={msg("auto.features.sidebar.components.sidebar.literal.7")}
          data-tutorial="sidebar-nav"
        >
          {NAV_ITEMS.map(({ href, label, icon: Icon }) => (
            <NavItem
              key={href}
              href={href}
              label={label}
              Icon={Icon}
              active={href === "/" ? pathname === "/" : pathname.startsWith(href)}
              badge={href === "/" && renderedTab === "mine" && activeCount > 0 ? activeCount : null}
              compact={compact}
              tooltipSide={tooltipSide}
              isTagger={href === "/tagger"}
            />
          ))}
        </nav>

        {!compact && (
          <>
            <div aria-hidden="true" className="mx-3 h-px bg-sidebar-border/40" />

            <div
              role="tablist"
              aria-label={msg("sidebar.tab.aria")}
              className="relative mx-3 mt-2.5 mb-0.5 flex rounded-lg bg-muted p-1 gap-1"
            >
              <div
                aria-hidden="true"
                className="absolute top-1 bottom-1 w-[calc(50%-6px)] rounded-md bg-background shadow-sm transition-[inset-inline-start] duration-100 ease-out"
                style={{ insetInlineStart: tab === "mine" ? 4 : "calc(50% + 2px)" }}
              />
              {(["mine", "shared"] as const).map((key) => (
                <button
                  key={key}
                  type="button"
                  role="tab"
                  aria-selected={tab === key}
                  onClick={() => handleTabChange(key)}
                  className={cn(
                    "relative z-10 flex-1 cursor-pointer rounded-md px-2 py-1.5 text-center text-[0.6875rem] font-medium transition-colors duration-200",
                    tab === key ? "text-foreground" : "text-foreground/60 hover:text-foreground",
                  )}
                >
                  {msg(key === "mine" ? "sidebar.tab.mine" : "sidebar.tab.shared")}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-hidden flex flex-col min-h-0">
              <div ref={listRef} className="flex-1 overflow-y-auto px-3 pt-2 pb-2 no-scrollbar">
                <AnimatePresence mode="wait" initial={false}>
                  <motion.div
                    key={renderedTab}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.14, ease: "easeOut" }}
                  >
                    {groupedJobs.map((group) => (
                      <div key={group.label} className="mb-2">
                        <p className="flex items-center gap-1.5 text-[0.625rem] font-semibold uppercase tracking-[0.1em] text-muted-foreground/50 px-2 py-1.5">
                          <span>{group.label}</span>
                          <span className="tabular-nums text-muted-foreground/40 font-normal">
                            {group.jobs.length}
                          </span>
                        </p>
                        {group.jobs.map((job) => (
                          <JobRow
                            key={job.optimization_id}
                            job={job}
                            isShared={renderedTab === "shared"}
                            isActive={pathname === `/optimizations/${job.optimization_id}`}
                            activePair={
                              pathname === `/optimizations/${job.optimization_id}`
                                ? activePairIndex
                                : null
                            }
                            onDelete={handleDelete}
                            onRefresh={fetchData}
                          />
                        ))}
                      </div>
                    ))}
                    {loadedAll && groupedJobs.length === 0 && (
                      <EmptyState
                        variant="list"
                        icon={renderedTab === "shared" ? undefined : Send}
                        title={msg(
                          renderedTab === "shared" ? "sidebar.shared.empty" : "sidebar.mine.empty",
                        )}
                        description={msg(
                          renderedTab === "shared"
                            ? "sidebar.shared.empty.hint"
                            : "sidebar.mine.empty.hint",
                        )}
                      />
                    )}
                  </motion.div>
                </AnimatePresence>
                {loadingMore && (
                  <div className="px-1 pt-1 pb-2" aria-hidden="true">
                    <SidebarMoreSkeleton />
                  </div>
                )}
                {!loadedAll && <div ref={sentinelRef} aria-hidden="true" className="h-1 w-full" />}
              </div>
            </div>
          </>
        )}

        {compact && <div className="flex-1" aria-hidden="true" />}

        <div className="border-t border-sidebar-border/60">
          {!compact && <StorageMeter />}
          <div className={cn("py-2", compact ? "flex justify-center px-2" : "px-2")}>
            <AccountMenu collapsed={compact} />
          </div>
        </div>
      </div>

      <Dialog
        open={deleteConfirm !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteConfirm(null);
        }}
      >
        <DialogContent className="max-w-md sm:max-w-md" showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>
              {msg("auto.features.sidebar.components.sidebar.3")}
              {TERMS.optimization}
            </DialogTitle>
            <DialogDescription>
              {msg("auto.features.sidebar.components.sidebar.4")}
              {TERMS.optimization}{" "}
              <span className="font-semibold text-foreground break-words" dir="auto">
                {deleteJobInfo?.name}
              </span>
              ? {msg("delete.irreversible")}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-2 gap-3">
            <Button
              variant="outline"
              onClick={() => setDeleteConfirm(null)}
              disabled={deleteLoading}
              className="w-full justify-center"
            >
              {msg("auto.features.sidebar.components.sidebar.5")}
            </Button>
            <Button
              variant="destructive"
              onClick={confirmDelete}
              disabled={deleteLoading}
              className="w-full justify-center"
            >
              {deleteLoading ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                msg("auto.features.sidebar.components.sidebar.literal.10")
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </aside>
  );
}

/**
 * One primary-nav entry. Expanded shows icon + label (+ an optional count badge);
 * collapsed shrinks to an icon-only target wrapped in a tooltip that names the
 * destination, so the rail stays usable when minimized.
 */
function NavItem({
  href,
  label,
  Icon,
  active,
  badge,
  compact,
  tooltipSide,
  isTagger,
}: {
  href: string;
  label: string;
  Icon: React.ComponentType<{ className?: string }>;
  active: boolean;
  badge: number | null;
  compact: boolean;
  tooltipSide: "left" | "right";
  isTagger: boolean;
}) {
  const link = (
    <Link
      href={href}
      {...(isTagger ? { "data-tutorial": "sidebar-tagger" } : {})}
      aria-label={compact ? label : undefined}
      className={cn(
        "group relative flex items-center rounded-lg text-sm font-medium transition-all duration-200",
        compact ? "justify-center p-2.5" : "gap-3 px-3 py-2",
        active
          ? "text-primary"
          : cn(
              "text-sidebar-foreground/60 hover:bg-sidebar-accent/40 hover:text-sidebar-foreground",
              !compact && "hover:translate-x-[-2px]",
            ),
      )}
    >
      {active && (
        <motion.div
          layoutId="sidebar-active"
          className="absolute inset-0 rounded-lg bg-primary/[0.08] ring-1 ring-primary/10"
          style={compact ? undefined : { borderInlineStart: "3px solid var(--primary)" }}
          transition={{ type: "spring", stiffness: 350, damping: 28 }}
        />
      )}
      {compact ? (
        <span className="relative z-10 inline-flex">
          <Icon
            className={cn(
              "size-4 shrink-0 transition-colors duration-200",
              active ? "text-primary" : "group-hover:text-sidebar-foreground",
            )}
          />
          {badge != null && (
            <span
              aria-hidden="true"
              className="absolute -top-1 -end-1 size-1.5 rounded-full bg-primary"
            />
          )}
        </span>
      ) : (
        <span className="relative z-10 flex items-center gap-2.5 flex-1 min-w-0">
          <Icon
            className={cn(
              "size-4 shrink-0 transition-colors duration-200",
              active ? "text-primary" : "group-hover:text-sidebar-foreground",
            )}
          />
          <span className="truncate flex-1">{label}</span>
          {badge != null && (
            <span className="shrink-0 text-[0.625rem] font-bold bg-primary/10 text-primary px-1.5 py-0.5 rounded-full tabular-nums">
              {badge}
            </span>
          )}
        </span>
      )}
    </Link>
  );

  if (!compact) return link;
  return (
    <Tooltip>
      <TooltipTrigger asChild>{link}</TooltipTrigger>
      <TooltipContent side={tooltipSide}>{label}</TooltipContent>
    </Tooltip>
  );
}

function JobRow({
  job,
  isShared,
  isActive,
  activePair,
  onDelete,
  onRefresh,
}: {
  job: SidebarJobItem;
  isShared: boolean;
  isActive: boolean;
  activePair: number | null;
  onDelete: (e: React.MouseEvent, id: string) => void;
  onRefresh: () => void;
}) {
  const router = useRouter();
  const { prefs } = useUserPrefs();
  const isRtl = getActiveDir() === "rtl";
  const [menuOpen, setMenuOpen] = React.useState(false);
  const [renaming, setRenaming] = React.useState(false);
  const [renameValue, setRenameValue] = React.useState("");
  const [menuPos, setMenuPos] = React.useState<{ top: number; left: number } | null>(null);
  const [expanded, setExpanded] = React.useState(isActive && activePair !== null);
  const menuRef = React.useRef<HTMLDivElement>(null);
  const btnRef = React.useRef<HTMLButtonElement>(null);
  const renameRef = React.useRef<HTMLInputElement>(null);
  const isGridSearch =
    prefs.advancedMode && job.optimization_type === "grid_search" && (job.total_pairs ?? 0) > 0;
  const displayName =
    job.name ||
    [job.module_name, job.optimizer_name].filter(Boolean).join(" · ") ||
    job.optimization_id.slice(0, 8);

  // Shared rows carry the caller's grant role; gate row actions by what that
  // role permits server-side. Rename/rerun/pin are editor+; share/delete are
  // owner-only (and a shared-with-me row is never one the caller owns). On the
  // "mine" tab the caller is always owner/admin, so everything is allowed.
  const canEdit = !isShared || job.role === "editor" || job.role === "owner";

  const dropdownRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!menuOpen) return;
    const handler = (e: MouseEvent | TouchEvent) => {
      const target = e.target as Node;
      if (menuRef.current?.contains(target)) return;
      if (dropdownRef.current?.contains(target)) return;
      setMenuOpen(false);
    };
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setMenuOpen(false);
        btnRef.current?.focus();
      }
    };
    document.addEventListener("mousedown", handler);
    document.addEventListener("touchstart", handler);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", handler);
      document.removeEventListener("touchstart", handler);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [menuOpen]);

  React.useEffect(() => {
    if (renaming) renameRef.current?.focus();
  }, [renaming]);

  const [shareOpen, setShareOpen] = React.useState(false);

  // Open the same Drive-style sharing dialog the optimization page uses, instead
  // of a bare link copy — the dialog itself offers copy plus roles and visibility.
  const handleShare = () => {
    setMenuOpen(false);
    setShareOpen(true);
  };

  const handleClone = () => {
    setMenuOpen(false);
    router.push(`/submit?clone=${job.optimization_id}`);
  };

  const handleRetry = async () => {
    setMenuOpen(false);
    try {
      await restartJob(job.optimization_id);
      toast.success(msg("sidebar.rerun.success"));
      window.dispatchEvent(new Event("optimizations-changed"));
      onRefresh();
      // Restart re-runs the same id in place — open that run, not a new one.
      router.push(`/optimizations/${job.optimization_id}`);
    } catch (err) {
      // Surface the real backend reason (quota 429, wrong-status 409, …) like
      // the detail-view retry does, falling back to the generic message.
      toast.error(err instanceof Error ? err.message : msg("sidebar.rerun.failed"));
    }
  };

  const handleResume = async () => {
    setMenuOpen(false);
    try {
      await resumeJob(job.optimization_id);
      toast.success(msg("sidebar.resume.success"));
      // Resume continues the same run in place — refresh the list rather than
      // navigating to a new id.
      window.dispatchEvent(new Event("optimizations-changed"));
      onRefresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("sidebar.resume.failed"));
    }
  };

  // Enter triggers handleRename, then setRenaming(false) blurs the input,
  // which fires onBlur → handleRename again. Guard against the double-fire.
  const renameSubmittedRef = React.useRef(false);
  const handleRename = async () => {
    if (renameSubmittedRef.current) return;
    renameSubmittedRef.current = true;
    const newName = renameValue.trim();
    if (!newName || newName === (job.name ?? "")) {
      setRenaming(false);
      renameSubmittedRef.current = false;
      return;
    }
    try {
      await renameOptimization(job.optimization_id, newName);
      toast.success(msg("sidebar.rename.success"));
      window.dispatchEvent(
        new CustomEvent("optimization-renamed", {
          detail: { optimizationId: job.optimization_id, name: newName },
        }),
      );
      window.dispatchEvent(new Event("optimizations-changed"));
      onRefresh();
    } catch {
      toast.error(msg("sidebar.rename.failed"));
    }
    setRenaming(false);
    renameSubmittedRef.current = false;
  };

  const handlePin = async () => {
    try {
      const res = await togglePinOptimization(job.optimization_id);
      toast.success(res.pinned ? msg("sidebar.pin.on") : msg("sidebar.pin.off"));
      window.dispatchEvent(
        new CustomEvent("optimization-updated", {
          detail: { optimizationId: job.optimization_id },
        }),
      );
      window.dispatchEvent(new Event("optimizations-changed"));
      onRefresh();
    } catch {
      toast.error(msg("sidebar.generic_error"));
    }
    setMenuOpen(false);
  };

  if (renaming) {
    return (
      <div className="px-2 py-1.5">
        <input
          ref={renameRef}
          type="text"
          value={renameValue}
          onChange={(e) => setRenameValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              void handleRename();
            }
            if (e.key === "Escape") {
              renameSubmittedRef.current = true;
              setRenaming(false);
              renameSubmittedRef.current = false;
            }
          }}
          onBlur={handleRename}
          maxLength={120}
          className="w-full text-[0.6875rem] bg-sidebar-accent/30 border border-primary/30 rounded-md px-2 py-1 outline-none font-medium"
          dir="auto"
        />
      </div>
    );
  }

  return (
    <div className="relative" ref={menuRef}>
      <div
        className={cn(
          "flex items-center gap-1.5 rounded-lg px-2 py-2 text-[0.6875rem] transition-all duration-150",
          isActive
            ? "bg-primary/[0.07] text-foreground"
            : "text-muted-foreground hover:bg-sidebar-accent/30 hover:text-foreground",
        )}
      >
        <Link
          href={`/optimizations/${job.optimization_id}`}
          className="flex items-center gap-2 min-w-0 flex-1 overflow-hidden"
        >
          <span
            className="truncate font-medium leading-tight min-w-0 block text-start flex-1"
            title={displayName}
          >
            {displayName}
          </span>
          {isGridSearch && (
            <span
              className="inline-flex items-center gap-0.5 text-[9px] font-semibold text-muted-foreground/60 bg-muted/40 px-1 py-0.5 rounded shrink-0"
              title={formatMsg("auto.features.sidebar.components.sidebar.template.2", {
                p1: job.total_pairs ?? "?",
              })}
            >
              <Grid2x2 className="size-2.5" />
              {job.total_pairs ?? "?"}
            </span>
          )}
          {job.pinned && <Pin className="size-2.5 text-muted-foreground/60 shrink-0" />}
          <StatusDot status={job.status} />
        </Link>
        {isGridSearch && (
          <button
            type="button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              setExpanded((o) => !o);
            }}
            className="p-0.5 rounded cursor-pointer text-muted-foreground/40 hover:text-foreground transition-colors shrink-0"
            aria-label={
              expanded
                ? msg("auto.features.sidebar.components.sidebar.literal.11")
                : msg("auto.features.sidebar.components.sidebar.literal.12")
            }
          >
            <ChevronLeft
              className={cn("size-3.5 transition-transform duration-200", expanded && "-rotate-90")}
            />
          </button>
        )}
        {
          <button
            ref={btnRef}
            type="button"
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              if (!menuOpen && btnRef.current) {
                const rect = btnRef.current.getBoundingClientRect();
                const menuWidth = 140;
                const menuHeightEstimate = 200;
                const margin = 8;
                // Right-align by default; clamp to viewport so the menu
                // never overflows the screen edge on narrow windows.
                const left = Math.max(
                  margin,
                  Math.min(rect.right - menuWidth, window.innerWidth - menuWidth - margin),
                );
                const top =
                  rect.bottom + menuHeightEstimate + margin > window.innerHeight
                    ? Math.max(margin, rect.top - menuHeightEstimate - 4)
                    : rect.bottom + 4;
                setMenuPos({ top, left });
              }
              setMenuOpen((o) => !o);
            }}
            className="p-0.5 rounded cursor-pointer text-muted-foreground/40 hover:text-foreground transition-colors shrink-0"
            aria-label={formatMsg("auto.features.sidebar.components.sidebar.template.3", {
              p1: displayName,
            })}
            aria-haspopup="menu"
            aria-expanded={menuOpen}
          >
            <MoreHorizontal className="size-3.5" />
          </button>
        }
      </div>

      <AnimatePresence>
        {expanded && isGridSearch && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="ps-6 pe-2 pb-1">
              {Array.from({ length: job.total_pairs ?? 0 }, (_, i) => {
                const isPairActive = isActive && activePair === i;
                const pairStatus = derivePairStatus(
                  i,
                  job.status,
                  job.completed_pairs ?? 0,
                  job.failed_pairs ?? 0,
                );
                return (
                  <Link
                    key={i}
                    href={`/optimizations/${job.optimization_id}?pair=${i}`}
                    className={cn(
                      "flex items-center gap-2 rounded-md px-2 py-1.5 text-[0.625rem] transition-all duration-150",
                      isPairActive
                        ? "bg-primary/[0.07] text-foreground font-semibold"
                        : "text-muted-foreground/70 hover:bg-sidebar-accent/30 hover:text-foreground",
                    )}
                  >
                    <StatusDot status={pairStatus} />
                    <span>
                      {msg("auto.features.sidebar.components.sidebar.6")}
                      {i + 1}
                    </span>
                  </Link>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Dropdown menu — portaled to body to escape overflow clipping */}
      {menuOpen &&
        menuPos &&
        createPortal(
          <motion.div
            ref={dropdownRef}
            role="menu"
            initial={{ opacity: 0, scale: 0.95, y: -4 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -4 }}
            transition={{ duration: 0.12 }}
            className="fixed z-[9999] min-w-[140px] rounded-2xl border border-border/40 bg-card shadow-[0_4px_24px_rgba(28,22,18,0.1)] py-1.5"
            style={{ top: menuPos.top, left: menuPos.left, right: "auto" }}
          >
            {!isShared && (
              <button
                type="button"
                role="menuitem"
                onClick={handleShare}
                className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors"
              >
                <Share className="size-3.5 text-muted-foreground" />
                {msg("auto.features.sidebar.components.sidebar.7")}
              </button>
            )}

            {canEdit && (
              <button
                type="button"
                role="menuitem"
                onClick={() => {
                  setMenuOpen(false);
                  setRenameValue(job.name ?? displayName);
                  setRenaming(true);
                }}
                className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors"
              >
                <Pencil className="size-3.5 text-muted-foreground" />
                {msg("auto.features.sidebar.components.sidebar.8")}
              </button>
            )}

            <button
              type="button"
              role="menuitem"
              onClick={handleClone}
              className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors"
            >
              <CopyPlus className="size-3.5 text-muted-foreground" />
              {msg("auto.features.sidebar.components.sidebar.9")}
            </button>

            {canEdit &&
              (job.status === "failed" || job.status === "cancelled") &&
              (job.resumable ? (
                <button
                  type="button"
                  role="menuitem"
                  onClick={handleResume}
                  className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors"
                >
                  <Play className={cn("size-3.5 text-muted-foreground", isRtl && "-scale-x-100")} />
                  {msg("sidebar.resume")}
                </button>
              ) : (
                <button
                  type="button"
                  role="menuitem"
                  onClick={handleRetry}
                  className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors"
                >
                  <RotateCcw className="size-3.5 text-muted-foreground" />
                  {msg("sidebar.rerun")}
                </button>
              ))}

            {canEdit && (
              <>
                <div className="h-px bg-border/20 mx-2 my-1" />

                <button
                  type="button"
                  role="menuitem"
                  onClick={handlePin}
                  className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors"
                >
                  <Pin
                    className={cn(
                      "size-3.5",
                      job.pinned ? "text-foreground" : "text-muted-foreground",
                    )}
                  />
                  {job.pinned
                    ? msg("auto.features.sidebar.components.sidebar.literal.13")
                    : msg("auto.features.sidebar.components.sidebar.literal.14")}
                </button>
              </>
            )}

            {!isShared && (
              <button
                type="button"
                role="menuitem"
                onClick={(e) => {
                  setMenuOpen(false);
                  onDelete(e, job.optimization_id);
                }}
                className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-red-500 hover:bg-red-500/5 cursor-pointer transition-colors"
              >
                <Trash2 className="size-3.5" />
                {msg("auto.features.sidebar.components.sidebar.10")}
              </button>
            )}
          </motion.div>,
          document.body,
        )}

      {!isShared && (
        <ShareDialog
          optimizationId={job.optimization_id}
          open={shareOpen}
          onOpenChange={setShareOpen}
          hideTrigger
        />
      )}
    </div>
  );
}

function derivePairStatus(
  index: number,
  parentStatus: string,
  completedPairs: number,
  failedPairs: number,
): string {
  if (index < completedPairs) return "success";
  if (index < completedPairs + failedPairs) return "failed";
  return parentStatus;
}

function StatusDot({ status }: { status: string }) {
  const isRunning = isActiveStatus(status);
  return (
    <span className="relative flex size-2 shrink-0">
      {isRunning && (
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--warning)]/60" />
      )}
      <span
        className={cn(
          "relative inline-flex rounded-full size-2",
          status === "success"
            ? "bg-[var(--success)]"
            : status === "failed"
              ? "bg-[var(--danger)]"
              : status === "cancelled"
                ? "bg-[#6b6058]"
                : "bg-[var(--warning)]",
        )}
      />
    </span>
  );
}
