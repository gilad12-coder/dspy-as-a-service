"use client";

import * as React from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  SquaresFour,
  PaperPlaneTilt,
  Trash,
  DotsThree,
  ShareNetwork,
  PencilSimple,
  PushPin,
  CircleNotch,
  GridFour,
  CaretLeft,
  Compass,
  Copy,
  Database,
  ArrowCounterClockwise,
  Play,
} from "@/shared/ui/icons";
import { SidebarMoreSkeleton } from "./SidebarMoreSkeleton";
import { cn } from "@/shared/lib/utils";
import { Button } from "@/shared/ui/primitives/button";
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
import { AccountMenu } from "@/shared/layout/account-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { ShareDialog } from "@/features/optimizations";
import { StorageMeter } from "@/features/storage";
import { formatMsg, msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { sessionIdentity } from "@/shared/lib/session-identity";
import { recentResumableId } from "@/shared/lib/recent-session";
import { TERMS } from "@/shared/lib/terms";
import { sentenceCase } from "@/shared/lib/formatters";
import { EmptyState } from "@/shared/ui/empty-state";

const NAV_ITEMS = perLocale(
  () =>
    [
      {
        href: "/",
        label: msg("auto.features.sidebar.components.sidebar.literal.1"),
        icon: SquaresFour,
      },
      // One entry covers the whole Data hub: the dataset library and the
      // labeling-session chooser are tabs of the same surface, so both route
      // prefixes light it up.
      {
        href: "/datasets",
        label: msg("sidebar.nav.data"),
        icon: Database,
        match: ["/datasets", "/tagger"],
      },
      // The glossary term is lowercase for mid-sentence use; nav items are
      // sentence-cased ("Explore", "Data"), so this one matches.
      { href: "/submit", label: sentenceCase(TERMS.notificationNewOpt), icon: PaperPlaneTilt },
      { href: "/explore", label: msg("sidebar.nav.explore"), icon: Compass },
    ] as const,
);

const PAGE_SIZE = 20;

const SIDEBAR_MIN_WIDTH = 210;
const SIDEBAR_MAX_WIDTH = 420;
const SIDEBAR_DEFAULT_WIDTH = 240;
// Icon-rail width when collapsed: just enough to center a nav icon (and the
// account avatar / storage icon) with comfortable padding.
const SIDEBAR_COLLAPSED_WIDTH = 64;
// Drag-to-collapse hysteresis. While expanded, the rail shrinks with the cursor
// and rests at its min; only pulling the grip in past COLLAPSE_AT (well under
// the min) snaps it shut. While collapsed, pushing the grip back out past
// EXPAND_AT re-expands it. The gap between the two thresholds means a jittery
// hand near the edge can't rapidly toggle the state — that gap is what a single
// threshold lacked, so it flickered between the icon rail and the min width.
const SIDEBAR_COLLAPSE_AT = 150;
const SIDEBAR_EXPAND_AT = 196;
const SIDEBAR_WIDTH_STORAGE_KEY = "skynet.sidebar.width";
const SIDEBAR_COLLAPSED_STORAGE_KEY = "skynet.sidebar.collapsed";
const DESKTOP_MQ = "(min-width: 768px)";

/** Clamp a candidate sidebar width to the resizable range, rounded to a whole px. */
function clampSidebarWidth(n: number): number {
  return Math.max(SIDEBAR_MIN_WIDTH, Math.min(SIDEBAR_MAX_WIDTH, Math.round(n)));
}

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
  const [width, setWidth] = React.useState(SIDEBAR_DEFAULT_WIDTH);
  const [collapsed, setCollapsed] = React.useState(false);
  // Arms the collapse/expand width glide for the discrete snap only; continuous
  // resize drags leave it false so the rail tracks the cursor 1:1.
  const [snapping, setSnapping] = React.useState(false);
  const [isDesktop, setIsDesktop] = React.useState(true);
  // Collapse is a desktop-only affordance: the mobile drawer always shows the
  // full rail, so gate the icon-rail on the breakpoint.
  const isCollapsed = collapsed && isDesktop;
  const effectiveWidth = isCollapsed ? SIDEBAR_COLLAPSED_WIDTH : width;
  // Collapsed tooltips read toward the content area: right in LTR, left in RTL.
  const tooltipSide = isRtl ? "left" : "right";

  // Width is a desktop affordance — the mobile drawer uses a viewport-capped
  // width regardless. Hydrate the persisted width client-side (SSR can't read
  // localStorage without risking a mismatch) and track the desktop breakpoint.
  React.useEffect(() => {
    try {
      const raw = window.localStorage.getItem(SIDEBAR_WIDTH_STORAGE_KEY);
      const n = raw ? Number(raw) : NaN;
      if (Number.isFinite(n)) setWidth(clampSidebarWidth(n));
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

  const persistWidth = React.useCallback((next: number) => {
    const clamped = clampSidebarWidth(next);
    setWidth(clamped);
    try {
      window.localStorage.setItem(SIDEBAR_WIDTH_STORAGE_KEY, String(clamped));
    } catch {
      /* noop */
    }
  }, []);

  const setCollapsedPersist = React.useCallback((value: boolean) => {
    setCollapsed((prev) => {
      if (prev === value) return prev;
      try {
        window.localStorage.setItem(SIDEBAR_COLLAPSED_STORAGE_KEY, value ? "1" : "0");
      } catch {
        /* noop */
      }
      return value;
    });
  }, []);

  // Drag-resize doubles as the collapse control. The rail is pinned to the
  // inline-start edge (left in LTR, right in RTL), so the dragged inline-end
  // edge maps to clientX in LTR and to (innerWidth - clientX) in RTL.
  //
  // ``persistWidth`` clamps to [min, max], so while expanded the rail tracks the
  // cursor and simply rests at its min — pulling in further does nothing until
  // the cursor crosses COLLAPSE_AT, which snaps to the icon rail. ``dragCollapsed``
  // holds the drag's own collapsed flag so the hysteresis compares against where
  // this gesture last committed, not a stale render value; that's what keeps a
  // jittery hand from toggling on a single pixel.
  const resizingRef = React.useRef(false);
  const startResize = React.useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      resizingRef.current = true;
      let dragCollapsed = collapsed;
      const prevUserSelect = document.body.style.userSelect;
      const prevCursor = document.body.style.cursor;
      document.body.style.userSelect = "none";
      document.body.style.cursor = "col-resize";
      const onMove = (ev: MouseEvent) => {
        if (!resizingRef.current) return;
        const raw = isRtl ? window.innerWidth - ev.clientX : ev.clientX;
        if (dragCollapsed) {
          if (raw > SIDEBAR_EXPAND_AT) {
            dragCollapsed = false;
            // Discrete snap back to text mode — glide the width jump.
            setSnapping(true);
            setCollapsedPersist(false);
            persistWidth(raw);
          }
        } else if (raw < SIDEBAR_COLLAPSE_AT) {
          dragCollapsed = true;
          // Discrete snap to the icon rail — glide the width jump.
          setSnapping(true);
          setCollapsedPersist(true);
        } else {
          // Continuous resize within range: track the cursor 1:1, no transition.
          setSnapping(false);
          persistWidth(raw);
        }
      };
      const onUp = () => {
        resizingRef.current = false;
        document.body.style.userSelect = prevUserSelect;
        document.body.style.cursor = prevCursor;
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      };
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    },
    [collapsed, isRtl, persistWidth, setCollapsedPersist],
  );

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

  const [deleteConfirm, setDeleteConfirm] = React.useState<{ id: string } | null>(null);
  const [deleteLoading, setDeleteLoading] = React.useState(false);

  const handleDelete = React.useCallback((e: React.MouseEvent, optimizationId: string) => {
    e.preventDefault();
    e.stopPropagation();
    setDeleteConfirm({ id: optimizationId });
  }, []);

  const confirmDelete = async () => {
    if (!deleteConfirm) return;
    const { id } = deleteConfirm;
    setDeleteLoading(true);
    setJobs((prev) => prev.filter((j) => j.optimization_id !== id));
    try {
      await deleteJob(id);
      toast.success(msg("sidebar.delete.success"));
      window.dispatchEvent(new Event("optimizations-changed"));
      if (pathname === `/optimizations/${id}`) router.push("/");
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
    const job = jobs.find((j) => j.optimization_id === deleteConfirm.id);
    if (!job) return { name: deleteConfirm.id, id: deleteConfirm.id };
    const name =
      job.name ||
      [job.module_name, job.optimizer_name].filter(Boolean).join(" · ") ||
      job.optimization_id.slice(0, 8);
    return { name, id: job.optimization_id };
  }, [deleteConfirm, jobs]);

  // The sidebar is position:fixed on desktop (a locked rail), so it's out of the
  // shell's flow and can't reserve its own width. Publish the live width as a CSS
  // var the shell's <main> consumes for its inline margin — one source of truth,
  // updated as the rail is dragged so content and rail never overlap.
  //
  // ``data-sidebar-snapping`` arms the matched rail-width / main-margin glide
  // (globals.css) for the discrete collapse/expand snap only. useLayoutEffect,
  // not useEffect: the attribute must be on <html> *before* the browser paints
  // the new width, or the snap paints once — unanimated — before the transition
  // could arm, which is exactly the jump this smooths out. Cleared on the next
  // continuous drag and when the glide's transitionend fires.
  React.useLayoutEffect(() => {
    const root = document.documentElement;
    root.style.setProperty("--app-sidebar-width", `${effectiveWidth}px`);
    if (snapping) root.setAttribute("data-sidebar-snapping", "");
    else root.removeAttribute("data-sidebar-snapping");
  }, [effectiveWidth, snapping]);

  return (
    <aside
      className="app-sidebar-rail relative flex h-full shrink-0 flex-col border-e border-sidebar-border/60 bg-sidebar/80 backdrop-blur-xl overflow-hidden"
      style={{ width: isDesktop ? `${effectiveWidth}px` : `min(${width}px, 88vw)` }}
      onTransitionEnd={(e) => {
        // Disarm once the collapse/expand width glide lands so the next
        // continuous drag tracks the cursor 1:1. Guard against transitionend
        // bubbling up from descendant width animations (e.g. the storage bar).
        if (e.target === e.currentTarget && e.propertyName === "width") setSnapping(false);
      }}
      data-tutorial="sidebar-full"
    >
      {/* Drag-to-resize grip on the rail's inline-end edge (desktop only — the
          mobile drawer isn't resizable). Also the collapse control: dragging it
          past the threshold snaps the rail shut, and it stays grabbable on the
          collapsed rail's edge to drag back open. Invisible until hovered, then a
          primary hairline; the grab math is mirrored for RTL in ``startResize``. */}
      <button
        type="button"
        onMouseDown={startResize}
        aria-label={msg("auto.features.sidebar.components.sidebar.literal.15")}
        tabIndex={-1}
        className="group absolute inset-y-0 end-0 z-20 hidden w-1.5 cursor-col-resize md:block"
      >
        <span
          aria-hidden="true"
          className="absolute inset-y-0 end-0 w-px bg-transparent transition-colors duration-150 group-hover:bg-primary/30 group-active:bg-primary/50"
        />
      </button>
      <div className="flex flex-col h-full">
        <nav
          className={cn(
            "flex flex-col gap-1 pb-3 pt-3",
            isCollapsed ? "px-2" : "px-3",
          )}
          role="navigation"
          aria-label={msg("auto.features.sidebar.components.sidebar.literal.7")}
          data-tutorial="sidebar-nav"
        >
          {NAV_ITEMS.map((item) => (
            <NavItem
              key={item.href}
              href={item.href}
              label={item.label}
              Icon={item.icon}
              collapsed={isCollapsed}
              tooltipSide={tooltipSide}
              active={
                item.href === "/"
                  ? pathname === "/"
                  : ("match" in item ? item.match : [item.href]).some((prefix) =>
                      pathname.startsWith(prefix),
                    )
              }
              badge={
                item.href === "/" && renderedTab === "mine" && activeCount > 0 ? activeCount : null
              }
              tutorialId={item.href === "/datasets" ? "sidebar-data" : undefined}
              resume={
                item.href === "/submit"
                  ? { kind: "optimization", detailBase: "/optimizations" }
                  : undefined
              }
            />
          ))}
        </nav>

        <div
          aria-hidden="true"
          className={cn("mx-3 h-px bg-sidebar-border/40", isCollapsed && "hidden")}
        />

        <div
          role="tablist"
          aria-label={msg("sidebar.tab.aria")}
          className={cn(
            "relative mx-3 mt-2.5 mb-0.5 flex rounded-lg bg-muted p-1 gap-1",
            isCollapsed && "hidden",
          )}
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

        {/* Collapsed rail hides the run list (no icon form for a text list), but
            the data layer keeps polling — the Dashboard nav badge count reads
            from the same fetch — so hide rather than unmount. */}
        <div
          className={cn("flex-1 overflow-hidden flex flex-col min-h-0", isCollapsed && "hidden")}
        >
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
                    icon={PaperPlaneTilt}
                    iconWrap="tile"
                    title={msg(
                      renderedTab === "shared" ? "sidebar.shared.empty" : "sidebar.mine.empty",
                    )}
                    description={msg(
                      renderedTab === "shared"
                        ? "sidebar.shared.empty.hint"
                        : "sidebar.mine.empty.hint",
                    )}
                    className="px-4 py-10"
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

        {/* Collapsed rail has no scrolling list to fill the middle, so a plain
            spacer keeps the footer pinned to the bottom. */}
        {isCollapsed && <div className="flex-1" aria-hidden="true" />}

        <div className="border-t border-sidebar-border/60">
          <StorageMeter collapsed={isCollapsed} />
          <div className={cn("py-2", isCollapsed ? "flex justify-center px-2" : "px-2")}>
            <AccountMenu collapsed={isCollapsed} />
          </div>
        </div>
      </div>

      <Dialog
        open={deleteConfirm !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteConfirm(null);
        }}
      >
        <DialogContent className="sm:max-w-md" showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>
              {`${msg("auto.features.sidebar.components.sidebar.3")}${TERMS.optimization}`}
            </DialogTitle>
            <DialogDescription>
              {`${msg("auto.features.sidebar.components.sidebar.4")}${TERMS.optimization}`}{" "}
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
                <CircleNotch className="size-4 animate-spin" />
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
 * One primary-nav entry: an icon + label row (with an optional count badge), gold
 * "you are here" treatment when active, and a subtle inline-start nudge on hover.
 */
function NavItem({
  href,
  label,
  Icon,
  active,
  badge,
  collapsed,
  tooltipSide,
  tutorialId,
  resume,
}: {
  href: string;
  label: string;
  Icon: React.ComponentType<{ className?: string }>;
  active: boolean;
  badge: number | null;
  /** Icon-only rail: hide the label and name the entry via a hover tooltip. */
  collapsed: boolean;
  /** Which side the collapsed tooltip opens toward (content-ward). */
  tooltipSide: "left" | "right";
  /** ``data-tutorial`` anchor for entries the tutorial spotlights. */
  tutorialId?: string;
  resume?: { kind: "tagger" | "optimization"; detailBase: string };
}) {
  const router = useRouter();
  // When this entry can resume a recent session, decide at click time: if the
  // user left one within the resume window, reopen it; otherwise fall through to
  // the plain href (start a new session). Checked on click so the time window is
  // measured live, not at render.
  const onClick = resume
    ? (e: React.MouseEvent) => {
        // Let modifier-clicks (open-in-new-tab) fall through to the plain href.
        if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
        const id = recentResumableId(resume.kind);
        if (!id) return;
        e.preventDefault();
        router.push(`${resume.detailBase}/${id}`);
      }
    : undefined;

  if (collapsed) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Link
            href={href}
            onClick={onClick}
            {...(tutorialId ? { "data-tutorial": tutorialId } : {})}
            aria-label={label}
            aria-current={active ? "page" : undefined}
            className={cn(
              "group relative flex items-center justify-center rounded-lg p-2.5 transition-colors duration-200",
              active
                ? "bg-primary/[0.08] text-primary ring-1 ring-primary/10"
                : "text-sidebar-foreground/60 hover:bg-sidebar-accent/40 hover:text-sidebar-foreground",
            )}
          >
            <Icon
              className={cn(
                "size-5 shrink-0 transition-colors duration-200",
                active ? "text-primary" : "group-hover:text-sidebar-foreground",
              )}
            />
            {badge != null && (
              <span className="absolute -end-0.5 -top-0.5 grid min-w-4 place-items-center rounded-full bg-primary px-1 text-[0.5625rem] font-bold leading-4 text-primary-foreground tabular-nums">
                {badge}
              </span>
            )}
          </Link>
        </TooltipTrigger>
        <TooltipContent side={tooltipSide}>{label}</TooltipContent>
      </Tooltip>
    );
  }

  return (
    <Link
      href={href}
      onClick={onClick}
      {...(tutorialId ? { "data-tutorial": tutorialId } : {})}
      className={cn(
        "group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-all duration-200",
        active
          ? "text-primary"
          : "text-sidebar-foreground/60 hover:translate-x-[-2px] hover:bg-sidebar-accent/40 hover:text-sidebar-foreground",
      )}
    >
      {active && (
        <motion.div
          layoutId="sidebar-active"
          className="absolute inset-0 rounded-lg bg-primary/[0.08] ring-1 ring-primary/10"
          style={{ borderInlineStart: "3px solid var(--primary)" }}
          transition={{ type: "spring", stiffness: 350, damping: 28 }}
        />
      )}
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
    </Link>
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
  const isRtl = getActiveDir() === "rtl";
  const [menuOpen, setMenuOpen] = React.useState(false);
  const [renaming, setRenaming] = React.useState(false);
  const [renameValue, setRenameValue] = React.useState("");
  const [menuPos, setMenuPos] = React.useState<{ top: number; left: number } | null>(null);
  const [expanded, setExpanded] = React.useState(isActive && activePair !== null);
  const menuRef = React.useRef<HTMLDivElement>(null);
  const btnRef = React.useRef<HTMLButtonElement>(null);
  const renameRef = React.useRef<HTMLInputElement>(null);
  const isGridSearch = job.optimization_type === "grid_search" && (job.total_pairs ?? 0) > 0;
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
              <GridFour className="size-2.5" />
              {job.total_pairs ?? "?"}
            </span>
          )}
          {job.pinned && <PushPin className="size-2.5 text-muted-foreground/60 shrink-0" />}
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
            <CaretLeft
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
            <DotsThree
              className={cn(
                "size-3.5 transition-[transform,color] duration-200 ease-[cubic-bezier(0.2,0.8,0.2,1)] motion-reduce:transition-none motion-reduce:transform-none",
                menuOpen && "rotate-90 scale-110 text-foreground",
              )}
              aria-hidden="true"
            />
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
                <ShareNetwork className="size-3.5 text-muted-foreground" />
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
                <PencilSimple className="size-3.5 text-muted-foreground" />
                {msg("auto.features.sidebar.components.sidebar.8")}
              </button>
            )}

            <button
              type="button"
              role="menuitem"
              onClick={handleClone}
              className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[0.6875rem] text-foreground hover:bg-muted/40 cursor-pointer transition-colors"
            >
              <Copy className="size-3.5 text-muted-foreground" />
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
                  <ArrowCounterClockwise className="size-3.5 text-muted-foreground" />
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
                  <PushPin
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
                <Trash className="size-3.5" />
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
