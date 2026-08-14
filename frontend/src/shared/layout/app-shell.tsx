"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Popover as PopoverPrimitive } from "radix-ui";
import { List, GraduationCap, Lightbulb, Feather } from "@/shared/ui/icons";
import { ConceptsGuide, registerTutorialHook, TutorialMenu } from "@/features/tutorial";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/shared/ui/primitives/tooltip";
import { AnimatedWordmark } from "@/shared/ui/animated-wordmark";
import { GlobalSearch } from "@/shared/layout/global-search";
import { useLocale } from "@/shared/providers";
import { dirForLocale } from "@/shared/lib/locale";
import { msg } from "@/shared/lib/messages";
import { JobsStreamProvider } from "@/shared/hooks/use-jobs-stream";
import { PageContainer } from "@/shared/layout/page-container";
import { useUserPrefs, LiteModeHint } from "@/features/settings";
import {
  GeneralistPanel,
  GeneralistPanelProvider,
  WizardStateProvider,
  isGeneralistAgentEnabled,
} from "@/features/agent-panel";

const Sidebar = dynamic(() => import("@/features/sidebar").then((m) => m.Sidebar), { ssr: false });

const HEADER_HEIGHT_PX = 53;
const SIDEBAR_ID = "app-sidebar";
const DESKTOP_BP = "(min-width: 768px)";

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { locale } = useLocale();
  const dir = dirForLocale(locale);

  // Shared-optimization pages render bare — no sidebar, agent panel, or other
  // app chrome — to keep the focus on the shared item. The recipient is still
  // authenticated (the route is login-gated like the rest of the app).
  if (pathname.startsWith("/share/")) {
    return (
      <main className="min-h-screen" dir={dir}>
        {children}
      </main>
    );
  }

  if (pathname === "/login") {
    return (
      <main className="min-h-screen" dir={dir}>
        {children}
      </main>
    );
  }

  // The legal pages are public, standalone English documents that ship their
  // own header and footer, so they render bare (no sidebar or app chrome) and
  // force dir="ltr" regardless of the UI locale.
  if (pathname === "/terms" || pathname === "/privacy") {
    return (
      <main className="min-h-screen" dir="ltr">
        {children}
      </main>
    );
  }

  return <ShellChrome>{children}</ShellChrome>;
}

function ShellChrome({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);
  const [conceptsOpen, setConceptsOpen] = React.useState(false);
  const [isDesktop, setIsDesktop] = React.useState(false);
  const pathname = usePathname();
  const { locale } = useLocale();
  const dir = dirForLocale(locale);
  const isRtl = dir === "rtl";
  const { prefs, setPref } = useUserPrefs();
  const generalistEnabled = isGeneralistAgentEnabled();
  const progressRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    setSidebarOpen(false);
  }, [pathname]);

  // Let the tutorial slide the sidebar drawer in/out so its spotlight steps
  // have an on-screen target below 768px, where the sidebar is off-canvas.
  React.useEffect(() => registerTutorialHook("setSidebarOpen", setSidebarOpen), []);

  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setSidebarOpen(false);
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, []);

  React.useEffect(() => {
    const mq = window.matchMedia(DESKTOP_BP);
    const update = () => setIsDesktop(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  React.useEffect(() => {
    const el = progressRef.current;
    if (!el) return;
    const onScroll = () => {
      const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
      const scrollHeight =
        document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const progress = scrollHeight > 0 ? scrollTop / scrollHeight : 0;
      el.style.setProperty("--scroll-progress", String(Math.min(progress, 1)));
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const sidebarHidden = !isDesktop && !sidebarOpen;

  const shell = (
    <div
      className="flex min-h-screen flex-col"
      style={{ ["--header-height" as string]: `${HEADER_HEIGHT_PX}px` }}
    >
      <div ref={progressRef} className="scroll-progress" aria-hidden="true" />

      <div className="ambient-bg" aria-hidden="true">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
        <div className="orb orb-4" />
      </div>

      <motion.header
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className="fixed inset-x-0 top-0 z-30 flex items-center justify-between bg-background/60 backdrop-blur-2xl backdrop-saturate-[1.8] px-4 py-2.5 border-b border-border/40 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_4px_12px_rgba(0,0,0,0.03)]"
        dir="ltr"
        style={{
          borderImage:
            "linear-gradient(to right, transparent, var(--border) 20%, var(--border) 80%, transparent) 1",
        }}
      >
        <div className="flex items-center gap-1.5 cursor-default">
          {/* The morphing wordmark anchors the header at sm+; below sm it collapses
              to a plain SKYNET wordmark where the full SVG would crowd the row. */}
          <div className="hidden sm:block">
            <AnimatedWordmark size={16} autoMorph autoMorphDuration={10000} morphSpeed={250} />
          </div>
          <span
            className="sm:hidden text-sm font-bold tracking-[0.14em] uppercase text-foreground cursor-default"
            style={{ fontFamily: "var(--font-ui)" }}
          >
            SKYNET
          </span>
          {prefs.liteMode && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  onClick={() => setPref("liteMode", false)}
                  className="inline-flex items-center gap-1 rounded-full border border-border bg-accent/60 px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide text-muted-foreground hover:bg-accent hover:text-foreground cursor-pointer"
                  aria-label={msg("app.shell.lite.exit_aria")}
                >
                  <Feather className="size-3" aria-hidden="true" />
                  {msg("app.shell.lite.badge")}
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" dir={dir}>
                {msg("app.shell.lite.tooltip")}
              </TooltipContent>
            </Tooltip>
          )}
        </div>

        <div className="flex flex-1 justify-center px-2">
          <GlobalSearch />
        </div>

        <div className="flex items-center gap-1.5" dir={dir}>
          {/* Tutorials are grouped by workflow so this button opens a replayable
              guide chooser rather than starting one long product tour. */}
          <PopoverPrimitive.Root>
            <Tooltip>
              <TooltipTrigger asChild>
                <PopoverPrimitive.Trigger asChild>
                  <button
                    type="button"
                    className="rounded-lg p-1.5 hover:bg-accent/80 active:scale-95 transition-all duration-200 cursor-pointer text-muted-foreground hover:text-foreground inline-flex items-center justify-center"
                    aria-label={msg("app.shell.tour_aria")}
                  >
                    <GraduationCap className="size-4" />
                  </button>
                </PopoverPrimitive.Trigger>
              </TooltipTrigger>
              <TooltipContent side="bottom" dir={dir}>
                {msg("app.shell.tour_tooltip")}
              </TooltipContent>
            </Tooltip>
            <TutorialMenu />
          </PopoverPrimitive.Root>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={() => setConceptsOpen(true)}
                className="rounded-lg p-1.5 hover:bg-accent/80 active:scale-95 transition-all duration-200 cursor-pointer text-muted-foreground hover:text-foreground inline-flex items-center justify-center"
                aria-label={msg("app.shell.concepts_aria")}
              >
                <Lightbulb className="size-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" dir={dir}>
              {msg("app.shell.concepts_tooltip")}
            </TooltipContent>
          </Tooltip>
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            className="rounded-lg p-2 hover:bg-accent/80 active:scale-95 transition-all duration-200 md:hidden"
            aria-label={msg("app.shell.menu")}
            aria-expanded={sidebarOpen}
            aria-controls={SIDEBAR_ID}
          >
            <List className="size-5" />
          </button>
        </div>
      </motion.header>

      {/* The header is position:fixed, so it's out of flow — reserve its height
          here so content starts below it. The fixed sidebar inside this row is
          unaffected by the padding and stays pinned at top:var(--header-height). */}
      <div className="flex flex-1 pt-[var(--header-height)]" dir={dir}>
        <button
          type="button"
          aria-label={msg("app.shell.menu_close")}
          onClick={() => setSidebarOpen(false)}
          className={`fixed inset-0 z-40 bg-black/40 backdrop-blur-sm md:hidden transition-all duration-300 ease-out ${sidebarOpen ? "opacity-100" : "opacity-0 pointer-events-none"}`}
          tabIndex={sidebarOpen ? 0 : -1}
          aria-hidden={!sidebarOpen}
        />

        <main className="app-main flex-1 overflow-auto min-w-0 page-gradient grid-pattern">
          {/* The tagger picks its own container per phase (full-width working
              surface vs the capped session chooser), so its routes render bare
              and wrap themselves in PageContainer — nesting it inside the
              shell's box would double the inline padding. */}
          {pathname.startsWith("/tagger") ? (
            children
          ) : (
            <PageContainer>{children}</PageContainer>
          )}
        </main>

        {/* The sidebar lives after <main> in source order (content-first for
            assistive tech) but renders at the inline-start edge — right in
            Hebrew/RTL, left in English/LTR. On md+ it's position:fixed below the
            header: a locked rail that never scrolls with the page, with <main>
            reserving its width via the --app-sidebar-width inline margin. Below md
            it's an off-canvas drawer pinned to that same inline-start edge. */}
        <div
          id={SIDEBAR_ID}
          className={`fixed inset-y-0 ${isRtl ? "right-0" : "left-0"} z-50 transform transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] md:top-[var(--header-height)] md:z-10 md:translate-x-0 md:shadow-none ${sidebarOpen ? "translate-x-0" : isRtl ? "translate-x-full" : "-translate-x-full"}`}
          aria-hidden={sidebarHidden ? true : undefined}
        >
          <Sidebar />
        </div>
      </div>

      {conceptsOpen && <ConceptsGuide open onClose={() => setConceptsOpen(false)} />}
      <LiteModeHint />
    </div>
  );

  if (!generalistEnabled) return <JobsStreamProvider>{shell}</JobsStreamProvider>;

  return (
    <WizardStateProvider>
      <GeneralistPanelProvider>
        <JobsStreamProvider>{shell}</JobsStreamProvider>
        <GeneralistPanel />
      </GeneralistPanelProvider>
    </WizardStateProvider>
  );
}
