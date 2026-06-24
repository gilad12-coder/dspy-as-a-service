"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { usePathname } from "next/navigation";
import { motion } from "framer-motion";
import { Menu, LogOut, GraduationCap, Lightbulb, Feather, Sparkles } from "lucide-react";
import { useSession, signOut } from "next-auth/react";
import { AnimatedWordmark } from "@/shared/ui/animated-wordmark";
import { useTutorialContext, ConceptsGuide, registerTutorialHook } from "@/features/tutorial";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/shared/ui/primitives/tooltip";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { LanguageSwitcher } from "@/shared/ui/language-switcher";
import { useLocale } from "@/shared/providers";
import { dirForLocale } from "@/shared/lib/locale";
import { msg } from "@/shared/lib/messages";
import { JobsStreamProvider } from "@/shared/hooks/use-jobs-stream";
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

  return <ShellChrome>{children}</ShellChrome>;
}

function ShellChrome({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = React.useState(false);
  const [conceptsOpen, setConceptsOpen] = React.useState(false);
  const [isDesktop, setIsDesktop] = React.useState(false);
  const { data: session } = useSession();
  const pathname = usePathname();
  const { locale } = useLocale();
  const dir = dirForLocale(locale);
  const isRtl = dir === "rtl";
  const { prefs, setPref } = useUserPrefs();
  const { startDeepDive } = useTutorialContext();
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
        className="sticky top-0 z-30 flex items-center justify-between bg-background/60 backdrop-blur-2xl backdrop-saturate-[1.8] px-4 py-2.5 border-b border-border/40 shadow-[0_1px_3px_rgba(0,0,0,0.04),0_4px_12px_rgba(0,0,0,0.03)]"
        dir="ltr"
        style={{
          borderImage:
            "linear-gradient(to right, transparent, var(--border) 20%, var(--border) 80%, transparent) 1",
        }}
      >
        <div className="flex items-center gap-1.5 cursor-default">
          <div className="hidden sm:block">
            <AnimatedWordmark size={16} autoMorph morphSpeed={250} />
          </div>
          <span
            className="sm:hidden text-sm font-bold tracking-[0.14em] uppercase text-foreground cursor-default"
            style={{ fontFamily: '"Inter Variable", system-ui, sans-serif' }}
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
          {prefs.advancedMode && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  onClick={() => setPref("advancedMode", false)}
                  className="inline-flex items-center gap-1 rounded-full border border-border bg-accent/60 px-2 py-0.5 text-[0.65rem] font-semibold uppercase tracking-wide text-muted-foreground hover:bg-accent hover:text-foreground cursor-pointer"
                  aria-label={msg("app.shell.advanced.exit_aria")}
                >
                  <Sparkles className="size-3" aria-hidden="true" />
                  {msg("app.shell.advanced.badge")}
                </button>
              </TooltipTrigger>
              <TooltipContent side="bottom" dir={dir}>
                {msg("app.shell.advanced.tooltip")}
              </TooltipContent>
            </Tooltip>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={startDeepDive}
                className="rounded-lg p-1.5 hover:bg-accent/80 active:scale-95 transition-all duration-200 cursor-pointer text-muted-foreground hover:text-foreground inline-flex items-center justify-center"
                aria-label={msg("app.shell.tour_aria")}
              >
                <GraduationCap className="size-4" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom" dir={dir}>
              {msg("app.shell.tour_tooltip")}
            </TooltipContent>
          </Tooltip>
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
        </div>

        <div className="flex items-center gap-1.5">
          <LanguageSwitcher />
          {session?.user && (
            <>
              <span className="hidden sm:inline text-xs font-semibold text-foreground">
                {session.user.name ?? session.user.email}
              </span>
              <TooltipButton tooltip={msg("app.shell.logout")} side="bottom">
                <button
                  type="button"
                  onClick={() => signOut({ callbackUrl: "/login" })}
                  className="rounded-lg p-2 hover:bg-accent/80 active:scale-95 transition-all duration-200 cursor-pointer hidden sm:block"
                  aria-label={msg("app.shell.logout")}
                >
                  <LogOut className="size-4" />
                </button>
              </TooltipButton>
            </>
          )}
          <button
            type="button"
            onClick={() => setSidebarOpen(true)}
            className="rounded-lg p-2 hover:bg-accent/80 active:scale-95 transition-all duration-200 md:hidden"
            aria-label={msg("app.shell.menu")}
            aria-expanded={sidebarOpen}
            aria-controls={SIDEBAR_ID}
          >
            <Menu className="size-5" />
          </button>
        </div>
      </motion.header>

      <div className="flex flex-1" dir={dir}>
        <button
          type="button"
          aria-label={msg("app.shell.menu_close")}
          onClick={() => setSidebarOpen(false)}
          className={`fixed inset-0 z-40 bg-black/40 backdrop-blur-sm md:hidden transition-all duration-300 ease-out ${sidebarOpen ? "opacity-100" : "opacity-0 pointer-events-none"}`}
          tabIndex={sidebarOpen ? 0 : -1}
          aria-hidden={!sidebarOpen}
        />

        <main className="flex-1 overflow-auto min-w-0 page-gradient grid-pattern">
          <div
            className="relative z-[1] mx-auto max-w-7xl py-6 md:py-8"
            style={{ paddingInline: "clamp(1rem, 5vw - 0.5rem, 2rem)" }}
          >
            {children}
          </div>
        </main>

        {/* The sidebar lives after <main> in source order (content-first for
            assistive tech) but renders at the inline-start edge via md:order-first
            — right in Hebrew/RTL, left in English/LTR. Below md it's an off-canvas
            drawer pinned to that same inline-start edge. */}
        <div
          id={SIDEBAR_ID}
          className={`fixed inset-y-0 ${isRtl ? "right-0" : "left-0"} z-50 transform transition-all duration-300 ease-[cubic-bezier(0.16,1,0.3,1)] md:sticky md:inset-auto md:top-[var(--header-height)] md:order-first md:self-start md:h-[calc(100dvh-var(--header-height))] md:z-10 md:translate-x-0 md:shadow-none ${sidebarOpen ? "translate-x-0" : isRtl ? "translate-x-full" : "-translate-x-full"}`}
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
