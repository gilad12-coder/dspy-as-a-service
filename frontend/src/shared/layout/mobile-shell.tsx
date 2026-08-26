"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Compass, House, UserCircle, type Icon } from "@/shared/ui/icons";
import { GlobalSearch } from "@/shared/layout/global-search";
import { AccountMenu } from "@/shared/layout/account-menu";
import { PageContainer } from "@/shared/layout/page-container";
import { DesktopOnlyNotice } from "@/shared/layout/desktop-only-notice";
import { useLocale } from "@/shared/providers";
import { dirForLocale } from "@/shared/lib/locale";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { isDesktopOnlyPath } from "@/shared/lib/device-class";
import { JobsStreamProvider } from "@/shared/hooks/use-jobs-stream";
import {
  AgentPillDock,
  GeneralistPanel,
  GeneralistPanelProvider,
  WizardStateProvider,
  isGeneralistAgentEnabled,
} from "@/features/agent-panel";

const HEADER_HEIGHT_PX = 53;
const TABBAR_HEIGHT_PX = 56;

const TAB_CLASS =
  "flex h-full w-full flex-col items-center justify-center gap-0.5 text-[0.6875rem] font-medium transition-colors duration-150 cursor-pointer focus-visible:outline-none focus-visible:bg-accent/60";

function TabLink({
  href,
  icon: TabIcon,
  label,
  active,
}: {
  href: string;
  icon: Icon;
  label: string;
  active: boolean;
}) {
  return (
    <Link
      href={href}
      aria-current={active ? "page" : undefined}
      className={cn(TAB_CLASS, active ? "text-primary" : "text-muted-foreground")}
    >
      <TabIcon className="size-5" weight={active ? "fill" : "bold"} aria-hidden="true" />
      {label}
    </Link>
  );
}

/**
 * The phone shell (viewport ≤767px): a slim top bar (wordmark + search), the
 * page, and a three-tab bottom bar — Home, Explore, Account. No sidebar, no
 * ambient orbs, no tutorial chrome: phones are for checking on runs, not for
 * authoring them, so desktop-only routes render the notice in place of the page.
 * Providers mirror the desktop shell so streams and the agent panel behave the
 * same on both.
 */
export function MobileShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const { locale } = useLocale();
  const dir = dirForLocale(locale);
  const generalistEnabled = isGeneralistAgentEnabled();

  const shell = (
    <div
      className="flex min-h-screen flex-col"
      style={{
        ["--header-height" as string]: `${HEADER_HEIGHT_PX}px`,
        ["--tabbar-height" as string]: `calc(${TABBAR_HEIGHT_PX}px + env(safe-area-inset-bottom))`,
      }}
    >
      <header
        className="fixed inset-x-0 top-0 z-30 flex h-[var(--header-height)] items-center justify-between border-b border-border/40 bg-background/80 px-3 backdrop-blur-xl"
        dir="ltr"
      >
        <Link
          href="/"
          className="text-sm font-bold uppercase tracking-[0.14em] text-foreground"
          style={{ fontFamily: "var(--font-ui)" }}
        >
          SKYNET
        </Link>
        <GlobalSearch />
      </header>

      <main
        className="min-w-0 flex-1 pt-[var(--header-height)] pb-[var(--tabbar-height)]"
        dir={dir}
      >
        <PageContainer>
          {isDesktopOnlyPath(pathname) ? <DesktopOnlyNotice /> : children}
        </PageContainer>
      </main>

      {/* The agent's floating pill defaults to the viewport corner, which the tab
          bar now owns; dock it just above the bar instead. */}
      {generalistEnabled && (
        <AgentPillDock className="fixed end-4 bottom-[calc(var(--tabbar-height)+0.75rem)] z-30" />
      )}

      <nav
        aria-label={msg("mobile.nav.aria")}
        className="fixed inset-x-0 bottom-0 z-30 border-t border-border/40 bg-background/90 pb-[env(safe-area-inset-bottom)] backdrop-blur-xl"
        dir={dir}
      >
        <ul className="grid h-[56px] grid-cols-3">
          <li>
            <TabLink
              href="/"
              icon={House}
              label={msg("mobile.nav.home")}
              active={pathname === "/"}
            />
          </li>
          <li>
            <TabLink
              href="/explore"
              icon={Compass}
              label={msg("sidebar.nav.explore")}
              active={pathname.startsWith("/explore")}
            />
          </li>
          <li>
            <AccountMenu
              align="end"
              trigger={
                <button
                  type="button"
                  aria-label={msg("app.shell.account.aria")}
                  className={cn(TAB_CLASS, "text-muted-foreground")}
                >
                  <UserCircle className="size-5" aria-hidden="true" />
                  {msg("mobile.nav.account")}
                </button>
              }
            />
          </li>
        </ul>
      </nav>
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
