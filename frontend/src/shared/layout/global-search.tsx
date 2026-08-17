"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { useSession } from "next-auth/react";
import {
  ArrowUpRight,
  ChartBar,
  Compass,
  Database,
  GearSix,
  HardDrive,
  MagnifyingGlass,
  Robot,
  Sparkle,
  SquaresFour,
  Tag,
} from "@/shared/ui/icons";
import type { Icon } from "@/shared/ui/icons";

import { useSettingsModal } from "@/features/settings";
import { useIsPhone } from "@/shared/hooks/use-device-class";
import { isDesktopOnlyPath } from "@/shared/lib/device-class";
import { useLocale } from "@/shared/providers";
import { dirForLocale } from "@/shared/lib/locale";
import { msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";
import { cn } from "@/shared/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";

type SearchGroup = "quick" | "navigate" | "settings";

type SearchItem = {
  id: string;
  group: SearchGroup;
  label: string;
  description?: string;
  keywords: string[];
  icon: Icon;
  href?: string;
  settingsTab?: string;
};

const SETTINGS_ITEMS: Array<{
  id: string;
  label: string;
  kwKey: string;
  icon: Icon;
  settingsTab: string;
}> = [
  {
    id: "settings-wizard",
    label: "settings.tab.wizard",
    kwKey: "app.shell.search.kw.wizard",
    icon: Sparkle,
    settingsTab: "wizard",
  },
  {
    id: "settings-tagging",
    label: "settings.tab.tagging",
    kwKey: "app.shell.search.kw.tagging",
    icon: Tag,
    settingsTab: "tagging",
  },
  {
    id: "settings-agent",
    label: "settings.tab.agent",
    kwKey: "app.shell.search.kw.agent",
    icon: Robot,
    settingsTab: "agent",
  },
  {
    id: "settings-account",
    label: "settings.tab.account",
    kwKey: "app.shell.search.kw.account",
    icon: GearSix,
    settingsTab: "account",
  },
  {
    id: "settings-security",
    label: "settings.tab.security",
    kwKey: "app.shell.search.kw.security",
    icon: GearSix,
    settingsTab: "security",
  },
  {
    id: "settings-privacy",
    label: "settings.tab.privacy",
    kwKey: "app.shell.search.kw.privacy",
    icon: GearSix,
    settingsTab: "privacy",
  },
  {
    id: "settings-billing",
    label: "settings.tab.billing",
    kwKey: "app.shell.search.kw.billing",
    icon: GearSix,
    settingsTab: "billing",
  },
  {
    id: "settings-usage",
    label: "settings.tab.usage",
    kwKey: "app.shell.search.kw.usage",
    icon: ChartBar,
    settingsTab: "usage",
  },
  {
    id: "settings-providers",
    label: "settings.tab.providers",
    kwKey: "app.shell.search.kw.providers",
    icon: GearSix,
    settingsTab: "providers",
  },
  {
    id: "settings-api",
    label: "settings.tab.api",
    kwKey: "app.shell.search.kw.api",
    icon: GearSix,
    settingsTab: "api",
  },
  {
    id: "settings-about",
    label: "settings.tab.about",
    kwKey: "app.shell.search.kw.about",
    icon: GearSix,
    settingsTab: "about",
  },
];

const GROUP_ORDER: SearchGroup[] = ["quick", "navigate", "settings"];

function keyLabel(value: string): string {
  return value.startsWith("settings.") ? msg(value as Parameters<typeof msg>[0]) : value;
}

/** Render the global navigation/search trigger and its command palette. */
export function GlobalSearch() {
  const router = useRouter();
  const { locale } = useLocale();
  const dir = dirForLocale(locale);
  const { data: session } = useSession();
  const { open: settingsOpen, openTo } = useSettingsModal();
  const isPhone = useIsPhone();
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState("");
  const [activeIndex, setActiveIndex] = React.useState(0);
  const inputRef = React.useRef<HTMLInputElement | null>(null);

  const items = React.useMemo<SearchItem[]>(() => {
    const quickActions: SearchItem[] = [
      {
        id: "new-optimization",
        group: "quick",
        label: TERMS.notificationNewOpt,
        description: msg("app.shell.search.new_optimization_description"),
        keywords: msg("app.shell.search.kw.new_optimization")
          .split(/[\s,]+/)
          .filter(Boolean),
        icon: Sparkle,
        href: "/submit",
      },
      {
        id: "tagging",
        group: "quick",
        label: msg("data.tabs.sessions"),
        description: msg("app.shell.search.tagging_description"),
        keywords: msg("app.shell.search.kw.tagging")
          .split(/[\s,]+/)
          .filter(Boolean),
        icon: Tag,
        href: "/tagger",
      },
      {
        id: "settings",
        group: "quick",
        label: msg("app.shell.account.settings"),
        description: msg("settings.subtitle"),
        keywords: msg("app.shell.search.kw.account")
          .split(/[\s,]+/)
          .filter(Boolean),
        icon: GearSix,
        settingsTab: "account",
      },
    ];
    const navigation: SearchItem[] = [
      {
        id: "dashboard",
        group: "navigate",
        label: msg("app.shell.search.dashboard"),
        keywords: msg("app.shell.search.kw.dashboard")
          .split(/[\s,]+/)
          .filter(Boolean),
        icon: SquaresFour,
        href: "/",
      },
      {
        id: "data",
        group: "navigate",
        label: msg("sidebar.nav.data"),
        keywords: msg("app.shell.search.kw.data")
          .split(/[\s,]+/)
          .filter(Boolean),
        icon: Database,
        href: "/datasets",
      },
      {
        id: "explore",
        group: "navigate",
        label: msg("sidebar.nav.explore"),
        keywords: msg("app.shell.search.kw.explore")
          .split(/[\s,]+/)
          .filter(Boolean),
        icon: Compass,
        href: "/explore",
      },
      {
        id: "storage",
        group: "navigate",
        label: msg("app.shell.search.storage"),
        keywords: msg("app.shell.search.kw.storage")
          .split(/[\s,]+/)
          .filter(Boolean),
        icon: HardDrive,
        href: "/storage",
      },
    ];
    const settings = (
      session?.user?.role === "admin"
        ? [
            ...SETTINGS_ITEMS,
            {
              id: "settings-admin",
              label: "settings.tab.admin",
              // i18n-driven like the other settings items
              kwKey: "app.shell.search.kw.admin",
              icon: HardDrive,
              settingsTab: "admin",
            } as unknown as (typeof SETTINGS_ITEMS)[number],
          ]
        : SETTINGS_ITEMS
    ).map((item) => ({
      ...item,
      group: "settings" as const,
      label: keyLabel(item.label),
      keywords:
        "kwKey" in item && typeof (item as { kwKey?: unknown }).kwKey === "string"
          ? msg((item as { kwKey: string }).kwKey as Parameters<typeof msg>[0])
              .split(/[\s,]+/)
              .filter(Boolean)
          : ((item as unknown as SearchItem).keywords ?? []),
      description: msg("app.shell.search.settings_description"),
    }));
    const all: SearchItem[] = [...quickActions, ...navigation, ...settings];
    // The phone shell replaces authoring routes with a notice; don't offer them.
    return isPhone ? all.filter((item) => !item.href || !isDesktopOnlyPath(item.href)) : all;
  }, [session?.user?.role, isPhone]);

  const filteredItems = React.useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    if (!normalized) return items.filter((item) => item.group !== "settings");
    return items.filter((item) =>
      [item.label, item.description, ...item.keywords]
        .filter(Boolean)
        .join(" ")
        .toLocaleLowerCase()
        .includes(normalized),
    );
  }, [items, query]);

  React.useEffect(() => {
    setActiveIndex(0);
  }, [filteredItems.length, query]);

  React.useEffect(() => {
    if (!open) {
      setQuery("");
      return;
    }
    const frame = requestAnimationFrame(() => inputRef.current?.focus());
    return () => cancelAnimationFrame(frame);
  }, [open]);

  React.useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        if (settingsOpen) return;
        event.preventDefault();
        setOpen(true);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [settingsOpen]);

  const selectItem = (item: SearchItem) => {
    setOpen(false);
    if (item.settingsTab) {
      openTo(item.settingsTab);
      return;
    }
    if (item.href) router.push(item.href);
  };

  const handleInputKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveIndex((index) => (index + 1) % Math.max(filteredItems.length, 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveIndex(
        (index) =>
          (index - 1 + Math.max(filteredItems.length, 1)) % Math.max(filteredItems.length, 1),
      );
    } else if (event.key === "Enter" && filteredItems[activeIndex]) {
      event.preventDefault();
      selectItem(filteredItems[activeIndex]);
    }
  };

  const groupLabels: Record<SearchGroup, string> = {
    quick: msg("app.shell.search.quick_actions"),
    navigate: msg("app.shell.search.navigate"),
    settings: msg("app.shell.search.settings"),
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        dir={dir}
        className="group inline-flex size-[44px] shrink-0 items-center justify-center gap-3 rounded-xl border border-border/70 bg-background/80 px-2.5 text-muted-foreground shadow-none transition-[background-color,border-color,color,box-shadow] duration-150 hover:border-border hover:bg-accent/50 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/35 lg:h-8 lg:w-64 lg:justify-between"
        aria-label={msg("app.shell.search.button_aria")}
        aria-keyshortcuts="Control+K Meta+K"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls="global-search-dialog"
      >
        <span className="flex min-w-0 items-center gap-1.5">
          <MagnifyingGlass
            className="size-4 shrink-0 text-muted-foreground transition-colors duration-150 group-hover:text-primary"
            aria-hidden="true"
          />
          <span
            dir="auto"
            className="hidden truncate text-start text-[0.8rem] font-normal tracking-tight text-muted-foreground lg:block"
          >
            {msg("app.shell.search.label")}
          </span>
        </span>
        <span dir="ltr" className="hidden shrink-0 items-center gap-1 lg:flex" aria-hidden="true">
          <kbd className="inline-flex h-[18px] min-w-5 items-center justify-center rounded-md border border-border/70 bg-muted/55 px-1 text-[0.6875rem] font-medium text-muted-foreground">
            {msg("app.shell.search.command_key")}
          </kbd>
          <kbd className="inline-flex h-[18px] min-w-5 items-center justify-center rounded-md border border-border/70 bg-muted/55 px-1 text-[0.6875rem] font-medium text-muted-foreground">
            {msg("app.shell.search.k_key")}
          </kbd>
        </span>
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent
          id="global-search-dialog"
          showCloseButton={false}
          dir={dir}
          className="max-w-[calc(100%-1.5rem)] gap-0 overflow-hidden rounded-2xl border border-[#DDD4C8]/75 bg-[#FAF8F5] p-0 shadow-[0_16px_48px_rgba(28,22,18,0.16)] sm:max-w-xl"
        >
          <DialogHeader className="sr-only">
            <DialogTitle>{msg("app.shell.search.title")}</DialogTitle>
            <DialogDescription>{msg("app.shell.search.description")}</DialogDescription>
          </DialogHeader>
          <div className="group flex h-[58px] items-center gap-3 border-b border-border/60 px-4 focus-within:bg-white/50">
            <MagnifyingGlass
              className="size-5 shrink-0 text-primary/60 transition-colors group-focus-within:text-primary"
              aria-hidden="true"
            />
            <input
              ref={inputRef}
              id="global-search-input"
              name="global-search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={handleInputKeyDown}
              placeholder={msg("app.shell.search.placeholder")}
              aria-label={msg("app.shell.search.placeholder")}
              aria-controls="global-search-results"
              aria-activedescendant={
                filteredItems[activeIndex]
                  ? `global-search-${filteredItems[activeIndex].id}`
                  : undefined
              }
              role="combobox"
              aria-expanded="true"
              autoComplete="off"
              spellCheck={false}
              dir={dir}
              className="h-full min-w-0 flex-1 bg-transparent text-start text-[0.95rem] text-foreground outline-none placeholder:text-start placeholder:text-muted-foreground/70"
            />
          </div>

          <div
            id="global-search-results"
            role="listbox"
            aria-label={msg("app.shell.search.results")}
            className="max-h-[min(28rem,60vh)] overflow-y-auto p-2"
          >
            {filteredItems.length > 0 ? (
              GROUP_ORDER.map((group) => {
                const groupItems = filteredItems.filter((item) => item.group === group);
                if (groupItems.length === 0) return null;
                return (
                  <section key={group} className="pb-2 last:pb-0">
                    <p className="px-2 pb-1.5 pt-2 text-[0.65rem] font-semibold uppercase tracking-[0.12em] text-muted-foreground/65">
                      {groupLabels[group]}
                    </p>
                    <div className="space-y-0.5">
                      {groupItems.map((item) => {
                        const index = filteredItems.indexOf(item);
                        const selected = activeIndex === index;
                        const Icon = item.icon;
                        return (
                          <button
                            key={item.id}
                            id={`global-search-${item.id}`}
                            type="button"
                            role="option"
                            aria-selected={selected}
                            onMouseEnter={() => setActiveIndex(index)}
                            onClick={() => selectItem(item)}
                            className={cn(
                              "group/item flex w-full items-center gap-3 rounded-xl px-2.5 py-2 text-start outline-none transition-[background-color,color] duration-100",
                              selected
                                ? "bg-primary/[0.09] text-foreground"
                                : "text-foreground/80 hover:bg-accent/55",
                            )}
                          >
                            <span
                              className={cn(
                                "flex size-9 shrink-0 items-center justify-center rounded-xl transition-colors duration-100",
                                selected
                                  ? "bg-primary/[0.12] text-primary"
                                  : "bg-muted/70 text-muted-foreground group-hover/item:text-primary",
                              )}
                            >
                              <Icon className="size-[1.05rem]" aria-hidden="true" />
                            </span>
                            <span className="min-w-0 flex-1">
                              <span className="block truncate text-sm font-medium">
                                {item.label}
                              </span>
                              {item.description && (
                                <span className="mt-0.5 block truncate text-xs text-muted-foreground/75">
                                  {item.description}
                                </span>
                              )}
                            </span>
                            {selected && (
                              <span className="hidden shrink-0 items-center gap-1.5 text-xs font-medium text-muted-foreground sm:flex">
                                {msg("app.shell.search.open")}
                                <ArrowUpRight className="size-3.5" aria-hidden="true" />
                              </span>
                            )}
                          </button>
                        );
                      })}
                    </div>
                  </section>
                );
              })
            ) : (
              <div className="flex min-h-36 flex-col items-center justify-center gap-2 px-6 text-center">
                <MagnifyingGlass className="size-5 text-primary/30" aria-hidden="true" />
                <p className="text-sm font-medium text-foreground/75">
                  {msg("app.shell.search.no_results")}
                </p>
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}
