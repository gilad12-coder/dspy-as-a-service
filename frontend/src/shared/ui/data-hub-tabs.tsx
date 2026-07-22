"use client";

import Link from "next/link";

import { cn } from "@/shared/lib/utils";
import { msg } from "@/shared/lib/messages";

/**
 * Segmented navigation between the two halves of the Data hub: the dataset
 * library (/datasets) and the labeling-session chooser (/tagger). Rendered
 * only above those two list surfaces — deeper surfaces (dataset editor, live
 * tagging session) drop it. The segments are real links, so modifier-clicks
 * and browser history behave like navigation, not like in-page tabs.
 */
export function DataHubTabs({ active }: { active: "datasets" | "sessions" }) {
  const tabs = [
    { key: "datasets" as const, href: "/datasets", label: msg("sidebar.nav.datasets") },
    { key: "sessions" as const, href: "/tagger", label: msg("data.tabs.sessions") },
  ];
  return (
    <nav aria-label={msg("sidebar.nav.data")} className="mb-5 flex w-fit rounded-xl bg-muted p-1">
      {tabs.map((tab) => (
        <Link
          key={tab.key}
          href={tab.href}
          aria-current={active === tab.key ? "page" : undefined}
          className={cn(
            "rounded-lg px-4 py-1.5 text-sm font-medium transition-colors duration-150",
            "focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40",
            active === tab.key
              ? "bg-background text-foreground shadow-sm"
              : "text-foreground/60 hover:text-foreground",
          )}
        >
          {tab.label}
        </Link>
      ))}
    </nav>
  );
}
