"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { motion, useReducedMotion } from "framer-motion";
import { Database, Tag } from "@/shared/ui/icons";

import { Tabs, TabsList, TabsTrigger } from "@/shared/ui/primitives/tabs";
import { msg } from "@/shared/lib/messages";

// Mirrors DASHBOARD_TAB_CLASS in DashboardView: the active background is a
// single shared pill that slides between triggers via Framer's layoutId, so
// the trigger itself stays transparent and only fades text color + reacts to
// the press transform.
const DATA_HUB_TAB_CLASS =
  "relative z-10 min-h-10 rounded-full px-3 py-2 text-sm font-semibold cursor-pointer border-none bg-transparent text-foreground/65 shadow-none transition-[color,transform] data-[state=active]:bg-transparent data-[state=active]:text-foreground data-[state=active]:border-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 sm:px-4";

/**
 * Segmented navigation between the two halves of the Data hub: the dataset
 * library (/datasets) and the labeling-session chooser (/tagger). Rendered
 * only above those two list surfaces — deeper surfaces (dataset editor, live
 * tagging session) drop it. Uses the same tab system as the dashboard's
 * jobs/analytics tabs; selecting a tab navigates to the other route, with the
 * pill sliding optimistically before the route swap remounts the surface.
 */
export function DataHubTabs({ active }: { active: "datasets" | "sessions" }) {
  const router = useRouter();
  const prefersReducedMotion = useReducedMotion();
  const [value, setValue] = React.useState<string>(active);

  React.useEffect(() => {
    setValue(active);
  }, [active]);

  const tabPillTransition = prefersReducedMotion
    ? { duration: 0 }
    : { type: "tween" as const, duration: 0.2, ease: [0.2, 0.8, 0.2, 1] as const };

  const tabs = [
    { key: "datasets", href: "/datasets", label: msg("sidebar.nav.datasets"), Icon: Database },
    { key: "sessions", href: "/tagger", label: msg("data.tabs.sessions"), Icon: Tag },
  ] as const;

  return (
    <Tabs
      value={value}
      onValueChange={(next) => {
        setValue(next);
        const target = tabs.find((tab) => tab.key === next);
        if (target && next !== active) router.push(target.href);
      }}
      className="mb-5"
    >
      <TabsList
        aria-label={msg("sidebar.nav.data")}
        className="inline-flex h-auto w-full gap-1 rounded-full border border-border/60 bg-muted/50 p-1 shadow-[inset_0_1px_0_rgba(255,255,255,0.5)]"
      >
        {tabs.map(({ key, label, Icon }) => (
          <TabsTrigger key={key} value={key} className={DATA_HUB_TAB_CLASS}>
            {value === key && (
              <motion.span
                layoutId="dataHubTabPill"
                transition={tabPillTransition}
                className="absolute inset-0 z-0 rounded-full bg-background shadow-sm"
                aria-hidden="true"
              />
            )}
            <span className="relative z-10 inline-flex items-center gap-1.5">
              <Icon className="size-3.5" />
              {label}
            </span>
          </TabsTrigger>
        ))}
      </TabsList>
    </Tabs>
  );
}
