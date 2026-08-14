"use client";

import * as React from "react";
import { Popover as PopoverPrimitive } from "radix-ui";
import { Compass, Database, Lightning, TrendUp } from "@/shared/ui/icons";
import { useTutorialContext } from "./tutorial-provider";
import type { TutorialTrack } from "../lib/steps";
import { getLoadedTrack, loadStepsModule } from "../lib/steps-loader";
import { formatMsg, msg } from "@/shared/lib/messages";

/** How long each track is — filled in once the lazy steps module resolves. */
type TrackSize = { steps: number; minutes: number };

const ITEM_CLS =
  "flex min-h-14 w-full items-start gap-2.5 px-4 py-2.5 text-xs text-foreground hover:bg-muted/40 cursor-pointer transition-colors";
const ICON_CLS = "mt-0.5 size-4 shrink-0 text-muted-foreground/60";
const META_CLS =
  "ms-auto shrink-0 whitespace-nowrap font-mono text-[0.625rem] text-muted-foreground/60";

const TRACKS = [
  {
    id: "quick",
    icon: Lightning,
    nameKey: "tutorial.track.quick.name",
    descKey: "tutorial.track.quick.desc",
  },
  {
    id: "data",
    icon: Database,
    nameKey: "tutorial.track.data.name",
    descKey: "tutorial.track.data.desc",
  },
  {
    id: "results",
    icon: TrendUp,
    nameKey: "tutorial.track.results.name",
    descKey: "tutorial.track.results.desc",
  },
  {
    id: "workspace",
    icon: Compass,
    nameKey: "tutorial.track.workspace.name",
    descKey: "tutorial.track.workspace.desc",
  },
] as const satisfies ReadonlyArray<{
  id: TutorialTrack;
  icon: typeof Lightning;
  nameKey: Parameters<typeof msg>[0];
  descKey: Parameters<typeof msg>[0];
}>;

/**
 * The tutorial's workflow chooser — the popover half of the header button,
 * which supplies the `Popover.Root` and trigger around it.
 */
export function TutorialMenu() {
  const { startTrack } = useTutorialContext();
  const [sizes, setSizes] = React.useState<Partial<Record<TutorialTrack, TrackSize>>>({});

  // Content mounts only while the popover is open, so this runs on open. The
  // step definitions are lazily imported; until they land the items render
  // without their duration rather than blocking on it.
  React.useEffect(() => {
    let cancelled = false;
    void loadStepsModule().then(() => {
      if (cancelled) return;
      const next: Partial<Record<TutorialTrack, TrackSize>> = {};
      for (const track of TRACKS) {
        const definition = getLoadedTrack(track.id);
        if (definition) {
          next[track.id] = {
            steps: definition.stepCount,
            minutes: definition.estimatedMinutes,
          };
        }
      }
      setSizes(next);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <PopoverPrimitive.Portal>
      <PopoverPrimitive.Content
        align="end"
        side="bottom"
        sideOffset={6}
        className="z-50 w-[min(calc(100vw-24px),360px)] max-w-none rounded-2xl border border-border/40 bg-card py-1.5 shadow-[0_4px_24px_rgba(28,22,18,0.1)] animate-in fade-in-0 zoom-in-95"
      >
        <p className="px-4 pb-1.5 pt-1 text-[0.6875rem] leading-relaxed text-muted-foreground">
          {msg("tutorial.menu.subtitle")}
        </p>
        <div role="separator" className="mx-3 mb-1 h-px bg-border/40" />
        {TRACKS.map(({ id, icon: Icon, nameKey, descKey }) => (
          <TrackItem
            key={id}
            track={id}
            Icon={Icon}
            label={msg(nameKey)}
            description={msg(descKey)}
            size={sizes[id]}
            onStart={startTrack}
          />
        ))}
      </PopoverPrimitive.Content>
    </PopoverPrimitive.Portal>
  );
}

/** One track in the chooser: what it is called and how long it runs. */
function TrackItem({
  track,
  Icon,
  label,
  description,
  size,
  onStart,
}: {
  track: TutorialTrack;
  Icon: typeof Lightning;
  label: string;
  description: string;
  size?: TrackSize;
  onStart: (track: TutorialTrack) => void;
}) {
  return (
    <PopoverPrimitive.Close asChild>
      <button type="button" onClick={() => onStart(track)} className={ITEM_CLS}>
        <Icon className={ICON_CLS} />
        <span className="min-w-0 flex-1 text-start">
          <span className="flex items-center gap-2">
            <span className="font-medium">{label}</span>
            {size && (
              <span className={META_CLS}>
                {formatMsg("tutorial.menu.meta", { p1: size.steps, p2: size.minutes })}
              </span>
            )}
          </span>
          <span className="mt-0.5 block text-[0.6875rem] leading-snug text-muted-foreground">
            {description}
          </span>
        </span>
      </button>
    </PopoverPrimitive.Close>
  );
}
