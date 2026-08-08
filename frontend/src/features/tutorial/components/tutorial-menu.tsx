"use client";

import * as React from "react";
import { Popover as PopoverPrimitive } from "radix-ui";
import { Lightning, Compass } from "@/shared/ui/icons";
import { useTutorialContext } from "./tutorial-provider";
import type { TutorialTrack } from "../lib/steps";
import { getLoadedTrack, loadStepsModule } from "../lib/steps-loader";
import { formatMsg, msg } from "@/shared/lib/messages";

/** How long each track is — filled in once the lazy steps module resolves. */
type TrackSize = { steps: number; minutes: number };

const ITEM_CLS =
  "flex w-full items-center gap-2.5 whitespace-nowrap px-4 py-2 text-xs text-foreground hover:bg-muted/40 cursor-pointer transition-colors";
const ICON_CLS = "size-4 shrink-0 text-muted-foreground/60";
const META_CLS = "ms-auto shrink-0 whitespace-nowrap font-mono text-[0.625rem] text-muted-foreground/60";

/**
 * The tour's track chooser — the popover half of the header's tour button,
 * which supplies the `Popover.Root` and trigger around it.
 *
 * Both tracks walk the same ordered steps; the short one stops at the
 * essentials. The question gets asked on every press rather than remembered,
 * because the answer depends on how much time the user has right now.
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
      const quick = getLoadedTrack("quick");
      if (quick) next.quick = { steps: quick.stepCount, minutes: quick.estimatedMinutes };
      const deep = getLoadedTrack("deep-dive");
      if (deep) next["deep-dive"] = { steps: deep.stepCount, minutes: deep.estimatedMinutes };
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
        className="z-50 min-w-[230px] max-w-[min(280px,90vw)] rounded-2xl border border-border/40 bg-card py-1.5 shadow-[0_4px_24px_rgba(28,22,18,0.1)] animate-in fade-in-0 zoom-in-95"
      >
        <TrackItem
          track="quick"
          Icon={Lightning}
          label={msg("tutorial.track.quick.name")}
          size={sizes.quick}
          onStart={startTrack}
        />
        <TrackItem
          track="deep-dive"
          Icon={Compass}
          label={msg("tutorial.track.full.name")}
          size={sizes["deep-dive"]}
          onStart={startTrack}
        />
      </PopoverPrimitive.Content>
    </PopoverPrimitive.Portal>
  );
}

/** One track in the chooser: what it is called and how long it runs. */
function TrackItem({
  track,
  Icon,
  label,
  size,
  onStart,
}: {
  track: TutorialTrack;
  Icon: typeof Lightning;
  label: string;
  size?: TrackSize;
  onStart: (track: TutorialTrack) => void;
}) {
  return (
    <PopoverPrimitive.Close asChild>
      <button type="button" onClick={() => onStart(track)} className={ITEM_CLS}>
        <Icon className={ICON_CLS} />
        <span className="flex-1 whitespace-nowrap text-start">{label}</span>
        {size && (
          <span className={META_CLS}>{formatMsg("tutorial.menu.meta", { p1: size.steps, p2: size.minutes })}</span>
        )}
      </button>
    </PopoverPrimitive.Close>
  );
}
