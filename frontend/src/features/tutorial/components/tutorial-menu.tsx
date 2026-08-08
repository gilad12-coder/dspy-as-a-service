"use client";

import * as React from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, GraduationCap, Lightning, Compass } from "@/shared/ui/icons";
import { useTutorialContext } from "./tutorial-provider";
import type { TutorialTrack } from "../lib/steps";
import { getLoadedTrack, loadStepsModule } from "../lib/steps-loader";
import { formatMsg, msg } from "@/shared/lib/messages";

/** How long each track is — filled in once the lazy steps module resolves. */
type TrackSize = { steps: number; minutes: number };

const TRACK_ICONS: Record<TutorialTrack, typeof Lightning> = {
  quick: Lightning,
  "deep-dive": Compass,
};

export function TutorialMenu() {
  const { state, startTrack, closeMenu } = useTutorialContext();
  const dialogRef = React.useRef<HTMLDivElement | null>(null);
  const titleId = React.useId();
  const [sizes, setSizes] = React.useState<Partial<Record<TutorialTrack, TrackSize>>>({});

  // Names and blurbs render straight from the catalog so the chooser is
  // readable the instant it opens; only the step counts have to wait for the
  // step definitions, and they arrive as an addition rather than a reflow.
  React.useEffect(() => {
    if (!state.isMenuOpen) return;
    let cancelled = false;
    void loadStepsModule().then(() => {
      if (cancelled) return;
      const next: Partial<Record<TutorialTrack, TrackSize>> = {};
      for (const id of Object.keys(TRACK_ICONS) as TutorialTrack[]) {
        const track = getLoadedTrack(id);
        if (track) next[id] = { steps: track.stepCount, minutes: track.estimatedMinutes };
      }
      setSizes(next);
    });
    return () => {
      cancelled = true;
    };
  }, [state.isMenuOpen]);

  React.useEffect(() => {
    if (!state.isMenuOpen) return;
    // Land on the first track rather than the close button: the dialog exists
    // to be answered, and the shorter track is the safer default answer.
    dialogRef.current?.querySelector<HTMLButtonElement>("[data-track]")?.focus();

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        closeMenu();
        return;
      }
      // Trap focus inside the dialog. Without this, Tab walks out into
      // the page chrome behind the modal backdrop.
      if (e.key !== "Tab") return;
      const root = dialogRef.current;
      if (!root) return;
      const focusables = root.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
      );
      if (focusables.length === 0) return;
      const first = focusables[0]!;
      const last = focusables[focusables.length - 1]!;
      const active = document.activeElement;
      if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [state.isMenuOpen, closeMenu]);

  return (
    <AnimatePresence>
      {state.isMenuOpen && (
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby={titleId}
        >
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="absolute inset-0 bg-[#1C1612]/50 backdrop-blur-sm"
            onClick={closeMenu}
            aria-hidden="true"
          />

          <motion.div
            ref={dialogRef}
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 12 }}
            transition={{ duration: 0.25, ease: [0.2, 0.8, 0.2, 1] }}
            className="relative w-full max-w-sm rounded-2xl border border-[#E5DDD4] bg-gradient-to-b from-[#FAF8F5] to-[#F5F1EC] shadow-[0_16px_48px_rgba(28,22,18,0.18)] overflow-hidden"
          >
            <button
              type="button"
              onClick={closeMenu}
              className="close-button absolute top-4 end-4 z-10"
              aria-label={msg("auto.features.tutorial.components.tutorial.menu.literal.1")}
            >
              <X />
            </button>

            <div className="flex flex-col items-center px-6 pt-7 pb-6 text-center">
              <div className="size-12 rounded-xl bg-[#F0EBE4] flex items-center justify-center mb-4">
                <GraduationCap className="size-6 text-[#8C7A6B]" />
              </div>
              <h2 id={titleId} className="text-lg font-bold text-[#3D2E22] mb-1">
                {msg("auto.features.tutorial.components.tutorial.menu.1")}
              </h2>
              <p className="text-xs text-[#8C7A6B] leading-relaxed mb-5">
                {msg("tutorial.menu.subtitle")}
              </p>

              <div className="flex w-full flex-col gap-2.5">
                <TrackOption
                  track="quick"
                  name={msg("tutorial.track.quick.name")}
                  description={msg("tutorial.track.quick.desc")}
                  size={sizes.quick}
                  onStart={startTrack}
                />
                <TrackOption
                  track="deep-dive"
                  name={msg("tutorial.track.full.name")}
                  description={msg("tutorial.track.full.desc")}
                  size={sizes["deep-dive"]}
                  onStart={startTrack}
                />
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}

/** One card in the chooser: a track's name, what it covers, and how long it is. */
function TrackOption({
  track,
  name,
  description,
  size,
  onStart,
}: {
  track: TutorialTrack;
  name: string;
  description: string;
  size?: TrackSize;
  onStart: (track: TutorialTrack) => void;
}) {
  const Icon = TRACK_ICONS[track];
  return (
    <button
      type="button"
      data-track={track}
      onClick={() => onStart(track)}
      className="group w-full cursor-pointer rounded-xl border border-[#E5DDD4] bg-[#FDFCFA] p-3.5 text-start transition-colors hover:border-[#C9BCAE] hover:bg-[#F7F3EE] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3D2E22]/30"
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-[#F0EBE4] text-[#8C7A6B] transition-colors group-hover:bg-[#E7DFD5] group-hover:text-[#3D2E22]">
          <Icon className="size-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-sm font-semibold text-[#3D2E22]">{name}</span>
            {size && (
              <span className="shrink-0 text-[0.6875rem] tabular-nums text-[#A8998A]">
                {formatMsg("tutorial.menu.meta", { p1: size.steps, p2: size.minutes })}
              </span>
            )}
          </div>
          <p className="mt-1 text-xs leading-relaxed text-[#8C7A6B]">{description}</p>
        </div>
      </div>
    </button>
  );
}
