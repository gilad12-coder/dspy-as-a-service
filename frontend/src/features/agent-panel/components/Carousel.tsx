"use client";

import * as React from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";

import { cn } from "@/shared/lib/utils";

interface CarouselProps<T> {
  /** Slides to page through. */
  items: readonly T[];
  /** Stable key per item — drives the enter/exit animation. */
  itemKey: (item: T, index: number) => string;
  /** Renders the slide body for an item. */
  renderItem: (item: T, index: number) => React.ReactNode;
  /** Leading-side header label; omit to show only the ``i / n`` position counter. */
  title?: React.ReactNode;
  /** Accessible name for the carousel region. */
  ariaLabel: string;
  /** Slide-area sizing — e.g. ``h-[132px]`` (the default). Ignored when
   *  {@link fluid} is set. */
  bodyClassName?: string;
  /**
   * Let each slide grow to its own content height instead of living in a fixed
   * box. Slides flow normally and cross-fade (no inner scrollbar — the embedding
   * container scrolls); the nav strip follows the active slide's height. Use for
   * tall, variable content; leave off for the fixed-height tour cards.
   */
  fluid?: boolean;
  /**
   * Per-dot accent colour (inline ``background-color``) keyed by slide index;
   * return null for the default faint dot. Lets a caller paint slide status onto
   * the nav strip (e.g. which tools changed) so it doubles as a change map. Size
   * still encodes the active slide, so the cue never relies on colour alone.
   */
  dotTone?: (index: number) => string | null;
  /**
   * Slide indices worth landing on (e.g. the changed ones). When non-empty the
   * carousel opens on the first such slide instead of slide 1, so compare mode
   * starts on a change; paired with {@link dotTone} the coloured dots then map
   * the rest for one-tap jumps.
   */
  jumpIndices?: readonly number[];
  /**
   * Wrap the whole carousel — position counter, slide, dot strip and nav — in one
   * bordered card so the chrome reads as part of a single tool card (mirrors the
   * curated tour's popover frame). Leave off when an outer container already
   * supplies the frame; the per-slide content should then carry no border itself.
   */
  framed?: boolean;
  /** Merged onto the root (width, padding). */
  className?: string;
}

/**
 * The shared carousel chrome — position counter, slide animation, dot strip and
 * RTL-aware prev/next nav — with the slide body supplied by the caller. Lets
 * the curated tool tour and the trajectory drawer's tool-description pager share
 * one paging shell instead of each re-implementing the navigation.
 */
export function Carousel<T>({
  items,
  itemKey,
  renderItem,
  fluid = false,
  title,
  ariaLabel,
  bodyClassName,
  dotTone,
  jumpIndices,
  framed = false,
  className,
}: CarouselProps<T>) {
  const count = items.length;
  const isRtl = getActiveDir() === "rtl";
  // Open on the first flagged slide so compare mode lands on a change, not slide 1.
  const [idx, setIdx] = React.useState(() => jumpIndices?.[0] ?? 0);
  const [dir, setDir] = React.useState<1 | -1>(-1);
  const reduceMotion = useReducedMotion();

  // The slide set can shrink between renders (e.g. switching candidates), so the
  // stored index may point past the end — clamp for this render and resync state.
  const clampedIdx = Math.max(0, Math.min(idx, count - 1));
  React.useEffect(() => {
    if (idx !== clampedIdx) setIdx(clampedIdx);
  }, [idx, clampedIdx]);

  const go = React.useCallback(
    (next: number) => {
      const clamped = Math.max(0, Math.min(count - 1, next));
      // Forward slides enter from the inline-end edge — leftward in RTL, rightward
      // in LTR — so the animation's x-sign tracks the active direction.
      const forward = clamped > clampedIdx;
      setDir(forward === isRtl ? -1 : 1);
      setIdx(clamped);
    },
    [clampedIdx, count, isRtl],
  );

  const onKey = React.useCallback(
    (e: React.KeyboardEvent) => {
      // The arrow pointing toward the inline-end edge advances: ArrowLeft in RTL,
      // ArrowRight in LTR.
      const forwardKey = isRtl ? "ArrowLeft" : "ArrowRight";
      const backKey = isRtl ? "ArrowRight" : "ArrowLeft";
      if (e.key === forwardKey) {
        e.preventDefault();
        go(clampedIdx + 1);
      } else if (e.key === backKey) {
        e.preventDefault();
        go(clampedIdx - 1);
      }
    },
    [go, clampedIdx, isRtl],
  );

  const active = items[clampedIdx];
  if (active === undefined) return null;

  return (
    <div
      className={cn(
        "select-none rounded-xl",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3D2E22]/40 focus-visible:ring-offset-1",
        framed && "overflow-hidden border border-[#DDD4C8]/40 bg-background/80",
        className,
      )}
      dir={isRtl ? "rtl" : "ltr"}
      role="region"
      aria-label={ariaLabel}
      tabIndex={0}
      onKeyDown={onKey}
    >
      <div
        className={cn(
          "mb-2.5 flex items-baseline justify-between gap-2",
          framed && "px-2.5 pt-2.5",
        )}
      >
        {title !== undefined ? (
          <span className="text-[0.8125rem] font-medium text-foreground">{title}</span>
        ) : (
          <span aria-hidden="true" />
        )}
        <span className="font-mono tabular-nums text-[0.625rem] text-muted-foreground/70">
          {clampedIdx + 1} / {count}
        </span>
      </div>

      <div
        className={cn(
          "relative overflow-hidden rounded-xl",
          fluid ? bodyClassName : (bodyClassName ?? "h-[132px]"),
        )}
      >
        <AnimatePresence custom={dir} mode={fluid ? "wait" : "popLayout"} initial={false}>
          <motion.div
            key={itemKey(active, clampedIdx)}
            custom={dir}
            variants={{
              enter: (d: 1 | -1) => ({
                x: reduceMotion ? 0 : d * 28,
                opacity: 0,
              }),
              center: { x: 0, opacity: 1 },
              exit: (d: 1 | -1) => ({
                x: reduceMotion ? 0 : d * -28,
                opacity: 0,
              }),
            }}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
            className={fluid ? undefined : "absolute inset-0"}
          >
            {renderItem(active, clampedIdx)}
          </motion.div>
        </AnimatePresence>
      </div>

      <div
        className={cn(
          "mt-2.5 flex flex-wrap items-center justify-center gap-1",
          framed && "px-2.5",
        )}
      >
        {Array.from({ length: count }, (_, i) => {
          const tone = dotTone?.(i) ?? null;
          const isActive = i === clampedIdx;
          return (
            <button
              key={i}
              type="button"
              onClick={() => go(i)}
              aria-label={formatMsg(
                "auto.features.agent.panel.components.toolscarousel.template.11",
                { p1: i + 1, p2: count },
              )}
              aria-current={isActive ? "true" : undefined}
              // Tone paints status; width (1.5 → 2.5 → 4) carries the active and
              // flagged cues so the strip never leans on colour alone.
              style={tone ? { backgroundColor: tone } : undefined}
              className={cn(
                "h-1.5 rounded-full transition-all duration-200 cursor-pointer",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3D2E22]/40 focus-visible:ring-offset-1",
                isActive
                  ? cn("w-4", !tone && "bg-[#3D2E22]/70")
                  : tone
                    ? "w-2.5 opacity-70 hover:opacity-100"
                    : "w-1.5 bg-[#3D2E22]/20 hover:bg-[#3D2E22]/40",
              )}
            />
          );
        })}
      </div>

      <div
        className={cn(
          "mt-2.5 flex items-center justify-between gap-2",
          framed && "px-2.5 pb-2.5",
        )}
      >
        {/* Prev sits at the inline-start edge — right in RTL, left in LTR. */}
        <CarouselNav
          direction="prev"
          disabled={clampedIdx === 0}
          onClick={() => go(clampedIdx - 1)}
        />
        <CarouselNav
          direction="next"
          disabled={clampedIdx >= count - 1}
          onClick={() => go(clampedIdx + 1)}
        />
      </div>
    </div>
  );
}

function CarouselNav({
  direction,
  disabled,
  onClick,
}: {
  direction: "prev" | "next";
  disabled: boolean;
  onClick: () => void;
}) {
  // "prev" points back toward the inline-start edge, "next" toward inline-end:
  // rightward/leftward chevrons in RTL, mirrored in LTR.
  const isRtl = getActiveDir() === "rtl";
  const Icon =
    direction === "prev"
      ? isRtl
        ? ChevronRight
        : ChevronLeft
      : isRtl
        ? ChevronLeft
        : ChevronRight;
  const label =
    direction === "prev"
      ? msg("auto.features.agent.panel.components.toolscarousel.literal.14")
      : msg("auto.features.agent.panel.components.toolscarousel.literal.15");
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      className={cn(
        "inline-flex size-7 items-center justify-center rounded-full",
        "border border-border/50 bg-background/85",
        "transition-all duration-150 cursor-pointer",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#3D2E22]/40 focus-visible:ring-offset-1",
        "hover:bg-accent/60 hover:border-border active:scale-[0.96]",
        "disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-background/85",
      )}
    >
      <Icon className="size-3.5 text-foreground/70" aria-hidden="true" />
    </button>
  );
}
