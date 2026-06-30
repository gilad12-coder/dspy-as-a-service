"use client";

import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Animated "SKYNET" wordmark — matching the story.foundation morphing SVG style.
 *
 * Each letter is a <g> with 4 variant sub-groups (default, glyph, serif, sans).
 * On hover a rapid interval (250ms) randomly picks 2-3 letters each tick and
 * switches each to a random variant via opacity. On mouse leave everything
 * resets to the default variant.
 */

type RectDef = { x: number; y: number; w: number; h: number };

type LetterVariants = {
  /** The standard base logo variant (bold, chunky, geometric) */
  default: string;
  /** The geometric square block representation */
  glyph: RectDef[];
  /** High-contrast Serif-style alternate */
  serif: string;
  /** Thinner / lighter uniform sans-serif alternate */
  sans: string;
};

type LetterDef = {
  offset: number;
  variants: LetterVariants;
};

// Translated & custom-calculated to match the STORY design language exactly,
// all scaled within a 92-unit high viewBox.

const LETTERS: LetterDef[] = [
  {
    offset: 0,
    variants: {
      default:
        "M34.9,92C54,92,69.5,80.5,69.5,61.1C69.5,43,56,30.3,34.9,30.3v13.4c-9.7,0-16.9-4.3-16.9-13.1 c0-8.8,6.2-14,17.4-14c9.2,0,14.7,3.8,16.1,8.8h17C67.2,11.4,54,0,35,0C14.9,0,0.6,12.6,0.6,31c0,18.4,14.9,29.5,34.3,29.5V47.8 c10.3,0,17.4,4.6,17.4,13.7c0,9-7.2,14.1-17.3,14.1c-9.1,0-15.4-4-17.3-9.5H0C2.5,80.6,15.8,92,34.9,92z",
      glyph: [
        { x: 37.8, y: 36.6, w: 22.6, h: 22.6 },
        { x: 10.4, y: 9.1, w: 22.6, h: 22.6 },
        { x: 10.4, y: 64, w: 22.6, h: 22.6 },
      ],
      serif:
        "M34,91.9c-15.2,0-27.8-7.7-31.1-21l11.8-3.7c2.4,12.6,10.8,21,20.8,21c9.4,0,15.7-6.5,15.7-15.8 C51.3,45.8,5.7,55.6,5.7,24.3C5.7,9.8,16.7,0,33.7,0c13.4,0,25.7,7,29.5,19.4l-11.9,3.7C48.7,11.3,40.7,3.7,32,3.7 c-8.8,0-14.8,5.8-14.8,14.4c0,24,46.5,15.8,46.5,47.8C63.7,81.8,52.1,91.9,34,91.9z",
      sans: "M35.3,92c-10.4,0-18.8-2.4-25-7.3C4.1,79.9,0.9,73.3,0.9,65H13c0,5.7,2.1,10,6.2,13c4.1,2.9,9.5,4.4,16.2,4.4 c3,0,5.9-0.3,8.5-0.9c2.6-0.6,4.9-1.6,7-2.8c2.1-1.3,3.7-3,4.9-5.2c1.2-2.2,1.8-4.8,1.8-7.6c0-1.5-0.2-2.8-0.5-4 c-0.4-1.2-0.8-2.2-1.4-3.1c-0.5-0.9-1.4-1.7-2.5-2.5c-1.2-0.8-2.2-1.5-3.3-1.9c-1-0.5-2.5-1.1-4.6-1.7c-2-0.6-3.7-1.1-5.3-1.5 c-1.5-0.4-3.7-0.9-6.5-1.6c-2.2-0.5-4-0.9-5.3-1.3c-1.3-0.3-3-0.8-5-1.4s-3.6-1.1-4.9-1.6c-1.2-0.5-2.7-1.2-4.4-1.9 c-1.6-0.8-3-1.6-4-2.4c-1-0.8-2.1-1.8-3.1-2.9c-1.1-1.2-2-2.4-2.5-3.7c-0.6-1.2-1.1-2.7-1.5-4.3c-0.4-1.7-0.6-3.4-0.6-5.2 c0-7.4,3-13.5,9-18.2c5.9-4.6,13.7-7,23-7c9.8,0,17.8,2.4,24,7.3s9.3,11.3,9.3,19H56c0-5-2-9-6-12.1c-4-3.1-9.2-4.7-15.6-4.7 c-6,0-11,1.4-14.9,4s-5.9,6.4-5.9,11.4c0,1.4,0.2,2.6,0.5,3.7c0.4,1.1,0.8,2.1,1.4,2.9c0.6,0.8,1.4,1.7,2.6,2.4 c1.2,0.7,2.4,1.4,3.4,1.9c1.1,0.5,2.7,1.1,4.7,1.7c2.1,0.6,3.9,1.1,5.5,1.6c1.7,0.4,3.9,0.9,6.8,1.7c2.2,0.5,3.9,0.9,5.1,1.3 c1.2,0.3,2.9,0.7,4.9,1.3c2,0.6,3.6,1.1,4.8,1.6c1.2,0.5,2.5,1.1,4.2,1.9c1.6,0.8,2.9,1.5,3.8,2.4c0.9,0.8,2,1.8,3,2.9 c1.1,1.1,1.9,2.4,2.5,3.7c0.6,1.3,1,2.8,1.4,4.4c0.4,1.7,0.6,3.5,0.6,5.4c0,7.6-3,13.9-8.9,18.9C54.1,89.5,45.9,92,35.3,92z",
    },
  },
  {
    offset: 75,
    variants: {
      default:
        "M 0 2.1 h 22 v 87.9 h -22 Z M 20 52 L 55 2.1 h 22 L 20 62 Z M 36 45 L 65 90 h -25 L 20 50 Z",
      glyph: [
        { x: 0, y: 35, w: 22.6, h: 22.6 },
        { x: 45, y: 2.1, w: 22.6, h: 22.6 },
        { x: 45, y: 68, w: 22.6, h: 22.6 },
      ],
      serif:
        "M 8 2.1 h 18 v 3.8 h -5 v 78.3 h 5 v 3.8 h -18 v -3.8 h 5 v -78.3 h -5 Z M 18 52 L 65 2.1 h 18 v 3.8 L 22 55 Z M 35 40 L 72 86.2 h 10 v 3.8 h -25 L 18 50 Z",
      sans: "M 2 2.1 h 11 v 87.9 h -11 Z M 6 46 L 52 2.1 h 14 L 6 54 Z M 28 35 L 66 90 h -14 L 16 45 Z",
    },
  },
  {
    offset: 155,
    variants: {
      default:
        "M36.8,90 h18.4 V52.9 L83,2.2 h-21.2 l-25,46.9 V90 z M20.7,40.6 h20.7 L20.7,2.2 h-20.7 L20.7,40.6 z",
      glyph: [
        { x: 30.7, y: 35.4, w: 22.2, h: 22.2 },
        { x: 3.7, y: 8.5, w: 22.2, h: 22.2 },
        { x: 57.7, y: 62.5, w: 22.2, h: 22.2 },
      ],
      serif:
        "M26.8,90 v-1.8 c6.9,-0.7 10.2,-2.4 10.2,-9.5 V49.4 l-19.6,-34.1 c-4.4,-7.9 -7.2,-10.4 -12.2,-11.2 V2.2 h31.7 V4 c-8.1,0.7 -8.9,3.1 -4.4,11.1 l16.5,29 l16.4,-29 c4.4,-8 3,-10.5 -6.2,-11.1 V2.2 h23.3 V4 c-5.8,0.8 -8.7,3.3 -13.1,11.1 l-19.4,34.1 v29.3 c0,7.1 3.4,8.8 10.3,9.5 v1.8 L26.8,90 Z",
      sans: "M82.9,2.2 l-32.9,59.5 V90 h-11.5 V61.7 L5.8,2.2 h12.7 L44.3,50 l25.9,-47.8 H82.9 z",
    },
  },
  {
    offset: 245,
    variants: {
      default:
        "M 0 2.1 h 22 v 87.9 h -22 Z M 52 2.1 h 22 v 87.9 h -22 Z M 0 2.1 l 74 87.9 h -22 l -52 -87.9 Z",
      glyph: [
        { x: 0, y: 68, w: 22.6, h: 22.6 },
        { x: 25, y: 35, w: 22.6, h: 22.6 },
        { x: 50, y: 2.1, w: 22.6, h: 22.6 },
      ],
      serif:
        "M 8 2.1 h 18 v 3.8 h -5 v 78.3 h 5 v 3.8 h -18 v -3.8 h 5 v -78.3 h -5 Z M 56 2.1 h 18 v 3.8 h -5 v 78.3 h 5 v 3.8 h -18 v -3.8 h 5 v -78.3 h -5 Z M 16 2.1 l 45 87.9 h 13 l -45 -87.9 Z",
      sans: "M 2 2.1 h 11 v 87.9 h -11 Z M 52 2.1 h 11 v 87.9 h -11 Z M 2 2.1 l 61 87.9 h -11 l -50 -87.9 Z",
    },
  },
  {
    offset: 325,
    variants: {
      default: "M 0 2.1 h 62 v 20 h -40 v 14 h 32 v 18 h -32 v 16 h 42 v 20 h -64 Z",
      glyph: [
        { x: 2, y: 2.1, w: 22.6, h: 22.6 },
        { x: 24, y: 35, w: 22.6, h: 22.6 },
        { x: 2, y: 68, w: 22.6, h: 22.6 },
      ],
      serif:
        "M 10 2.1 h 52 v 22 h -3.8 c -1,-8 -4,-12 -12,-12 h -20 v 32 h 16 c 4,0 6,-2 8,-8 h 3.8 v 24 h -3.8 c -2,-6 -4,-8 -8,-8 h -16 v 34 h 22 c 8,0 12,-4 14,-14 h 3.8 v 20 h -60 v -3.8 h 6 v -78.3 h -6 Z",
      sans: "M 2 2.1 h 50 v 10.2 h -39 v 28 h 32 v 10.2 h -32 v 29.3 h 41 v 10.2 h -52 Z",
    },
  },
  {
    offset: 395,
    variants: {
      default: "M27.6,90 L46.9,90 L46.9,19.5 L74.5,19.5 L74.5,2.1 L0,2.1 L0,19.5 L27.6,19.5 Z",
      glyph: [
        { x: 26.4, y: 35.3, w: 21.7, h: 21.7 },
        { x: 52.7, y: 9, w: 21.7, h: 21.7 },
        { x: 0.1, y: 61.7, w: 21.7, h: 21.7 },
      ],
      serif:
        "M71.1,2.1 l3,24 h-1.7 c-2.8,-12.8 -9.3,-20.3 -21.2,-20.3 h-7.3 v73.2 c0,6.6 2.9,8.7 10.7,9.5 v1.8 H20 v-1.8 c7.9,-0.8 10.7,-2.9 10.7,-9.5 V5.8 h-7.3 c-11.8,0 -18.5,7.5 -21.2,20.3 h-1.7 l3,-24 C3.5,2.1 71.1,2.1 71.1,2.1 z",
      sans: "M2,12.3 V2.1 h70.3 v10.2 h-29.4 V90 h-11.5 V12.3 H2 z",
    },
  },
];

const TOTAL_WIDTH = 475;
const VARIANT_NAMES = ["default", "glyph", "serif", "sans"] as const;
type VariantName = (typeof VARIANT_NAMES)[number];

function pickRandom(total: number, count: number): number[] {
  const indices: number[] = [];
  const pool = Array.from({ length: total }, (_, i) => i);
  for (let i = 0; i < Math.min(count, total); i++) {
    const r = Math.floor(Math.random() * pool.length);
    indices.push(pool[r]!);
    pool.splice(r, 1);
  }
  return indices;
}

function randomOtherVariant(current: VariantName): VariantName {
  const others = VARIANT_NAMES.filter((v) => v !== current);
  return others[Math.floor(Math.random() * others.length)]!;
}

// Uses cubic-bezier equivalent to 'ease-out-cubic' for incredibly smooth overlaps
const TRANSITION = "opacity 300ms cubic-bezier(0.215, 0.610, 0.355, 1.000)";

function VariantGroup({
  variant,
  content,
  active,
}: {
  variant: VariantName;
  content: LetterVariants;
  active: boolean;
}) {
  const opacity = active ? 1 : 0;
  const style = { opacity, transition: TRANSITION };

  if (variant === "glyph") {
    return (
      <g style={style}>
        {content.glyph.map((r, i) => (
          <rect key={i} x={r.x} y={r.y} width={r.w} height={r.h} />
        ))}
      </g>
    );
  }

  const d =
    variant === "default" ? content.default : variant === "serif" ? content.serif : content.sans;

  return <path d={d} style={style} fillRule="nonzero" />;
}

export function AnimatedWordmark({
  size = 28,
  className,
  autoMorph = false,
  autoMorphDuration,
  morphSpeed = 250,
  fluid = false,
}: {
  size?: number;
  className?: string;
  /** Morph continuously from mount, independent of hover (for splash/idle states) */
  autoMorph?: boolean;
  /** When set (ms), an auto-morphing wordmark settles back to its default
   *  variant after this many ms instead of morphing forever — for persistent
   *  surfaces (navbar, landing) where perpetual motion gets distracting.
   *  Ignored unless `autoMorph` is on. */
  autoMorphDuration?: number;
  /** Interval in ms between morph ticks (default 250) */
  morphSpeed?: number;
  /** Fill the parent's width (height derives from the viewBox) instead of using `size`. */
  fluid?: boolean;
}) {
  const [activeVariants, setActiveVariants] = useState<VariantName[]>(() =>
    LETTERS.map(() => "default"),
  );
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const stopTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reducedMotionRef = useRef(false);
  // Lite mode is signalled by LiteModeProvider via a data-lite attribute on
  // <html>. Read it straight from the DOM rather than the settings hook so this
  // shared/ui leaf imports nothing from features/: a feature import here closes
  // a client-module cycle (wordmark → settings → prefs → tutorial barrel →
  // tutorial-overlay → wordmark) that traps Turbopack in an endless recompile.
  const [lite, setLite] = useState(false);

  useEffect(() => {
    const mql = window.matchMedia("(prefers-reduced-motion: reduce)");
    reducedMotionRef.current = mql.matches;
    const handler = (e: MediaQueryListEvent) => {
      reducedMotionRef.current = e.matches;
    };
    mql.addEventListener("change", handler);
    return () => mql.removeEventListener("change", handler);
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    const read = () => setLite(root.hasAttribute("data-lite"));
    read();
    const observer = new MutationObserver(read);
    observer.observe(root, { attributes: true, attributeFilter: ["data-lite"] });
    return () => observer.disconnect();
  }, []);

  const startMorph = useCallback(() => {
    if (intervalRef.current) return;

    intervalRef.current = setInterval(() => {
      const count = 2 + Math.floor(Math.random() * 2); // Picks 2 or 3 letters to change
      const indices = pickRandom(LETTERS.length, count);

      setActiveVariants((prev) => {
        const next = [...prev];
        for (const idx of indices) {
          next[idx] = randomOtherVariant(prev[idx] ?? "default");
        }
        return next;
      });
    }, morphSpeed);
  }, [morphSpeed]);

  const stopMorph = useCallback(() => {
    if (intervalRef.current != null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setActiveVariants(LETTERS.map(() => "default"));
  }, []);

  const handleMouseEnter = useCallback(() => {
    // When autoMorph owns the lifecycle the wordmark is already morphing — hover
    // must not spin up a second interval. Lite mode disables the morph outright.
    if (autoMorph || lite) return;
    if (!reducedMotionRef.current) startMorph();
  }, [autoMorph, lite, startMorph]);

  const handleMouseLeave = useCallback(() => {
    // Never reset a continuously-morphing wordmark on leave; it should keep
    // morphing regardless of hover.
    if (autoMorph) return;
    stopMorph();
  }, [autoMorph, stopMorph]);

  useEffect(() => {
    // Lite mode keeps the wordmark static — kill any running morph and never
    // start one, even for callers that opt into autoMorph (splash/idle states).
    if (lite) {
      stopMorph();
      return;
    }
    if (autoMorph && !reducedMotionRef.current) {
      // Clear any stale ref before starting to avoid the guard in startMorph
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      startMorph();
      if (autoMorphDuration != null) {
        stopTimeoutRef.current = setTimeout(stopMorph, autoMorphDuration);
      }
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (stopTimeoutRef.current) {
        clearTimeout(stopTimeoutRef.current);
        stopTimeoutRef.current = null;
      }
    };
  }, [autoMorph, autoMorphDuration, lite, startMorph, stopMorph]);

  const aspectRatio = TOTAL_WIDTH / 92;
  const svgWidth = size * aspectRatio;

  return (
    <span
      dir="ltr"
      className={`${fluid ? "flex w-full justify-center" : "inline-flex"} items-center select-none cursor-default ${className ?? ""}`}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      role="img"
      aria-label="SKYNET"
    >
      <svg
        className="overflow-visible transition-all duration-300 pointer-events-none"
        width={fluid ? undefined : svgWidth}
        height={fluid ? undefined : size}
        viewBox={`0 0 ${TOTAL_WIDTH} 92`}
        fill="currentColor"
        stroke="none"
        xmlns="http://www.w3.org/2000/svg"
        style={fluid ? { width: "100%", height: "auto", color: "var(--foreground)" } : { color: "var(--foreground)" }}
      >
        {LETTERS.map((letter, li) => (
          <g key={li} transform={`translate(${letter.offset}, 0)`}>
            {VARIANT_NAMES.map((name) => (
              <VariantGroup
                key={name}
                variant={name}
                content={letter.variants}
                active={activeVariants[li] === name}
              />
            ))}
          </g>
        ))}
      </svg>
    </span>
  );
}
