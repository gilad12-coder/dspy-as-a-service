"use client";

import { memo } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { getActiveLocale } from "@/shared/lib/runtime-locale";
import { HALO_CARDS, type HaloCard } from "../login-samples";
import { HALO_CARDS_EN } from "../login-samples.en";

const EASE_OUT_EXPO = [0.16, 1, 0.3, 1] as const;

interface LocalizedHaloCard extends HaloCard {
  titleEn: string;
}

// Zip the Hebrew source with its English overlay by index, once at module load,
// so the render path just reads the active locale's title.
const LOCALIZED_CARDS: LocalizedHaloCard[] = HALO_CARDS.map((card, i) => ({
  ...card,
  titleEn: HALO_CARDS_EN[i] ?? card.title,
}));

// The subset that also frames the login on small portrait screens.
const MOBILE_HALO_CARDS = LOCALIZED_CARDS.filter((card) => card.mobilePos);

/**
 * One floating "finished-run" chip: settles in once, then drifts gently and
 * forever, and scales up under the pointer. Positioning lives on the plain
 * wrapper and rotation is animated numerically, so framer-motion only ever
 * touches transform/opacity — never the positional unit-conversion path
 * (`positionalValues[name]`) that throws on re-render. `i` seeds the drift so
 * each chip desyncs without Math.random (which would break SSR hydration).
 */
function HaloChip({
  card,
  i,
  pos,
  title,
  dir,
  wrapperClassName,
  reduce,
}: {
  card: HaloCard;
  i: number;
  pos: HaloCard["pos"];
  title: string;
  dir: "rtl" | "ltr";
  wrapperClassName: string;
  reduce: boolean | null;
}) {
  const amp = 4 + (i % 3) * 2;
  const xamp = i % 2 === 0 ? 3 : -3;
  const wob = i % 2 === 0 ? 0.8 : -0.8;
  const dur = 5 + (i % 6);
  const fdelay = (i % 5) * 0.5;

  return (
    <div className={wrapperClassName} style={pos}>
      <motion.div
        className="pointer-events-auto cursor-default"
        initial={reduce ? { opacity: 1 } : { opacity: 0, scale: 0.9, y: 10 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={
          reduce
            ? { duration: 0 }
            : { duration: 0.7, delay: 0.15 + i * 0.03, ease: EASE_OUT_EXPO }
        }
      >
        <motion.div
          dir={dir}
          className="flex w-max max-w-[14rem] items-center rounded-xl border border-border/50 bg-card px-3 py-2 shadow-[0_8px_24px_-14px_rgba(28,22,18,0.28)]"
          animate={
            reduce
              ? { rotate: card.rot }
              : {
                  y: [0, -amp, 0],
                  x: [0, xamp, 0],
                  rotate: [card.rot, card.rot + wob, card.rot],
                }
          }
          transition={
            reduce
              ? { duration: 0 }
              : {
                  y: { duration: dur, repeat: Infinity, ease: "easeInOut", delay: fdelay },
                  x: { duration: dur * 1.5, repeat: Infinity, ease: "easeInOut", delay: fdelay * 0.6 },
                  rotate: { duration: dur * 1.25, repeat: Infinity, ease: "easeInOut", delay: fdelay },
                }
          }
          whileHover={
            reduce ? undefined : { scale: 1.1, transition: { duration: 0.35, ease: EASE_OUT_EXPO } }
          }
        >
          <span className="truncate text-[0.72rem] font-medium text-foreground/85">
            {title}
          </span>
        </motion.div>
      </motion.div>
    </div>
  );
}

/**
 * The scattered "product halo" behind the login panel: fake finished-run cards,
 * each just a task name, no badge and no scores.
 *
 * The full set fans out around the edges and bleeds off them — that reads well
 * only with room to spare, so it renders only when there's both width (`md`+)
 * and height (≥640px); short landscape phones don't qualify. Below `md`, a
 * curated few (the ones with `mobilePos`) reposition into a sparse top/bottom
 * frame so phones still show a wall of Skynet's runs without the desktop layout
 * collapsing into clipped corner slivers. That frame is portrait-only — landscape
 * phones have no vertical room, so they fall back to a clean, focused login.
 *
 * Decorative only (hidden from assistive tech) and clipped so cards bleed off
 * the edges. The whole layer is memoized (it has no props) so typing in the form
 * never re-renders these animated nodes.
 */
function LoginHaloImpl() {
  const reduce = useReducedMotion();
  const locale = getActiveLocale();
  const dir = locale === "en" ? "ltr" : "rtl";
  const titleOf = (card: LocalizedHaloCard) => (locale === "en" ? card.titleEn : card.title);

  return (
    <div aria-hidden className="pointer-events-none absolute inset-0 z-0 overflow-hidden">
      {LOCALIZED_CARDS.map((card, i) => (
        <HaloChip
          key={`d${i}`}
          card={card}
          i={i}
          pos={card.pos}
          title={titleOf(card)}
          dir={dir}
          wrapperClassName="absolute hidden md:[@media(min-height:640px)]:block"
          reduce={reduce}
        />
      ))}
      {MOBILE_HALO_CARDS.map((card, i) => (
        <HaloChip
          key={`m${i}`}
          card={card}
          i={i}
          pos={card.mobilePos!}
          title={titleOf(card)}
          dir={dir}
          wrapperClassName="absolute md:hidden landscape:hidden"
          reduce={reduce}
        />
      ))}
    </div>
  );
}

export const LoginHalo = memo(LoginHaloImpl);
