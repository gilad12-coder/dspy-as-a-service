"use client";

import * as React from "react";
import { motion, useReducedMotion } from "framer-motion";

import { msg } from "@/shared/lib/messages";

// Delay before the elapsed counter fades in. A first token that arrives faster
// than this never flashes a number — the ember shows for a beat and the bubble
// morphs straight into the reply. Longer waits reveal the honest count.
const TIMER_REVEAL_MS = 600;

/**
 * Skynet "Ember" — the time-to-first-token indicator shown in the assistant
 * bubble during the send → first-token gap. A breathing brand-ink core inside
 * expanding warm-tan ripples, an honest "thinking" label, and a live elapsed
 * counter that reveals itself only once the wait is real. The transcript
 * unmounts it the instant any token, tool call, or reasoning arrives, and its
 * bubble shell matches {@link AgentBubble} so the swap into streamed text reads
 * as one surface warming up rather than a flicker.
 */
export function TtftIndicator() {
  const shouldReduceMotion = useReducedMotion();
  const [startTs] = React.useState(() => Date.now());
  const [elapsedMs, setElapsedMs] = React.useState(0);

  React.useEffect(() => {
    const id = setInterval(() => setElapsedMs(Date.now() - startTs), 100);
    return () => clearInterval(id);
  }, [startTs]);

  const showTimer = elapsedMs >= TIMER_REVEAL_MS;
  const seconds = (elapsedMs / 1000).toFixed(1);

  return (
    <div
      className="inline-flex items-center gap-2.5 rounded-[22px] rounded-ee-[4px] bg-muted/60 px-4 py-3 shadow-sm"
      role="status"
      aria-live="polite"
    >
      <EmberGlyph reduce={Boolean(shouldReduceMotion)} />
      <span className="text-xs font-medium text-[#3D2E22]">{msg("shared.agent.thinking")}</span>
      {showTimer && (
        <motion.span
          initial={shouldReduceMotion ? false : { opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={shouldReduceMotion ? { duration: 0 } : { duration: 0.3, ease: [0.2, 0.8, 0.2, 1] }}
          dir="ltr"
          aria-hidden="true"
          className="font-mono tabular-nums text-[0.625rem] text-muted-foreground/55"
        >
          {seconds}
          {msg("shared.agent.seconds_short")}
        </motion.span>
      )}
    </div>
  );
}

// The glyph: a warm-tan double ripple radiating from a brand-ink core that
// breathes with a soft gold glow. It replaces the panel's flat `animate-ping`
// halo (see ThinkingSection's ThinkingIndicator, which now reuses this) so the
// whole pre-first-token wait — pure gap *and* reasoning — reads as one distinct,
// premium Skynet moment rather than a plain pulsing dot.
export function EmberGlyph({ reduce }: { reduce: boolean }) {
  if (reduce) {
    return (
      <span className="relative inline-flex size-4 items-center justify-center shrink-0" aria-hidden="true">
        <span className="absolute inset-0 rounded-full bg-[#C8A882]/20" />
        <span className="relative size-2 rounded-full bg-[#3D2E22]" />
      </span>
    );
  }
  return (
    <span className="relative inline-flex size-4 items-center justify-center shrink-0" aria-hidden="true">
      {[0, 0.8].map((delay) => (
        <motion.span
          key={delay}
          className="absolute inset-0 rounded-full border border-[#C8A882]"
          initial={{ scale: 0.6, opacity: 0.55 }}
          animate={{ scale: 1.9, opacity: 0 }}
          transition={{ duration: 1.6, ease: "easeOut", repeat: Infinity, delay }}
        />
      ))}
      <motion.span
        className="relative size-2 rounded-full bg-[#3D2E22]"
        animate={{
          scale: [1, 1.18, 1],
          boxShadow: [
            "0 0 0px rgba(200,168,130,0)",
            "0 0 7px rgba(200,168,130,0.65)",
            "0 0 0px rgba(200,168,130,0)",
          ],
        }}
        transition={{ duration: 1.6, ease: "easeInOut", repeat: Infinity }}
      />
    </span>
  );
}
