"use client";

import * as React from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Check, ChevronDown } from "lucide-react";

import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";

import type { AgentThinking } from "./types";

// Within this many px of the bottom still counts as "at the bottom", so
// momentum scrolling and fractional positions don't break follow mode.
const STICKY_BOTTOM_SLACK_PX = 48;

export function ThinkingSection({ thinking }: { thinking: AgentThinking }) {
  const { reasoning, startedAt, endedAt, streaming } = thinking;
  const isThinking = streaming && !endedAt && Boolean(startedAt);
  const hasFinished = Boolean(endedAt && startedAt);
  const [open, setOpen] = React.useState(true);
  const [nowTs, setNowTs] = React.useState(() => Date.now());
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const autoCollapsedRef = React.useRef(false);
  const shouldReduceMotion = useReducedMotion();

  React.useEffect(() => {
    if (!isThinking) return;
    const id = setInterval(() => setNowTs(Date.now()), 200);
    return () => clearInterval(id);
  }, [isThinking]);

  React.useEffect(() => {
    if (hasFinished && !autoCollapsedRef.current) {
      autoCollapsedRef.current = true;
      setOpen(false);
    }
  }, [hasFinished]);

  // Follow the stream only while the user is at the bottom. Once they scroll
  // up to read earlier reasoning their position persists — new tokens must
  // not yank them back down. Scrolling back to the bottom (or collapsing and
  // reopening) re-engages following. The programmatic pin lands at distance
  // 0, so it keeps the flag true rather than fighting the user.
  const stickToBottomRef = React.useRef(true);

  const handleScroll = React.useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    stickToBottomRef.current =
      el.scrollHeight - el.scrollTop - el.clientHeight < STICKY_BOTTOM_SLACK_PX;
  }, []);

  React.useEffect(() => {
    if (!open) {
      stickToBottomRef.current = true;
      return;
    }
    const el = scrollRef.current;
    if (el && stickToBottomRef.current) el.scrollTop = el.scrollHeight;
  }, [reasoning, open]);

  const tail = React.useMemo(() => {
    if (!reasoning) return "";
    const cleaned = reasoning.replace(/\s+/g, " ").trim();
    return cleaned.length > 90 ? `…${cleaned.slice(-89)}` : cleaned;
  }, [reasoning]);

  if (!reasoning && !isThinking) return null;

  const elapsedMs = startedAt ? (endedAt ?? nowTs) - startedAt : 0;
  const elapsedSec = Math.max(0, Math.round(elapsedMs / 100) / 10);
  const label = isThinking
    ? msg("shared.agent.thinking")
    : formatMsg("shared.agent.thought_seconds", { seconds: elapsedSec.toFixed(1) });

  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2.5 px-4 py-2.5 text-start hover:bg-[#3D2E22]/[0.035] transition-colors cursor-pointer"
        aria-expanded={open}
      >
        <ThinkingIndicator active={isThinking} />
        <div className="flex items-baseline gap-2 shrink-0">
          <span
            className={cn(
              "text-xs font-medium",
              isThinking
                ? "text-[#3D2E22] animate-pulse motion-reduce:animate-none"
                : "text-foreground/70",
            )}
          >
            {label}
          </span>
          {isThinking && (
            <span className="font-mono tabular-nums text-[0.625rem] text-muted-foreground/55">
              {elapsedSec.toFixed(1)}
              {msg("shared.agent.seconds_short")}
            </span>
          )}
        </div>
        {!open && tail && (
          <span
            className="flex-1 min-w-0 text-[0.6875rem] text-muted-foreground/45 font-mono truncate"
            dir="ltr"
          >
            {tail}
          </span>
        )}
        <ChevronDown
          className={cn(
            "ms-auto size-3.5 text-muted-foreground/50 transition-transform shrink-0",
            open ? "rotate-0" : "rotate-90",
          )}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && reasoning && (
          <motion.div
            initial={shouldReduceMotion ? false : { height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={shouldReduceMotion ? undefined : { height: 0, opacity: 0 }}
            transition={
              shouldReduceMotion ? { duration: 0 } : { duration: 0.22, ease: [0.2, 0.8, 0.2, 1] }
            }
            className="overflow-hidden"
          >
            <div
              ref={scrollRef}
              onScroll={handleScroll}
              dir="ltr"
              className="max-h-44 overflow-y-auto border-t border-[#3D2E22]/[0.08] px-4 py-3 text-[0.6875rem] leading-[1.65] text-[#5C4D40]/80 font-mono whitespace-pre-wrap break-words"
            >
              {reasoning}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ThinkingIndicator({ active }: { active: boolean }) {
  if (active) {
    return (
      <span className="relative inline-flex size-4 items-center justify-center shrink-0">
        <span className="absolute inset-0 rounded-full bg-[#3D2E22]/15 animate-ping motion-reduce:animate-none" />
        <span className="relative size-2 rounded-full bg-[#3D2E22]" />
      </span>
    );
  }
  return (
    <span className="inline-flex size-4 items-center justify-center rounded-full bg-[#3D2E22]/15 shrink-0">
      <Check className="size-2.5 text-[#3D2E22]" strokeWidth={3} />
    </span>
  );
}
