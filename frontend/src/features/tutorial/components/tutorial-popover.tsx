"use client";

import { motion, useReducedMotion } from "framer-motion";
import * as React from "react";
import { ArrowLeft, ArrowRight, X, Play, Pause } from "@/shared/ui/icons";
import { cn } from "@/shared/lib/utils";
import type { TutorialStep } from "../lib/steps";
import { msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { Button } from "@/shared/ui/primitives/button";

interface TutorialPopoverProps {
  step: TutorialStep;
  stepNumber: number;
  totalSteps: number;
  position: { top: number; left: number; placement: "top" | "bottom" | "left" | "right" };
  onNext: () => void;
  onPrev: () => void;
  onExit: () => void;
  isFirst: boolean;
  isLast: boolean;
  isAutoPlaying: boolean;
  onToggleAutoPlay: () => void;
}

export function TutorialPopover({
  step,
  stepNumber,
  totalSteps,
  position,
  onNext,
  onPrev,
  onExit,
  isFirst,
  isLast,
  isAutoPlaying,
  onToggleAutoPlay,
}: TutorialPopoverProps) {
  const prefersReduced = useReducedMotion();
  // Back points toward the start, Next toward the end — the physical arrow
  // direction flips with the locale (left/right swap in RTL).
  const rtl = getActiveDir() === "rtl";
  const BackArrow = rtl ? ArrowRight : ArrowLeft;
  const NextArrow = rtl ? ArrowLeft : ArrowRight;

  return (
    <motion.div
      initial={prefersReduced ? false : { opacity: 0, scale: 0.97, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={prefersReduced ? { opacity: 0 } : { opacity: 0, scale: 0.97, y: 6 }}
      transition={{ duration: prefersReduced ? 0 : 0.18, ease: [0.2, 0.8, 0.2, 1] }}
      className="fixed z-[9999] pointer-events-auto"
      style={{ top: position.top, left: position.left }}
    >
      <div className="relative w-[min(90vw,360px)] rounded-2xl border border-[#E5DDD4] bg-gradient-to-b from-[#FAF8F5] to-[#F5F1EC] shadow-[0_8px_32px_rgba(28,22,18,0.14)] overflow-hidden">
        <div className="flex items-start justify-between gap-3 px-5 pt-4 pb-2">
          <div className="flex-1 min-w-0">
            <h3 className="text-sm font-bold text-[#3D2E22] leading-tight">{step.title}</h3>
            <div className="flex items-center gap-1.5 mt-0.5">
              <p className="text-[0.625rem] font-medium text-[#8C7A6B]/70 tabular-nums">
                {stepNumber}
                {msg("auto.features.tutorial.components.tutorial.popover.1")}
                {totalSteps}
              </p>
              <Button
                variant="ghost"
                size="icon-xs"
                onClick={onToggleAutoPlay}
                className="text-[#8C7A6B] hover:bg-[#E5DDD4]/60 hover:text-[#3D2E22]"
                aria-label={
                  isAutoPlaying
                    ? msg("auto.features.tutorial.components.tutorial.popover.literal.1")
                    : msg("auto.features.tutorial.components.tutorial.popover.literal.2")
                }
              >
                {isAutoPlaying ? <Pause className="size-3" /> : <Play className="size-3" />}
              </Button>
            </div>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onExit}
            aria-label={msg("auto.features.tutorial.components.tutorial.popover.literal.5")}
          >
            <X className="size-4" />
          </Button>
        </div>

        <div className="px-5 pb-3">
          <p className="text-xs text-[#3D2E22]/75 leading-relaxed">{step.description}</p>
        </div>

        <div className="px-5 pb-3">
          <div className="h-1 bg-[#E5DDD4]/50 rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-[#3D2E22] rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${(stepNumber / totalSteps) * 100}%` }}
              transition={{ duration: 0.35, ease: [0.2, 0.8, 0.2, 1] }}
            />
          </div>
        </div>

        <div
          className={cn(
            "flex items-center gap-2 px-5 pb-4",
            isFirst ? "justify-end" : "justify-between",
          )}
        >
          {!isFirst && (
            <Button variant="outline" size="sm" onClick={onPrev} className="text-xs">
              <BackArrow className="size-3" />
              {msg("auto.features.tutorial.components.tutorial.popover.2")}
            </Button>
          )}

          <Button size="sm" onClick={onNext} className="text-xs">
            {isLast
              ? msg("auto.features.tutorial.components.tutorial.popover.literal.6")
              : msg("auto.features.tutorial.components.tutorial.popover.literal.7")}
            {!isLast && <NextArrow className="size-3" />}
          </Button>
        </div>

        {isAutoPlaying && (
          <motion.div
            className="absolute bottom-0 inset-x-0 h-[2px] bg-[#3D2E22]/30 origin-right"
            initial={{ scaleX: 1 }}
            animate={{ scaleX: 0 }}
            transition={{ duration: step.readingTimeSec ?? 10, ease: "linear" }}
          />
        )}
      </div>
    </motion.div>
  );
}
