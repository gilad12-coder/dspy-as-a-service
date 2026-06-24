"use client";

import { motion } from "framer-motion";
import * as React from "react";
import { ArrowLeft, ArrowRight, X, Play, Pause } from "lucide-react";
import { cn } from "@/shared/lib/utils";
import type { TutorialStep } from "../lib/steps";
import { msg } from "@/shared/lib/messages";

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
  const spring = { type: "spring", stiffness: 400, damping: 35, mass: 0.8 } as const;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96, y: 4 }}
      animate={{
        opacity: 1,
        scale: 1,
        y: 0,
        top: position.top,
        left: position.left,
      }}
      exit={{ opacity: 0, scale: 0.96, y: 4 }}
      transition={{
        opacity: { duration: 0.18, ease: [0.16, 1, 0.3, 1] },
        scale: { duration: 0.22, ease: [0.16, 1, 0.3, 1] },
        y: { duration: 0.22, ease: [0.16, 1, 0.3, 1] },
        top: spring,
        left: spring,
      }}
      className="fixed z-[9999] pointer-events-auto"
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
              <button
                type="button"
                onClick={onToggleAutoPlay}
                className="p-0.5 rounded hover:bg-[#E5DDD4]/60 text-[#8C7A6B] hover:text-[#3D2E22] transition-colors cursor-pointer"
                aria-label={
                  isAutoPlaying
                    ? msg("auto.features.tutorial.components.tutorial.popover.literal.1")
                    : msg("auto.features.tutorial.components.tutorial.popover.literal.2")
                }
                title={
                  isAutoPlaying
                    ? msg("auto.features.tutorial.components.tutorial.popover.literal.3")
                    : msg("auto.features.tutorial.components.tutorial.popover.literal.4")
                }
              >
                {isAutoPlaying ? <Pause className="size-2.5" /> : <Play className="size-2.5" />}
              </button>
            </div>
          </div>
          <button
            type="button"
            onClick={onExit}
            className="close-button shrink-0"
            aria-label={msg("auto.features.tutorial.components.tutorial.popover.literal.5")}
          >
            <X />
          </button>
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
            <button
              type="button"
              onClick={onPrev}
              className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold bg-[#3D2E22] text-[#FAF8F5] hover:bg-[#2C2018] transition-colors cursor-pointer"
            >
              <ArrowRight className="size-3" />
              {msg("auto.features.tutorial.components.tutorial.popover.2")}
            </button>
          )}

          <button
            type="button"
            onClick={onNext}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold bg-[#3D2E22] text-[#FAF8F5] hover:bg-[#2C2018] transition-colors cursor-pointer"
          >
            {isLast
              ? msg("auto.features.tutorial.components.tutorial.popover.literal.6")
              : msg("auto.features.tutorial.components.tutorial.popover.literal.7")}
            {!isLast && <ArrowLeft className="size-3" />}
          </button>
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
