"use client";

import { Check } from "@/shared/ui/icons";
import { motion } from "framer-motion";
import { cn } from "@/shared/lib/utils";

import { STEPS } from "../constants";
import type { SubmitWizardContext } from "../hooks/use-submit-wizard";

export function SubmitStepper({ w }: { w: SubmitWizardContext }) {
  const { step, maxReachableStep, validateStep, handleTabClick } = w;

  return (
    <div className="relative" data-tutorial="wizard-stepper">
      <div className="flex items-center justify-between">
        {STEPS.map((s, i) => {
          const reachable = i <= maxReachableStep;
          const completed = i < step && validateStep(i);
          const active = i === step;
          const clickable = i <= step || reachable;
          const isLast = i === STEPS.length - 1;
          const segmentDone = i + 1 <= step;
          return (
            <div key={s.id} className="flex flex-col items-center relative z-10 flex-1">
              {!isLast ? (
                <div
                  aria-hidden="true"
                  className="absolute top-[18px] sm:top-5 h-px overflow-hidden rounded-full bg-muted/40"
                  style={{
                    insetInlineStart: "calc(50% + 22px)",
                    insetInlineEnd: "calc(-50% + 22px)",
                  }}
                >
                  <motion.div
                    className="h-full rounded-full opacity-40"
                    style={{ background: "linear-gradient(90deg, #c8a882, #a68b6b, #d4b896)" }}
                    initial={{ width: 0 }}
                    animate={{ width: segmentDone ? "100%" : "0%" }}
                    transition={{ duration: 0.45, ease: [0.2, 0.8, 0.2, 1] }}
                  />
                </div>
              ) : null}
              <button
                type="button"
                onClick={() => handleTabClick(i)}
                disabled={!clickable}
                className={cn(
                  "relative z-10 flex items-center justify-center rounded-full transition-all duration-300 cursor-pointer",
                  "size-9 sm:size-10 text-sm font-semibold",
                  active
                    ? "bg-primary text-primary-foreground shadow-[0_0_16px_rgba(124,99,80,0.4)] scale-110"
                    : completed
                      ? "bg-primary/15 text-primary hover:bg-primary/25"
                      : reachable
                        ? "bg-muted text-muted-foreground hover:bg-muted/80 hover:text-foreground"
                        : "bg-muted/50 text-muted-foreground/30 cursor-not-allowed",
                )}
              >
                {completed ? <Check className="size-4" /> : i + 1}
                {active && (
                  <motion.span
                    layoutId="step-ring"
                    className="absolute inset-0 rounded-full border-2 border-primary"
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
              </button>
              <span
                className={cn(
                  "mt-2 text-[0.6875rem] font-medium transition-colors duration-200 hidden sm:block text-center",
                  active ? "text-foreground" : completed ? "text-primary" : "text-muted-foreground",
                )}
              >
                {s.label()}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
