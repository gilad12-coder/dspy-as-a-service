"use client";

import * as React from "react";

import { cn } from "@/shared/lib/utils";
import type { QuestionChoice } from "./types";

interface QuestionChoicesProps {
  options: QuestionChoice[];
  /** Send the picked option's label as the user's answer. */
  onSelect: (label: string) => void;
  disabled?: boolean;
  /** Muted line pointing at the composer for a free-text answer. */
  hint?: string;
  /** Localized accessible name for the option group. */
  ariaLabel?: string;
  className?: string;
}

/**
 * The Claude Code / Codex-style multiple-choice answer picker: 2-4 option
 * cards, each a short label plus a one-line description, selectable by click,
 * digit key, or arrow-then-Enter. Picking one sends its label as the answer;
 * the composer below is always the free-text ("something else") path, so this
 * never renders an explicit "other" card. Rendered in place of a plain quick-
 * reply row beneath an interviewer's question.
 */
export function QuestionChoices({
  options,
  onSelect,
  disabled,
  hint,
  ariaLabel,
  className,
}: QuestionChoicesProps) {
  const buttonsRef = React.useRef<Array<HTMLButtonElement | null>>([]);
  const [focusedIndex, setFocusedIndex] = React.useState(0);

  // A fresh question resets the roving focus to its first option.
  React.useEffect(() => {
    setFocusedIndex(0);
    buttonsRef.current = buttonsRef.current.slice(0, options.length);
  }, [options]);

  const moveFocus = (delta: number) => {
    setFocusedIndex((prev) => {
      const next = (prev + delta + options.length) % options.length;
      buttonsRef.current[next]?.focus();
      return next;
    });
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveFocus(1);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      moveFocus(-1);
    } else if (/^[1-9]$/.test(event.key)) {
      const choice = options[Number(event.key) - 1];
      if (choice) {
        event.preventDefault();
        onSelect(choice.label);
      }
    }
  };

  if (options.length === 0) return null;

  return (
    <div
      role="group"
      aria-label={ariaLabel}
      onKeyDown={handleKeyDown}
      className={cn(
        "flex flex-col gap-1.5 border-t border-border/40 px-4 pb-1 pt-3",
        "motion-safe:animate-in motion-safe:fade-in-0 motion-safe:slide-in-from-bottom-1",
        className,
      )}
    >
      {options.map((option, index) => (
        <button
          key={`${index}-${option.label}`}
          ref={(el) => {
            buttonsRef.current[index] = el;
          }}
          type="button"
          disabled={disabled}
          tabIndex={index === focusedIndex ? 0 : -1}
          onFocus={() => setFocusedIndex(index)}
          onClick={() => onSelect(option.label)}
          className={cn(
            "group flex w-full items-start gap-2.5 rounded-lg border px-3 py-2.5 text-start",
            "border-border bg-background transition-colors duration-100 motion-reduce:transition-none",
            "cursor-pointer hover:border-primary/50 hover:bg-primary/5",
            "focus-visible:border-primary/60 focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40",
            "disabled:pointer-events-none disabled:opacity-50",
          )}
        >
          <span
            aria-hidden
            className={cn(
              "mt-px flex size-5 shrink-0 items-center justify-center rounded-md text-[11px] font-medium tabular-nums",
              "border border-border/70 bg-muted text-muted-foreground",
              "transition-colors duration-100 motion-reduce:transition-none",
              "group-hover:border-primary/40 group-hover:bg-primary/10 group-hover:text-primary",
            )}
          >
            {index + 1}
          </span>
          <span className="flex min-w-0 flex-col gap-0.5">
            <span className="text-sm font-medium leading-snug text-foreground" dir="auto">
              {option.label}
            </span>
            {option.description && (
              <span className="text-xs leading-snug text-muted-foreground" dir="auto">
                {option.description}
              </span>
            )}
          </span>
        </button>
      ))}
      {hint && <p className="px-0.5 pt-0.5 text-xs text-muted-foreground/70">{hint}</p>}
    </div>
  );
}

/**
 * Placeholder rendered while the interviewer is still composing its answer
 * options: two pulsing option-card shapes in `QuestionChoices`'s exact
 * geometry, so the real choices land without a layout shift.
 */
export function QuestionChoicesSkeleton({ className }: { className?: string }) {
  return (
    <div
      aria-hidden
      className={cn(
        "flex flex-col gap-1.5 border-t border-border/40 px-4 pb-1 pt-3",
        "motion-safe:animate-in motion-safe:fade-in-0",
        className,
      )}
    >
      {[0, 1].map((index) => (
        <div
          key={index}
          className="flex w-full items-start gap-2.5 rounded-lg border border-border/60 bg-background px-3 py-2.5"
        >
          <span className="mt-px size-5 shrink-0 rounded-md bg-muted motion-safe:animate-pulse" />
          <span className="flex min-w-0 flex-1 flex-col gap-1.5 py-0.5">
            <span className="h-3.5 w-2/5 rounded bg-muted motion-safe:animate-pulse" />
            <span className="h-3 w-3/4 rounded bg-muted/70 motion-safe:animate-pulse" />
          </span>
        </div>
      ))}
    </div>
  );
}
