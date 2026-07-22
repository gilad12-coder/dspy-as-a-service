"use client";

import * as React from "react";
import { Check } from "lucide-react";

import { cn } from "@/shared/lib/utils";

interface SelectCheckboxProps {
  checked: boolean;
  /** ``shiftKey`` is true on shift-click, for range selection. */
  onToggle: (shiftKey: boolean) => void;
  ariaLabel: string;
  disabled?: boolean;
}

/**
 * The rounded multi-select checkbox used by the storage cleanup drawer,
 * extracted so card lists (labeling sessions, datasets) select the same way:
 * plain click toggles, shift-click extends a range. Stops its events so a
 * click or Space press never activates the clickable row behind it.
 */
export function SelectCheckbox({ checked, onToggle, ariaLabel, disabled }: SelectCheckboxProps) {
  return (
    <button
      type="button"
      role="checkbox"
      aria-checked={checked}
      aria-label={ariaLabel}
      disabled={disabled}
      onClick={(e) => {
        e.stopPropagation();
        onToggle(e.shiftKey);
      }}
      onKeyDown={(e) => e.stopPropagation()}
      className={cn(
        "grid size-5 shrink-0 cursor-pointer place-items-center rounded-md border transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45",
        checked
          ? "border-transparent bg-foreground text-background"
          : "border-border/70 bg-background hover:border-foreground/40",
      )}
    >
      {checked && <Check className="size-3.5" strokeWidth={3} aria-hidden="true" />}
    </button>
  );
}
