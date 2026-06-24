"use client";

import * as React from "react";
import { Calendar, ChevronLeft, ChevronRight } from "lucide-react";
import {
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@/shared/ui/primitives/popover";
import { msg, formatMsg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";

// Hebrew single-letter weekday labels, indexed by JS getDay() (0=Sun..6=Sat).
const WEEKDAY_LABELS = [
  msg("explore.datepicker.weekday.sun"),
  msg("explore.datepicker.weekday.mon"),
  msg("explore.datepicker.weekday.tue"),
  msg("explore.datepicker.weekday.wed"),
  msg("explore.datepicker.weekday.thu"),
  msg("explore.datepicker.weekday.fri"),
  msg("explore.datepicker.weekday.sat"),
] as const;

const MONTH_HEADER_FMT = new Intl.DateTimeFormat("he-IL", {
  month: "long",
  year: "numeric",
});

const DAY_ARIA_FMT = new Intl.DateTimeFormat("he-IL", {
  weekday: "long",
  day: "numeric",
  month: "long",
  year: "numeric",
});

const DISPLAY_FMT = new Intl.DateTimeFormat("he-IL", {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
});

export function parseISODate(s: string | null | undefined): Date | null {
  if (!s) return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  if (!match) return null;
  const d = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
  return Number.isNaN(d.getTime()) ? null : d;
}

export function toISODate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function startOfDay(d: Date): Date {
  return new Date(d.getFullYear(), d.getMonth(), d.getDate());
}

function startOfMonth(d: Date): Date {
  return new Date(d.getFullYear(), d.getMonth(), 1);
}

function addMonths(d: Date, n: number): Date {
  return new Date(d.getFullYear(), d.getMonth() + n, 1);
}

function isSameDay(a: Date, b: Date): boolean {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  );
}

function clampToBounds(d: Date, min: Date | null, max: Date | null): Date {
  if (min && d < min) return new Date(min);
  if (max && d > max) return new Date(max);
  return d;
}

function isOutOfRange(d: Date, min: Date | null, max: Date | null): boolean {
  if (min && startOfDay(d) < startOfDay(min)) return true;
  if (max && startOfDay(d) > startOfDay(max)) return true;
  return false;
}

function buildMonthGrid(viewDate: Date): Array<Date | null> {
  const first = startOfMonth(viewDate);
  const lastDay = new Date(
    viewDate.getFullYear(),
    viewDate.getMonth() + 1,
    0,
  ).getDate();
  // JS getDay(): 0=Sunday matches the Hebrew week start, so we can use the
  // raw weekday index as the leading-pad count.
  const startWeekday = first.getDay();
  const cells: Array<Date | null> = [];
  for (let i = 0; i < startWeekday; i++) cells.push(null);
  for (let d = 1; d <= lastDay; d++) {
    cells.push(new Date(viewDate.getFullYear(), viewDate.getMonth(), d));
  }
  while (cells.length % 7 !== 0) cells.push(null);
  return cells;
}

interface SkynetDatePickerProps {
  value: string | null;
  onChange: (next: string | null) => void;
  min?: string | null;
  max?: string | null;
  placeholder?: string;
  ariaLabel?: string;
  className?: string;
}

/**
 * Calendar-style date picker tailored to the Skynet visual language.
 * RTL-first: weekday labels read right-to-left starting on Sunday, the
 * "previous month" chevron points right and "next month" points left, and
 * dates are surfaced as YYYY-MM-DD ISO strings so callers stay timezone-safe.
 */
export function SkynetDatePicker({
  value,
  onChange,
  min,
  max,
  placeholder,
  ariaLabel,
  className,
}: SkynetDatePickerProps) {
  const selectedDate = React.useMemo(() => parseISODate(value), [value]);
  const minDate = React.useMemo(() => parseISODate(min), [min]);
  const maxDate = React.useMemo(() => parseISODate(max), [max]);
  const today = React.useMemo(() => startOfDay(new Date()), []);

  const [open, setOpen] = React.useState(false);
  const [viewDate, setViewDate] = React.useState<Date>(() =>
    startOfMonth(selectedDate ?? clampToBounds(today, minDate, maxDate)),
  );
  const [focusedDate, setFocusedDate] = React.useState<Date | null>(null);
  const buttonRefs = React.useRef<Map<string, HTMLButtonElement | null>>(
    new Map(),
  );

  // Pull the view back to the selected month whenever the value changes
  // externally (e.g. cleared via a filter chip).
  React.useEffect(() => {
    if (selectedDate) setViewDate(startOfMonth(selectedDate));
  }, [selectedDate]);

  // On open, seed the focused day so arrow-key navigation has an anchor.
  React.useEffect(() => {
    if (!open) {
      setFocusedDate(null);
      return;
    }
    setFocusedDate(
      selectedDate ?? clampToBounds(today, minDate, maxDate),
    );
  }, [open, selectedDate, today, minDate, maxDate]);

  // Keep DOM focus on the focused cell during arrow nav, but only when
  // focus is already inside the popover — otherwise we'd steal focus from
  // whatever opened it.
  React.useEffect(() => {
    if (!open || !focusedDate) return;
    const el = buttonRefs.current.get(toISODate(focusedDate));
    if (!el || el === document.activeElement) return;
    const popoverEl = el.closest('[data-slot="popover-content"]');
    if (popoverEl?.contains(document.activeElement)) {
      el.focus();
    }
  }, [open, focusedDate]);

  const handleSelect = React.useCallback(
    (d: Date) => {
      if (isOutOfRange(d, minDate, maxDate)) return;
      onChange(toISODate(d));
      setOpen(false);
    },
    [minDate, maxDate, onChange],
  );

  const handleClear = React.useCallback(() => {
    onChange(null);
    setOpen(false);
  }, [onChange]);

  const goToday = React.useCallback(() => {
    if (!isOutOfRange(today, minDate, maxDate)) {
      handleSelect(today);
    } else {
      setViewDate(startOfMonth(clampToBounds(today, minDate, maxDate)));
    }
  }, [today, minDate, maxDate, handleSelect]);

  const onGridKeyDown = (e: React.KeyboardEvent) => {
    if (!focusedDate) return;
    let next: Date | null = null;
    // In RTL, ArrowLeft is visually "next" (forward in time) and ArrowRight
    // is "previous". Up/Down still mean previous/next week — they're not
    // affected by writing direction.
    if (e.key === "ArrowLeft") {
      next = new Date(focusedDate);
      next.setDate(next.getDate() + 1);
    } else if (e.key === "ArrowRight") {
      next = new Date(focusedDate);
      next.setDate(next.getDate() - 1);
    } else if (e.key === "ArrowUp") {
      next = new Date(focusedDate);
      next.setDate(next.getDate() - 7);
    } else if (e.key === "ArrowDown") {
      next = new Date(focusedDate);
      next.setDate(next.getDate() + 7);
    } else if (e.key === "PageUp") {
      next = new Date(focusedDate);
      next.setMonth(next.getMonth() - 1);
    } else if (e.key === "PageDown") {
      next = new Date(focusedDate);
      next.setMonth(next.getMonth() + 1);
    } else if (e.key === "Home") {
      next = new Date(focusedDate.getFullYear(), focusedDate.getMonth(), 1);
    } else if (e.key === "End") {
      next = new Date(focusedDate.getFullYear(), focusedDate.getMonth() + 1, 0);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleSelect(focusedDate);
      return;
    } else {
      return;
    }
    e.preventDefault();
    if (minDate && next < startOfDay(minDate)) next = startOfDay(minDate);
    if (maxDate && next > startOfDay(maxDate)) next = startOfDay(maxDate);
    setFocusedDate(next);
    if (
      next.getMonth() !== viewDate.getMonth() ||
      next.getFullYear() !== viewDate.getFullYear()
    ) {
      setViewDate(startOfMonth(next));
    }
  };

  const grid = React.useMemo(() => buildMonthGrid(viewDate), [viewDate]);
  const monthHeader = MONTH_HEADER_FMT.format(viewDate);
  const displayValue = selectedDate ? DISPLAY_FMT.format(selectedDate) : null;
  const triggerPlaceholder = placeholder ?? msg("explore.datepicker.placeholder");

  const prevMonthDisabled = Boolean(
    minDate && addMonths(viewDate, -1) < startOfMonth(minDate),
  );
  const nextMonthDisabled = Boolean(
    maxDate && addMonths(viewDate, 1) > startOfMonth(maxDate),
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={ariaLabel ?? msg("explore.datepicker.open")}
          data-has-value={displayValue ? "true" : "false"}
          className={`group inline-flex w-full items-center justify-between gap-2 rounded-lg border border-border bg-background px-3 py-2 text-[13px] text-foreground transition-colors cursor-pointer hover:border-foreground/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 data-[state=open]:border-foreground/40 ${className ?? ""}`}
        >
          <span
            dir={displayValue ? "ltr" : undefined}
            className={`tabular-nums ${displayValue ? "" : "text-foreground/40"}`}
          >
            {displayValue ?? triggerPlaceholder}
          </span>
          <Calendar
            className="size-3.5 shrink-0 text-foreground/55 transition-colors group-hover:text-foreground/75 group-data-[state=open]:text-foreground/75"
            aria-hidden="true"
          />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={8}
        className="w-[min(296px,92vw)] p-0"
      >
        <div dir={getActiveDir()} className="flex flex-col">
          <div className="flex items-center justify-between gap-2 border-b border-border/60 px-3 py-2">
            <button
              type="button"
              onClick={() => setViewDate(addMonths(viewDate, -1))}
              aria-label={msg("explore.datepicker.previous_month")}
              disabled={prevMonthDisabled}
              className="inline-flex size-7 items-center justify-center rounded-md text-foreground/60 transition-colors cursor-pointer hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:text-foreground/25 disabled:hover:bg-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              <ChevronRight className="size-4" aria-hidden="true" />
            </button>
            <div
              className="text-[13px] font-medium text-foreground"
              aria-live="polite"
            >
              {monthHeader}
            </div>
            <button
              type="button"
              onClick={() => setViewDate(addMonths(viewDate, 1))}
              aria-label={msg("explore.datepicker.next_month")}
              disabled={nextMonthDisabled}
              className="inline-flex size-7 items-center justify-center rounded-md text-foreground/60 transition-colors cursor-pointer hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:text-foreground/25 disabled:hover:bg-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              <ChevronLeft className="size-4" aria-hidden="true" />
            </button>
          </div>

          <div
            role="grid"
            aria-label={monthHeader}
            onKeyDown={onGridKeyDown}
            className="px-2 py-2"
          >
            <div role="row" className="grid grid-cols-7 pb-1">
              {WEEKDAY_LABELS.map((d) => (
                <div
                  key={d}
                  role="columnheader"
                  className="inline-flex h-7 items-center justify-center text-[11px] font-medium text-foreground/45"
                >
                  {d}
                </div>
              ))}
            </div>
            <div className="grid grid-cols-7">
              {grid.map((cell, idx) => {
                if (!cell) {
                  return (
                    <div
                      key={`empty-${idx}`}
                      className="size-9"
                      aria-hidden="true"
                    />
                  );
                }
                const oor = isOutOfRange(cell, minDate, maxDate);
                const isSelected = selectedDate
                  ? isSameDay(cell, selectedDate)
                  : false;
                const isToday = isSameDay(cell, today);
                const isFocused = focusedDate
                  ? isSameDay(cell, focusedDate)
                  : false;
                const key = toISODate(cell);
                return (
                  <button
                    key={key}
                    ref={(el) => {
                      if (el) buttonRefs.current.set(key, el);
                      else buttonRefs.current.delete(key);
                    }}
                    type="button"
                    role="gridcell"
                    aria-label={formatMsg("explore.datepicker.day.aria", {
                      date: DAY_ARIA_FMT.format(cell),
                    })}
                    aria-selected={isSelected}
                    aria-current={isToday ? "date" : undefined}
                    disabled={oor}
                    tabIndex={isFocused ? 0 : -1}
                    onClick={() => handleSelect(cell)}
                    onFocus={() => setFocusedDate(cell)}
                    className={`relative inline-flex size-9 items-center justify-center rounded-md text-[12.5px] tabular-nums transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 ${
                      oor
                        ? "cursor-not-allowed text-foreground/25"
                        : isSelected
                          ? "cursor-pointer bg-foreground text-background hover:bg-foreground/90"
                          : isToday
                            ? "cursor-pointer text-foreground ring-1 ring-foreground/30 ring-inset hover:bg-accent"
                            : "cursor-pointer text-foreground/80 hover:bg-accent hover:text-foreground"
                    }`}
                  >
                    {cell.getDate()}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex items-center justify-between gap-2 border-t border-border/60 px-3 py-2">
            <button
              type="button"
              onClick={goToday}
              disabled={isOutOfRange(today, minDate, maxDate)}
              className="inline-flex items-center justify-center rounded-md px-2 py-1 text-[12px] text-foreground/65 transition-colors cursor-pointer hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:text-foreground/25 disabled:hover:bg-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              {msg("explore.datepicker.today")}
            </button>
            <button
              type="button"
              onClick={handleClear}
              disabled={!selectedDate}
              className="inline-flex items-center justify-center rounded-md px-2 py-1 text-[12px] text-foreground/65 transition-colors cursor-pointer hover:bg-accent hover:text-foreground disabled:cursor-not-allowed disabled:text-foreground/25 disabled:hover:bg-transparent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              {msg("explore.datepicker.clear")}
            </button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
