"use client";

import * as React from "react";
import { Minus, Plus } from "lucide-react";
import { cn } from "@/shared/lib/utils";

interface NumberInputProps {
  id?: string;
  value: number | "";
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  className?: string;
  disabled?: boolean;
}

export function NumberInput({
  id,
  value,
  onChange,
  min,
  max,
  step = 1,
  className,
  disabled,
}: NumberInputProps) {
  const numValue = typeof value === "number" ? value : 0;
  const decimals = step < 1 ? Math.max(String(step).split(".")[1]?.length ?? 0, 2) : 0;
  const round = (n: number) => (decimals ? parseFloat(n.toFixed(decimals)) : n);
  const clamp = (n: number) => {
    if (min != null && n < min) return min;
    if (max != null && n > max) return max;
    return n;
  };
  const format = (n: number) => (decimals ? n.toFixed(decimals) : String(n));

  // The field owns its text while focused so a mid-edit value — an empty box, a
  // trailing "0.", "0.6" before its final digit — survives keystrokes instead of
  // being reparsed and reformatted out from under the caret on every change (the
  // old behaviour made a typed digit read as a tiny nudge rather than the number
  // you meant). External value changes are mirrored in only when not editing;
  // blur normalizes back to the canonical format.
  const [text, setText] = React.useState(typeof value === "number" ? format(value) : "");
  const editing = React.useRef(false);

  React.useEffect(() => {
    if (editing.current) return;
    setText(typeof value === "number" ? (decimals ? value.toFixed(decimals) : String(value)) : "");
  }, [value, decimals]);

  const commitText = (raw: string) => {
    if (raw === "" || raw === ".") return;
    const n = parseFloat(raw);
    if (!isNaN(n)) onChange(round(clamp(n)));
  };

  const setValue = (next: number) => {
    onChange(next);
    setText(format(next));
  };

  const decrement = () => {
    const next = round(numValue - step);
    if (min != null && next < min) return;
    setValue(next);
  };

  const increment = () => {
    const next = round(numValue + step);
    if (max != null && next > max) return;
    setValue(next);
  };

  return (
    <div
      className={cn(
        "flex items-center h-9 rounded-xl border border-input/90 bg-background/75 shadow-[inset_0_1px_0_rgba(255,255,255,0.72),0_12px_26px_-24px_rgba(15,23,42,0.45)] backdrop-blur-sm overflow-hidden",
        disabled && "opacity-50 pointer-events-none",
        className,
      )}
    >
      <button
        type="button"
        onClick={decrement}
        disabled={disabled || (min != null && numValue <= min)}
        className="flex items-center justify-center size-9 shrink-0 text-muted-foreground hover:text-foreground hover:bg-accent/60 transition-colors disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
        aria-label="Decrease"
      >
        <Minus className="size-3" />
      </button>
      <input
        id={id}
        type="text"
        inputMode={decimals ? "decimal" : "numeric"}
        value={text}
        onFocus={() => {
          editing.current = true;
        }}
        onChange={(e) => {
          const raw = e.target.value.replace(/[^0-9.]/g, "");
          setText(raw);
          commitText(raw);
        }}
        onBlur={() => {
          editing.current = false;
          setText(typeof value === "number" ? format(value) : "");
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowUp") {
            e.preventDefault();
            increment();
          } else if (e.key === "ArrowDown") {
            e.preventDefault();
            decrement();
          }
        }}
        disabled={disabled}
        className="flex-1 min-w-0 h-full bg-transparent text-center text-sm tabular-nums outline-none"
        dir="ltr"
      />
      <button
        type="button"
        onClick={increment}
        disabled={disabled || (max != null && numValue >= max)}
        className="flex items-center justify-center size-9 shrink-0 text-muted-foreground hover:text-foreground hover:bg-accent/60 transition-colors disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
        aria-label="Increase"
      >
        <Plus className="size-3" />
      </button>
    </div>
  );
}
