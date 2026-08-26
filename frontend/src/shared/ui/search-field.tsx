"use client";

import * as React from "react";
import { X } from "@/shared/ui/icons";

import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";

interface SearchFieldProps {
  value: string;
  onValueChange: (next: string) => void;
  placeholder: string;
  className?: string;
}

/**
 * The house search input: the Explore bar's rounded surface — border, focus
 * glow, quiet placeholder, inline clear — sized for list-filter toolbars, so
 * every search field in the app reads as the same control.
 */
export function SearchField({ value, onValueChange, placeholder, className }: SearchFieldProps) {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const isActive = value.trim().length > 0;

  return (
    <div
      className={cn(
        "group relative flex h-11 items-center gap-1 rounded-2xl border border-border bg-background ps-4 pe-1 transition-[border-color,box-shadow] duration-150 ease-out focus-within:border-foreground/40 focus-within:shadow-[0_2px_24px_-12px_oklch(0.25_0.04_45/.18)]",
        isActive && "border-foreground/25",
        className,
      )}
    >
      <input
        ref={inputRef}
        type="text"
        inputMode="search"
        autoComplete="off"
        spellCheck={false}
        dir="auto"
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            if (value) onValueChange("");
            else inputRef.current?.blur();
          }
        }}
        placeholder={placeholder}
        aria-label={placeholder}
        className="min-w-0 flex-1 bg-transparent px-2 py-1.5 text-[15px] tracking-tight text-foreground placeholder:text-foreground/40 focus:outline-none"
      />
      {value.length > 0 && (
        <button
          type="button"
          onClick={() => {
            onValueChange("");
            inputRef.current?.focus();
          }}
          aria-label={msg("search.clear")}
          className="inline-flex size-8 shrink-0 items-center justify-center rounded-lg text-foreground/55 transition-[background-color,color] cursor-pointer hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
        >
          <X className="size-4" aria-hidden="true" />
        </button>
      )}
    </div>
  );
}
