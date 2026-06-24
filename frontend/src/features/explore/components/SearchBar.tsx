"use client";

import * as React from "react";
import { motion } from "framer-motion";
import {
  FilterX,
  Globe,
  Loader2,
  SlidersHorizontal,
  User,
  Users,
  X,
} from "lucide-react";
import { msg } from "@/shared/lib/messages";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import type { ExploreCorpus } from "../hooks/use-semantic-search";
import { SearchSuggestions } from "./SearchSuggestions";

const PILL_TRANSITION = { type: "tween", duration: 0.18, ease: [0.22, 1, 0.36, 1] } as const;

interface SearchBarProps {
  /** The committed query — what's actually being searched (mirrors the URL). */
  text: string;
  /** Fires on debounced typing pause, explicit Enter, or clear. */
  onSubmit: (next: string) => void;
  corpus: ExploreCorpus;
  onCorpusChange: (next: ExploreCorpus) => void;
  /** Disables the session-scoped tabs (Mine, Shared) when no logged-in user. */
  signedIn: boolean;
  filtersCount: number;
  onOpenFilters: () => void;
  /** Quick-clears the metadata filters (preserving the text query). */
  onClearFilters: () => void;
  /** True while a search request is in flight — drives the inline spinner. */
  loading: boolean;
  /**
   * Result keyboard-nav handler (↑/↓/Enter/Escape). Called first on keydown;
   * returns true when it consumed the event so the input skips its defaults.
   */
  onResultKeyDown: (event: React.KeyboardEvent) => boolean;
  /** Index of the keyboard-highlighted result, or -1 — drives aria + panel. */
  activeResultIndex: number;
  /** Recently-used queries shown when the field is focused and empty. */
  recentQueries: string[];
  onClearRecent: () => void;
  /** Trending public queries (ranked server-side) shown alongside recent queries. */
  suggestions: string[];
}

// Wait this long after the user stops typing before committing the query.
// This is the *only* debounce on the path (the fetch fires immediately off
// the committed value), so it's tuned to the 250–300ms sweet spot for
// server-backed search: a burst of keystrokes collapses to one request, yet
// results land while the user's attention is still on the field. Enter
// bypasses the wait entirely.
const TYPING_DEBOUNCE_MS = 250;

/**
 * The page's center of gravity. A segmented corpus toggle sits on top so the
 * user can pick where to search (their own jobs, runs shared with them, or
 * other users' public ones), and the rounded input surface below carries the
 * free-text query and
 * filters affordance. Keyboard: pressing "/" anywhere focuses the input;
 * Enter fires the search immediately.
 *
 * Typing into the input auto-submits after a short pause so the embedding
 * API isn't hammered with one request per keystroke — Enter exists as an
 * escape hatch that skips the wait.
 */
export function SearchBar({
  text,
  onSubmit,
  corpus,
  onCorpusChange,
  signedIn,
  filtersCount,
  onOpenFilters,
  onClearFilters,
  loading,
  onResultKeyDown,
  activeResultIndex,
  recentQueries,
  onClearRecent,
  suggestions,
}: SearchBarProps) {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const [draft, setDraft] = React.useState(text);
  const [focused, setFocused] = React.useState(false);
  React.useEffect(() => {
    setDraft(text);
  }, [text]);
  const inputDir = detectInputDir(draft);
  const isActive = text.trim().length > 0 || filtersCount > 0;

  // The suggestions dropdown only competes for attention on a blank field;
  // once the user starts arrowing through results (activeResultIndex >= 0) it
  // yields so the two affordances never overlap.
  const activeResultId =
    activeResultIndex >= 0 ? `explore-result-${activeResultIndex}` : undefined;
  const suggestOpen =
    focused &&
    draft.trim().length === 0 &&
    activeResultIndex < 0 &&
    (recentQueries.length > 0 || suggestions.length > 0);

  // Debounce typing into committed submissions. Compare against the
  // already-committed ``text`` so a draft that already matches the URL
  // doesn't re-fire the search on focus changes or external resets.
  React.useEffect(() => {
    if (draft === text) return;
    const handle = window.setTimeout(() => {
      onSubmit(draft);
    }, TYPING_DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
  }, [draft, text, onSubmit]);

  const submitDraft = React.useCallback(() => {
    onSubmit(draft);
  }, [draft, onSubmit]);

  const clearAll = React.useCallback(() => {
    setDraft("");
    onSubmit("");
    inputRef.current?.focus();
  }, [onSubmit]);

  const selectSuggestion = React.useCallback(
    (value: string) => {
      setDraft(value);
      onSubmit(value);
      // Blur to dismiss the panel; the committed value stays in the field.
      inputRef.current?.blur();
    },
    [onSubmit],
  );

  // "/" focuses the input from anywhere on the page (Google's shortcut).
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "/") return;
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || target?.isContentEditable) return;
      e.preventDefault();
      inputRef.current?.focus();
      inputRef.current?.select();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div data-tutorial="explore-search" className="mx-auto flex w-full max-w-3xl flex-col gap-2.5">
      <div className="flex items-center justify-center">
        <CorpusToggle
          value={corpus}
          onChange={onCorpusChange}
          signedIn={signedIn}
        />
      </div>
      <div
        className={`group relative flex items-center gap-1 rounded-2xl border border-border bg-background ps-4 pe-1 py-1.5 transition-[border-color,box-shadow] duration-150 ease-out focus-within:border-foreground/40 focus-within:shadow-[0_2px_24px_-12px_oklch(0.25_0.04_45/.18)] ${
          isActive ? "border-foreground/25" : ""
        }`}
      >
        <input
          ref={inputRef}
          dir="auto"
          type="text"
          inputMode="search"
          autoComplete="off"
          spellCheck={false}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          onKeyDown={(e) => {
            if (onResultKeyDown(e)) return;
            if (e.key === "Enter") {
              e.preventDefault();
              submitDraft();
            } else if (e.key === "Escape") {
              if (draft) clearAll();
              else inputRef.current?.blur();
            }
          }}
          placeholder={msg("explore.search.placeholder")}
          aria-label={msg("explore.search.aria")}
          aria-busy={loading}
          aria-controls="explore-results"
          aria-activedescendant={activeResultId}
          style={{ textAlign: inputDir === "rtl" ? "right" : "left" }}
          className="min-w-0 flex-1 bg-transparent px-2 py-1.5 text-[15px] tracking-tight text-foreground placeholder:text-foreground/40 focus:outline-none"
        />
        {loading && (
          <span
            className="inline-flex size-8 shrink-0 items-center justify-center"
            aria-hidden="true"
          >
            <Loader2 className="size-4 animate-spin text-foreground/40" />
          </span>
        )}
        {draft.length > 0 && (
          <button
            type="button"
            onClick={clearAll}
            aria-label={msg("explore.search.clear")}
            className="inline-flex size-8 shrink-0 items-center justify-center rounded-lg text-foreground/55 transition-[background-color,color] cursor-pointer hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
          >
            <X className="size-4" aria-hidden="true" />
          </button>
        )}
        {filtersCount > 0 && (
          <TooltipButton tooltip={msg("explore.filters.reset")} side="bottom">
            <button
              type="button"
              onClick={onClearFilters}
              aria-label={msg("explore.filters.reset")}
              className="inline-flex size-9 shrink-0 items-center justify-center rounded-lg text-foreground/55 transition-[background-color,color] cursor-pointer hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              <FilterX className="size-[1.05rem]" aria-hidden="true" />
            </button>
          </TooltipButton>
        )}
        <button
          type="button"
          onClick={onOpenFilters}
          aria-label={msg("explore.filters.button")}
          className="inline-flex h-9 shrink-0 items-center gap-1.5 rounded-lg px-2.5 text-[13px] text-foreground/70 transition-[background-color,color] cursor-pointer hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
        >
          <SlidersHorizontal className="size-[1.125rem]" aria-hidden="true" />
          {filtersCount > 0 && (
            <span
              dir="ltr"
              className="inline-flex min-w-5 items-center justify-center rounded-full bg-foreground px-1.5 text-[10px] font-semibold leading-tight text-background tabular-nums"
            >
              {filtersCount}
            </span>
          )}
        </button>
        {suggestOpen && (
          <SearchSuggestions
            recent={recentQueries}
            suggestions={suggestions}
            onSelect={selectSuggestion}
            onClearRecent={onClearRecent}
          />
        )}
      </div>
      <div className="mx-1 flex items-center justify-end gap-3 text-[12px] text-foreground/55">
        <span className="hidden text-[11px] text-foreground/40 md:inline">
          {msg("explore.search.kbd_hint")}
        </span>
      </div>
    </div>
  );
}

/**
 * Walk the typed text and return "rtl" or "ltr" based on the first strong
 * directional character. Defaults to "rtl" when the value is empty so the
 * Hebrew placeholder renders correctly. More reliable than `dir="auto"`,
 * which falls back to the parent direction inconsistently across browsers
 * when the input is empty or starts with whitespace/punctuation.
 */
function detectInputDir(text: string): "rtl" | "ltr" {
  for (const ch of text) {
    const code = ch.codePointAt(0);
    if (code === undefined) continue;
    if (
      (code >= 0x0590 && code <= 0x05ff) ||
      (code >= 0x0600 && code <= 0x06ff) ||
      (code >= 0x0700 && code <= 0x074f) ||
      (code >= 0xfb1d && code <= 0xfdff) ||
      (code >= 0xfe70 && code <= 0xfeff)
    ) {
      return "rtl";
    }
    if (
      (code >= 0x0041 && code <= 0x005a) ||
      (code >= 0x0061 && code <= 0x007a) ||
      (code >= 0x00c0 && code <= 0x024f) ||
      (code >= 0x0250 && code <= 0x02af)
    ) {
      return "ltr";
    }
  }
  return "rtl";
}

function CorpusToggle({
  value,
  onChange,
  signedIn,
}: {
  value: ExploreCorpus;
  onChange: (next: ExploreCorpus) => void;
  signedIn: boolean;
}) {
  const segments: ReadonlyArray<{
    value: ExploreCorpus;
    label: string;
    icon: React.ComponentType<{ className?: string; "aria-hidden"?: boolean | "true" }>;
    aria: string;
    disabled?: boolean;
  }> = [
    {
      value: "mine",
      label: msg("explore.corpus.mine"),
      icon: User,
      aria: msg("explore.corpus.mine.aria"),
      disabled: !signedIn,
    },
    {
      value: "shared",
      label: msg("explore.corpus.shared"),
      icon: Users,
      aria: msg("explore.corpus.shared.aria"),
      disabled: !signedIn,
    },
    {
      value: "public",
      label: msg("explore.corpus.public"),
      icon: Globe,
      aria: msg("explore.corpus.public.aria"),
    },
  ];

  return (
    <div
      role="radiogroup"
      aria-label={msg("explore.corpus.aria")}
      className="relative inline-flex items-center rounded-full border border-border/80 bg-muted/40 p-0.5"
    >
      {segments.map((seg) => {
        const active = seg.value === value;
        const Icon = seg.icon;
        const disabled = seg.disabled === true;
        return (
          <TooltipButton key={seg.value} tooltip={seg.aria} side="bottom">
            <button
              type="button"
              role="radio"
              aria-checked={active}
              aria-label={seg.aria}
              disabled={disabled}
              onClick={() => {
                if (disabled) return;
                if (!active) onChange(seg.value);
              }}
              className={`relative inline-flex items-center gap-1.5 rounded-full px-3.5 py-1.5 text-[12.5px] font-medium transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 ${
                active
                  ? "text-foreground"
                  : disabled
                  ? "cursor-not-allowed text-foreground/30"
                  : "cursor-pointer text-foreground/60 hover:text-foreground"
              }`}
            >
              {active && (
                <motion.span
                  layoutId="explore-corpus-pill"
                  className="absolute inset-0 rounded-full bg-background shadow-[0_1px_2px_oklch(0.25_0.04_45/.12)]"
                  transition={PILL_TRANSITION}
                  aria-hidden="true"
                />
              )}
              <span className="relative z-10 inline-flex items-center gap-1.5">
                <Icon className="size-3.5" aria-hidden="true" />
                <span>{seg.label}</span>
              </span>
            </button>
          </TooltipButton>
        );
      })}
    </div>
  );
}
