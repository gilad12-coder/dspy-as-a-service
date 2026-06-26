"use client";

import * as React from "react";
import { Check, Languages, Wand2 } from "lucide-react";
import { LOCALES, LOCALE_REGISTRY, type Locale } from "@/shared/lib/locale";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers/locale-provider";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/ui/primitives/popover";

/** Strip combining marks + case so "francais" matches "français". */
function normalize(value: string): string {
  return value
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .toLowerCase();
}

/**
 * Searchable language combobox. Replaces the old he/en segmented toggle in the
 * same header slot: the trigger shows the active locale's endonym, and the
 * popover lists every registry locale (endonym + dimmed English name) filtered
 * by a diacritic-insensitive search over endonym, English name, and tag. Each
 * row renders in its own writing direction so RTL endonyms read correctly. The
 * actual switch — cookie write + reload — stays in LocaleProvider.
 */
export function LanguageSwitcher({ className }: { className?: string }) {
  const { locale, setLocale, isAuto, resetToAuto } = useLocale();
  const [open, setOpen] = React.useState(false);
  const [query, setQuery] = React.useState("");
  const [highlight, setHighlight] = React.useState(0);
  const inputRef = React.useRef<HTMLInputElement>(null);

  const current = LOCALE_REGISTRY[locale];

  const results = React.useMemo(() => {
    const q = normalize(query.trim());
    if (!q) return LOCALES;
    return LOCALES.filter((l) => {
      const entry = LOCALE_REGISTRY[l];
      return (
        normalize(entry.nativeName).includes(q) ||
        normalize(entry.englishName).includes(q) ||
        l.toLowerCase().includes(q)
      );
    });
  }, [query]);

  React.useEffect(() => {
    setHighlight(0);
  }, [query]);

  React.useEffect(() => {
    if (!open) return;
    setQuery("");
    const id = window.requestAnimationFrame(() => inputRef.current?.focus());
    return () => window.cancelAnimationFrame(id);
  }, [open]);

  const choose = (next: Locale) => {
    setOpen(false);
    setLocale(next);
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      setHighlight((i) => Math.min(i + 1, results.length - 1));
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      setHighlight((i) => Math.max(i - 1, 0));
    } else if (event.key === "Enter") {
      event.preventDefault();
      const next = results[highlight];
      if (next) choose(next);
    }
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={msg("shared.language.switch_aria")}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-lg border border-border/70 px-2 py-1 text-xs font-semibold text-foreground transition-colors duration-200 cursor-pointer hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45",
            className,
          )}
        >
          <Languages className="size-3.5 text-muted-foreground" aria-hidden="true" />
          <span dir={current.dir}>{current.nativeName}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-64 p-1" onKeyDown={onKeyDown}>
        <input
          ref={inputRef}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={msg("shared.language.search_placeholder")}
          dir="auto"
          aria-label={msg("shared.language.switch_aria")}
          className="mb-1 w-full rounded-md border-b border-border/60 bg-transparent px-2 py-1.5 text-sm text-foreground outline-none placeholder:text-muted-foreground"
        />
        {!query.trim() && (
          <button
            type="button"
            role="option"
            aria-selected={isAuto}
            onClick={() => {
              setOpen(false);
              resetToAuto();
            }}
            className="mb-1 flex w-full items-center gap-2 rounded-md border-b border-border/40 px-2 py-1.5 text-start transition-colors cursor-pointer hover:bg-accent/60"
          >
            <Wand2 className="size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
            <span className="flex min-w-0 flex-1 flex-col">
              <span className="truncate text-sm text-foreground">
                {msg("shared.language.auto_detect")}
              </span>
              {isAuto && (
                <span dir={current.dir} className="truncate text-[0.6875rem] text-muted-foreground">
                  {current.nativeName}
                </span>
              )}
            </span>
            {isAuto && <Check className="size-4 shrink-0 text-[#C8A882]" aria-hidden="true" />}
          </button>
        )}
        <div role="listbox" className="max-h-72 overflow-y-auto">
          {results.length === 0 ? (
            <p className="px-2 py-3 text-center text-xs text-muted-foreground">
              {msg("shared.language.no_results")}
            </p>
          ) : (
            results.map((l, i) => {
              const entry = LOCALE_REGISTRY[l];
              const selected = !isAuto && l === locale;
              return (
                <button
                  key={l}
                  type="button"
                  role="option"
                  aria-selected={selected}
                  onMouseMove={() => setHighlight(i)}
                  onClick={() => choose(l)}
                  className={cn(
                    "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-start transition-colors cursor-pointer",
                    i === highlight ? "bg-accent" : "hover:bg-accent/60",
                  )}
                >
                  <span className="flex min-w-0 flex-1 flex-col">
                    <span dir={entry.dir} className="truncate text-sm text-foreground">
                      {entry.nativeName}
                    </span>
                    <span className="truncate text-[0.6875rem] text-muted-foreground">
                      {entry.englishName}
                    </span>
                  </span>
                  {selected && <Check className="size-4 shrink-0 text-[#C8A882]" aria-hidden="true" />}
                </button>
              );
            })
          )}
        </div>
      </PopoverContent>
    </Popover>
  );
}
