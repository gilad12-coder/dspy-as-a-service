"use client";

import { LOCALES, LOCALE_SHORT_LABEL } from "@/shared/lib/locale";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers/locale-provider";

/**
 * Compact segmented control for switching the UI language. Shows every locale by
 * its endonym (each language in its own script) with the active one highlighted,
 * so a user who can't read the current UI can still find their language. The
 * actual switch — cookie write + reload — lives in LocaleProvider.
 */
export function LanguageSwitcher({ className }: { className?: string }) {
  const { locale, setLocale } = useLocale();
  return (
    <div
      role="group"
      aria-label={msg("shared.language.switch_aria")}
      className={cn(
        "inline-flex items-center rounded-lg border border-border/70 p-0.5",
        className,
      )}
    >
      {LOCALES.map((option) => {
        const active = option === locale;
        return (
          <button
            key={option}
            type="button"
            onClick={() => setLocale(option)}
            aria-pressed={active}
            className={cn(
              "rounded-md px-1.5 py-0.5 text-xs font-semibold transition-colors duration-200 cursor-pointer",
              active
                ? "bg-accent text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {LOCALE_SHORT_LABEL[option]}
          </button>
        );
      })}
    </div>
  );
}
