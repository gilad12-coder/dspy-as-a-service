/**
 * Locale model shared by the server and client halves of the i18n layer.
 *
 * Skynet ships English-first; Hebrew is the opt-in second locale. Everything
 * here is framework-agnostic so it can be imported from React Server
 * Components, client components, and plain `.ts` libs alike (no `next/*` or
 * `react` imports). Request-time resolution and the sync `msg()` plumbing live
 * in `runtime-locale.ts`; this file is just the vocabulary.
 */

export const LOCALES = ["he", "en"] as const;
export type Locale = (typeof LOCALES)[number];

/** English is the product default — Hebrew is opt-in via the language switcher. */
export const DEFAULT_LOCALE: Locale = "en";

/**
 * Cookie that persists the user's chosen locale. Read server-side in the
 * `force-dynamic` root layout and written client-side by the locale switcher.
 */
export const LOCALE_COOKIE = "skynet_locale";

/** One year — a language choice is sticky until the user changes it again. */
export const LOCALE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

const LOCALE_DIR: Record<Locale, "rtl" | "ltr"> = { he: "rtl", en: "ltr" };

/** Native, self-referential label for each locale (used by the switcher). */
export const LOCALE_LABEL: Record<Locale, string> = { he: "עברית", en: "English" };

/** Short native code for compact controls such as a header toggle. */
export const LOCALE_SHORT_LABEL: Record<Locale, string> = { he: "עב", en: "EN" };

/** Narrow an arbitrary value to a supported `Locale`. */
export function isLocale(value: unknown): value is Locale {
  return typeof value === "string" && (LOCALES as readonly string[]).includes(value);
}

/** Writing direction for a locale: Hebrew is RTL, English is LTR. */
export function dirForLocale(locale: Locale): "rtl" | "ltr" {
  return LOCALE_DIR[locale];
}

/**
 * Pick the best supported locale from an `Accept-Language` header.
 *
 * Parses the comma-separated, q-weighted list, sorts by descending quality,
 * and returns the first entry whose primary subtag matches a supported locale.
 *
 * Args:
 *   header: Raw `Accept-Language` value, or null/undefined when absent.
 *
 * Returns:
 *   The matched `Locale`, or null when nothing supported is requested (the
 *   caller then falls back to `DEFAULT_LOCALE`).
 */
export function localeFromAcceptLanguage(header: string | null | undefined): Locale | null {
  if (!header) return null;
  const ranked = header
    .split(",")
    .map((part) => {
      const [rawTag = "", ...params] = part.trim().split(";");
      const qValue = params.find((p) => p.trim().startsWith("q="))?.split("=")[1];
      const weight = qValue ? Number.parseFloat(qValue) : 1;
      return { tag: rawTag.trim().toLowerCase(), weight: Number.isFinite(weight) ? weight : 0 };
    })
    .filter((entry) => entry.tag)
    .sort((a, b) => b.weight - a.weight);
  for (const { tag } of ranked) {
    const primary = tag.split("-")[0];
    if (isLocale(primary)) return primary;
  }
  return null;
}
