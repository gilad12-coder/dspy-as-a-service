/**
 * Locale model shared by the server and client halves of the i18n layer.
 *
 * A single `LOCALE_REGISTRY` is the source of truth: each locale declares its
 * writing direction, its endonym/exonym for the switcher, and the locale it
 * falls back to when a key is untranslated. Adding a language is one registry
 * row plus a translation catalog — nothing here is a hardcoded "is it Hebrew"
 * check. Everything is framework-agnostic (no `next/*` or `react` imports) so it
 * imports from RSC, client components, and plain `.ts` libs alike. Request-time
 * resolution and the sync `msg()` plumbing live in `runtime-locale.ts`.
 *
 * Direction is a property of the writing system, not the language, so the `dir`
 * field is authoritative rather than derived. Today the only RTL locale is
 * Hebrew; an RTL language such as Arabic slots in with `dir: "rtl"` and the
 * existing BiDi / logical-CSS / `<html dir>` plumbing picks it up unchanged.
 *
 * Fallback note: Hebrew is currently the only complete, hand-authored catalog,
 * so it terminates every chain (`he` has `fallback: null`) and English — itself
 * a partial overlay — points back at it. New locales fall through English to
 * Hebrew. Once the English overlay is verified complete the source can flip
 * (`en: null`, `he: "en"`) without touching any call site.
 */

export type Direction = "rtl" | "ltr";

/**
 * Source of truth for every supported locale. The key IS the canonical BCP-47
 * tag (used directly for `Intl.*`), so there is no separate tag indirection.
 * `dir` is validated against `Direction`; `fallback` is checked for validity
 * structurally by `fallbackChain` (a typo'd target fails to compile there).
 */
export const LOCALE_REGISTRY = {
  en:        { dir: "ltr", nativeName: "English",                  englishName: "English",                fallback: "he" },
  he:        { dir: "rtl", nativeName: "עברית",                    englishName: "Hebrew",                 fallback: null },
  "en-GB":   { dir: "ltr", nativeName: "British English",          englishName: "English (UK)",           fallback: "en" },
  "en-IN":   { dir: "ltr", nativeName: "Indian English",           englishName: "English (India)",        fallback: "en" },
  "zh-Hans": { dir: "ltr", nativeName: "简体中文",                  englishName: "Chinese (Simplified)",   fallback: "en" },
  yue:       { dir: "ltr", nativeName: "粵語",                      englishName: "Cantonese",              fallback: "zh-Hans" },
  fr:        { dir: "ltr", nativeName: "français",                 englishName: "French",                 fallback: "en" },
  "fr-CA":   { dir: "ltr", nativeName: "français canadien",        englishName: "French (Canada)",        fallback: "fr" },
  de:        { dir: "ltr", nativeName: "Deutsch",                  englishName: "German",                 fallback: "en" },
  "de-AT":   { dir: "ltr", nativeName: "Österreichisches Deutsch", englishName: "German (Austria)",       fallback: "de" },
  hi:        { dir: "ltr", nativeName: "हिन्दी",                     englishName: "Hindi",                  fallback: "en" },
  it:        { dir: "ltr", nativeName: "italiano",                 englishName: "Italian",                fallback: "en" },
  ja:        { dir: "ltr", nativeName: "日本語",                    englishName: "Japanese",               fallback: "en" },
  ko:        { dir: "ltr", nativeName: "한국어",                    englishName: "Korean",                 fallback: "en" },
  pt:        { dir: "ltr", nativeName: "português",                englishName: "Portuguese",             fallback: "en" },
  "pt-BR":   { dir: "ltr", nativeName: "português (Brasil)",       englishName: "Portuguese (Brazil)",    fallback: "pt" },
  "pt-PT":   { dir: "ltr", nativeName: "português (Portugal)",     englishName: "Portuguese (Portugal)",  fallback: "pt" },
  ru:        { dir: "ltr", nativeName: "русский",                  englishName: "Russian",                fallback: "en" },
  es:        { dir: "ltr", nativeName: "español",                  englishName: "Spanish",                fallback: "en" },
  "es-419":  { dir: "ltr", nativeName: "español (Latinoamérica)",  englishName: "Spanish (Latin America)", fallback: "es" },
  tr:        { dir: "ltr", nativeName: "Türkçe",                   englishName: "Turkish",                fallback: "en" },
  uk:        { dir: "ltr", nativeName: "українська",               englishName: "Ukrainian",              fallback: "en" },
} as const satisfies Record<
  string,
  { dir: Direction; nativeName: string; englishName: string; fallback: string | null }
>;

export type Locale = keyof typeof LOCALE_REGISTRY;

/** One registry row, as stored. */
export type LocaleEntry = (typeof LOCALE_REGISTRY)[Locale];

/** All supported locale tags, in registry (switcher) order. */
export const LOCALES = Object.keys(LOCALE_REGISTRY) as Locale[];

/** English is the product default; every other locale is opt-in via the switcher. */
export const DEFAULT_LOCALE: Locale = "en";

/**
 * Cookie that persists the user's chosen locale. Read server-side in the
 * `force-dynamic` root layout and written client-side by the language switcher.
 */
export const LOCALE_COOKIE = "skynet_locale";

/** One year — a language choice is sticky until the user changes it again. */
export const LOCALE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

/** Narrow an arbitrary value to a supported `Locale`. */
export function isLocale(value: unknown): value is Locale {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(LOCALE_REGISTRY, value);
}

/** Writing direction for a locale, read straight from its registry entry. */
export function dirForLocale(locale: Locale): Direction {
  return LOCALE_REGISTRY[locale].dir;
}

/**
 * BCP-47 tag an `Intl.*` formatter should use for a locale. The locale id is
 * already the canonical tag (e.g. "en", "he", "pt-BR", "es-419"), so it is
 * returned directly — date/number/relative-time output follows the active
 * locale, including regional variants.
 */
export function intlLocaleTag(locale: Locale): string {
  return locale;
}

/**
 * The ordered list of locales to consult for an untranslated key: the locale
 * itself, then each `fallback` pointer until a root (`fallback: null`). E.g.
 * `pt-BR -> pt -> en -> he`, `yue -> zh-Hans -> en -> he`, `he -> he`. The
 * `seen` guard makes a mis-pointed cycle terminate instead of looping.
 */
export function fallbackChain(locale: Locale): Locale[] {
  const chain: Locale[] = [];
  const seen = new Set<Locale>();
  let cur: Locale | null = locale;
  while (cur && !seen.has(cur)) {
    chain.push(cur);
    seen.add(cur);
    cur = LOCALE_REGISTRY[cur].fallback;
  }
  return chain;
}

/**
 * Pick the best supported locale from an `Accept-Language` header.
 *
 * Parses the comma-separated, q-weighted list, sorts by descending quality, and
 * for each requested tag tries an exact registry match first (so `en-GB` and
 * `pt-BR` are honored), then a primary-language match (so `en-AU` resolves to
 * `en` and a bare `zh` resolves to the first `zh-*` we ship).
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
    const exact = LOCALES.find((l) => l.toLowerCase() === tag);
    if (exact) return exact;
    const primary = tag.split("-")[0];
    const byLanguage = LOCALES.find((l) => l.toLowerCase().split("-")[0] === primary);
    if (byLanguage) return byLanguage;
  }
  return null;
}
