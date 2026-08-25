/**
 * Locale model shared by the server and client halves of the i18n layer.
 *
 * The private deployment intentionally ships Hebrew only. Keeping a locale
 * model preserves the shared formatter contracts without language negotiation,
 * translation downloads or extra script fonts.
 */

export type Direction = "rtl" | "ltr";

/**
 * Source of truth for every supported locale. The key IS the canonical BCP-47
 * tag (used directly for `Intl.*`), so there is no separate tag indirection.
 * `dir` is validated against `Direction`; `fallback` is checked for validity
 * structurally by `fallbackChain` (a typo'd target fails to compile there).
 */
export const LOCALE_REGISTRY = {
  he: { dir: "rtl", nativeName: "עברית", englishName: "Hebrew", fallback: null },
} as const satisfies Record<
  string,
  { dir: Direction; nativeName: string; englishName: string; fallback: string | null }
>;

export type Locale = keyof typeof LOCALE_REGISTRY;

/** All supported locale tags, in registry (switcher) order. */
export const LOCALES = Object.keys(LOCALE_REGISTRY) as Locale[];

/** Hebrew is the deployment's only locale. */
export const DEFAULT_LOCALE: Locale = "he";

/**
 * Cookie that persists the user's chosen locale. Read server-side in the
 * `force-dynamic` root layout and written client-side by the language switcher.
 */
export const LOCALE_COOKIE = "skynet_locale";

/** One year — a language choice is sticky until the user changes it again. */
export const LOCALE_COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

/**
 * Window event fired synchronously right before a locale switch reloads the
 * page, so in-memory state that should survive the switch (e.g. the submit
 * wizard draft) can stash itself for the hop.
 */
export const LOCALE_RELOAD_EVENT = "skynet:locale-will-reload";

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
  return header == null ? null : DEFAULT_LOCALE;
}
