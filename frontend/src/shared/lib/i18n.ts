import {
  I18N_MESSAGES,
  I18N_MESSAGES_EN,
  TERMS,
  TERMS_EN,
} from "@/shared/lib/generated/i18n-catalog";
import { dirForLocale, fallbackChain, type Locale } from "@/shared/lib/locale";
import { getActiveLocale } from "@/shared/lib/runtime-locale";

export {
  I18N_MESSAGES,
  I18N_KEY,
  type ErrorCode,
  type I18nMessageKey,
} from "@/shared/lib/generated/i18n-catalog";

const TERM_PATTERN = /\{term\.([A-Za-z0-9_]+)\}/g;

const FSI = "⁨";
const PDI = "⁩";
const LATIN_LIKE = /[A-Za-z0-9@./:_\-+#]/;

function reportMissingKey(key: string): void {
  if (typeof console !== "undefined" && typeof console.warn === "function") {
    console.warn(`[i18n] missing translation for ${key}`);
  }
}

function isolate(value: unknown, locale: Locale): string {
  const str = String(value);
  // BiDi isolation only matters in RTL: wrapping Latin-script / numeric runs in
  // FSI/PDI stops them flipping surrounding Hebrew punctuation. In an LTR locale
  // the base direction already matches, so the wrappers are pure noise.
  if (dirForLocale(locale) === "ltr") return str;
  if (LATIN_LIKE.test(str)) return `${FSI}${str}${PDI}`;
  return str;
}

// Glossary catalogs keyed by locale: Hebrew base + English overlay. Walked via
// the registry fallback chain, so a new locale inherits English/Hebrew terms
// until it ships its own overlay.
const TERM_CATALOGS: Partial<Record<Locale, Record<string, string>>> = {
  en: TERMS_EN as Record<string, string>,
  he: TERMS as Record<string, string>,
};

function resolveTerms(template: string, locale: Locale): string {
  return template.replace(TERM_PATTERN, (match, key: string) => {
    for (const loc of fallbackChain(locale)) {
      const value = TERM_CATALOGS[loc]?.[key];
      if (value !== undefined) return value;
    }
    return match;
  });
}

function findMatchingBrace(source: string, start: number): number {
  let depth = 0;
  for (let i = start; i < source.length; i++) {
    const ch = source.charAt(i);
    if (ch === "{") depth++;
    else if (ch === "}") {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function parseBranches(body: string): Record<string, string> {
  const branches: Record<string, string> = {};
  let i = 0;
  while (i < body.length) {
    while (i < body.length && /\s/.test(body.charAt(i))) i++;
    const nameStart = i;
    while (i < body.length && /[\w=]/.test(body.charAt(i))) i++;
    const name = body.slice(nameStart, i);
    while (i < body.length && /\s/.test(body.charAt(i))) i++;
    if (body.charAt(i) !== "{") break;
    const close = findMatchingBrace(body, i);
    if (close === -1) break;
    branches[name] = body.slice(i + 1, close);
    i = close + 1;
  }
  return branches;
}

function pickPluralCategory(count: number, locale: Locale): Intl.LDMLPluralRule {
  try {
    return new Intl.PluralRules(locale).select(count);
  } catch {
    return count === 1 ? "one" : "other";
  }
}

function resolveSubstitutions(
  template: string,
  params: Record<string, unknown>,
  locale: Locale,
): string {
  let out = "";
  let i = 0;
  while (i < template.length) {
    const ch = template.charAt(i);
    if (ch !== "{") {
      out += ch;
      i++;
      continue;
    }
    const close = findMatchingBrace(template, i);
    if (close === -1) {
      out += ch;
      i++;
      continue;
    }
    const inner = template.slice(i + 1, close);
    const pluralMatch = /^(\w+)\s*,\s*plural\s*,\s*([\s\S]+)$/.exec(inner);
    if (pluralMatch && pluralMatch[1] && pluralMatch[2]) {
      const name = pluralMatch[1];
      const branchesStr = pluralMatch[2];
      const raw = params[name];
      const count = typeof raw === "number" ? raw : Number(raw);
      if (!Number.isFinite(count)) {
        out += template.slice(i, close + 1);
      } else {
        const branches = parseBranches(branchesStr);
        const exact = `=${count}`;
        const category = pickPluralCategory(count, locale);
        const body =
          branches[exact] ??
          branches[category] ??
          branches.other ??
          "";
        out += resolveSubstitutions(
          body.replace(/#/g, isolate(count, locale)),
          params,
          locale,
        );
      }
    } else if (/^[A-Za-z0-9_]+$/.test(inner)) {
      if (inner in params) {
        const value = params[inner];
        out += value == null ? template.slice(i, close + 1) : isolate(value, locale);
      } else {
        out += template.slice(i, close + 1);
      }
    } else {
      out += template.slice(i, close + 1);
    }
    i = close + 1;
  }
  return out;
}

/**
 * Render a template string with ICU plurals and BiDi-isolated interpolation.
 * Used by both ``tI18n`` and ``formatMsg`` so backend codes and frontend
 * UI strings share identical formatting semantics. Plural categories, term
 * lookups, and BiDi isolation all follow ``locale`` (defaults to the active
 * request/render locale).
 */
export function formatTemplate(
  template: string,
  params: Record<string, unknown> | undefined,
  locale: Locale = getActiveLocale(),
): string {
  const resolved = resolveTerms(template, locale);
  if (!params) return resolved;
  return resolveSubstitutions(resolved, params, locale);
}

// Backend-code catalogs keyed by locale: Hebrew base + English overlay, walked
// via the registry fallback chain.
const I18N_CATALOGS: Partial<Record<Locale, Record<string, string>>> = {
  en: I18N_MESSAGES_EN as Record<string, string>,
  he: I18N_MESSAGES as Record<string, string>,
};

/**
 * Resolve a backend i18n code into the active locale, with `{term.x}` term
 * lookups, ICU plural support (`{count, plural, one {} two {} other {}}`), and
 * locale-aware BiDi isolation around every interpolated value.
 *
 * Walks the locale's fallback chain over the per-locale code catalogs, so a
 * code with no translation for the active locale degrades to its fallback.
 * Returns the code unchanged when no template exists at all, so drift is
 * dev-visible.
 */
export function tI18n(
  code: string,
  params?: Record<string, unknown>,
  locale: Locale = getActiveLocale(),
): string {
  let template: string | undefined;
  for (const loc of fallbackChain(locale)) {
    const value = I18N_CATALOGS[loc]?.[code];
    if (value !== undefined) {
      template = value;
      break;
    }
  }
  if (template === undefined) {
    reportMissingKey(code);
    return code;
  }
  return formatTemplate(template, params, locale);
}
