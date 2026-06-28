/**
 * Canonical Hebrew vocabulary for Skynet.
 *
 * Source of truth for the TERMS map: ``i18n/locales/he.json`` (under
 * ``terms``) and ``i18n/glossary.yml``. UI message strings are NOT
 * sourced from here — they live in the per-feature slice files under
 * ``frontend/src/features/<name>/messages.ts`` and are hand-edited.
 *
 * Hebrew copy addresses the user in the singular, supporting both genders via
 * slash forms: "לחץ/י", "בחר/י", "ראה/ראי". Avoid plural imperatives such as
 * "לחצו" or "בחרו": the user is one person (male or female), and the agent
 * ("סוכן") is masculine singular.
 *
 * Keep established borrowed ML terms: "אופטימיזציה", "אופטימייזר",
 * "דאטאסט", and "מודול". Use native Hebrew only for the owner-approved
 * exceptions: baseline -> "בסיס" / "תוצאות בסיס", and metric ->
 * "פונקציית מדידה".
 *
 * Any block that holds user-generated text, model identifiers (`openai/...`),
 * numbers, or code should set `dir="ltr"`. The app shell is RTL by default.
 *
 * Adding a new term:
 * 1. Edit i18n/locales/he.json and i18n/glossary.yml.
 * 2. Run `python scripts/generate_i18n.py`.
 * 3. Reference `TERMS.<key>` from messages, tooltips, constants, and copy
 *    catalogs instead of hardcoding the Hebrew term.
 * 4. Grep for hardcoded variants and migrate or document any deliberate
 *    exceptions in the same change.
 */
import { TERMS as TERMS_HE, TERMS_BY_LOCALE } from "./generated/i18n-catalog";
import { fallbackChain } from "./locale";
import { getActiveLocale } from "./runtime-locale";

export type { TermKey } from "./generated/i18n-catalog";

// Glossary overlays by locale, generated from i18n/locales/*.json. Each locale's
// full term catalog layers over the Hebrew base (the Proxy target); a read walks
// the active locale's fallback chain and takes the first overlay that defines the
// term. New locales appear here automatically via codegen — nothing to wire by
// hand. (`he` is present too and equals the base, harmlessly.) This is the same
// registry i18n.ts resolves `{term.x}` placeholders against.
const TERM_OVERLAYS: Record<string, Record<string, string>> = TERMS_BY_LOCALE;

/**
 * Locale-aware view over the generated glossary.
 *
 * The generated `TERMS_HE` map is the Hebrew base (the Proxy target); every
 * other locale layers in via `TERM_OVERLAYS`. Reading any property
 * resolves against the active locale's fallback chain AT ACCESS TIME — the first
 * overlay that defines the term wins, otherwise the Hebrew base. This is what
 * makes plain `TERMS.optimizationPlural` render the right language in any locale
 * without every call site threading a locale or routing through a `{term.x}`
 * message template; bare access used to be Hebrew-forever, which leaked Hebrew
 * terms all over the non-Hebrew UI.
 *
 * The handler traps reads only — the app never enumerates this map (no
 * `Object.keys`/spread), so a `get` trap is sufficient and keeps the exported
 * shape and `TermKey` typing identical to the raw catalog.
 */
export const TERMS: typeof TERMS_HE = new Proxy(TERMS_HE, {
  get(target, key, receiver) {
    if (typeof key === "string") {
      for (const loc of fallbackChain(getActiveLocale())) {
        const value = TERM_OVERLAYS[loc]?.[key];
        if (value !== undefined) return value;
      }
    }
    return Reflect.get(target, key, receiver);
  },
});
