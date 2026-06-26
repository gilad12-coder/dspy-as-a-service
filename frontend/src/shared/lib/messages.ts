/**
 * `msg()` — resolve a user-facing UI string by key in the active locale.
 *
 * The strings themselves are no longer bundled here. They live in
 * ``i18n/locales/ui/<locale>.json`` (Hebrew is the complete base; other locales
 * are overlays), are compiled into ``generated/ui-catalog.ts`` by
 * ``scripts/generate_i18n.py``, and reach the runtime as the *active* locale's
 * merged catalog only — built server-side (``messages.server.ts``) and read
 * through the ``runtime-messages`` slot. So the browser ships one locale's
 * strings, not all of them. Adding a language is: drop a JSON catalog, add a
 * registry row in ``locale.ts``, regenerate.
 *
 * Backend i18n codes (errors, validations) take a different path: they live in
 * ``i18n/locales/he.json``, regenerate into ``generated/i18n-catalog.ts``, and
 * resolve via ``tI18n``.
 */

import { formatTemplate } from "@/shared/lib/i18n";
import { getActiveLocale } from "@/shared/lib/runtime-locale";
import { getActiveMessages } from "@/shared/lib/runtime-messages";

export type { MessageKey } from "@/shared/lib/generated/ui-catalog";

import type { MessageKey } from "@/shared/lib/generated/ui-catalog";

type MessageParams = Record<string, string | number>;

/**
 * Look up a user-facing string by key and optionally interpolate placeholders.
 *
 * Reads the active locale's pre-merged catalog (already resolved down the
 * fallback chain server-side), so a key is present whenever it exists for any
 * locale in the chain; a genuinely unknown key degrades to itself rather than
 * throwing. The template is always run through `formatTemplate` so `{term.x}`
 * vocabulary placeholders and ICU plurals resolve rather than leak to the UI.
 */
export function msg(key: MessageKey, params?: MessageParams): string {
  const template = getActiveMessages()[key] ?? key;
  return formatTemplate(template, params, getActiveLocale());
}

export function formatMsg(key: MessageKey, params: MessageParams): string {
  return msg(key, params);
}
