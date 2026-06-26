/**
 * Server-only loader that merges a request's locale fallback chain into one flat
 * UI catalog.
 *
 * SERVER-ONLY BY CONTRACT: this module value-imports the full per-locale catalog
 * registry (`generated/ui-catalog`), so importing it from a client component
 * would pull every locale's strings into the browser bundle — defeating the
 * whole point of lazy per-locale delivery. Only the root layout imports it, to
 * build the active catalog it then injects (window shim) and pins (server slot).
 * The boundary is verified by a post-build bundle check: no Hebrew catalog
 * string may appear in any client chunk.
 */

import { UI_CATALOGS } from "@/shared/lib/generated/ui-catalog";
import { fallbackChain, type Locale } from "@/shared/lib/locale";
import type { UiCatalog } from "@/shared/lib/runtime-messages";

/**
 * Merge the active locale's fallback chain into a single complete catalog.
 *
 * The chain runs most-specific → base (e.g. `pt-BR → pt → en → he`); merging
 * base-first lets a more specific overlay override, while the complete Hebrew
 * base guarantees every key is present so `msg()` never falls through to a raw
 * key. Locales in the chain without their own catalog file are skipped.
 *
 * Args:
 *   locale: The request's resolved locale.
 *
 * Returns:
 *   A flat `{key: value}` map covering every `MessageKey` for this locale.
 */
export function buildActiveCatalog(locale: Locale): UiCatalog {
  const chain = fallbackChain(locale);
  const merged: UiCatalog = {};
  for (let i = chain.length - 1; i >= 0; i--) {
    const tag = chain[i];
    const catalog = tag ? UI_CATALOGS[tag] : undefined;
    if (catalog) Object.assign(merged, catalog);
  }
  return merged;
}
