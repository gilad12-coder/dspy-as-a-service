/**
 * Centralized tooltip copy.
 *
 * Tooltips that describe the same concept should share a single string.
 * Call sites import `tip(key)` and pass the result to <HelpTip text={...}>.
 *
 *     import { tip } from "@/shared/lib/tooltips";
 *     <HelpTip text={tip("score.baseline")}>{TERMS.baselineScore}</HelpTip>
 *
 * Keys are grouped by domain concept, not by feature slice — the same
 * definition of "baseline score" should read identically on the overview
 * tab and the pair detail view.
 *
 * The copy itself lives in the UI catalog (`i18n/locales/ui/<locale>.json`)
 * under the `tooltip.` key prefix, so tooltips get the same per-locale
 * translation, `{term.x}` vocabulary interpolation, and locale fallback
 * chain as every other UI string — and only the active locale's copy ships
 * to the client. Adding a tooltip is: add `tooltip.<key>` to the Hebrew and
 * English catalogs, regenerate, and call `tip("<key>")`.
 */

import type { MessageKey } from "@/shared/lib/generated/ui-catalog";
import { msg } from "@/shared/lib/messages";

type TooltipMessageKey = Extract<MessageKey, `tooltip.${string}`>;

/** Unprefixed tooltip ids — `tip("score.baseline")` reads `tooltip.score.baseline`. */
export type TooltipKey = TooltipMessageKey extends `tooltip.${infer K}` ? K : never;

/**
 * Look up tooltip copy by key in the active locale.
 *
 * Resolves `tooltip.<key>` through the merged UI catalog, so translation,
 * vocabulary interpolation, and locale fallback behave exactly like `msg()`.
 */
export function tip(key: TooltipKey): string {
  return msg(`tooltip.${key}` as TooltipMessageKey);
}
