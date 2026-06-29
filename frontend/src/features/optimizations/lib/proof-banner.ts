import type { RunBillingOutcome } from "@/shared/types/api";

/**
 * Read the typed billing outcome the worker stamps under `result.details.billing`.
 *
 * Returns the validated outcome, or `null` when the run hasn't settled or the
 * stamp is malformed — so callers can treat "no billing yet" and "no card"
 * identically.
 *
 * @param details The run result's `details` bag, or `undefined`.
 * @returns The parsed billing outcome, or `null` when absent/invalid.
 */
export function readBilling(
  details: Record<string, unknown> | undefined,
): RunBillingOutcome | null {
  const billing = details?.billing;
  if (
    billing &&
    typeof billing === "object" &&
    "outcome" in billing &&
    "credits" in billing &&
    ((billing as RunBillingOutcome).outcome === "billed" ||
      (billing as RunBillingOutcome).outcome === "refunded")
  ) {
    return billing as RunBillingOutcome;
  }
  return null;
}

/**
 * Decide which proof banner a settled run earns.
 *
 * The worker stamps `outcome === "billed"` whenever a credit was debited and the
 * guarantee did not refund — but that fires for re-runs, runs with no comparable
 * scores, and any regression on an already-spent guarantee slot, none of which
 * beat the baseline. So the lift-claiming variant is honest only when
 * `improvement > 0`; a billed run without measured lift falls back to neutral
 * receipt copy rather than asserting a beat that did not happen. Returns `null`
 * when no banner should render.
 *
 * @param billing The stamped billing outcome, or `null` if the run has not settled.
 * @param improvement Test-split lift in percentage points, or `undefined` if unscored.
 * @returns The banner variant to render, or `null` for no banner.
 */
export function proofBannerVariant(
  billing: RunBillingOutcome | null,
  improvement: number | undefined,
): "refunded" | "billed-lift" | "billed-neutral" | null {
  if (billing == null) return null;
  if (billing.outcome === "refunded") return "refunded";
  return improvement != null && improvement > 0 ? "billed-lift" : "billed-neutral";
}
