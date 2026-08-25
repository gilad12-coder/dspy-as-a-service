import type { RunBillingOutcome } from "@/shared/types/api";

/**
 * Read the typed billing stamp the worker writes under `result.details.billing`.
 *
 * Returns the validated stamp, or `null` when the run hasn't settled or the
 * stamp is malformed — so callers can treat "no billing yet" and "no chip"
 * identically. Only `outcome === "billed"` is accepted: every run bills, and a
 * legacy "refunded" stamp (from the retired no-lift guarantee) is ignored
 * rather than rendered as a charge it never was.
 *
 * @param details The run result's `details` bag, or `undefined`.
 * @returns The parsed billing stamp, or `null` when absent/invalid.
 */
export function readBilling(
  details: Record<string, unknown> | undefined,
): RunBillingOutcome | null {
  const billing = details?.billing;
  if (
    billing &&
    typeof billing === "object" &&
    "credits" in billing &&
    (billing as RunBillingOutcome).outcome === "billed"
  ) {
    return billing as RunBillingOutcome;
  }
  return null;
}
