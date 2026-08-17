/**
 * Per-model credit pricing — the frontend mirror of backend `core.billing.pricing`.
 *
 * A run's credit cost is the real provider cost of its tokens (per-model
 * input/output rates from the catalog) times the platform `MARKUP`, converted to
 * credits at `CREDIT_USD_VALUE`. The same function prices a *projected* token
 * volume here (the pre-run estimate) that the backend prices on *measured*
 * tokens (the charge), so the estimate and the bill reconcile by construction.
 *
 * Keep `MARKUP` and the default rates in step with the backend module — they are
 * the margin lever and must not drift between the estimate and the charge.
 */

import type { CatalogModel } from "@/shared/types/api";
import { CREDIT_USD_VALUE } from "./credit";

/** Margin multiplier on raw provider cost — mirrors backend `pricing.MARKUP`.
 * 1.50 = 1.09 × 1.10 × 1.25: ~9% covers the payment-processing fees (OpenRouter
 * deposit + Stripe), ~10% the CPU/storage a run consumes, and 25% is a profit
 * margin that offsets the fixed hosting bill (1.20 was break-even). */
export const MARKUP = 1.5;

/** Fallback per-token costs (USD) for a model the catalog doesn't price. */
export const DEFAULT_INPUT_COST_PER_TOKEN = 1e-6;
export const DEFAULT_OUTPUT_COST_PER_TOKEN = 3e-6;

/** Share of a run's full credit cost charged as the platform fee on a BYOK run —
 * mirrors the backend. The infra (CPU/storage) + margin share of the markup,
 * grossed up for Stripe's cut: the tokens are on the user's own key, but the
 * compute isn't. */
export const PLATFORM_FEE_FRACTION = 0.28;

/** Projected or measured token usage attributed to one model. */
export interface ModelTokenUsage {
  /** The model the tokens ran on; `null`/`undefined` prices at the defaults. */
  model?: CatalogModel | null;
  inputTokens: number;
  outputTokens: number;
}

/**
 * A model's `(input, output)` per-token cost in USD, falling back to the module
 * defaults when the catalog leaves a rate unpriced (so a model is never free).
 */
export function modelTokenCosts(model?: CatalogModel | null): { input: number; output: number } {
  const input = model?.input_cost_per_token;
  const output = model?.output_cost_per_token;
  return {
    input: typeof input === "number" && input > 0 ? input : DEFAULT_INPUT_COST_PER_TOKEN,
    output: typeof output === "number" && output > 0 ? output : DEFAULT_OUTPUT_COST_PER_TOKEN,
  };
}

/** Raw provider cost (USD, pre-markup) of per-model token usage. */
export function rawCostUsd(usages: ModelTokenUsage[]): number {
  let total = 0;
  for (const usage of usages) {
    const { input, output } = modelTokenCosts(usage.model);
    total += usage.inputTokens * input + usage.outputTokens * output;
  }
  return total;
}

/**
 * Convert per-model token usage to the credits it costs, rounding up. Mirrors
 * backend `credits_for_usage`: any non-zero usage costs at least one credit.
 */
export function creditsForUsage(usages: ModelTokenUsage[]): number {
  const cost = rawCostUsd(usages) * MARKUP;
  if (cost <= 0) return 0;
  return Math.max(1, Math.ceil(cost / CREDIT_USD_VALUE));
}

/**
 * The BYOK platform fee for a run's full credit cost, rounding up — mirrors
 * backend `platform_fee_credits_for_usage`. On a BYOK run the provider tokens are
 * paid on the user's own key, so only this fraction is charged in credits.
 */
export function platformFeeCredits(fullCredits: number): number {
  const fee = fullCredits * PLATFORM_FEE_FRACTION;
  if (fee <= 0) return 0;
  return Math.max(1, Math.ceil(fee));
}
