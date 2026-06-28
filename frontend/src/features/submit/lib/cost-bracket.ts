/**
 * Projected credit bracket for a pre-run estimate.
 *
 * A DSPy/GEPA job's token use is not linear: bootstrapping, compile steps,
 * dataset size, and validation loops all swing the total, so a single tight
 * number would imply a precision the optimizer can't honour. Instead we project
 * a *bracket* — a low/high credit range — and let the user cap the run with a
 * Max Cost Ceiling. The numbers here are deliberately coarse and operator-tunable;
 * they exist to set expectations and seed a sensible default cap, never to promise
 * an exact charge.
 *
 * Anchored to GEPA's metric-call budgets (light/medium/heavy → 500/2000/8000
 * calls, mirroring the backend `_AUTO_BUDGETS`) times a per-call token estimate,
 * converted to credits at the platform rate. Pure and side-effect-free so it is
 * trivially testable and safe to import anywhere.
 */

import { TOKENS_PER_CREDIT } from "@/features/billing";

/** GEPA metric-call budgets by `auto` tier — mirrors the backend `_AUTO_BUDGETS`. */
const AUTO_METRIC_CALLS: Record<string, number> = {
  light: 500,
  medium: 2000,
  heavy: 8000,
};

/** Fallback metric-call budget when no `auto` tier and no explicit eval count is set. */
const DEFAULT_METRIC_CALLS = 2000;

/**
 * Per-metric-call token estimate, low and high ends of the bracket. The spread
 * (≈6×) reflects how much one rollout's tokens vary with prompt/output length and
 * whether a reflection model is in the loop — the honest uncertainty the bracket
 * is meant to convey rather than hide.
 */
const TOKENS_PER_CALL_LOW = 700;
const TOKENS_PER_CALL_HIGH = 4500;

/** A reflection model adds a second LM pass per proposal — widen the high end. */
const REFLECTION_HIGH_MULTIPLIER = 1.5;

export interface CostBracketInput {
  /** GEPA `auto` tier ("light"/"medium"/"heavy"), or "" when an explicit eval count is used. */
  autoLevel: string;
  /** Explicit `max_full_evals` as a string from the wizard (used when `autoLevel` is empty). */
  maxFullEvals: string;
  /** Dataset row count — scales the per-eval work; 0 when no dataset is loaded yet. */
  datasetRows: number;
  /** Whether a reflection model is configured (adds a second LM pass per proposal). */
  hasReflection: boolean;
  /** For a grid search, the number of (gen × refl) model pairs swept; 1 for a single run. */
  pairs?: number;
}

export interface CostBracket {
  /** Low end of the projected credit range. */
  lowCredits: number;
  /** High end of the projected credit range. */
  highCredits: number;
}

/** Resolve the metric-call budget the bracket scales from. */
function resolveMetricCalls(autoLevel: string, maxFullEvals: string): number {
  const tier = autoLevel ? AUTO_METRIC_CALLS[autoLevel] : undefined;
  if (tier !== undefined) return tier;
  const evals = parseInt(maxFullEvals, 10);
  // max_full_evals counts full valset passes; a pass is on the order of a few
  // hundred calls, so scale it into the same metric-call space as the auto tiers.
  if (Number.isFinite(evals) && evals > 0) return evals * 250;
  return DEFAULT_METRIC_CALLS;
}

/**
 * Project a low/high credit bracket for a run from its GEPA budget and dataset.
 *
 * Returns rounded credit bounds; the high end seeds the default Max Cost Ceiling.
 * Always returns `highCredits >= lowCredits >= 1` so the bracket reads sensibly
 * even before a dataset is loaded.
 */
export function projectCostBracket(input: CostBracketInput): CostBracket {
  const { autoLevel, maxFullEvals, datasetRows, hasReflection, pairs = 1 } = input;
  const calls = resolveMetricCalls(autoLevel, maxFullEvals);
  // Larger datasets mean longer prompts and more baseline/eval rollouts; fold in
  // a gentle, sub-linear factor so a big dataset widens the bracket without
  // exploding it. Clamp the row factor so an empty/tiny dataset still projects.
  const rowFactor = 1 + Math.min(datasetRows, 2000) / 2000;
  const sweep = Math.max(pairs, 1);

  const lowTokens = calls * TOKENS_PER_CALL_LOW * rowFactor * sweep;
  const highTokens =
    calls *
    TOKENS_PER_CALL_HIGH *
    rowFactor *
    sweep *
    (hasReflection ? REFLECTION_HIGH_MULTIPLIER : 1);

  const lowCredits = Math.max(1, Math.round(lowTokens / TOKENS_PER_CREDIT));
  const highCredits = Math.max(lowCredits, Math.round(highTokens / TOKENS_PER_CREDIT));
  return { lowCredits, highCredits };
}

/**
 * Default Max Cost Ceiling for a freshly-projected bracket: the high end with a
 * little headroom, rounded to a friendly figure. Headroom keeps the cap from
 * tripping on an ordinary run that lands near the top of the bracket while still
 * bounding a runaway.
 */
export function defaultCeilingForBracket(bracket: CostBracket): number {
  const withHeadroom = Math.ceil(bracket.highCredits * 1.15);
  return roundToNiceCap(withHeadroom);
}

/** Round a credit cap up to a readable step (10s under 500, 50s above). */
export function roundToNiceCap(credits: number): number {
  if (credits <= 0) return 1;
  const step = credits < 500 ? 10 : 50;
  return Math.ceil(credits / step) * step;
}
