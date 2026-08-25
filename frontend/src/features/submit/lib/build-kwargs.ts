/**
 * Pure builders for optimizer_kwargs.
 * Kept side-effect-free so they're trivially testable.
 */

export interface OptimizerKwargsInput {
  autoLevel: string;
  maxFullEvals: string;
  maxMetricCalls?: string;
  reflectionMinibatchSize: string;
  useMerge: boolean;
  pxnParents?: string;
  pxnProposals?: string;
}

export function buildOptimizerKwargs(input: OptimizerKwargsInput): Record<string, unknown> {
  const {
    autoLevel,
    maxFullEvals,
    maxMetricCalls,
    reflectionMinibatchSize,
    useMerge,
    pxnParents,
    pxnProposals,
  } = input;
  const kw: Record<string, unknown> = {};
  // GEPA requires exactly one of: auto, max_full_evals, max_metric_calls.
  // An explicit metric-call budget outranks max_full_evals because the latter
  // always holds its "6" default — the budget field is opt-in.
  if (autoLevel) {
    kw.auto = autoLevel;
  } else if (maxMetricCalls) {
    kw.max_metric_calls = parseInt(maxMetricCalls, 10);
  } else if (maxFullEvals) {
    kw.max_full_evals = parseInt(maxFullEvals, 10);
  }
  if (reflectionMinibatchSize) kw.reflection_minibatch_size = parseInt(reflectionMinibatchSize, 10);
  kw.use_merge = useMerge;
  // PxN batched sampling: the server turns these two integers into
  // PxNSampling(p, n) — the strategy object itself can't cross JSON. Send them
  // only when they deviate from 1x1, so an untouched form keeps inheriting the
  // server-wide GEPA_PXN_* defaults instead of pinning them to 1.
  const p = pxnParents ? parseInt(pxnParents, 10) : 1;
  const n = pxnProposals ? parseInt(pxnProposals, 10) : 1;
  if (Number.isFinite(p) && Number.isFinite(n) && (p > 1 || n > 1)) {
    kw.pxn_parents = p;
    kw.pxn_proposals = n;
  }
  return Object.keys(kw).length > 0 ? kw : {};
}
