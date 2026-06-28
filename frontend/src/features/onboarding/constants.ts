/**
 * Pre-baked onboarding demo — a real-shaped before/after a new user can feel in
 * seconds, before uploading anything. The numbers are illustrative of a typical
 * managed run (baseline vs optimized on a held-out test split); they are not a
 * promise, which is why the surrounding copy frames it as a sample task.
 */
export interface OnboardingDemo {
  /** Baseline score on the held-out test split, 0–100. */
  baseline: number;
  /** Optimized score on the same held-out test split, 0–100. */
  optimized: number;
}

export const ONBOARDING_DEMO: OnboardingDemo = {
  baseline: 71.4,
  optimized: 88.2,
};

/** Smallest dataset we'll carry into the wizard from the onboarding upload. */
export const MIN_ONBOARDING_ROWS = 2;
