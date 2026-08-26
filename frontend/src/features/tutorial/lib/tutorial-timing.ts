/** Keep the quick-start handoff and demo run inside one two-second budget. */
export const TUTORIAL_OPTIMIZATION_TOTAL_MS = 2_000;
export const TUTORIAL_SUBMIT_SPLASH_MS = 500;
export const TUTORIAL_DEMO_RUN_MS =
  TUTORIAL_OPTIMIZATION_TOTAL_MS - TUTORIAL_SUBMIT_SPLASH_MS;
