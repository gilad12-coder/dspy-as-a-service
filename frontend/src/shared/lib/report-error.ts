/**
 * Report a handled (caught) error to Sentry without surfacing it to the user.
 *
 * `try/catch` blocks that show a toast or a fallback UI swallow the failure
 * from Sentry's point of view — only uncaught exceptions reach it by default.
 * Route those through here so real breakage (a 5xx from the API, a network
 * drop, a parse failure) is still counted, tagged `handled:true` so it can be
 * filtered apart from crashes. Never throws; a Sentry failure is not an error.
 */

import * as Sentry from "@sentry/nextjs";

export interface ReportErrorOptions {
  /** Low-cardinality labels for grouping/filtering (endpoint, feature, status). */
  tags?: Record<string, string | number | boolean | undefined>;
  /** Free-form structured context attached to the event; must be PII-free. */
  extra?: Record<string, unknown>;
}

export function reportHandledError(error: unknown, options: ReportErrorOptions = {}): void {
  try {
    Sentry.captureException(error, {
      level: "error",
      tags: { handled: true, ...options.tags },
      extra: options.extra,
    });
  } catch {
    // Reporting must never become a second failure.
  }
}
