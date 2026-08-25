/**
 * Report a handled error to the deployment's first-party telemetry store.
 *
 * Only error type and bounded structural tags are recorded; messages and stack
 * traces can contain user content and remain in the browser console. Never
 * throws because observability must not become a second product failure.
 */

import { track, TelemetryEvent } from "@/shared/lib/telemetry";

export interface ReportErrorOptions {
  /** Low-cardinality labels for grouping/filtering (endpoint, feature, status). */
  tags?: Record<string, string | number | boolean | undefined>;
  /** Free-form structured context attached to the event; must be PII-free. */
  extra?: Record<string, unknown>;
}

export function reportHandledError(error: unknown, options: ReportErrorOptions = {}): void {
  try {
    const tags = Object.fromEntries(
      Object.entries(options.tags ?? {}).filter((entry) => entry[1] !== undefined),
    );
    track(TelemetryEvent.ClientError, {
      handled: true,
      error_type: error instanceof Error ? error.name : typeof error,
      ...tags,
    });
    console.error(error);
  } catch {
    // Reporting must never become a second failure.
  }
}
