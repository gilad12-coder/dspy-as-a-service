/**
 * Single source of truth for job-status presentation and lifecycle classification.
 *
 * `STATUS_LIFECYCLE` is the exhaustive (`satisfies Record<JobStatus, …>`)
 * active/terminal classification — adding a new `JobStatus` member without
 * updating it fails at type-check, so a status cannot silently belong to
 * neither set. `ACTIVE_STATUSES` and `TERMINAL_STATUSES` are derived from it.
 *
 * `STATUS_LABELS` and `JOB_TYPE_LABELS` are frozen (`as const`) i18n lookups
 * pulling from `@/shared/lib/terms` (the Hebrew vocabulary catalogue).
 *
 * `getStatusLabel` and `getJobTypeLabel` deliberately accept `string` — see
 * their JSDoc for the fallback contract.
 */

import { TERMS } from "@/shared/lib/terms";
import type { JobStatus, OptimizationType } from "@/shared/types/api";

type StatusLifecycle = "active" | "terminal";

const STATUS_LIFECYCLE = {
  pending: "active",
  validating: "active",
  running: "active",
  success: "terminal",
  failed: "terminal",
  cancelled: "terminal",
  // Paused is lifecycle-terminal (no worker holds it) but resumable — it carries
  // a Resume affordance, unlike cancelled/failed which offer Resume-or-Restart.
  paused: "terminal",
} as const satisfies Record<JobStatus, StatusLifecycle>;

const statusesWith = (kind: StatusLifecycle): ReadonlySet<JobStatus> =>
  new Set(
    (Object.keys(STATUS_LIFECYCLE) as JobStatus[]).filter((s) => STATUS_LIFECYCLE[s] === kind),
  );

export const ACTIVE_STATUSES: ReadonlySet<JobStatus> = statusesWith("active");
export const TERMINAL_STATUSES: ReadonlySet<JobStatus> = statusesWith("terminal");

const STATUS_LABELS = {
  pending: TERMS.statusPending,
  validating: TERMS.statusValidating,
  running: TERMS.statusRunning,
  success: TERMS.statusSuccess,
  failed: TERMS.statusFailed,
  cancelled: TERMS.statusCancelled,
  paused: TERMS.statusPaused,
} as const satisfies Record<JobStatus, string>;

const JOB_TYPE_LABELS = {
  run: TERMS.optimizationTypeRun,
  grid_search: TERMS.optimizationTypeGrid,
} as const satisfies Record<OptimizationType, string>;

/**
 * Resolve a Hebrew display label for a status string.
 *
 * Accepts `string` (not just `JobStatus`) because some callers receive raw
 * values from API responses or chart-data keys (e.g. `Object.entries` on
 * `status_counts`). Unknown values fall through as-is so the UI shows the
 * source value rather than blanking out.
 */
export function getStatusLabel(status: string): string {
  return isJobStatus(status) ? STATUS_LABELS[status] : status;
}

/**
 * Resolve a Hebrew display label for a job-type string.
 *
 * Same fallback contract as `getStatusLabel` — unknown values pass through
 * unchanged.
 */
export function getJobTypeLabel(type: string): string {
  return isOptimizationType(type) ? JOB_TYPE_LABELS[type] : type;
}

export function isActiveStatus(status: string): status is JobStatus {
  return isJobStatus(status) && ACTIVE_STATUSES.has(status);
}

function isJobStatus(status: string): status is JobStatus {
  return Object.prototype.hasOwnProperty.call(STATUS_LABELS, status);
}

function isOptimizationType(type: string): type is OptimizationType {
  return Object.prototype.hasOwnProperty.call(JOB_TYPE_LABELS, type);
}
