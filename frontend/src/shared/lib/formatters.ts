import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveIntlLocale } from "@/shared/lib/runtime-locale";

/**
 * Format ISO date string in the active locale
 * @example "2026-04-11T12:30:00" → "11/04/2026, 12:30:00" (he) or "4/11/2026, 12:30:00 PM" (en)
 */
export function formatDate(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString(getActiveIntlLocale());
}

/**
 * Format ISO date as relative time in Hebrew
 * @example localized relative time for minutes, hours, or days ago
 */
export function formatRelativeTime(iso: string): string {
  try {
    const diff = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return msg("auto.shared.lib.formatters.literal.1");
    if (mins < 60) return formatMsg("auto.shared.lib.formatters.template.1", { p1: mins });
    const hours = Math.floor(mins / 60);
    if (hours < 24) return formatMsg("auto.shared.lib.formatters.template.2", { p1: hours });
    const days = Math.floor(hours / 24);
    if (days < 7) return formatMsg("auto.shared.lib.formatters.template.3", { p1: days });
    return formatDate(iso);
  } catch {
    return formatDate(iso);
  }
}

const pad2 = (n: number) => String(n).padStart(2, "0");

/**
 * Parse a backend timestamp into a Date in the viewer's local zone.
 *
 * The backend persists UTC. When the ISO string carries no timezone designator
 * we append "Z" so it is read as UTC instead of local — otherwise the displayed
 * wall-clock silently drifts by the viewer's offset (e.g. a UTC "23:29" rendered
 * verbatim instead of the local "02:29").
 */
function parseBackendTimestamp(ts: string): Date {
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(ts);
  return new Date(hasZone ? ts : `${ts}Z`);
}

/**
 * Format log timestamp for display in the viewer's local time
 * @example "2026-03-30T23:32:45.971393+00:00" → "31/03 02:32:45" (UTC+3 viewer)
 */
export function formatLogTimestamp(ts: string): string {
  const d = parseBackendTimestamp(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return `${pad2(d.getDate())}/${pad2(d.getMonth() + 1)} ${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
}

/**
 * Group log timestamps by minute for filtering, in the viewer's local time
 * @example "2026-03-30T23:32:45+00:00" → "31/03 02:32" (UTC+3 viewer)
 */
export function logTimeBucket(ts: string): string {
  const d = parseBackendTimestamp(ts);
  if (Number.isNaN(d.getTime())) return ts;
  return `${pad2(d.getDate())}/${pad2(d.getMonth() + 1)} ${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
}

/**
 * Format duration in seconds as HH:MM:SS or MM:SS
 * @example 3665 → "1:01:05", 125 → "2:05"
 */
export function formatDuration(seconds: number | undefined | null): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return "—";
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  const pad = (n: number) => String(n).padStart(2, "0");
  if (hrs > 0) return `${hrs}:${pad(mins)}:${pad(secs)}`;
  return `${mins}:${pad(secs)}`;
}

export const formatElapsed = formatDuration;

/**
 * Format an aggregate metric on the canonical 0–100 percentage scale.
 * @example 85.6 → "85.6%", 60.3 → "60.3%"
 */
export function formatPercent(n: number | undefined | null): string {
  if (n == null) return "—";
  return `${n.toFixed(1)}%`;
}

/**
 * Format a 0–100-scale improvement delta with an explicit sign. A 0.3-point
 * gain renders "+0.3%", not "+30%": the value is already in percentage points,
 * so it is never rescaled.
 * @example 15 → "+15.0%", -5 → "-5.0%"
 */
export function formatImprovement(v: number | undefined | null): string {
  if (v == null) return "—";
  return v >= 0 ? `+${v.toFixed(1)}%` : `${v.toFixed(1)}%`;
}

/**
 * Format a byte count as a compact human-readable size.
 * @example 0 → "0 B", 1536 → "1.5 KB", 5_242_880 → "5 MB"
 */
export function formatBytes(bytes: number | undefined | null): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const exp = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exp;
  const rounded = value >= 100 || exp === 0 ? Math.round(value) : Math.round(value * 10) / 10;
  return `${rounded} ${units[exp]}`;
}

/**
 * Format a byte count for the unified storage budget, never escalating past MB.
 * Budgets are small (hundreds of MB), so GB reads as noise; sub-MB values still
 * fall back to KB/B. Whole MB render with up to 4 digits (e.g. 9999 MB).
 * @example 262_144_000 → "250 MB", 5_242_880 → "5 MB", 524_288 → "512 KB"
 */
export function formatStorageSize(bytes: number | undefined | null): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB"];
  const exp = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / 1024 ** exp;
  const rounded = value >= 100 || exp === 0 ? Math.round(value) : Math.round(value * 10) / 10;
  return `${rounded} ${units[exp]}`;
}

export function formatOutput(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string") return v;
  if (typeof v === "number" || typeof v === "boolean") return String(v);
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

export function formatId(id: string): string {
  return id;
}

/**
 * Friendly label for a DSPy module name. The backend may store either the
 * short alias ("cot") or the resolved dotted path ("dspy.ChainOfThought"), so
 * match on keywords rather than exact strings.
 * @example "dspy.ChainOfThought" → "CoT", "react" → "ReAct"
 */
export function moduleLabel(raw: string | null | undefined): string {
  if (!raw) return "—";
  const v = raw.toLowerCase();
  if (v === "workflow") return "Workflow";
  if (v.includes("react")) return "ReAct";
  if (v.includes("chain") || v === "cot") return "CoT";
  if (v.includes("predict")) return "Predict";
  return raw;
}

/**
 * Bare model name of a provider-routed model id, for display. Ids arrive as
 * gateway paths ("openrouter/openai/gpt-5.4-mini"); everything before the last
 * slash is routing, not identity, so UI text shows only the final segment.
 * Keep the full id in a tooltip/title when two providers could collide, and
 * never trim ids used as filter values, API payloads, or exports.
 * @example "openrouter/openai/gpt-5.4-mini" → "gpt-5.4-mini", "auto:intelligent" → "auto:intelligent"
 */
export function modelDisplayName(id: string | null | undefined): string {
  if (!id) return "";
  const tail = id.split("/").pop();
  return tail || id;
}


/**
 * Sentence-case a lowercase glossary term for standalone UI positions (nav
 * items, wizard step labels) where surrounding labels are capitalized.
 * Caseless scripts (Hebrew) pass through unchanged.
 * @example "new optimization" → "New optimization"
 */
export function sentenceCase(term: string): string {
  return term.charAt(0).toLocaleUpperCase(getActiveIntlLocale()) + term.slice(1);
}
