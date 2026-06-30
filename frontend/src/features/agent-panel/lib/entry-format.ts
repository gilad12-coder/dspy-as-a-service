/**
 * Pure formatting helpers for the generalist agent's argument /
 * result rows. Hebrew labels for known tool keys, scalar / array /
 * object value formatting, UUID detection, and plain-object guards.
 *
 * Lives in ``lib/`` because there is no JSX here — the matching
 * ``components/EntryRow.tsx`` consumes these helpers to render rows.
 */

import { formatMsg, msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";

export const ARG_LABELS: Record<string, string> = {
  optimization_id: msg("auto.features.agent.panel.lib.entry.row.literal.1"),
  job_id: msg("auto.features.agent.panel.lib.entry.row.literal.3"),
  ids: msg("auto.features.agent.panel.lib.entry.row.literal.4"),
  optimization_ids: msg("auto.features.agent.panel.lib.entry.row.literal.5"),
  name: msg("auto.features.agent.panel.lib.entry.row.literal.6"),
  new_name: msg("auto.features.agent.panel.lib.entry.row.literal.7"),
  pinned: msg("auto.features.agent.panel.lib.entry.row.literal.8"),
  archived: msg("auto.features.agent.panel.lib.entry.row.literal.9"),
  model: TERMS.model,
  optimizer: TERMS.optimizer,
  signature: TERMS.signature,
  metric: TERMS.metric,
  code: msg("auto.features.agent.panel.lib.entry.row.literal.10"),
  signature_code: formatMsg("auto.features.agent.panel.lib.entry.row.template.1", {
    p1: TERMS.signature,
  }),
  metric_code: formatMsg("auto.features.agent.panel.lib.entry.row.template.2", {
    p1: TERMS.metric,
  }),
  target: msg("auto.features.agent.panel.lib.entry.row.literal.11"),
  kind: msg("auto.features.agent.panel.lib.entry.row.literal.12"),
  count: msg("auto.features.agent.panel.lib.entry.row.literal.13"),
  name_prefix: msg("auto.features.agent.panel.lib.entry.row.literal.14"),
  value: msg("auto.features.agent.panel.lib.entry.row.literal.15"),
  sample_id: msg("auto.features.agent.panel.lib.entry.row.literal.16"),
  dataset_columns: msg("auto.features.agent.panel.lib.entry.row.literal.17"),
  dataset: TERMS.dataset,
  column_roles: msg("auto.features.agent.panel.lib.entry.row.literal.18"),
  roles: msg("auto.features.agent.panel.lib.entry.row.literal.19"),
  job_name: msg("auto.features.agent.panel.lib.entry.row.literal.20"),
  description: msg("auto.features.agent.panel.lib.entry.row.literal.21"),
  config: msg("auto.features.agent.panel.lib.entry.row.literal.22"),
  model_config: TERMS.modelConfig,
  model_config_override: TERMS.modelConfig,
  optimizer_kwargs: formatMsg("auto.features.agent.panel.lib.entry.row.template.3", {
    p1: TERMS.optimizer,
  }),
  reflection_model_config: TERMS.reflectionModel,
  grid: msg("auto.features.agent.panel.lib.entry.row.literal.23"),
  grid_params: msg("auto.features.agent.panel.lib.entry.row.literal.24"),
  username: msg("auto.features.agent.panel.lib.entry.row.literal.25"),
  status: msg("auto.features.agent.panel.lib.entry.row.literal.26"),
  detail: msg("auto.features.agent.panel.lib.entry.row.literal.27"),
  result: msg("auto.features.agent.panel.lib.entry.row.literal.28"),
  id: msg("auto.features.agent.panel.lib.entry.row.literal.29"),
  message: msg("auto.features.agent.panel.lib.entry.row.literal.30"),
  goal: msg("auto.features.agent.panel.lib.entry.row.literal.31"),
  current_signature: formatMsg("auto.features.agent.panel.lib.entry.row.template.4", {
    p1: TERMS.signature,
  }),
  current_metric: formatMsg("auto.features.agent.panel.lib.entry.row.template.5", {
    p1: TERMS.metric,
  }),
  sample_rows: msg("auto.features.agent.panel.lib.entry.row.literal.32"),
  sample_row: msg("auto.features.agent.panel.lib.entry.row.literal.33"),
  assistant_message: msg("auto.features.agent.panel.lib.entry.row.literal.34"),
  column_mapping: msg("auto.features.agent.panel.lib.entry.row.literal.35"),
  optimizer_name: formatMsg("auto.features.agent.panel.lib.entry.row.template.6", {
    p1: TERMS.optimizer,
  }),
  valid: msg("auto.features.agent.panel.lib.entry.row.literal.36"),
  errors: msg("auto.features.agent.panel.lib.entry.row.literal.37"),
  warnings: msg("auto.features.agent.panel.lib.entry.row.literal.38"),
  error: msg("auto.features.agent.panel.lib.entry.row.literal.39"),
  warning: msg("auto.features.agent.panel.lib.entry.row.literal.40"),
  signature_fields: formatMsg("auto.features.agent.panel.lib.entry.row.template.7", {
    p1: TERMS.signature,
  }),
  inputs: msg("auto.features.agent.panel.lib.entry.row.literal.41"),
  outputs: msg("auto.features.agent.panel.lib.entry.row.literal.42"),
  score: TERMS.score,
  feedback: msg("auto.features.agent.panel.lib.entry.row.literal.43"),
  baseline: msg("auto.features.agent.panel.lib.entry.row.literal.44"),
  optimized: formatMsg("auto.features.agent.panel.lib.entry.row.template.8", {
    p1: TERMS.optimization,
  }),
  task_type: msg("auto.features.agent.panel.lib.entry.row.literal.45"),
  rationale: msg("auto.features.agent.panel.lib.entry.row.literal.46"),
  summary: msg("auto.features.agent.panel.lib.entry.row.literal.47"),
  explanation: msg("auto.features.agent.panel.lib.entry.row.literal.48"),
  reason: msg("auto.features.agent.panel.lib.entry.row.literal.49"),
};

export function hasHebrewLabel(key: string): boolean {
  return Object.prototype.hasOwnProperty.call(ARG_LABELS, key);
}

export const CODE_KEYS = new Set([
  "code",
  "signature_code",
  "metric_code",
  "current_signature",
  "current_metric",
]);

export const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function shortenId(s: string): string {
  return UUID_RE.test(s) ? s.slice(0, 8) : s;
}

export function formatValue(v: unknown): string {
  if (typeof v === "string") {
    if (v.length > 240) return `${v.slice(0, 237)}…`;
    return v;
  }
  if (typeof v === "boolean")
    return v
      ? msg("auto.features.agent.panel.lib.entry.row.literal.50")
      : msg("auto.features.agent.panel.lib.entry.row.literal.51");
  if (typeof v === "number") return String(v);
  if (v == null) return "—";
  if (Array.isArray(v)) {
    if (v.length === 0) return "—";
    return v.map((x) => (typeof x === "string" ? shortenId(x) : formatValue(x))).join(", ");
  }
  try {
    const json = JSON.stringify(v);
    return json.length > 240 ? `${json.slice(0, 237)}…` : json;
  } catch {
    return String(v);
  }
}

export function isPlainObject(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === "object" && !Array.isArray(v);
}
