import type { WorkflowSpec } from "@/shared/types/api";

/** Trust mode — how much the agent must ask before acting. */
export type TrustMode = "ask" | "auto_safe" | "yolo";

/** One prior turn sent with every generalist SSE request. */
export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

/**
 * Snapshot of the submit wizard exposed to the generalist agent.
 *
 * The backend's phased-exposure gate reads five boolean-ish keys
 * (dataset_ready, columns_configured, signature_code, metric_code,
 * model_configured) to decide which MCP tools are callable on each turn.
 * The remaining fields are diagnostic for the agent's context and form
 * the write surface for ``update_wizard_state`` — every editable wizard
 * field lives here so agent patches flow 1:1 into local wizard state.
 */
export interface WizardState {
  dataset_ready?: boolean;
  columns_configured?: boolean;
  signature_code?: string;
  metric_code?: string;
  model_configured?: boolean;
  staged_dataset_id?: string;
  job_name?: string;
  job_description?: string;
  job_type?: "run" | "grid_search";
  optimizer_name?: string;
  module_name?: string;
  // Authored graph for a ``workflow`` (multi-module) run — carried in place of
  // signature_code and injected into the run request on submit.
  workflow?: WorkflowSpec;
  // React (ReAct-agent) run config — the live tool source. Excludes the secret
  // mcp_auth_header, which is never shared.
  react_config?: Record<string, unknown>;
  dataset_columns?: string[];
  column_roles?: Record<string, string>;
  model_config?: Record<string, unknown>;
  reflection_model_config?: Record<string, unknown>;
  generation_models?: Array<Record<string, unknown>>;
  reflection_models?: Array<Record<string, unknown>>;
  use_all_generation_models?: boolean;
  use_all_reflection_models?: boolean;
  split_fractions?: { train: number; val: number; test: number };
  split_mode?: "auto" | "manual";
  seed?: number;
  shuffle?: boolean;
  is_private?: boolean;
  optimizer_kwargs?: Record<string, unknown>;
}

export interface ToolStartPayload {
  id: string;
  tool: string;
  reason: string;
  arguments: Record<string, unknown>;
}

export interface ToolEndPayload {
  id: string;
  tool: string;
  status: string;
  result?: unknown;
}

export interface PendingApprovalPayload {
  id: string;
  tool: string;
  arguments: Record<string, unknown>;
}

export interface ApprovalResolvedPayload {
  id: string;
  tool: string;
  approved: boolean;
}
