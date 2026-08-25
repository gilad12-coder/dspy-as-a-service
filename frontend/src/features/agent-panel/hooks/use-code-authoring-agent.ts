"use client";

import * as React from "react";

import { validateCode } from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { useCodeAgent, type CodeAgentState } from "@/shared/hooks/use-code-agent";
import { autoLayoutSpec, defaultWorkflowSpec } from "@/features/submit/workflow/model";
import type { ParsedDataset } from "@/shared/lib/parse-dataset";
import type { ValidationResult } from "@/shared/ui/code-editor";
import type { ColumnMapping, ValidateCodeResponse, WorkflowSpec } from "@/shared/types/api";

import type { ConfirmedDataset } from "../components/DatasetUploadCard";

/**
 * The code agent's state plus the artifact fields the panel mirror renders and
 * the auto-handoff gates on. ``useCodeAgent`` keeps the code + validation in the
 * caller's own state, so this wrapper surfaces them alongside the agent state.
 */
export interface CodeAuthoringAgentState extends CodeAgentState {
  signatureCode: string;
  metricCode: string;
  signatureValidation: ValidateCodeResponse | null;
  metricValidation: ValidateCodeResponse | null;
  /** True when the run targets the multi-module ``workflow`` module. */
  isWorkflow: boolean;
  /** The authored graph the inline canvas renders; null until seeded. */
  workflowSpec: WorkflowSpec | null;
  /** Bumps on external graph replacements so the canvas remounts cleanly. */
  workflowRevision: number;
  /** Node the agent just changed — the canvas pulses it briefly. */
  agentPulseNodeId: string | null;
  /** Commit a user edit from the canvas back onto the graph. */
  updateWorkflowSpec: (spec: WorkflowSpec) => void;
}

export interface UseCodeAuthoringAgentArgs {
  /** Dataset the user confirmed in-panel; supplies columns, rows, and roles. */
  dataset: ConfirmedDataset | null;
  /**
   * Whether the generalist has actually called ``request_code_authoring``.
   * Gates the code agent's own auto-seed effect so it fires on the tool call,
   * not the moment a dataset is attached.
   */
  armed: boolean;
  /** DSPy module the run targets; only steers the seed's expected shape. */
  moduleName?: string;
  /** Optimizer name, forwarded to validation for optimizer-specific checks. */
  optimizerName?: string;
  /**
   * Catalog model id + effort the code author runs on. The panel forwards the
   * conversation's chosen composer model so code authoring follows it instead
   * of always taking the server default (which auto-routes and can rate-limit).
   */
  model?: string | null;
  reasoningEffort?: string | null;
}

const NOOP = () => {};

/** Build the validator's column mapping from the panel's role assignments. */
function rolesToColumnMapping(
  columnRoles: Record<string, "input" | "output" | "ignore">,
): ColumnMapping {
  const inputs: Record<string, string> = {};
  const outputs: Record<string, string> = {};
  for (const [col, role] of Object.entries(columnRoles)) {
    if (role === "input") inputs[col] = col;
    else if (role === "output") outputs[col] = col;
  }
  return { inputs, outputs };
}

/**
 * Host the canonical wizard code agent (``useCodeAgent``) at the generalist
 * panel level so the panel's authoring card mirrors exactly what the code agent
 * does — same streaming, same thinking timer, same validation + auto-fix.
 *
 * This is glue around the shared hook, not a second engine: it owns the
 * code/validation state the hook writes into, adapts the panel's
 * ``ConfirmedDataset`` to the wizard's ``ParsedDataset``, and runs the same
 * ``validateCode`` checks the wizard runs.
 *
 * Args:
 *   args: Dataset, the ``armed`` gate, and optional module/optimizer context.
 *
 * Returns:
 *   The code agent state plus the surfaced signature/metric code + validation.
 */
export function useCodeAuthoringAgent(
  args: UseCodeAuthoringAgentArgs,
): CodeAuthoringAgentState {
  const { dataset, armed, moduleName = "predict", optimizerName, model, reasoningEffort } = args;

  const [signatureCode, setSignatureCode] = React.useState("");
  const [metricCode, setMetricCode] = React.useState("");
  const [signatureManuallyEdited, setSignatureManuallyEdited] = React.useState(false);
  const [metricManuallyEdited, setMetricManuallyEdited] = React.useState(false);
  const [signatureValidation, setSignatureValidation] =
    React.useState<ValidateCodeResponse | null>(null);
  const [metricValidation, setMetricValidation] =
    React.useState<ValidateCodeResponse | null>(null);

  const columnRoles = React.useMemo(() => dataset?.columnRoles ?? {}, [dataset]);
  const columnKinds = React.useMemo(() => dataset?.columnKinds ?? {}, [dataset]);

  const isWorkflow = (moduleName ?? "predict").toLowerCase() === "workflow";

  // Graph state mirrors the submit wizard's canvas ownership: the spec is the
  // single source of truth, `workflowRevision` bumps only on external replaces
  // (starter seed, agent authoring) so the canvas remounts without looping on
  // its own edits, and `workflowTouched` gates the code agent's auto-seed so a
  // user (or agent) edit is never clobbered by a re-seed.
  const [workflowSpec, setWorkflowSpec] = React.useState<WorkflowSpec | null>(null);
  const [workflowRevision, setWorkflowRevision] = React.useState(0);
  const [workflowTouched, setWorkflowTouched] = React.useState(false);
  const [agentPulseNodeId, setAgentPulseNodeId] = React.useState<string | null>(null);
  const pulseClearRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // Seed a starter graph the moment the run is armed in workflow mode — the
  // code agent's workflow seed only fires once a graph exists to draft over.
  React.useEffect(() => {
    if (!armed || !isWorkflow || !dataset || workflowSpec) return;
    setWorkflowSpec(defaultWorkflowSpec(columnRoles, columnKinds));
    setWorkflowRevision((r) => r + 1);
  }, [armed, isWorkflow, dataset, workflowSpec, columnRoles, columnKinds]);

  const applyAgentWorkflow = React.useCallback(
    (spec: WorkflowSpec, changedNodeId: string | null) => {
      // Agent-authored nodes arrive without canvas positions; lay the whole
      // graph out so they never pile on top of each other.
      const laid = spec.nodes.some((n) => !n.position) ? autoLayoutSpec(spec) : spec;
      setWorkflowSpec(laid);
      setWorkflowRevision((r) => r + 1);
      setAgentPulseNodeId(changedNodeId);
      if (pulseClearRef.current) clearTimeout(pulseClearRef.current);
      if (changedNodeId) {
        pulseClearRef.current = setTimeout(() => setAgentPulseNodeId(null), 1600);
      }
    },
    [],
  );

  const updateWorkflowSpec = React.useCallback((spec: WorkflowSpec) => {
    // A user edit from the canvas: keep the spec but do NOT bump the revision
    // (the canvas already holds this state) and mark it touched so the seed
    // never overwrites the user's work.
    setWorkflowTouched(true);
    setWorkflowSpec(spec);
  }, []);

  // Gate the seed on ``armed``: until the generalist requests authoring there is
  // no dataset for the code agent, so its auto-seed effect stays dormant.
  const parsedDataset = React.useMemo<ParsedDataset | null>(() => {
    if (!armed || !dataset) return null;
    return { columns: dataset.columns, rows: dataset.rows, rowCount: dataset.rowCount };
  }, [armed, dataset]);

  // Validation runners read live dataset/code via a ref so their identity stays
  // stable for ``useCodeAgent`` (which captures them once) while still seeing
  // the current sample row, roles, and edited code.
  const ctxRef = React.useRef({ dataset, columnRoles, optimizerName, signatureCode, metricCode });
  React.useEffect(() => {
    ctxRef.current = { dataset, columnRoles, optimizerName, signatureCode, metricCode };
  });

  const runValidation = React.useCallback(
    async (kind: "signature" | "metric", code: string): Promise<ValidationResult | null> => {
      const { dataset: ds, columnRoles: roles, optimizerName: opt } = ctxRef.current;
      if (!ds || ds.rows.length === 0) return null;
      try {
        const result = (await validateCode({
          signature_code: kind === "signature" ? code : undefined,
          metric_code: kind === "metric" ? code : undefined,
          column_mapping: rolesToColumnMapping(roles),
          sample_row: ds.rows[0] as Record<string, unknown>,
          optimizer_name: opt,
        })) as ValidateCodeResponse;
        if (kind === "signature") setSignatureValidation(result);
        else setMetricValidation(result);
        return result;
      } catch (err) {
        return {
          valid: false,
          errors: [err instanceof Error ? err.message : msg("agent.validation.failed")],
          warnings: [],
        };
      }
    },
    [],
  );

  const runSignatureValidation = React.useCallback(
    (overrideCode?: string) =>
      runValidation("signature", overrideCode ?? ctxRef.current.signatureCode),
    [runValidation],
  );
  const runMetricValidation = React.useCallback(
    (overrideCode?: string) =>
      runValidation("metric", overrideCode ?? ctxRef.current.metricCode),
    [runValidation],
  );

  const agent = useCodeAgent({
    codeAssistMode: "auto",
    setCodeAssistMode: NOOP,
    columnRoles,
    columnKinds,
    parsedDataset,
    moduleName,
    signatureCode,
    metricCode,
    setSignatureCode,
    setMetricCode,
    signatureManuallyEdited,
    metricManuallyEdited,
    setSignatureManuallyEdited,
    setMetricManuallyEdited,
    setSignatureValidation,
    setMetricValidation,
    signatureValidation,
    metricValidation,
    runSignatureValidation,
    runMetricValidation,
    model,
    reasoningEffort,
    isWorkflow,
    workflowSpec,
    workflowTouched,
    applyAgentWorkflow,
  });

  // ``useCodeAgent.reset`` clears its own state but not the caller-owned code
  // (the wizard clears that separately); wrap it so a fresh conversation drops
  // the previous run's artifacts too.
  const reset = React.useCallback(() => {
    agent.reset();
    setSignatureCode("");
    setMetricCode("");
    setSignatureValidation(null);
    setMetricValidation(null);
    setSignatureManuallyEdited(false);
    setMetricManuallyEdited(false);
    setWorkflowSpec(null);
    setWorkflowRevision(0);
    setWorkflowTouched(false);
    setAgentPulseNodeId(null);
  }, [agent.reset]);

  return {
    ...agent,
    reset,
    signatureCode,
    metricCode,
    signatureValidation,
    metricValidation,
    isWorkflow,
    workflowSpec,
    workflowRevision,
    agentPulseNodeId,
    updateWorkflowSpec,
  };
}
