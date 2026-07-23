"use client";

import * as React from "react";

import { validateCode } from "@/shared/lib/api";
import { useCodeAgent, type CodeAgentState } from "@/shared/hooks/use-code-agent";
import type { ParsedDataset } from "@/shared/lib/parse-dataset";
import type { ValidationResult } from "@/shared/ui/code-editor";
import type { ColumnMapping, ValidateCodeResponse } from "@/shared/types/api";

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
          errors: [err instanceof Error ? err.message : "Validation failed"],
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
  }, [agent.reset]);

  return {
    ...agent,
    reset,
    signatureCode,
    metricCode,
    signatureValidation,
    metricValidation,
  };
}
