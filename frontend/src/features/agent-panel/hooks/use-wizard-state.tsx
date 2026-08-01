"use client";

import * as React from "react";

import type { WizardState } from "../lib/types";

type WizardKey =
  | "dataset_ready"
  | "columns_configured"
  | "signature_code"
  | "metric_code"
  | "model_configured"
  | "job_name"
  | "job_description"
  | "job_type"
  | "optimizer_name"
  | "module_name"
  | "workflow"
  | "react_config"
  | "dataset_columns"
  | "column_roles"
  | "model_config"
  | "reflection_model_config"
  | "generation_models"
  | "reflection_models"
  | "split_fractions"
  | "split_mode"
  | "seed"
  | "shuffle"
  | "is_private"
  | "optimizer_kwargs"
  | "target_score"
  | "staged_dataset_id"
  | "source_dataset_id";
type WriteSource = "user" | "agent";

interface WizardStateContextValue {
  state: WizardState;
  overriddenFields: string[];
  /** Bumped whenever the agent writes a field so consumers can pulse. */
  agentPulseTick: number;
  /**
   * Keys touched by the most recent agent write. Persists until the next agent
   * write replaces it; consumers gate on ``agentPulseTick`` (not on a clearing
   * tick) to detect new pulses.
   */
  agentPulseKeys: readonly string[];
  setField: (key: WizardKey, value: WizardState[WizardKey], source?: WriteSource) => void;
  applyAgentPatch: (patch: Partial<WizardState>) => void;
  clearField: (key: WizardKey) => void;
  /**
   * Wipe all shared wizard state back to empty. Called after a submit so the
   * next optimization (and the agent's readiness gate) starts clean instead of
   * inheriting the just-submitted run's dataset, code, and readiness flags.
   */
  reset: () => void;
}

const Ctx = React.createContext<WizardStateContextValue | null>(null);

/**
 * Shared wizard state between the submit wizard and the generalist
 * agent. Every field write records the source; user writes after the
 * agent wrote are marked as overridden so the UI can surface a dot.
 */
export function WizardStateProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = React.useState<WizardState>({});
  const [overridden, setOverridden] = React.useState<Set<string>>(() => new Set());
  const [pulseTick, setPulseTick] = React.useState(0);
  const [pulseKeys, setPulseKeys] = React.useState<readonly string[]>([]);
  const agentWrittenRef = React.useRef<Set<string>>(new Set());

  const setField = React.useCallback<WizardStateContextValue["setField"]>(
    (key, value, source = "user") => {
      setState((prev) => {
        if (prev[key] === value) return prev;
        return { ...prev, [key]: value } as WizardState;
      });
      if (source === "user") {
        if (agentWrittenRef.current.has(key)) {
          setOverridden((prev) => {
            if (prev.has(key)) return prev;
            const next = new Set(prev);
            next.add(key);
            return next;
          });
        }
      } else {
        agentWrittenRef.current.add(key);
        setOverridden((prev) => {
          if (!prev.has(key)) return prev;
          const next = new Set(prev);
          next.delete(key);
          return next;
        });
        setPulseKeys([key]);
        setPulseTick((t) => t + 1);
      }
    },
    [],
  );

  const applyAgentPatch = React.useCallback((patch: Partial<WizardState>) => {
    const keys = Object.keys(patch) as WizardKey[];
    if (keys.length === 0) return;
    setState((prev) => {
      let changed = false;
      const next = { ...prev };
      for (const k of keys) {
        const v = patch[k];
        if (v !== undefined && prev[k] !== v) {
          (next as Record<string, unknown>)[k] = v;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
    for (const k of keys) agentWrittenRef.current.add(k);
    setOverridden((prev) => {
      let changed = false;
      const next = new Set(prev);
      for (const k of keys) {
        if (next.has(k)) {
          next.delete(k);
          changed = true;
        }
      }
      return changed ? next : prev;
    });
    setPulseKeys(keys);
    setPulseTick((t) => t + 1);
  }, []);

  const clearField = React.useCallback<WizardStateContextValue["clearField"]>((key) => {
    setState((prev) => {
      if (!(key in prev)) return prev;
      const next = { ...prev };
      delete (next as Record<string, unknown>)[key];
      return next;
    });
    setOverridden((prev) => {
      if (!prev.has(key)) return prev;
      const next = new Set(prev);
      next.delete(key);
      return next;
    });
  }, []);

  const reset = React.useCallback(() => {
    agentWrittenRef.current = new Set();
    setState({});
    setOverridden(new Set());
    // Clear the pulse keys (but don't bump the tick) so a wizard mounting
    // afterwards early-returns from its incoming-sync effect instead of
    // replaying the last agent write onto an otherwise-empty form.
    setPulseKeys([]);
  }, []);

  const overriddenFields = React.useMemo(() => Array.from(overridden).sort(), [overridden]);

  const value = React.useMemo<WizardStateContextValue>(
    () => ({
      state,
      overriddenFields,
      agentPulseTick: pulseTick,
      agentPulseKeys: pulseKeys,
      setField,
      applyAgentPatch,
      clearField,
      reset,
    }),
    [state, overriddenFields, pulseTick, pulseKeys, setField, applyAgentPatch, clearField, reset],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

/** Safe variant — returns null when outside a provider. */
export function useWizardStateOptional(): WizardStateContextValue | null {
  return React.useContext(Ctx);
}

/**
 * Extract the subset of a tool result that updates the wizard. Defensive:
 * accepts any object and picks only known wizard keys with expected types.
 */
export function extractWizardPatch(result: unknown): Partial<WizardState> {
  if (!result || typeof result !== "object") return {};
  const r = result as Record<string, unknown>;
  const wrap =
    r.wizard_state && typeof r.wizard_state === "object"
      ? (r.wizard_state as Record<string, unknown>)
      : r;
  const patch: Partial<WizardState> = {};

  if (typeof wrap.dataset_ready === "boolean") patch.dataset_ready = wrap.dataset_ready;
  if (typeof wrap.columns_configured === "boolean")
    patch.columns_configured = wrap.columns_configured;
  if (typeof wrap.model_configured === "boolean") patch.model_configured = wrap.model_configured;

  if (typeof wrap.staged_dataset_id === "string" && wrap.staged_dataset_id) {
    patch.staged_dataset_id = wrap.staged_dataset_id;
  }
  if (typeof wrap.source_dataset_id === "string" && wrap.source_dataset_id) {
    patch.source_dataset_id = wrap.source_dataset_id;
  }

  if (typeof wrap.signature_code === "string") patch.signature_code = wrap.signature_code;
  if (typeof wrap.metric_code === "string") patch.metric_code = wrap.metric_code;

  if (typeof wrap.job_name === "string") patch.job_name = wrap.job_name;
  if (typeof wrap.job_description === "string") patch.job_description = wrap.job_description;
  if (wrap.job_type === "run" || wrap.job_type === "grid_search") patch.job_type = wrap.job_type;

  if (typeof wrap.optimizer_name === "string") patch.optimizer_name = wrap.optimizer_name;
  if (typeof wrap.module_name === "string") patch.module_name = wrap.module_name;

  if (
    wrap.workflow &&
    typeof wrap.workflow === "object" &&
    !Array.isArray(wrap.workflow) &&
    Array.isArray((wrap.workflow as Record<string, unknown>).nodes)
  ) {
    patch.workflow = wrap.workflow as WizardState["workflow"];
  }

  if (
    wrap.react_config &&
    typeof wrap.react_config === "object" &&
    !Array.isArray(wrap.react_config)
  ) {
    patch.react_config = wrap.react_config as Record<string, unknown>;
  }

  if (
    Array.isArray(wrap.dataset_columns) &&
    wrap.dataset_columns.every((x) => typeof x === "string")
  ) {
    patch.dataset_columns = wrap.dataset_columns as string[];
  }
  if (
    wrap.column_roles &&
    typeof wrap.column_roles === "object" &&
    !Array.isArray(wrap.column_roles)
  ) {
    patch.column_roles = wrap.column_roles as Record<string, string>;
  }

  if (
    wrap.model_config &&
    typeof wrap.model_config === "object" &&
    !Array.isArray(wrap.model_config)
  ) {
    patch.model_config = wrap.model_config as Record<string, unknown>;
  }
  if (
    wrap.reflection_model_config &&
    typeof wrap.reflection_model_config === "object" &&
    !Array.isArray(wrap.reflection_model_config)
  ) {
    patch.reflection_model_config = wrap.reflection_model_config as Record<string, unknown>;
  }

  if (
    Array.isArray(wrap.generation_models) &&
    wrap.generation_models.every((m) => m && typeof m === "object" && !Array.isArray(m))
  ) {
    patch.generation_models = wrap.generation_models as Array<Record<string, unknown>>;
  }
  if (
    Array.isArray(wrap.reflection_models) &&
    wrap.reflection_models.every((m) => m && typeof m === "object" && !Array.isArray(m))
  ) {
    patch.reflection_models = wrap.reflection_models as Array<Record<string, unknown>>;
  }
  if (
    wrap.split_fractions &&
    typeof wrap.split_fractions === "object" &&
    !Array.isArray(wrap.split_fractions)
  ) {
    const sf = wrap.split_fractions as Record<string, unknown>;
    const train = typeof sf.train === "number" ? sf.train : NaN;
    const val = typeof sf.val === "number" ? sf.val : NaN;
    const test = typeof sf.test === "number" ? sf.test : NaN;
    if (!Number.isNaN(train) && !Number.isNaN(val) && !Number.isNaN(test)) {
      patch.split_fractions = { train, val, test };
    }
  }
  if (wrap.split_mode === "auto" || wrap.split_mode === "manual") {
    patch.split_mode = wrap.split_mode;
  }
  if (typeof wrap.seed === "number" && Number.isFinite(wrap.seed)) patch.seed = wrap.seed;
  if (typeof wrap.shuffle === "boolean") patch.shuffle = wrap.shuffle;
  if (typeof wrap.is_private === "boolean") patch.is_private = wrap.is_private;

  if (
    wrap.optimizer_kwargs &&
    typeof wrap.optimizer_kwargs === "object" &&
    !Array.isArray(wrap.optimizer_kwargs)
  ) {
    patch.optimizer_kwargs = wrap.optimizer_kwargs as Record<string, unknown>;
  }

  return patch;
}
