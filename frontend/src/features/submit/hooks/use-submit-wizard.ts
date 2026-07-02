"use client";

import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useSession } from "next-auth/react";
import { toast } from "react-toastify";
import { track, TelemetryEvent } from "@/shared/lib/telemetry";

import {
  submitRun,
  submitGridSearch,
  validateCode,
  validateDataset,
  getOptimizationPayload,
  getJob,
  getSharedOptimization,
  getPublicOptimization,
  stageDatasetForAgent,
  getStagedDataset,
  getDatasetRows,
  isStorageQuotaError,
  isInsufficientCreditsError,
  type DatasetSummary,
} from "@/shared/lib/api";
import type {
  ModelConfig,
  SplitFractions,
  ValidateCodeResponse,
  ValidateDatasetResponse,
  DatasetProfile,
  SplitPlan,
  RunRequest,
  ToolSource,
} from "@/shared/types/api";
import { parseDatasetFile, type ParsedDataset } from "@/shared/lib/parse-dataset";
import type { ValidationResult as EditorValidationResult } from "@/shared/ui/code-editor";
import { registerTutorialHook } from "@/features/tutorial";
import { formatMsg, msg } from "@/shared/lib/messages";
import { useWizardStateOptional } from "@/features/agent-panel";
import { readPref, useUserPrefs } from "@/features/settings";
import { useCredits } from "@/features/billing";

import { STEPS, emptyModelConfig, defaultSplit, defaultReactConfig } from "../constants";
import type { ReactConfig, ColumnRole } from "../constants";
import { buildSignatureTemplate } from "../lib/build-signature";
import { buildMetricTemplate } from "../lib/build-metric";
import { buildOptimizerKwargs } from "../lib/build-kwargs";
import {
  projectCostBracket,
  defaultCeilingForBracket,
  chargeableBracket,
  type CostBracket,
} from "../lib/cost-bracket";
import {
  saveWizardDraft,
  readWizardDraft,
  clearWizardDraft,
  type WizardDraftData,
} from "../lib/wizard-draft";
import { useCodeAgent } from "@/shared/hooks/use-code-agent";
import {
  buildColumnMapping,
  useDatasetProfiling,
  useModelCatalog,
  useRecentModelConfigs,
} from "./use-submit-wizard-data";

const COLUMN_ROLES = new Set<string>(["input", "output", "ignore"]);

/** Type guard for a valid dataset column role (signature I/O). */
function isColumnRole(value: unknown): value is ColumnRole {
  return typeof value === "string" && COLUMN_ROLES.has(value);
}

export function useSubmitWizard() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { data: session } = useSession();
  const { prefs } = useUserPrefs();
  // The active token-source mode (managed credits vs the user's own key). Sent
  // on every submit so the guarantee is enforced server-side, not just shown in
  // the wizard.
  const { wallet } = useCredits();

  const [step, setStep] = useState(0);
  const [direction, setDirection] = useState(0);
  const [furthestReachedStep, setFurthestReachedStep] = useState(0);
  const [summaryTab, setSummaryTab] = useState(0);
  const [summaryCodeTab, setSummaryCodeTab] = useState<string>("signature");

  const [jobType, setOptimizationType] = useState<"run" | "grid_search">("run");
  const [isPrivate, setIsPrivate] = useState(true);

  const username = session?.user?.name ?? "";
  const [jobName, setJobName] = useState("");
  const [jobDescription, setJobDescription] = useState("");
  const [moduleName, setModuleName] = useState("predict");
  const [optimizerName, setOptimizerName] = useState("gepa");

  // React (ReAct-agent) tool roster. Only sent when moduleName is "react".
  // React is generic — it is scored by the same standard metric_code as
  // predict/cot, and this config only carries the live tool source.
  const [reactConfig, setReactConfig] = useState<ReactConfig>(defaultReactConfig);
  const updateReactConfig = useCallback(
    (patch: Partial<ReactConfig>) => setReactConfig((prev) => ({ ...prev, ...patch })),
    [],
  );
  const isReact = moduleName.toLowerCase() === "react";

  const [signatureCode, setSignatureCode] = useState(() => buildSignatureTemplate({}));
  const [metricCode, setMetricCode] = useState(() => buildMetricTemplate({}));

  const [parsedDataset, setParsedDataset] = useState<ParsedDataset | null>(null);
  const [datasetFileName, setDatasetFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // A by-reference submit (source_dataset_id) is only valid while the on-screen
  // rows are still the ones we loaded from the library. Every other dataset
  // source — upload, clone, agent-staged — replaces ``parsedDataset`` with a
  // fresh object, so this identity-bound ref naturally goes stale and the submit
  // falls back to inlining rows without clearing a flag at each call site.
  const librarySourceRef = useRef<{ id: string; parsed: ParsedDataset } | null>(null);

  const [columnRoles, setColumnRoles] = useState<Record<string, ColumnRole>>({});
  // Manual override for input column modality. The dataset profiler auto-fills
  // entries for every input column (kind = "text" | "image"); the user can
  // flip a column manually via the DatasetStep toggle.
  const [columnKinds, setColumnKinds] = useState<Record<string, "text" | "image">>({});
  const [signatureManuallyEdited, setSignatureManuallyEdited] = useState(false);
  const [metricManuallyEdited, setMetricManuallyEdited] = useState(false);
  const [codeAssistMode, setCodeAssistMode] = useState<"auto" | "manual">(() =>
    readPref("wizardCodeAssist"),
  );

  const [globalBaseUrl, setGlobalBaseUrl] = useState("");
  const [globalApiKey, setGlobalApiKey] = useState("");

  const [modelConfig, setModelConfig] = useState<ModelConfig>(emptyModelConfig());
  const [secondModelConfig, setSecondModelConfig] = useState<ModelConfig | null>(null);

  const [editingModel, setEditingModel] = useState<{
    config: ModelConfig;
    onSave: (c: ModelConfig) => void;
    label: string;
    onSelectAllAvailable?: () => void;
  } | null>(null);

  const { recentConfigs, saveToRecent, clearRecentConfigs, removeRecentConfig } =
    useRecentModelConfigs();

  const catalog = useModelCatalog();

  const anyProviderHasEnvKey = catalog?.providers.some((p) => p.has_env_key) ?? false;

  const [generationModels, setGenerationModels] = useState<ModelConfig[]>([emptyModelConfig()]);
  const [reflectionModels, setReflectionModels] = useState<ModelConfig[]>([emptyModelConfig()]);
  const [useAllGenerationModels, setUseAllGenerationModels] = useState(false);
  const [useAllReflectionModels, setUseAllReflectionModels] = useState(false);

  const [split, setSplit] = useState<SplitFractions>(defaultSplit);

  // Dataset profile + recommended split plan (non-blocking; the user can always
  // override or ignore). A ref mirrors the manual-edit flag so the auto-profile
  // effect can read it without stale-closure re-runs.
  const [datasetProfile, setDatasetProfile] = useState<DatasetProfile | null>(null);
  const [splitPlan, setSplitPlan] = useState<SplitPlan | null>(null);
  const [profileLoading, setProfileLoading] = useState(false);
  const [splitMode, setSplitModeState] = useState<"auto" | "manual">(() =>
    readPref("wizardSplitMode"),
  );
  const splitModeRef = useRef<"auto" | "manual">(readPref("wizardSplitMode"));

  // Mirror live pref changes into local wizard state so changes the user
  // makes in the settings modal while the wizard is mounted take effect
  // without a remount. Skip the first run because UserPrefsProvider boots
  // with DEFAULT_PREFS and hydrates from localStorage in a useEffect — our
  // useState initializers above already used readPref() (sync), so the first
  // render's `prefs.*` values would clobber them with defaults.
  const codeAssistFirstRunRef = useRef(true);
  useEffect(() => {
    if (codeAssistFirstRunRef.current) {
      codeAssistFirstRunRef.current = false;
      return;
    }
    setCodeAssistMode(prefs.wizardCodeAssist);
  }, [prefs.wizardCodeAssist]);

  const splitModeFirstRunRef = useRef(true);
  useEffect(() => {
    if (splitModeFirstRunRef.current) {
      splitModeFirstRunRef.current = false;
      return;
    }
    splitModeRef.current = prefs.wizardSplitMode;
    setSplitModeState(prefs.wizardSplitMode);
  }, [prefs.wizardSplitMode]);

  const [seed, setSeed] = useState<number | undefined>(undefined);

  const [signatureValidation, setSignatureValidation] = useState<ValidateCodeResponse | null>(null);
  const [metricValidation, setMetricValidation] = useState<ValidateCodeResponse | null>(null);
  const [datasetValidation, setDatasetValidation] = useState<ValidateDatasetResponse | null>(null);

  const [autoLevel, setAutoLevel] = useState<string>("light");
  const [reflectionMinibatchSize, setReflectionMinibatchSize] = useState<string>("3");
  const [maxFullEvals, setMaxFullEvals] = useState<string>("6");
  const [useMerge, setUseMerge] = useState(true);
  const [shuffle, setShuffle] = useState(true);
  // User-set Max Cost Ceiling, in credits — null until the user opts into a cap.
  // The run is hard-stopped server-side once spend exceeds it (Phase 2 [FG-1]),
  // and the run button carries the cap. `null` ⇒ no ceiling sent.
  const [maxCostCredits, setMaxCostCredits] = useState<number | null>(null);

  const [submitting, setSubmitting] = useState(false);
  const [submitPhase, setSubmitPhase] = useState<"idle" | "sending" | "splash" | "done">("idle");

  // Guard against double-clicks on the Next button while the code-step
  // validation network call is in flight (~5s). Without this, two
  // sequential `setStep((s) => s + 1)` calls advance the wizard twice.
  const [advancing, setAdvancing] = useState(false);
  const advancingRef = useRef(false);

  // Memoise validation results per (kind, code, mapping, sample_row,
  // optimizer) tuple. Storing the in-flight promise itself also dedupes
  // concurrent calls against the same key — useful when both signature and
  // metric run in parallel and one is re-triggered before the other lands.
  const validationCacheRef = useRef(new Map<string, Promise<ValidateCodeResponse>>());

  const [cloneLoading, setCloneLoading] = useState(false);
  const cloneRan = useRef(false);

  // Register setters with the typed tutorial bridge so the tutorial system
  // can drive the wizard from plain-JS steps (see lib/tutorial-bridge.ts).
  useEffect(() => {
    const unregister = [
      registerTutorialHook("setWizardStep", setStep),
      registerTutorialHook("setParsedDataset", setParsedDataset),
      registerTutorialHook("setColumnRoles", setColumnRoles),
      registerTutorialHook("setDatasetFileName", setDatasetFileName),
      registerTutorialHook("setSignatureCode", setSignatureCode),
      registerTutorialHook("setMetricCode", setMetricCode),
      registerTutorialHook("setOptimizerName", setOptimizerName),
    ];
    return () => unregister.forEach((fn) => fn());
  }, []);

  // Shared wizard-state bridge: the generalist agent writes wizard fields
  // into WizardStateContext. We mirror those agent writes into the local
  // wizard state, and push local edits back so the agent's phased-exposure
  // gate sees them. Echo is avoided by only pushing when the value actually
  // differs from shared.
  const wizardCtx = useWizardStateOptional();
  const { agentPulseTick, agentPulseKeys, sharedState } = {
    agentPulseTick: wizardCtx?.agentPulseTick ?? 0,
    agentPulseKeys: wizardCtx?.agentPulseKeys ?? [],
    sharedState: wizardCtx?.state,
  };

  // After a successful submit, navigation unmounts this form. Clear the shared
  // wizard state on that unmount so a later agent turn doesn't inherit the
  // just-submitted run's readiness + staged dataset and offer a duplicate.
  // Gated on ``submittedRef`` so merely navigating away from a half-filled
  // wizard preserves the in-progress state. Deferred to unmount (rather than
  // run inline in handleSubmit) so the outgoing sync effects below can't
  // re-push the stale values back into the context after the reset.
  const wizardCtxRef = useRef(wizardCtx);
  useEffect(() => {
    wizardCtxRef.current = wizardCtx;
  }, [wizardCtx]);
  const submittedRef = useRef(false);

  // Mirror the full serializable wizard snapshot into a ref every commit so the
  // unmount cleanup below parks the *latest* values — a []-deps cleanup would
  // otherwise close over the first render's state.
  const draftRef = useRef<WizardDraftData | null>(null);
  useEffect(() => {
    draftRef.current = {
      step,
      furthestReachedStep,
      summaryTab,
      summaryCodeTab,
      jobType,
      isPrivate,
      jobName,
      jobDescription,
      moduleName,
      optimizerName,
      reactConfig,
      signatureCode,
      metricCode,
      signatureManuallyEdited,
      metricManuallyEdited,
      parsedDataset,
      datasetFileName,
      columnRoles,
      columnKinds,
      globalBaseUrl,
      globalApiKey,
      modelConfig,
      secondModelConfig,
      generationModels,
      reflectionModels,
      useAllGenerationModels,
      useAllReflectionModels,
      split,
      seed,
      autoLevel,
      reflectionMinibatchSize,
      maxFullEvals,
      useMerge,
      shuffle,
      maxCostCredits,
    };
  });

  // Restore a parked draft on mount so switching to another sidebar tab and
  // coming back lands the user on the same step with inputs intact. Skipped when
  // a clone/share URL owns hydration — that flow populates the form itself.
  const restoredRef = useRef(false);
  useEffect(() => {
    if (restoredRef.current) return;
    restoredRef.current = true;
    if (searchParams.get("clone") || searchParams.get("shareToken")) return;
    const d = readWizardDraft();
    if (!d) return;
    setStep(d.step);
    setFurthestReachedStep(d.furthestReachedStep);
    setSummaryTab(d.summaryTab);
    setSummaryCodeTab(d.summaryCodeTab);
    setOptimizationType(d.jobType);
    setIsPrivate(d.isPrivate);
    setJobName(d.jobName);
    setJobDescription(d.jobDescription);
    setModuleName(d.moduleName);
    setOptimizerName(d.optimizerName);
    setReactConfig(d.reactConfig);
    setSignatureCode(d.signatureCode);
    setMetricCode(d.metricCode);
    setSignatureManuallyEdited(d.signatureManuallyEdited);
    setMetricManuallyEdited(d.metricManuallyEdited);
    setParsedDataset(d.parsedDataset);
    setDatasetFileName(d.datasetFileName);
    setColumnRoles(d.columnRoles);
    setColumnKinds(d.columnKinds);
    setGlobalBaseUrl(d.globalBaseUrl);
    setGlobalApiKey(d.globalApiKey);
    setModelConfig(d.modelConfig);
    setSecondModelConfig(d.secondModelConfig);
    setGenerationModels(d.generationModels);
    setReflectionModels(d.reflectionModels);
    setUseAllGenerationModels(d.useAllGenerationModels);
    setUseAllReflectionModels(d.useAllReflectionModels);
    setSplit(d.split);
    setSeed(d.seed);
    setAutoLevel(d.autoLevel);
    setReflectionMinibatchSize(d.reflectionMinibatchSize);
    setMaxFullEvals(d.maxFullEvals);
    setUseMerge(d.useMerge);
    setShuffle(d.shuffle);
    setMaxCostCredits(d.maxCostCredits);
  }, []);

  useEffect(
    () => () => {
      if (submittedRef.current) {
        // A submitted run navigated away on purpose — reset the shared agent state
        // and drop any draft parked on an earlier nav-away.
        wizardCtxRef.current?.reset();
        clearWizardDraft();
        return;
      }
      // Park a half-filled form (skip a pristine one) so the round-trip restores.
      const d = draftRef.current;
      if (
        d &&
        (d.step > 0 ||
          d.parsedDataset !== null ||
          d.datasetFileName !== null ||
          d.jobName.trim() !== "")
      ) {
        saveWizardDraft(d);
      } else {
        clearWizardDraft();
      }
    },
    [],
  );

  // Incoming: apply agent patches to local state whenever the pulse bumps.
  useEffect(() => {
    if (!sharedState || agentPulseKeys.length === 0) return;
    for (const key of agentPulseKeys) {
      if (key === "job_name" && typeof sharedState.job_name === "string") {
        setJobName(sharedState.job_name);
      } else if (key === "job_description" && typeof sharedState.job_description === "string") {
        setJobDescription(sharedState.job_description);
      } else if (
        key === "job_type" &&
        (sharedState.job_type === "run" || sharedState.job_type === "grid_search")
      ) {
        setOptimizationType(sharedState.job_type);
      } else if (key === "optimizer_name" && typeof sharedState.optimizer_name === "string") {
        setOptimizerName(sharedState.optimizer_name);
      } else if (key === "module_name" && typeof sharedState.module_name === "string") {
        setModuleName(sharedState.module_name);
      } else if (key === "react_config" && sharedState.react_config) {
        const rc = sharedState.react_config as Record<string, unknown>;
        setReactConfig((prev) => {
          const next = { ...prev };
          if (rc.toolSourceKind === "live_mcp" || rc.toolSourceKind === "dataset_snapshot") {
            next.toolSourceKind = rc.toolSourceKind;
          }
          for (const field of ["mcpUrl", "toolFilter"] as const) {
            if (typeof rc[field] === "string") next[field] = rc[field] as string;
          }
          return next;
        });
      } else if (key === "signature_code" && typeof sharedState.signature_code === "string") {
        setSignatureCode(sharedState.signature_code);
        setSignatureManuallyEdited(true);
        setSignatureValidation(null);
      } else if (key === "metric_code" && typeof sharedState.metric_code === "string") {
        setMetricCode(sharedState.metric_code);
        setMetricManuallyEdited(true);
        setMetricValidation(null);
      } else if (key === "column_roles" && sharedState.column_roles) {
        setColumnRoles((prev) => {
          const next = { ...prev };
          for (const [col, role] of Object.entries(sharedState.column_roles ?? {})) {
            if (isColumnRole(role)) next[col] = role;
          }
          return next;
        });
      } else if (key === "model_config" && sharedState.model_config) {
        setModelConfig({
          ...emptyModelConfig(),
          ...(sharedState.model_config as Partial<ModelConfig>),
        });
      } else if (key === "reflection_model_config" && sharedState.reflection_model_config) {
        setSecondModelConfig({
          ...emptyModelConfig(),
          ...(sharedState.reflection_model_config as Partial<ModelConfig>),
        });
      } else if (key === "generation_models" && Array.isArray(sharedState.generation_models)) {
        setGenerationModels(
          sharedState.generation_models.map((m) => ({
            ...emptyModelConfig(),
            ...(m as Partial<ModelConfig>),
          })),
        );
      } else if (key === "reflection_models" && Array.isArray(sharedState.reflection_models)) {
        setReflectionModels(
          sharedState.reflection_models.map((m) => ({
            ...emptyModelConfig(),
            ...(m as Partial<ModelConfig>),
          })),
        );
      } else if (
        key === "use_all_generation_models" &&
        typeof sharedState.use_all_generation_models === "boolean"
      ) {
        setUseAllGenerationModels(sharedState.use_all_generation_models);
      } else if (
        key === "use_all_reflection_models" &&
        typeof sharedState.use_all_reflection_models === "boolean"
      ) {
        setUseAllReflectionModels(sharedState.use_all_reflection_models);
      } else if (key === "split_fractions" && sharedState.split_fractions) {
        setSplit(sharedState.split_fractions);
      } else if (
        key === "split_mode" &&
        (sharedState.split_mode === "auto" || sharedState.split_mode === "manual")
      ) {
        splitModeRef.current = sharedState.split_mode;
        setSplitModeState(sharedState.split_mode);
      } else if (key === "seed" && typeof sharedState.seed === "number") {
        setSeed(sharedState.seed);
      } else if (key === "shuffle" && typeof sharedState.shuffle === "boolean") {
        setShuffle(sharedState.shuffle);
      } else if (key === "is_private" && typeof sharedState.is_private === "boolean") {
        setIsPrivate(sharedState.is_private);
      } else if (key === "optimizer_kwargs" && sharedState.optimizer_kwargs) {
        const kw = sharedState.optimizer_kwargs as Record<string, unknown>;
        if (typeof kw.auto === "string") setAutoLevel(kw.auto);
        if (typeof kw.reflection_minibatch_size === "number") {
          setReflectionMinibatchSize(String(kw.reflection_minibatch_size));
        }
        if (typeof kw.max_full_evals === "number") {
          setMaxFullEvals(String(kw.max_full_evals));
        }
        if (typeof kw.use_merge === "boolean") setUseMerge(kw.use_merge);
      }
    }
  }, [agentPulseTick]);

  // Outgoing: push relevant local state back into the shared context so the
  // agent's tool-gate (dataset_ready, columns_configured, model_configured)
  // reflects what the user actually has. Guarded by value equality to avoid
  // echo after incoming patches.
  useEffect(() => {
    if (!wizardCtx) return;
    const datasetReady = !!parsedDataset && parsedDataset.rowCount > 0;
    if (wizardCtx.state.dataset_ready !== datasetReady) {
      wizardCtx.setField("dataset_ready", datasetReady, "user");
    }
    const columns = parsedDataset?.columns ?? [];
    const shared = wizardCtx.state.dataset_columns;
    const changed =
      !shared || shared.length !== columns.length || columns.some((c, i) => shared[i] !== c);
    if (changed && columns.length > 0) {
      wizardCtx.setField("dataset_columns", columns, "user");
    }
  }, [parsedDataset, wizardCtx]);

  // Projected pre-run credit bracket [FG-1]: a DSPy job's token use isn't linear,
  // so we show a range rather than a false-precision single number and seed the
  // Max Cost Ceiling from its high end. For a grid, count the (gen × refl) pairs
  // so the bracket reflects the whole sweep.
  const gridPairs =
    jobType === "grid_search"
      ? Math.max(
          1,
          (useAllGenerationModels
            ? (catalog?.models.length ?? 1)
            : generationModels.filter((m) => m.name.trim()).length || 1) *
            (useAllReflectionModels
              ? (catalog?.models.length ?? 1)
              : reflectionModels.filter((m) => m.name.trim()).length || 1),
        )
      : 1;
  const costBracket: CostBracket = useMemo(() => {
    // Price the projection on the actually-chosen models (a representative
    // gen/refl pair for a grid). Looked up in the catalog so each model's real
    // $/token moves the estimate; unresolved names price at the engine defaults.
    const taskName =
      jobType === "grid_search"
        ? generationModels.find((m) => m.name.trim())?.name
        : modelConfig.name;
    const reflName =
      jobType === "grid_search"
        ? reflectionModels.find((m) => m.name.trim())?.name
        : secondModelConfig?.name;
    const taskModel = taskName?.trim()
      ? (catalog?.models.find((m) => m.value === taskName) ?? null)
      : null;
    const reflectionModel = reflName?.trim()
      ? (catalog?.models.find((m) => m.value === reflName) ?? null)
      : null;
    return projectCostBracket({
      autoLevel,
      maxFullEvals,
      datasetRows: parsedDataset?.rowCount ?? 0,
      pairs: gridPairs,
      taskModel,
      reflectionModel,
    });
  }, [
    autoLevel,
    maxFullEvals,
    parsedDataset?.rowCount,
    jobType,
    modelConfig.name,
    secondModelConfig,
    generationModels,
    reflectionModels,
    catalog,
    gridPairs,
  ]);

  // Default the cap to the bracket's high end (with headroom) the first time a
  // user opens the ceiling control; once they set a value we leave it alone.
  const suggestedCeiling = useMemo(() => defaultCeilingForBracket(costBracket), [costBracket]);

  // Stage the parsed rows on the backend so the agent can submit by id
  // without inlining tens of thousands of rows into its tool arguments.
  // Re-runs whenever the user uploads a new file or clones a different
  // job; identity-equality on ``parsedDataset`` is enough — every parse
  // path replaces the object.
  const lastStagedDatasetRef = useRef<ParsedDataset | null>(null);
  // Staged id that the current ``parsedDataset`` already corresponds to —
  // set when this wizard stages its own upload, or when it hydrates a dataset
  // the chat staged. Lets the incoming hydration effect below skip a dataset
  // we're already showing, and stops the outgoing effect from re-staging
  // (which would mint a *different* id for identical rows and ping-pong the
  // shared context against the panel).
  const parsedDatasetStagedIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (!wizardCtx) return;
    if (!parsedDataset || parsedDataset.rowCount === 0) {
      if (wizardCtx.state.staged_dataset_id !== undefined) {
        wizardCtx.clearField("staged_dataset_id");
      }
      lastStagedDatasetRef.current = null;
      parsedDatasetStagedIdRef.current = null;
      return;
    }
    if (lastStagedDatasetRef.current === parsedDataset) return;
    lastStagedDatasetRef.current = parsedDataset;
    let cancelled = false;
    stageDatasetForAgent({
      dataset: parsedDataset.rows as Array<Record<string, unknown>>,
      dataset_filename: datasetFileName || "dataset.json",
    })
      .then((res) => {
        if (cancelled) return;
        if (lastStagedDatasetRef.current !== parsedDataset) return;
        parsedDatasetStagedIdRef.current = res.staged_dataset_id;
        wizardCtx.setField("staged_dataset_id", res.staged_dataset_id, "user");
      })
      .catch(() => {
        if (cancelled) return;
        if (lastStagedDatasetRef.current === parsedDataset) {
          lastStagedDatasetRef.current = null;
        }
      });
    return () => {
      cancelled = true;
    };
  }, [parsedDataset, datasetFileName, wizardCtx]);

  // Incoming (chat → wizard): when the shared context points at a staged
  // dataset this wizard isn't already showing — e.g. the user attached a file
  // in the agent panel — fetch those exact rows and mirror them here. The
  // shared context survives client-side navigation (it lives in the app shell),
  // so this reliably rehydrates whether the wizard was already mounted or the
  // user navigated to /submit afterwards, with no dependence on sessionStorage.
  const hydratingStagedIdRef = useRef<string | null>(null);
  const sharedStagedId = sharedState?.staged_dataset_id;
  useEffect(() => {
    if (!wizardCtx) return;
    if (typeof sharedStagedId !== "string" || !sharedStagedId) return;
    if (sharedStagedId === parsedDatasetStagedIdRef.current) return;
    if (sharedStagedId === hydratingStagedIdRef.current) return;
    hydratingStagedIdRef.current = sharedStagedId;
    let cancelled = false;
    getStagedDataset(sharedStagedId)
      .then((res) => {
        if (cancelled || !res || res.rows.length === 0) return;
        const hydrated: ParsedDataset = {
          columns: res.columns.length > 0 ? res.columns : Object.keys(res.rows[0] ?? {}),
          rows: res.rows,
          rowCount: res.row_count,
        };
        // Mark the rows as already-staged under this id BEFORE setting them so
        // the outgoing staging effect early-returns instead of re-staging.
        parsedDatasetStagedIdRef.current = sharedStagedId;
        lastStagedDatasetRef.current = hydrated;
        setParsedDataset(hydrated);
        setDatasetFileName((prev) => prev ?? "dataset.json");
        setDatasetProfile(null);
        setSplitPlan(null);
        // The agent staged this dataset, so it owns code authoring for the
        // session (via request_code_authoring in the panel). Mark code as
        // already-authored so the wizard's own useCodeAgent auto-seed stands
        // down instead of authoring a second, racing Signature/Metric. The
        // agent's authored code then arrives as an applyAgentPatch.
        setSignatureManuallyEdited(true);
        setMetricManuallyEdited(true);
        const roles = wizardCtx.state.column_roles;
        if (roles && typeof roles === "object") {
          const next: Record<string, "input" | "output" | "ignore"> = {};
          for (const [col, role] of Object.entries(roles)) {
            if (role === "input" || role === "output" || role === "ignore") next[col] = role;
          }
          if (Object.keys(next).length > 0) setColumnRoles(next);
        }
      })
      .catch(() => {
        /* best-effort: a failed fetch leaves the wizard's own dataset intact */
      })
      .finally(() => {
        if (!cancelled && hydratingStagedIdRef.current === sharedStagedId) {
          hydratingStagedIdRef.current = null;
        }
      });
    return () => {
      cancelled = true;
    };
  }, [sharedStagedId, wizardCtx]);

  useEffect(() => {
    if (!wizardCtx) return;
    if (wizardCtx.state.job_name !== jobName) {
      wizardCtx.setField("job_name", jobName, "user");
    }
  }, [jobName, wizardCtx]);

  useEffect(() => {
    if (!wizardCtx) return;
    if (wizardCtx.state.signature_code !== signatureCode) {
      wizardCtx.setField("signature_code", signatureCode, "user");
    }
  }, [signatureCode, wizardCtx]);

  useEffect(() => {
    if (!wizardCtx) return;
    if (wizardCtx.state.metric_code !== metricCode) {
      wizardCtx.setField("metric_code", metricCode, "user");
    }
  }, [metricCode, wizardCtx]);

  useEffect(() => {
    if (!wizardCtx) return;
    const inputs = Object.values(columnRoles).filter((r) => r === "input").length;
    const outputs = Object.values(columnRoles).filter((r) => r === "output").length;
    const configured = inputs > 0 && outputs > 0;
    if (wizardCtx.state.columns_configured !== configured) {
      wizardCtx.setField("columns_configured", configured, "user");
    }
    const shared = wizardCtx.state.column_roles ?? {};
    const sameShape =
      Object.keys(shared).length === Object.keys(columnRoles).length &&
      Object.entries(columnRoles).every(([c, r]) => shared[c] === r);
    if (!sameShape && Object.keys(columnRoles).length > 0) {
      wizardCtx.setField("column_roles", columnRoles, "user");
    }
  }, [columnRoles, wizardCtx]);

  useEffect(() => {
    if (!wizardCtx) return;
    const configured = !!modelConfig.name.trim();
    if (wizardCtx.state.model_configured !== configured) {
      wizardCtx.setField("model_configured", configured, "user");
    }
    wizardCtx.setField("model_config", modelConfig as unknown as Record<string, unknown>, "user");
  }, [modelConfig, wizardCtx]);

  // Outgoing: scalar wizard fields the agent can read back for decisions.
  useEffect(() => {
    if (!wizardCtx) return;
    const s = wizardCtx.state;
    if (s.job_description !== jobDescription) {
      wizardCtx.setField("job_description", jobDescription, "user");
    }
    if (s.job_type !== jobType) {
      wizardCtx.setField("job_type", jobType, "user");
    }
    if (s.optimizer_name !== optimizerName) {
      wizardCtx.setField("optimizer_name", optimizerName, "user");
    }
    if (s.module_name !== moduleName) {
      wizardCtx.setField("module_name", moduleName, "user");
    }
    if (s.use_all_generation_models !== useAllGenerationModels) {
      wizardCtx.setField("use_all_generation_models", useAllGenerationModels, "user");
    }
    if (s.use_all_reflection_models !== useAllReflectionModels) {
      wizardCtx.setField("use_all_reflection_models", useAllReflectionModels, "user");
    }
    if (s.split_mode !== splitMode) {
      wizardCtx.setField("split_mode", splitMode, "user");
    }
    if (s.seed !== seed) {
      wizardCtx.setField("seed", seed, "user");
    }
    if (s.shuffle !== shuffle) {
      wizardCtx.setField("shuffle", shuffle, "user");
    }
    if (s.is_private !== isPrivate) {
      wizardCtx.setField("is_private", isPrivate, "user");
    }
  }, [
    jobDescription,
    jobType,
    optimizerName,
    moduleName,
    useAllGenerationModels,
    useAllReflectionModels,
    splitMode,
    seed,
    shuffle,
    isPrivate,
    wizardCtx,
  ]);

  // Outgoing: split fractions (compared component-wise to avoid object echo).
  useEffect(() => {
    if (!wizardCtx) return;
    const shared = wizardCtx.state.split_fractions;
    if (
      !shared ||
      shared.train !== split.train ||
      shared.val !== split.val ||
      shared.test !== split.test
    ) {
      wizardCtx.setField("split_fractions", split, "user");
    }
  }, [split, wizardCtx]);

  // Outgoing: object/array fields — setField's internal ref-dedupe keeps these cheap.
  useEffect(() => {
    if (!wizardCtx) return;
    wizardCtx.setField(
      "reflection_model_config",
      (secondModelConfig ?? undefined) as Record<string, unknown> | undefined,
      "user",
    );
  }, [secondModelConfig, wizardCtx]);

  useEffect(() => {
    if (!wizardCtx) return;
    wizardCtx.setField(
      "generation_models",
      generationModels as unknown as Array<Record<string, unknown>>,
      "user",
    );
  }, [generationModels, wizardCtx]);

  useEffect(() => {
    if (!wizardCtx) return;
    wizardCtx.setField(
      "reflection_models",
      reflectionModels as unknown as Array<Record<string, unknown>>,
      "user",
    );
  }, [reflectionModels, wizardCtx]);

  // Outgoing: react config (minus the secret mcp_auth_header) so the agent and
  // clone path can read/drive it. JSON-compared because a fresh object is built
  // each render; the secret never enters shared state.
  useEffect(() => {
    if (!wizardCtx) return;
    const { mcpAuthHeader: _omit, ...shareable } = reactConfig;
    const shared = wizardCtx.state.react_config;
    if (JSON.stringify(shared ?? null) !== JSON.stringify(shareable)) {
      wizardCtx.setField("react_config", shareable as Record<string, unknown>, "user");
    }
  }, [reactConfig, wizardCtx]);

  // Outgoing: optimizer_kwargs — rebuild from the quartet and compare entries.
  useEffect(() => {
    if (!wizardCtx) return;
    const kw = buildOptimizerKwargs({
      autoLevel,
      maxFullEvals,
      reflectionMinibatchSize,
      useMerge,
    });
    const shared = wizardCtx.state.optimizer_kwargs ?? {};
    const kwEntries = Object.entries(kw);
    const same =
      Object.keys(shared).length === kwEntries.length &&
      kwEntries.every(([k, v]) => shared[k] === v);
    if (!same) {
      wizardCtx.setField("optimizer_kwargs", kw, "user");
    }
  }, [autoLevel, maxFullEvals, reflectionMinibatchSize, useMerge, wizardCtx]);

  // Chat-driven dataset staging: when the user attaches a CSV/JSON/XLSX
  // file in the agent panel and confirms the column roles, the panel
  // dispatches ``wizard:dataset-staged`` with ``{dataset, dataset_filename,
  // wizard_state: {dataset_columns, column_roles, column_kinds}}``. The
  // panel ALSO stashes the same payload in sessionStorage under
  // ``wizard:staged-dataset`` so navigating from /explore to /submit
  // doesn't drop the event when the wizard hasn't mounted yet.
  useEffect(() => {
    const applyStaged = (detail: unknown) => {
      if (!detail || typeof detail !== "object") return;
      const d = detail as Record<string, unknown>;
      const rows = d.dataset;
      const ws = (d.wizard_state ?? {}) as Record<string, unknown>;
      if (!Array.isArray(rows) || rows.length === 0) return;
      const stagedColumns = ws.dataset_columns;
      const columns =
        Array.isArray(stagedColumns) && stagedColumns.length > 0
          ? (stagedColumns as string[])
          : Object.keys((rows[0] ?? {}) as Record<string, unknown>);
      const parsed: ParsedDataset = {
        columns,
        rows: rows as Array<Record<string, unknown>>,
        rowCount: rows.length,
      };
      // This same-page fast path already carries the rows; record the staged
      // id (when present) so the outgoing effect doesn't re-stage identical
      // rows under a new id and the incoming hydration effect skips a refetch.
      const eventStagedId = typeof d.staged_dataset_id === "string" ? d.staged_dataset_id : null;
      if (eventStagedId) parsedDatasetStagedIdRef.current = eventStagedId;
      lastStagedDatasetRef.current = parsed;
      setParsedDataset(parsed);
      const explicitFilename =
        typeof d.dataset_filename === "string" && d.dataset_filename ? d.dataset_filename : null;
      const jobBasedFilename =
        typeof ws.job_name === "string" && ws.job_name ? `${ws.job_name}.json` : null;
      setDatasetFileName(explicitFilename ?? jobBasedFilename ?? "sample.json");
      const stagedDefaultMode = readPref("wizardSplitMode");
      splitModeRef.current = stagedDefaultMode;
      setSplitModeState(stagedDefaultMode);
      setDatasetProfile(null);
      setSplitPlan(null);
      setSignatureValidation(null);
      setMetricValidation(null);
      // This dataset was staged by the agent panel; the agent owns code
      // authoring (request_code_authoring), so suppress the wizard's own
      // auto-seed to avoid a second, racing Signature/Metric. The agent's
      // authored code lands here via an applyAgentPatch.
      setSignatureManuallyEdited(true);
      setMetricManuallyEdited(true);
      if (ws.column_kinds && typeof ws.column_kinds === "object") {
        const stagedKinds: Record<string, "text" | "image"> = {};
        for (const [col, kind] of Object.entries(ws.column_kinds as Record<string, unknown>)) {
          stagedKinds[col] = kind === "image" ? "image" : "text";
        }
        setColumnKinds(stagedKinds);
      }
      if (ws.column_roles && typeof ws.column_roles === "object") {
        const stagedRoles: Record<string, "input" | "output" | "ignore"> = {};
        for (const [col, role] of Object.entries(ws.column_roles as Record<string, unknown>)) {
          if (role === "input" || role === "output" || role === "ignore") {
            stagedRoles[col] = role;
          }
        }
        if (Object.keys(stagedRoles).length > 0) setColumnRoles(stagedRoles);
      }
    };

    if (typeof window !== "undefined") {
      try {
        const raw = window.sessionStorage.getItem("wizard:staged-dataset");
        if (raw) {
          window.sessionStorage.removeItem("wizard:staged-dataset");
          applyStaged(JSON.parse(raw));
        }
      } catch {
        window.sessionStorage.removeItem("wizard:staged-dataset");
      }
    }

    const handler = (e: Event) => applyStaged((e as CustomEvent).detail);
    window.addEventListener("wizard:dataset-staged", handler);
    return () => window.removeEventListener("wizard:dataset-staged", handler);
  }, []);

  // Auto-seed columnKinds from the dataset profile. The profiler returns
  // ``inputs: [{name, kind}]`` once it has classified each input column; we
  // fill any column the user hasn't already overridden. A fresh dataset
  // upload clears columnKinds entirely, so this effect re-seeds on every
  // new file. Columns the user has manually flipped stay flipped.
  useEffect(() => {
    if (!datasetProfile?.inputs?.length) return;
    setColumnKinds((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const entry of datasetProfile.inputs) {
        if (next[entry.name] === undefined) {
          next[entry.name] = entry.kind === "image" ? "image" : "text";
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [datasetProfile]);

  // Auto-update signature template when column roles or modalities change.
  // ``columnKinds`` flips an input column to ``dspy.Image`` so the auto
  // template tracks the modality toggle without waiting for the AI agent.
  useEffect(() => {
    if (signatureManuallyEdited) return;
    const hasRoles = Object.values(columnRoles).some((r) => r === "input" || r === "output");
    if (!hasRoles) return;
    setSignatureCode(buildSignatureTemplate(columnRoles, columnKinds));
  }, [columnRoles, columnKinds, signatureManuallyEdited]);

  // Auto-update metric template when output roles change. The agent
  // overwrites this in auto mode; the manual-edit flag protects a user's
  // edits from being clobbered when they toggle between columns afterwards.
  useEffect(() => {
    if (metricManuallyEdited) return;
    const hasOutputs = Object.values(columnRoles).some((r) => r === "output");
    if (!hasOutputs) return;
    setMetricCode(buildMetricTemplate(columnRoles));
  }, [columnRoles, metricManuallyEdited]);

  useEffect(() => {
    const cloneId = searchParams.get("clone");
    if (!cloneId || cloneRan.current) return;
    cloneRan.current = true;
    const pairParam = searchParams.get("pair");
    const clonePairIndex = pairParam == null ? null : Number(pairParam);
    const shareToken = searchParams.get("shareToken");
    // Forking a public Explore run: there's no share token, so hydrate from the
    // by-id scrubbed public composite instead of the authed owner endpoints.
    const publicClone = searchParams.get("public") === "1";
    setCloneLoading(true);

    // Maps a cloned optimization payload into wizard state. Shared between the
    // authed clone path and the token-gated share clone path; the latter passes
    // a synthetic payload whose ``dataset`` is rebuilt from the public split.
    const applyClone = (
      optimization_type: string,
      payload: Record<string, unknown>,
      jobData: Awaited<ReturnType<typeof getJob>> | null,
    ) => {
      const clonePair =
        Number.isInteger(clonePairIndex) && jobData?.grid_result
          ? (jobData.grid_result.pair_results.find((p) => p.pair_index === clonePairIndex) ?? null)
          : null;
      setOptimizationType(
        clonePair ? "run" : optimization_type === "grid_search" ? "grid_search" : "run",
      );

      const displayName = jobData?.name || payload.name;
      if (displayName) setJobName(String(displayName));
      if (payload.description) setJobDescription(String(payload.description));
      if (payload.module_name) setModuleName(String(payload.module_name));
      if (payload.optimizer_name) setOptimizerName(String(payload.optimizer_name));
      if (payload.signature_code) {
        setSignatureCode(String(payload.signature_code));
        setSignatureManuallyEdited(true);
      }
      if (payload.metric_code) {
        setMetricCode(String(payload.metric_code));
        setMetricManuallyEdited(true);
      }

      if (Array.isArray(payload.dataset) && payload.dataset.length > 0) {
        const rows = payload.dataset as Array<Record<string, unknown>>;
        const rowKeys = Object.keys(rows[0] ?? {});
        // Restore the submitted column order from the persisted array (the
        // rows' own key order is scrambled by JSONB). Fall back to row keys,
        // and append any columns the saved order didn't cover.
        const savedOrder = Array.isArray(payload.column_order)
          ? (payload.column_order as string[]).filter((c) => rowKeys.includes(c))
          : [];
        const columns =
          savedOrder.length > 0
            ? [...savedOrder, ...rowKeys.filter((c) => !savedOrder.includes(c))]
            : rowKeys;
        setParsedDataset({ columns, rows, rowCount: rows.length });
        setDatasetFileName(
          String(
            (payload as Record<string, unknown>).dataset_filename || displayName || cloneId || "",
          ),
        );
      }

      const cm = payload.column_mapping as
        | { inputs?: Record<string, string>; outputs?: Record<string, string> }
        | undefined;
      if (cm) {
        const roles: Record<string, "input" | "output" | "ignore"> = {};
        if (cm.inputs)
          Object.keys(cm.inputs).forEach((k) => {
            roles[k] = "input";
          });
        if (cm.outputs)
          Object.keys(cm.outputs).forEach((k) => {
            roles[k] = "output";
          });
        setColumnRoles(roles);
      }

      const sf = payload.split_fractions as
        | { train?: number; val?: number; test?: number }
        | undefined;
      if (sf) {
        setSplit({ train: sf.train ?? 0.7, val: sf.val ?? 0.15, test: sf.test ?? 0.15 });
        // Cloned splits are intentional — pin the wizard to manual so the
        // auto-profile effect doesn't clobber them when the dataset reloads.
        splitModeRef.current = "manual";
        setSplitModeState("manual");
      }

      if (payload.shuffle != null) setShuffle(Boolean(payload.shuffle));
      if (payload.seed != null) setSeed(Number(payload.seed));
      if (payload.is_private != null) setIsPrivate(Boolean(payload.is_private));

      if (clonePair) {
        const findPairModel = (
          models: ModelConfig[] | undefined,
          name: string,
          reasoningEffort?: string | null,
        ) => {
          const match = models?.find(
            (model) =>
              model.name === name &&
              (!reasoningEffort || model.extra?.reasoning_effort === reasoningEffort),
          );
          if (match) return { ...emptyModelConfig(), ...match };
          return {
            ...emptyModelConfig(),
            name,
            extra: reasoningEffort ? { reasoning_effort: reasoningEffort } : undefined,
          };
        };
        setModelConfig(
          findPairModel(
            payload.generation_models as ModelConfig[] | undefined,
            clonePair.generation_model,
            clonePair.generation_reasoning_effort,
          ),
        );
        setSecondModelConfig(
          findPairModel(
            payload.reflection_models as ModelConfig[] | undefined,
            clonePair.reflection_model,
            clonePair.reflection_reasoning_effort,
          ),
        );
        setUseAllGenerationModels(false);
        setUseAllReflectionModels(false);
      } else {
        const mc = payload.model_config as ModelConfig | undefined;
        if (mc) setModelConfig({ ...emptyModelConfig(), ...mc });

        const smc = (payload.reflection_model_config ?? payload.task_model_config) as
          | ModelConfig
          | undefined;
        if (smc?.name) setSecondModelConfig({ ...emptyModelConfig(), ...smc });

        const gm = payload.generation_models as ModelConfig[] | undefined;
        if (gm?.length) setGenerationModels(gm.map((m) => ({ ...emptyModelConfig(), ...m })));

        const rm = payload.reflection_models as ModelConfig[] | undefined;
        if (rm?.length) setReflectionModels(rm.map((m) => ({ ...emptyModelConfig(), ...m })));

        if (payload.use_all_available_generation_models) setUseAllGenerationModels(true);
        if (payload.use_all_available_reflection_models) setUseAllReflectionModels(true);
      }

      const optKw = payload.optimizer_kwargs as Record<string, unknown> | undefined;
      if (optKw) {
        if (optKw.auto) setAutoLevel(String(optKw.auto));
        if (optKw.reflection_minibatch_size != null)
          setReflectionMinibatchSize(String(optKw.reflection_minibatch_size));
        if (optKw.max_full_evals != null) setMaxFullEvals(String(optKw.max_full_evals));
        if (optKw.use_merge != null) setUseMerge(Boolean(optKw.use_merge));
      }

      // React run config — hydrate tool source from the wire model. Scoring is
      // owned by metric_code (hydrated above), so there is no reward to restore.
      // mcp_auth_header is scrubbed from cloned/shared payloads, never present.
      const ts = payload.tool_source as Record<string, unknown> | undefined;
      if (ts) {
        setReactConfig((prev) => {
          const next = { ...prev };
          if (ts.kind) next.toolSourceKind = String(ts.kind) as ReactConfig["toolSourceKind"];
          if (ts.mcp_url != null) next.mcpUrl = String(ts.mcp_url);
          if (Array.isArray(ts.tool_filter)) next.toolFilter = ts.tool_filter.join(", ");
          return next;
        });
      }
      toast.success(msg("submit.clone.success"));
    };

    // Share / public clone: hydrate from the scrubbed composite — token-gated for
    // a share link, or by id for a public Explore run. Both return the same
    // shape; the payload is scrubbed (no api_key/base_url/username) and carries no
    // dataset rows, so reconstruct them from the full train/val/test split.
    const scrubbedComposite = shareToken
      ? getSharedOptimization(shareToken)
      : publicClone
        ? getPublicOptimization(cloneId)
        : null;
    const source = scrubbedComposite
      ? scrubbedComposite.then((shared) => {
          const splits = shared.dataset?.splits;
          const rows = splits
            ? [...splits.train, ...splits.val, ...splits.test]
                .sort((a, b) => a.index - b.index)
                .map((entry) => entry.row)
            : [];
          const payload: Record<string, unknown> = {
            ...shared.payload,
            ...(rows.length > 0 ? { dataset: rows } : {}),
            // The dataset endpoint carries the mapping too; fall back to it so
            // column roles hydrate even if the scrubbed payload omitted it.
            ...(shared.payload?.column_mapping == null && shared.dataset?.column_mapping
              ? { column_mapping: shared.dataset.column_mapping }
              : {}),
          };
          applyClone(shared.status.optimization_type, payload, shared.status);
        })
      : Promise.all([getOptimizationPayload(cloneId), getJob(cloneId).catch(() => null)]).then(
          ([{ optimization_type, payload }, jobData]) => {
            applyClone(optimization_type, payload as Record<string, unknown>, jobData);
          },
        );

    source
      .catch(() => {
        toast.error(msg("submit.clone.failed"));
      })
      .finally(() => setCloneLoading(false));
  }, []);

  const goNext = () => {
    if (step < STEPS.length - 1) {
      setDirection(1);
      setStep((s) => {
        const next = s + 1;
        setFurthestReachedStep((prev) => Math.max(prev, next));
        return next;
      });
    }
  };
  const goPrev = () => {
    if (step > 0) {
      setDirection(-1);
      setStep((s) => s - 1);
    }
  };
  const goTo = (idx: number) => {
    setDirection(idx > step ? 1 : -1);
    setStep(idx);
    setFurthestReachedStep((prev) => Math.max(prev, idx));
  };

  const currentColumnMapping = () => buildColumnMapping(columnRoles);

  useDatasetProfiling({
    parsedDataset,
    columnRoles,
    splitModeRef,
    setDatasetProfile,
    setSplitPlan,
    setProfileLoading,
    setSplit,
    setShuffle,
    setSeed,
  });

  const setSplitMode = useCallback(
    (mode: "auto" | "manual") => {
      splitModeRef.current = mode;
      setSplitModeState(mode);
      if (mode === "auto" && splitPlan) {
        setSplit(splitPlan.fractions);
        setShuffle(splitPlan.shuffle);
        setSeed(splitPlan.seed);
      }
    },
    [splitPlan],
  );

  /** Validates a wizard step; optionally surfaces toast errors. */
  const validateStep = (s: number, showToast = false): boolean => {
    switch (s) {
      case 0:
        if (!username.trim()) {
          if (showToast) toast.error(msg("submit.validation.username_required"));
          return false;
        }
        if (!jobName.trim()) {
          if (showToast) toast.error(msg("submit.validation.name_required"));
          return false;
        }
        return true;
      case 1: {
        if (!parsedDataset || parsedDataset.rowCount === 0) {
          if (showToast) toast.error(msg("submit.validation.dataset_required"));
          return false;
        }
        const m = currentColumnMapping();
        if (Object.keys(m.inputs).length === 0) {
          if (showToast) toast.error(msg("submit.validation.input_column_required"));
          return false;
        }
        if (Object.keys(m.outputs).length === 0) {
          if (showToast) toast.error(msg("submit.validation.output_column_required"));
          return false;
        }
        return true;
      }
      case 2:
        // React live-MCP runs need a tool endpoint; gate it here so the empty
        // URL is caught when leaving the params step instead of only at submit.
        if (isReact && reactConfig.toolSourceKind === "live_mcp" && !reactConfig.mcpUrl.trim()) {
          if (showToast) toast.error(msg("submit.validation.mcp_url_required"));
          return false;
        }
        if (datasetValidation && datasetValidation.errors.length > 0) {
          if (showToast) toast.error(msg("submit.validation.split_too_small"));
          return false;
        }
        return true;
      case 3:
        if (!signatureCode.trim()) {
          if (showToast) toast.error(msg("submit.validation.signature_required"));
          return false;
        }
        if (!signatureValidation || signatureValidation.errors.length > 0) {
          return false;
        }
        if (!metricCode.trim()) {
          if (showToast) toast.error(msg("submit.validation.metric_required"));
          return false;
        }
        if (!metricValidation || metricValidation.errors.length > 0) {
          return false;
        }
        return true;
      case 4: {
        if (jobType === "run") {
          if (!modelConfig.name.trim()) {
            if (showToast) toast.error(msg("submit.validation.model_required"));
            return false;
          }
          // Require api_key if provider has no env default AND no global key
          const hasApiKey = !!globalApiKey || !!modelConfig.extra?.api_key;
          if (!anyProviderHasEnvKey && !hasApiKey) {
            if (showToast) toast.error(msg("submit.validation.api_key_required"));
            return false;
          }
          const secondModel = secondModelConfig;
          if (!secondModel?.name?.trim()) {
            if (showToast) toast.error(msg("submit.validation.reflection_model_required"));
            return false;
          }
        }
        if (jobType === "grid_search") {
          if (!useAllGenerationModels && generationModels.every((m) => !m.name.trim())) {
            if (showToast) toast.error(msg("submit.validation.generation_model_required"));
            return false;
          }
          if (!useAllReflectionModels && reflectionModels.every((m) => !m.name.trim())) {
            if (showToast) toast.error(msg("submit.validation.reflection_models_required"));
            return false;
          }
          if (
            (useAllGenerationModels || useAllReflectionModels) &&
            (catalog?.models.length ?? 0) === 0
          ) {
            if (showToast) toast.error(msg("submit.validation.no_models_available"));
            return false;
          }
        }
        // Vision gate: if any input column is image-typed, every chosen
        // generation model must support vision. Mirrors the backend's
        // ``submission.vision_required`` rejection so we fail fast in the UI
        // instead of waiting for a 400 from /run.
        const imageInputs = Object.entries(columnKinds)
          .filter(([col, kind]) => kind === "image" && columnRoles[col] === "input")
          .map(([col]) => col);
        if (imageInputs.length > 0 && catalog?.models?.length) {
          const visionByValue = new Map(catalog.models.map((m) => [m.value, m.supports_vision]));
          const isVision = (id: string): boolean => visionByValue.get(id) ?? false;
          const candidates: string[] =
            jobType === "run"
              ? [modelConfig.name].filter((n) => n.trim())
              : useAllGenerationModels
                ? catalog.models.filter((m) => m.available).map((m) => m.value)
                : generationModels.map((m) => m.name).filter((n) => n.trim());
          const offenders = candidates.filter((id) => !isVision(id));
          if (offenders.length > 0) {
            if (showToast) {
              toast.error(
                formatMsg("submit.validation.vision_required", {
                  fields: imageInputs.join(", "),
                  model: offenders.join(", "),
                }),
              );
            }
            return false;
          }
        }
        return true;
      }
      default:
        return true;
    }
  };

  const maxReachableStep = furthestReachedStep;

  const validateBlock = async (
    kind: "signature" | "metric",
    overrideCode?: string,
  ): Promise<ValidateCodeResponse | EditorValidationResult> => {
    const code = overrideCode ?? (kind === "signature" ? signatureCode : metricCode);
    if (!code.trim()) {
      return { valid: false, errors: [`Missing ${kind} code`], warnings: [] };
    }
    if (!parsedDataset || parsedDataset.rowCount === 0) {
      return { valid: false, errors: ["Upload a dataset before validating code"], warnings: [] };
    }
    const mapping = currentColumnMapping();
    const sampleRow = parsedDataset.rows[0] as Record<string, unknown>;
    const cacheKey = JSON.stringify([kind, code, mapping, sampleRow, optimizerName, moduleName]);
    const cached = validationCacheRef.current.get(cacheKey);
    if (cached) return cached;
    const pending = validateCode({
      signature_code: kind === "signature" ? code : undefined,
      metric_code: kind === "metric" ? code : undefined,
      column_mapping: mapping,
      sample_row: sampleRow,
      optimizer_name: optimizerName,
      module_name: moduleName,
    });
    validationCacheRef.current.set(cacheKey, pending);
    pending.catch(() => validationCacheRef.current.delete(cacheKey));
    return pending;
  };

  const runSignatureValidation = async (
    overrideCode?: string,
  ): Promise<EditorValidationResult | null> => {
    try {
      const result = await validateBlock("signature", overrideCode);
      setSignatureValidation(result as ValidateCodeResponse);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Signature validation failed";
      return { valid: false, errors: [errorMessage], warnings: [] };
    }
  };

  const runMetricValidation = async (
    overrideCode?: string,
  ): Promise<EditorValidationResult | null> => {
    try {
      const result = await validateBlock("metric", overrideCode);
      setMetricValidation(result as ValidateCodeResponse);
      return result;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Metric validation failed";
      return { valid: false, errors: [errorMessage], warnings: [] };
    }
  };

  const handleValidateCode = async (): Promise<boolean> => {
    if (!parsedDataset || parsedDataset.rowCount === 0) {
      toast.error(msg("submit.validation.dataset_before_code"));
      return false;
    }
    try {
      const [sigRes, metRes] = await Promise.all([
        signatureCode.trim() ? runSignatureValidation() : Promise.resolve(null),
        metricCode.trim() ? runMetricValidation() : Promise.resolve(null),
      ]);
      const sigOk = !sigRes || sigRes.errors.length === 0;
      const metOk = !metRes || metRes.errors.length === 0;
      if (sigOk && metOk) return true;
      toast.error(msg("submit.validation.code_has_errors"));
      return false;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("submit.code_validation_failed"));
      return false;
    }
  };

  const handleValidateDataset = async (): Promise<boolean> => {
    if (!parsedDataset || parsedDataset.rowCount === 0) {
      toast.error(msg("submit.validation.dataset_required"));
      return false;
    }
    const effectiveFractions =
      splitModeRef.current === "auto" && splitPlan ? splitPlan.fractions : split;
    const sum = effectiveFractions.train + effectiveFractions.val + effectiveFractions.test;
    if (Math.abs(sum - 1) > 0.001) {
      toast.error(msg("submit.validation.split_must_sum_to_one"));
      return false;
    }
    try {
      const result = await validateDataset({
        row_count: parsedDataset.rowCount,
        fractions: effectiveFractions,
      });
      setDatasetValidation(result);
      if (result.errors.length === 0) return true;
      toast.error(msg("submit.validation.split_too_small"));
      return false;
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("submit.validation.split_too_small"));
      return false;
    }
  };

  const handleNext = async () => {
    if (advancingRef.current) return;
    advancingRef.current = true;
    setAdvancing(true);
    try {
      if (step === 2 && parsedDataset && parsedDataset.rowCount > 0) {
        const passed = await handleValidateDataset();
        if (!passed) return;
      }
      if (step === 3 && signatureCode.trim() && parsedDataset && metricCode.trim()) {
        const passed = await handleValidateCode();
        if (!passed) return;
      }
      if (validateStep(step, true)) goNext();
    } finally {
      advancingRef.current = false;
      setAdvancing(false);
    }
  };

  const handleTabClick = (idx: number) => {
    if (idx <= step) {
      goTo(idx);
      return;
    }
    if (idx <= maxReachableStep) {
      goTo(idx);
      return;
    }
    validateStep(step, true);
  };

  const handleFileUpload = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const parsed = await parseDatasetFile(file);
      setParsedDataset(parsed);
      setDatasetFileName(file.name);
      const roles: Record<string, "input" | "output" | "ignore"> = {};
      parsed.columns.forEach((col) => {
        roles[col] = "input";
      });
      setColumnRoles(roles);
      // Drop any prior modality overrides — the profiler effect will re-seed
      // from the new dataset's detected kinds.
      setColumnKinds({});
      // A fresh dataset deserves a fresh recommendation — reset split mode to
      // the user's saved preference so the auto-profile effect applies the new
      // plan when they prefer auto, and stays out of the way when they prefer
      // manual.
      const uploadDefaultMode = readPref("wizardSplitMode");
      splitModeRef.current = uploadDefaultMode;
      setSplitModeState(uploadDefaultMode);
      setDatasetProfile(null);
      setSplitPlan(null);
      // A new dataset invalidates any cloned or user-authored code — clear
      // the manual-edit flags so the template effects and the code agent
      // can repopulate for the new schema.
      setSignatureManuallyEdited(false);
      setMetricManuallyEdited(false);
      setSignatureValidation(null);
      setMetricValidation(null);
      toast.success(
        formatMsg("auto.features.submit.hooks.use.submit.wizard.template.1", {
          p1: parsed.rowCount,
          p2: file.name,
        }),
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("submit.dataset.file_error"));
    }
  }, []);

  const handlePickFromLibrary = useCallback(async (dataset: DatasetSummary) => {
    try {
      const res = await getDatasetRows(dataset.id);
      if (res.rows.length === 0) {
        toast.error(msg("submit.dataset.library_empty"));
        return;
      }
      const columns = res.columns.length > 0 ? res.columns : Object.keys(res.rows[0] ?? {});
      const parsed: ParsedDataset = { columns, rows: res.rows, rowCount: res.row_count };
      // Bind the reference to this exact object so handleSubmit can tell a live
      // library pick from a later upload/clone by identity (see librarySourceRef).
      librarySourceRef.current = { id: dataset.id, parsed };
      setParsedDataset(parsed);
      setDatasetFileName(dataset.name);
      // Restore the saved roles; default any column the schema didn't cover to
      // input so the column step opens fully populated.
      const savedRoles = res.column_schema.column_roles ?? {};
      const roles: Record<string, "input" | "output" | "ignore"> = {};
      columns.forEach((col) => {
        roles[col] = savedRoles[col] ?? "input";
      });
      setColumnRoles(roles);
      // Seed saved modality kinds; the profiler effect only fills gaps, so these
      // survive its auto-detection pass.
      setColumnKinds(res.column_schema.column_kinds ?? {});
      const pickedDefaultMode = readPref("wizardSplitMode");
      splitModeRef.current = pickedDefaultMode;
      setSplitModeState(pickedDefaultMode);
      setDatasetProfile(null);
      setSplitPlan(null);
      setSignatureManuallyEdited(false);
      setMetricManuallyEdited(false);
      setSignatureValidation(null);
      setMetricValidation(null);
      toast.success(
        formatMsg("submit.dataset.library_loaded", {
          name: dataset.name,
          count: parsed.rowCount,
        }),
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("submit.dataset.file_error"));
    }
  }, []);

  const updateSplit = (field: keyof SplitFractions, value: string) => {
    if (splitModeRef.current === "auto") return;
    const num = parseFloat(value);
    if (isNaN(num) || num < 0 || num > 1) return;
    setSplit((prev) => ({ ...prev, [field]: num }));
  };
  const splitSum = +(split.train + split.val + split.test).toFixed(4);

  const handleSubmit = async () => {
    if (!username.trim()) {
      toast.error(msg("submit.validation.username_required"));
      goTo(0);
      return;
    }
    if (!parsedDataset || parsedDataset.rowCount === 0) {
      toast.error(msg("submit.validation.dataset_required_short"));
      goTo(1);
      return;
    }
    if (!signatureCode.trim()) {
      toast.error(msg("submit.validation.signature_required"));
      goTo(3);
      return;
    }
    if (!metricCode.trim()) {
      toast.error(msg("submit.validation.metric_required"));
      goTo(3);
      return;
    }
    if (isReact && reactConfig.toolSourceKind === "live_mcp" && !reactConfig.mcpUrl.trim()) {
      toast.error(msg("submit.validation.mcp_url_required"));
      goTo(2);
      return;
    }

    const columnMapping = currentColumnMapping();
    if (Object.keys(columnMapping.inputs).length === 0) {
      toast.error(msg("submit.validation.input_column_required"));
      goTo(1);
      return;
    }
    if (Object.keys(columnMapping.outputs).length === 0) {
      toast.error(msg("submit.validation.output_column_required"));
      goTo(1);
      return;
    }

    setSubmitting(true);
    setSubmitPhase("sending");
    try {
      const optKw = buildOptimizerKwargs({
        autoLevel,
        maxFullEvals,
        reflectionMinibatchSize,
        useMerge,
      });
      // Submit by reference when the on-screen rows are still the library dataset
      // we loaded — the server inlines the rows and records the link back to it.
      // Any other dataset source replaced the object identity, so fall back to
      // sending the rows inline. The two are mutually exclusive server-side.
      const librarySourceId =
        librarySourceRef.current && librarySourceRef.current.parsed === parsedDataset
          ? librarySourceRef.current.id
          : null;
      // Persist the chargeable bracket the user just saw so the post-run proof
      // moment can reconcile it against the actual charge. Same bracket the cost
      // surface and review recap showed (managed: full per-model; byok: fee).
      const estimate = chargeableBracket(costBracket, wallet.mode);
      const base = {
        name: jobName.trim() || undefined,
        description: jobDescription.trim() || undefined,
        username: username.trim(),
        module_name: moduleName,
        signature_code: signatureCode,
        metric_code: metricCode,
        optimizer_name: optimizerName,
        ...(librarySourceId
          ? { source_dataset_id: librarySourceId }
          : { dataset: parsedDataset.rows as Array<Record<string, unknown>> }),
        dataset_filename: datasetFileName || undefined,
        column_mapping: columnMapping,
        // Preserve the on-screen column order (file order) so a clone restores
        // it — JSONB would otherwise scramble the dataset's object-key order.
        column_order: parsedDataset.columns,
        split_fractions: split,
        shuffle,
        is_private: isPrivate,
        token_source: wallet.mode,
        estimated_credits_low: estimate.lowCredits,
        estimated_credits_high: estimate.highCredits,
        ...(maxCostCredits != null && { max_cost_credits: maxCostCredits }),
        ...(seed != null && { seed }),
        ...(Object.keys(optKw).length > 0 && { optimizer_kwargs: optKw }),
      };

      // Reshape the flat UI react config into the backend ToolSource wire
      // model. mcp_auth_header is forwarded once on the wire but never persisted
      // (backend) or mirrored into shared agent state.
      const buildReactFields = (): { tool_source: ToolSource } => {
        const filter = reactConfig.toolFilter
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
        const tool_source: ToolSource = {
          kind: reactConfig.toolSourceKind,
          ...(reactConfig.toolSourceKind === "live_mcp" && reactConfig.mcpUrl.trim()
            ? { mcp_url: reactConfig.mcpUrl.trim() }
            : {}),
          ...(reactConfig.mcpAuthHeader.trim()
            ? { mcp_auth_header: reactConfig.mcpAuthHeader.trim() }
            : {}),
          ...(filter.length > 0 ? { tool_filter: filter } : {}),
        };
        return { tool_source };
      };

      // Inject global API key + base URL into any model config that doesn't override
      const applyGlobals = (mc: ModelConfig): ModelConfig => {
        const out = { ...mc };
        if (globalBaseUrl && !out.base_url) out.base_url = globalBaseUrl;
        if (globalApiKey && !out.extra?.api_key)
          out.extra = { ...out.extra, api_key: globalApiKey };
        return out;
      };

      let result;
      if (jobType === "run") {
        if (!modelConfig.name.trim()) {
          toast.error(msg("submit.validation.model_required"));
          goTo(4);
          setSubmitting(false);
          setSubmitPhase("idle");
          return;
        }
        const secondApplied = secondModelConfig?.name?.trim()
          ? applyGlobals(secondModelConfig)
          : undefined;
        const runPayload: RunRequest = {
          ...base,
          model_config: applyGlobals(modelConfig),
          ...(secondApplied ? { reflection_model_config: secondApplied } : {}),
          ...(isReact ? buildReactFields() : {}),
        };
        result = await submitRun(runPayload);
        track(TelemetryEvent.RunSubmitted, {
          react: isReact,
          has_reflection: Boolean(secondApplied),
        });
      } else {
        const validGen = generationModels.filter((m) => m.name.trim()).map(applyGlobals);
        const validRef = reflectionModels.filter((m) => m.name.trim()).map(applyGlobals);
        if (!useAllGenerationModels && validGen.length === 0) {
          toast.error(msg("submit.validation.generation_model_required"));
          goTo(4);
          setSubmitting(false);
          setSubmitPhase("idle");
          return;
        }
        if (!useAllReflectionModels && validRef.length === 0) {
          toast.error(msg("submit.validation.reflection_models_required"));
          goTo(4);
          setSubmitting(false);
          setSubmitPhase("idle");
          return;
        }
        if (
          (useAllGenerationModels || useAllReflectionModels) &&
          (catalog?.models.length ?? 0) === 0
        ) {
          toast.error(msg("submit.validation.no_models_available"));
          goTo(4);
          setSubmitting(false);
          setSubmitPhase("idle");
          return;
        }
        result = await submitGridSearch({
          ...base,
          generation_models: useAllGenerationModels ? [] : validGen,
          reflection_models: useAllReflectionModels ? [] : validRef,
          ...(useAllGenerationModels && { use_all_available_generation_models: true }),
          ...(useAllReflectionModels && { use_all_available_reflection_models: true }),
        });
        track(TelemetryEvent.GridSearchSubmitted, {
          generation_models: validGen.length,
          reflection_models: validRef.length,
          use_all_generation: useAllGenerationModels,
          use_all_reflection: useAllReflectionModels,
        });
      }

      // Wipe any pasted API key from the in-memory wizard state right
      // after the submit succeeds. The wire payload already left the
      // browser; keeping the plaintext in React state would leak it to
      // the next submit, the recents cache, and any debug surface.
      const stripKey = (cfg: ModelConfig): ModelConfig => {
        if (!cfg.extra || !("api_key" in cfg.extra)) return cfg;
        const { api_key: _omit, ...rest } = cfg.extra;
        return { ...cfg, extra: Object.keys(rest).length > 0 ? rest : undefined };
      };
      setModelConfig(stripKey);
      setSecondModelConfig((prev) => (prev ? stripKey(prev) : prev));
      setGenerationModels((prev) => prev.map(stripKey));
      setReflectionModels((prev) => prev.map(stripKey));

      // Mark the submit so the unmount cleanup clears the shared wizard state
      // once navigation tears this form down.
      submittedRef.current = true;
      const jobUrl = `/optimizations/${result.optimization_id}`;
      setSubmitPhase("splash");
      // Collapse sidebar before navigating so the job page opens with full width
      window.dispatchEvent(new Event("sidebar:collapse"));
      setTimeout(() => {
        setSubmitPhase("done");
        router.push(jobUrl);
      }, 1500);
    } catch (err) {
      // The storage-budget 409 and the credit-gate 402 each open their own shared
      // modal centrally; suppress the redundant toast so the modal is the single
      // surface for both.
      if (!isStorageQuotaError(err) && !isInsufficientCreditsError(err)) {
        toast.error(err instanceof Error ? err.message : msg("submit.submit_failed"));
      }
      setSubmitPhase("idle");
      setSubmitting(false);
    }
  };

  // Hoisted to wizard scope so the seed pass fires as soon as the user
  // has a dataset + I/O roles — by the time they reach the code step,
  // the Signature + metric are already filled (or streaming in).
  const agent = useCodeAgent({
    codeAssistMode,
    setCodeAssistMode,
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
  });

  return {
    step,
    setStep,
    direction,
    setDirection,
    summaryTab,
    setSummaryTab,
    summaryCodeTab,
    setSummaryCodeTab,
    goNext,
    goPrev,
    goTo,
    maxReachableStep,
    validateStep,
    handleNext,
    handleTabClick,
    jobType,
    setOptimizationType,
    isPrivate,
    setIsPrivate,
    username,
    jobName,
    setJobName,
    jobDescription,
    setJobDescription,
    moduleName,
    setModuleName,
    isReact,
    reactConfig,
    updateReactConfig,
    optimizerName,
    setOptimizerName,
    signatureCode,
    setSignatureCode,
    setSignatureManuallyEdited,
    metricCode,
    setMetricCode,
    setMetricManuallyEdited,
    codeAssistMode,
    setCodeAssistMode,
    signatureValidation,
    setSignatureValidation,
    metricValidation,
    setMetricValidation,
    runSignatureValidation,
    runMetricValidation,
    parsedDataset,
    setParsedDataset,
    datasetFileName,
    setDatasetFileName,
    fileInputRef,
    handleFileUpload,
    handlePickFromLibrary,
    columnRoles,
    setColumnRoles,
    columnKinds,
    setColumnKinds,
    globalBaseUrl,
    setGlobalBaseUrl,
    globalApiKey,
    setGlobalApiKey,
    anyProviderHasEnvKey,
    modelConfig,
    setModelConfig,
    secondModelConfig,
    setSecondModelConfig,
    editingModel,
    setEditingModel,
    recentConfigs,
    saveToRecent,
    clearRecentConfigs,
    removeRecentConfig,
    catalog,
    generationModels,
    setGenerationModels,
    reflectionModels,
    setReflectionModels,
    useAllGenerationModels,
    setUseAllGenerationModels,
    useAllReflectionModels,
    setUseAllReflectionModels,
    split,
    updateSplit,
    splitSum,
    datasetProfile,
    splitPlan,
    profileLoading,
    splitMode,
    setSplitMode,
    seed,
    shuffle,
    setShuffle,
    autoLevel,
    setAutoLevel,
    reflectionMinibatchSize,
    setReflectionMinibatchSize,
    maxFullEvals,
    setMaxFullEvals,
    useMerge,
    setUseMerge,
    maxCostCredits,
    setMaxCostCredits,
    costBracket,
    suggestedCeiling,
    submitting,
    submitPhase,
    advancing,
    handleSubmit,
    cloneLoading,
    agent,
  };
}

export type SubmitWizardContext = ReturnType<typeof useSubmitWizard>;
