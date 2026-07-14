"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { AnimatePresence, motion } from "framer-motion";
import { Bot, Brain, Check, Repeat2, Sparkles, Workflow, Zap } from "lucide-react";
import { formatMsg, msg } from "@/shared/lib/messages";

import { Label } from "@/shared/ui/primitives/label";
import { Separator } from "@/shared/ui/primitives/separator";
import { HelpTip } from "@/shared/ui/help-tip";
import { Skeleton } from "@/shared/ui/skeleton";
import { tip } from "@/shared/lib/tooltips";
import { cn } from "@/shared/lib/utils";
import { TERMS } from "@/shared/lib/terms";

import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";
import type { ArtifactStatus } from "@/shared/hooks/use-code-agent";
import { CodeAgentPanel, VersionStepper } from "./CodeAgentPanel";
import { CodeInterviewPanel } from "./CodeInterviewPanel";
import { ReactConfigSection } from "./ReactConfigSection";
import { workflowUsesTools } from "../../workflow/model";

// The four DSPy module choices offered by the picker. Names are technical
// terms kept in English; descriptions reuse the localized tooltip copy.
const MODULE_META = [
  { value: "predict", label: "Predict", icon: Zap, tipKey: "module.predict" },
  { value: "cot", label: "Chain of Thought", icon: Brain, tipKey: "module.cot" },
  { value: "react", label: "ReAct", icon: Bot, tipKey: "module.react" },
  { value: "workflow", label: "Workflow", icon: Workflow, tipKey: "module.workflow" },
] as const;

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={200} borderRadius={8} />,
});

// React Flow ships ~100KB of canvas code; only workflow runs pay for it.
const WorkflowCanvas = dynamic(
  () => import("../../workflow/WorkflowCanvas").then((m) => m.WorkflowCanvas),
  { ssr: false, loading: () => <Skeleton height={480} borderRadius={8} /> },
);

export function CodeStep({ w }: { w: SubmitWizardContext }) {
  const {
    isWorkflow,
    isReact,
    moduleName,
    moduleSelectionRequired,
    chooseModule,
    reopenModulePicker,
    workflowSpec,
    setWorkflowSpec,
    workflowRevision,
    agentPulseNodeId,
    workflowSampleInputs,
    workflowDryRunDisabledReason,
    workflowDryRunNeedsModel,
    openDryRunModelPicker,
    modelConfig,
    runWorkflowDryRun,
    signatureCode,
    setSignatureCode,
    setSignatureManuallyEdited,
    signatureValidation,
    setSignatureValidation,
    metricCode,
    setMetricCode,
    setMetricManuallyEdited,
    metricValidation,
    setMetricValidation,
    runSignatureValidation,
    runMetricValidation,
    codeAssistMode,
    setCodeAssistMode,
    parsedDataset,
    columnRoles,
    agent,
    interview,
    interviewEligible,
  } = w;
  const hasContext = React.useMemo(() => {
    if (!parsedDataset || parsedDataset.rowCount === 0) return false;
    const hasInput = Object.values(columnRoles).some((r) => r === "input");
    const hasOutput = Object.values(columnRoles).some((r) => r === "output");
    return hasInput && hasOutput;
  }, [parsedDataset, columnRoles]);

  // The interview owns the agent-panel slot until it resolves — but never
  // over an existing conversation (a locale-switch reload restores the
  // agent transcript; re-interviewing over generated code would be noise).
  const interviewActive =
    interviewEligible &&
    !interview.resolved &&
    agent.messages.length === 0 &&
    agent.signatureVersions.length === 0 &&
    agent.metricVersions.length === 0;

  const disabledReason = !hasContext
    ? parsedDataset
      ? msg("auto.features.submit.components.steps.codestep.literal.1")
      : formatMsg("auto.features.submit.components.steps.codestep.template.1", {
          p1: TERMS.dataset,
        })
    : undefined;

  const moduleChip = {
    label: MODULE_META.find((m) => m.value === moduleName.toLowerCase())?.label ?? moduleName,
    onChangeModule: reopenModulePicker,
  };

  // The step opens as the default module's editor; the chip in the header
  // reopens the picker to switch modules. The three views share one
  // AnimatePresence so picking or switching a module cross-fades instead of
  // hard-swapping the card.
  const view = moduleSelectionRequired ? "picker" : isWorkflow ? "workflow" : "code";

  let content: React.ReactNode;
  if (view === "picker") {
    content = <ModulePicker onChoose={chooseModule} />;
  } else if (view === "workflow") {
    content = (
      <div className="overflow-hidden rounded-2xl border border-border/50 bg-card/80 backdrop-blur-xl shadow-lg">
        <ModeToggle
          value={codeAssistMode}
          onChange={setCodeAssistMode}
          disabledReason={disabledReason}
          module={moduleChip}
        />
        <div
          className={cn(
            "grid grid-cols-1",
            codeAssistMode === "auto" && "lg:grid-cols-[400px_minmax(0,1fr)]",
          )}
        >
          {codeAssistMode === "auto" && (
            <div className="relative min-h-[560px] self-stretch overflow-hidden border-b border-border/40 lg:border-b-0 lg:border-e">
              {interviewActive ? (
                <CodeInterviewPanel interview={interview} className="absolute inset-0" />
              ) : (
                <CodeAgentPanel
                  agent={agent}
                  disabled={!hasContext}
                  disabledReason={disabledReason}
                  className="absolute inset-0"
                />
              )}
            </div>
          )}
          <div className="flex min-w-0 flex-col self-stretch">
            <div className="border-b border-border/30 px-6 py-3">
              <h3 className="inline-flex text-lg font-semibold tracking-tight text-foreground">
                <HelpTip text={tip("module.workflow")}>{msg("workflow.step.title")}</HelpTip>
              </h3>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {msg("workflow.step.subtitle")}
              </p>
            </div>
            {workflowSpec && (
              <WorkflowCanvas
                spec={workflowSpec}
                specRevision={workflowRevision}
                onSpecChange={setWorkflowSpec}
                pulseNodeId={agentPulseNodeId}
                dryRun={{
                  disabledReason: workflowDryRunDisabledReason,
                  needsModel: workflowDryRunNeedsModel,
                  pickModel: openDryRunModelPicker,
                  modelName: modelConfig.name || null,
                  sampleInputs: workflowSampleInputs,
                  run: runWorkflowDryRun,
                }}
                agentPanel={
                  codeAssistMode === "auto" ? (
                    interviewActive ? (
                      <CodeInterviewPanel interview={interview} />
                    ) : (
                      <CodeAgentPanel
                        agent={agent}
                        disabled={!hasContext}
                        disabledReason={disabledReason}
                      />
                    )
                  ) : undefined
                }
              />
            )}
            <div
              className="space-y-2 border-t border-border/30 px-6 py-4"
              data-tutorial="metric-editor"
            >
              <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                <HelpTip text={tip("code.metric")}>{msg("workflow.step.metric_title")}</HelpTip>
              </Label>
              <p className="text-[0.6875rem] leading-relaxed text-muted-foreground">
                {msg("workflow.step.metric_hint")}
              </p>
              <CodeEditor
                value={metricCode}
                onChange={(v) => {
                  setMetricCode(v);
                  setMetricManuallyEdited(true);
                  setMetricValidation(null);
                }}
                height="180px"
                onRun={runMetricValidation}
                validationResult={metricValidation}
                streaming={codeAssistMode === "auto" && agent.metricStatus === "writing"}
                flashLines={codeAssistMode === "auto" ? agent.metricFlashLines : undefined}
              />
            </div>
            {workflowSpec && workflowUsesTools(workflowSpec) && (
              <div className="border-t border-border/30 px-6 py-4">
                <ReactConfigSection w={w} />
              </div>
            )}
          </div>
        </div>
      </div>
    );
  } else {
    content = (
      <div className="overflow-hidden rounded-2xl border border-border/50 bg-card/80 backdrop-blur-xl shadow-lg">
        <ModeToggle
          value={codeAssistMode}
          onChange={setCodeAssistMode}
          disabledReason={disabledReason}
          module={moduleChip}
        />
        <div
          className={cn(
            "grid grid-cols-1",
            codeAssistMode === "auto" && "lg:grid-cols-[400px_minmax(0,1fr)]",
          )}
        >
          {codeAssistMode === "auto" && (
            <div className="relative min-h-[560px] self-stretch overflow-hidden border-b border-border/40 lg:border-b-0 lg:border-e">
              {interviewActive ? (
                <CodeInterviewPanel interview={interview} className="absolute inset-0" />
              ) : (
                <CodeAgentPanel
                  agent={agent}
                  disabled={!hasContext}
                  disabledReason={disabledReason}
                  className="absolute inset-0"
                />
              )}
            </div>
          )}
          <div className="flex min-w-0 flex-col self-stretch">
            <div className="shrink-0 border-b border-border/30 px-6 py-3">
              <h3 className="inline-flex text-lg font-semibold tracking-tight text-foreground">
                <HelpTip
                  text={
                    codeAssistMode === "auto"
                      ? formatMsg("auto.features.submit.components.steps.codestep.template.2", {
                          p1: TERMS.dataset,
                        })
                      : formatMsg("auto.features.submit.components.steps.codestep.template.3", {
                          p1: TERMS.signature,
                          p2: TERMS.metric,
                        })
                  }
                >
                  {msg("auto.features.submit.components.steps.codestep.1")}
                </HelpTip>
              </h3>
            </div>
            <div className="space-y-4 px-6 py-4">
              <div
                className={cn(
                  "space-y-2 transition-opacity duration-300",
                  codeAssistMode === "auto" && agent.metricStatus === "writing" && "opacity-50",
                )}
                data-tutorial="signature-editor"
              >
                <div className="flex items-center justify-between gap-2">
                  <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    <HelpTip text={tip("code.signature")}>
                      {msg("auto.features.submit.components.steps.codestep.2")}
                    </HelpTip>
                  </Label>
                  <div className="flex items-center gap-2">
                    {codeAssistMode === "auto" && (
                      <VersionStepper agent={agent} artifact="signature" />
                    )}
                    {codeAssistMode === "auto" && (
                      <ArtifactStatusChip status={agent.signatureStatus} />
                    )}
                  </div>
                </div>
                <CodeEditor
                  value={signatureCode}
                  onChange={(v) => {
                    setSignatureCode(v);
                    setSignatureManuallyEdited(true);
                    setSignatureValidation(null);
                  }}
                  height="180px"
                  onRun={runSignatureValidation}
                  validationResult={signatureValidation}
                  streaming={codeAssistMode === "auto" && agent.signatureStatus === "writing"}
                  flashLines={codeAssistMode === "auto" ? agent.signatureFlashLines : undefined}
                />
              </div>
              <Separator />
              <div
                className={cn(
                  "space-y-2 transition-opacity duration-300",
                  codeAssistMode === "auto" && agent.signatureStatus === "writing" && "opacity-50",
                )}
                data-tutorial="metric-editor"
              >
                <div className="flex items-center justify-between gap-2">
                  <Label className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    <HelpTip text={tip("code.metric")}>
                      {msg("auto.features.submit.components.steps.codestep.3")}
                    </HelpTip>
                  </Label>
                  <div className="flex items-center gap-2">
                    {codeAssistMode === "auto" && (
                      <VersionStepper agent={agent} artifact="metric" />
                    )}
                    {codeAssistMode === "auto" && (
                      <ArtifactStatusChip status={agent.metricStatus} />
                    )}
                  </div>
                </div>
                <CodeEditor
                  value={metricCode}
                  onChange={(v) => {
                    setMetricCode(v);
                    setMetricManuallyEdited(true);
                    setMetricValidation(null);
                  }}
                  height="180px"
                  onRun={runMetricValidation}
                  validationResult={metricValidation}
                  streaming={codeAssistMode === "auto" && agent.metricStatus === "writing"}
                  flashLines={codeAssistMode === "auto" ? agent.metricFlashLines : undefined}
                />
              </div>
              {isReact && (
                <>
                  <Separator />
                  <ReactConfigSection w={w} />
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div data-tutorial="wizard-step-4">
      <AnimatePresence mode="wait" initial={false}>
        <motion.div
          key={view}
          initial={{ opacity: 0, y: 10, scale: 0.99 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -10, scale: 0.99 }}
          transition={{ duration: 0.18, ease: "easeOut" }}
        >
          {content}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

function ModulePicker({ onChoose }: { onChoose: (module: string) => void }) {
  return (
    <div className="rounded-2xl border border-border/50 bg-card/80 px-6 py-10 shadow-lg backdrop-blur-xl sm:px-10">
      <div className="mx-auto max-w-2xl text-center">
        <h3 className="text-lg font-semibold tracking-tight text-foreground">
          <HelpTip text={tip("module.choice")}>{msg("submit.module.picker_title")}</HelpTip>
        </h3>
        <p className="mt-1 text-sm text-muted-foreground">{msg("submit.module.picker_subtitle")}</p>
      </div>
      <div
        className="mx-auto mt-6 grid max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2"
        data-tutorial="module-selector"
      >
        {MODULE_META.map(({ value, label, icon: Icon, tipKey }, index) => (
          <motion.button
            key={value}
            type="button"
            onClick={() => onChoose(value)}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, delay: index * 0.05, ease: "easeOut" }}
            className="group flex cursor-pointer flex-col items-start gap-2 rounded-xl border border-border/60 bg-background/60 p-4 text-start transition-all duration-150 hover:-translate-y-0.5 hover:border-[#C8A882] hover:shadow-md"
          >
            <span className="flex size-9 items-center justify-center rounded-lg bg-[#F3EDE3] text-[#3D2E22] transition-colors duration-150 group-hover:bg-[#3D2E22] group-hover:text-[#FAF8F5]">
              <Icon className="size-4" strokeWidth={2} />
            </span>
            <span className="text-sm font-semibold text-foreground">{label}</span>
            <span className="text-xs leading-relaxed text-muted-foreground">{tip(tipKey)}</span>
          </motion.button>
        ))}
      </div>
    </div>
  );
}

function ArtifactStatusChip({ status }: { status: ArtifactStatus }) {
  if (status === "idle") return null;
  if (status === "waiting") {
    return (
      <span className="inline-flex items-center gap-1 text-[0.6875rem] font-medium text-muted-foreground/70">
        {msg("auto.features.submit.components.steps.codestep.4")}
        <span className="size-1.5 rounded-full bg-muted-foreground/40" />
      </span>
    );
  }
  if (status === "writing") {
    return (
      <span className="inline-flex items-center gap-1 text-[0.6875rem] font-medium text-[#3D2E22]">
        {msg("auto.features.submit.components.steps.codestep.5")}
        <span className="size-1.5 rounded-full bg-[#3D2E22] animate-pulse" />
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-[0.6875rem] font-medium text-[#5A7247]">
      {msg("auto.features.submit.components.steps.codestep.6")}
      <Check className="size-3" />
    </span>
  );
}

interface ModeToggleProps {
  value: "auto" | "manual";
  onChange: (mode: "auto" | "manual") => void;
  disabledReason?: string;
  // Chip showing the chosen module with a click-to-switch affordance.
  module?: { label: string; onChangeModule: () => void } | null;
}

function ModeToggle({ value, onChange, disabledReason, module }: ModeToggleProps) {
  const autoDisabled = !!disabledReason && value !== "auto";

  return (
    <div className="flex items-center justify-between gap-3 border-b border-border/40 bg-[#FAF8F5] px-4 py-2.5">
      <div className="flex min-w-0 items-center gap-2.5">
        {module && (
          <button
            type="button"
            onClick={module.onChangeModule}
            data-tutorial="module-selector"
            className="group inline-flex shrink-0 cursor-pointer items-center gap-1.5 rounded-md border border-border/60 bg-background px-2 py-1 text-xs shadow-xs transition-colors hover:border-[#C8A882]"
          >
            <span className="font-semibold text-foreground">{module.label}</span>
            <span aria-hidden className="h-3 w-px bg-border/80" />
            <span className="flex items-center gap-1 font-medium text-muted-foreground transition-colors group-hover:text-foreground">
              {msg("submit.module.change")}
              <Repeat2 className="size-3" />
            </span>
          </button>
        )}
        <div className="flex items-center gap-1.5 text-xs text-[#5C4D40]">
          <Sparkles className="h-3.5 w-3.5 shrink-0 text-[#3D2E22]" />
          <span className="font-medium">
            {value === "auto"
              ? formatMsg("auto.features.submit.components.steps.codestep.template.4", {
                  p1: TERMS.dataset,
                })
              : msg("auto.features.submit.components.steps.codestep.literal.2")}
          </span>
        </div>
      </div>

      <div className="relative inline-grid grid-cols-2 rounded-lg bg-muted p-1 gap-1">
        <div
          aria-hidden
          className="absolute top-1 bottom-1 w-[calc(50%-6px)] rounded-md bg-background shadow-sm transition-[inset-inline-start] duration-150 ease-out pointer-events-none"
          style={{ insetInlineStart: value === "auto" ? 4 : "calc(50% + 2px)" }}
        />
        <button
          type="button"
          onClick={() => onChange("auto")}
          disabled={autoDisabled}
          title={autoDisabled ? disabledReason : undefined}
          aria-pressed={value === "auto"}
          className={cn(
            "relative z-[1] rounded-md px-4 py-1 text-xs font-medium leading-none text-center transition-colors cursor-pointer",
            value === "auto" ? "text-foreground" : "text-muted-foreground hover:text-foreground",
            autoDisabled && "opacity-40 cursor-not-allowed hover:text-muted-foreground",
          )}
        >
          {msg("auto.features.submit.components.steps.codestep.7")}
        </button>
        <button
          type="button"
          onClick={() => onChange("manual")}
          aria-pressed={value === "manual"}
          className={cn(
            "relative z-[1] rounded-md px-4 py-1 text-xs font-medium leading-none text-center transition-colors cursor-pointer",
            value === "manual" ? "text-foreground" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {msg("auto.features.submit.components.steps.codestep.8")}
        </button>
      </div>
    </div>
  );
}
