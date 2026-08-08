"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { AnimatePresence, motion } from "framer-motion";
import { Robot, Brain, Check, Repeat, Sparkle, FlowArrow, Lightning } from "@/shared/ui/icons";
import { formatMsg, msg } from "@/shared/lib/messages";

import { Button } from "@/shared/ui/primitives/button";
import { Label } from "@/shared/ui/primitives/label";
import { Separator } from "@/shared/ui/primitives/separator";
import { HelpTip } from "@/shared/ui/help-tip";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { Skeleton } from "@/shared/ui/skeleton";
import { tip, type TooltipKey } from "@/shared/lib/tooltips";
import { getActiveDir, getActiveIntlLocale } from "@/shared/lib/runtime-locale";
import { cn } from "@/shared/lib/utils";
import { TERMS } from "@/shared/lib/terms";
import { Carousel } from "@/features/agent-panel";

import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";
import type { ArtifactStatus } from "@/shared/hooks/use-code-agent";
import { CodeAgentPanel, VersionStepper } from "./CodeAgentPanel";
import { CodeInterviewPanel } from "./CodeInterviewPanel";
import { ReactConfigSection } from "./ReactConfigSection";
import { workflowUsesTools } from "../../workflow/model";

// The DSPy module choices offered by the picker. Names are technical
// terms kept in English; descriptions reuse the localized tooltip copy.
// Each carries the schematic drawn on its carousel slide's banner.
const MODULE_META = [
  {
    value: "predict",
    label: "Predict",
    icon: Lightning,
    tipKey: "module.predict",
    taglineKey: "submit.module.tagline.predict",
    Banner: PredictBanner,
  },
  {
    value: "cot",
    label: "Chain of Thought",
    icon: Brain,
    tipKey: "module.cot",
    taglineKey: "submit.module.tagline.cot",
    Banner: CotBanner,
  },
  {
    value: "react",
    label: "ReAct",
    icon: Robot,
    tipKey: "module.react",
    taglineKey: "submit.module.tagline.react",
    Banner: ReactBanner,
  },
  {
    value: "flex",
    label: "Flex",
    icon: Sparkle,
    tipKey: "module.flex",
    taglineKey: "submit.module.tagline.flex",
    Banner: FlexBanner,
  },
  {
    value: "workflow",
    label: "Workflow",
    icon: FlowArrow,
    tipKey: "module.workflow",
    taglineKey: "submit.module.tagline.workflow",
    Banner: WorkflowBanner,
  },
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

  // The step opens on the picker and stays there until a module is picked;
  // afterwards the chip in the header reopens it to switch. The three views
  // share one AnimatePresence so picking or switching a module cross-fades
  // instead of hard-swapping the card.
  const view = moduleSelectionRequired ? "picker" : isWorkflow ? "workflow" : "code";

  let content: React.ReactNode;
  if (view === "picker") {
    content = <ModulePicker current={moduleName} onChoose={chooseModule} />;
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
            <div className="relative min-h-[700px] self-stretch overflow-hidden border-b border-border/40 lg:border-b-0 lg:border-e">
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
            <div className="relative min-h-[700px] self-stretch overflow-hidden border-b border-border/40 lg:border-b-0 lg:border-e">
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
                  height="260px"
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
                  height="260px"
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

function ModulePicker({
  current,
  onChoose,
}: {
  current: string;
  onChoose: (module: string) => void;
}) {
  // Reopening the picker to switch modules opens on the one already in use.
  const currentIndex = MODULE_META.findIndex((m) => m.value === current.toLowerCase());
  return (
    // A container, not a breakpoint: the step card is max-w-5xl in auto mode
    // and max-w-2xl in manual, so only its own width can say whether the slide
    // has room to sit banner-beside-copy.
    <div className="@container rounded-2xl border border-border/50 bg-card/80 px-4 py-5 shadow-lg backdrop-blur-xl sm:px-8 sm:py-7">
      <div data-tutorial="module-selector">
        <Carousel
          items={MODULE_META}
          itemKey={(m) => m.value}
          renderItem={(m) => <ModuleSlide module={m} onChoose={onChoose} />}
          // The step heading rides the carousel's own header row, opposite the
          // position counter, rather than sitting above it as a second block.
          title={
            <span className="text-sm font-semibold tracking-tight">
              <HelpTip text={tip("module.choice")}>{msg("submit.module.picker_title")}</HelpTip>
            </span>
          }
          ariaLabel={msg("submit.module.carousel_aria")}
          jumpIndices={currentIndex >= 0 ? [currentIndex] : undefined}
          fluid
        />
      </div>
    </div>
  );
}

function ModuleSlide({
  module,
  onChoose,
}: {
  module: (typeof MODULE_META)[number];
  onChoose: (module: string) => void;
}) {
  const { value, label, icon: Icon, tipKey, taglineKey, Banner } = module;
  return (
    <div className="overflow-hidden rounded-xl border border-border/50 bg-background/60 @3xl:flex @3xl:min-h-64 @3xl:items-stretch">
      <Banner />
      <div className="flex flex-col items-center justify-center gap-2 px-6 pb-7 pt-6 text-center @3xl:flex-1 @3xl:px-10 @3xl:py-8">
        <div className="flex items-center gap-2.5">
          <span className="flex size-9 items-center justify-center rounded-lg bg-[#F3EDE3] text-[#3D2E22]">
            <Icon className="size-[1.125rem]" />
          </span>
          <h4 dir="ltr" className="text-lg font-semibold tracking-tight text-foreground">
            {label}
          </h4>
        </div>
        <p className="text-sm font-medium text-[#5C4D40]">{msg(taglineKey)}</p>
        {/* The measure keeps the copy readable at full width; the floor is the
            longest description's three lines, so the card holds one height as
            slides change instead of shifting the nav under the cursor. */}
        <p className="min-h-[4rem] max-w-md text-[0.8125rem] leading-relaxed text-muted-foreground">
          {moduleDescription(label, tipKey)}
        </p>
        <ChooseModuleButton label={label} onClick={() => onChoose(value)} />
      </div>
    </div>
  );
}

/**
 * The slide's commit control: a check that picks the module it sits on.
 *
 * Icon-only, so "Use <Module>" carries as the tooltip and as the button's
 * accessible name — the same string in both, which is what keeps the hover
 * label and the screen-reader label from drifting apart.
 */
function ChooseModuleButton({ label, onClick }: { label: string; onClick: () => void }) {
  const name = formatMsg("submit.module.choose", { p1: label });
  return (
    <TooltipButton tooltip={name} side="top" dir={getActiveDir()}>
      <Button
        variant="outline"
        size="icon-lg"
        className="mt-2 rounded-full"
        aria-label={name}
        onClick={onClick}
      >
        <Check />
      </Button>
    </TooltipButton>
  );
}

/**
 * A module's description, without the leading `"<Label> — "`.
 *
 * The tooltip copy is the one definition of each module in every locale, and
 * the slide already carries the label as its heading — so the opener is
 * stripped here rather than maintained as a second description per locale
 * that would drift from the tooltip. Copy without the opener passes through.
 * What follows the dash continues the label mid-sentence, so it is recased to
 * stand alone; scripts without case are unaffected.
 */
function moduleDescription(label: string, tipKey: TooltipKey): string {
  const text = tip(tipKey);
  const opener = `${label} — `;
  if (!text.startsWith(opener)) return text;
  const rest = text.slice(opener.length);
  return rest.charAt(0).toLocaleUpperCase(getActiveIntlLocale()) + rest.slice(1);
}

/**
 * The banner shell every module schematic is drawn into.
 *
 * The diagrams read input-to-output left-to-right in every locale, matching
 * the workflow canvas (which one of them depicts) rather than mirroring.
 */
function BannerFrame({ children }: { children: React.ReactNode }) {
  return (
    // Sized only by container queries — mixing in a `sm:` height would leave
    // two utilities competing for `height` from different variant groups.
    <div className="relative h-36 shrink-0 border-b border-border/50 bg-gradient-to-br from-[#F3EDE3] via-[#FAF8F5] to-[#EDE7DD] @xl:h-44 @3xl:h-auto @3xl:w-[45%] @3xl:border-b-0 @3xl:border-e">
      {/* The dot grid is CSS rather than an SVG pattern so it covers the whole
          banner — the schematic letterboxes inside the viewBox, a pattern fill
          would letterbox with it and leave bare bands at the sides. */}
      <div
        aria-hidden
        className="absolute inset-0"
        style={{
          backgroundImage: "radial-gradient(circle, rgba(61,46,34,0.07) 1px, transparent 1px)",
          backgroundSize: "12px 12px",
        }}
      />
      <svg
        viewBox="0 0 240 88"
        preserveAspectRatio="xMidYMid meet"
        className="absolute inset-0 h-full w-full"
        aria-hidden="true"
      >
        {children}
      </svg>
    </div>
  );
}

function GBox({
  x,
  y,
  w,
  h,
  accent = false,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  accent?: boolean;
}) {
  return (
    <rect
      x={x}
      y={y}
      width={w}
      height={h}
      rx={6}
      fill={accent ? "#C8A882" : "#FAF8F5"}
      fillOpacity={accent ? 0.45 : 0.9}
      stroke="#3D2E22"
      strokeOpacity={accent ? 0.4 : 0.22}
      strokeWidth={1.25}
    />
  );
}

function GWire({ d }: { d: string }) {
  return (
    <path
      d={d}
      fill="none"
      stroke="#3D2E22"
      strokeOpacity={0.3}
      strokeWidth={1.25}
      strokeLinecap="round"
    />
  );
}

/** A bar standing in for a line of text or code inside a node. */
function GBar({ x, y, w }: { x: number; y: number; w: number }) {
  return <rect x={x} y={y} width={w} height={2.5} rx={1.25} fill="#3D2E22" fillOpacity={0.3} />;
}

/** The arrowhead terminating a wire, tip at `(x, y)`. */
function GArrow({ x, y, dir }: { x: number; y: number; dir: "up" | "down" | "right" }) {
  const points =
    dir === "up"
      ? `${x},${y} ${x - 2.5},${y + 4.5} ${x + 2.5},${y + 4.5}`
      : dir === "down"
        ? `${x},${y} ${x - 2.5},${y - 4.5} ${x + 2.5},${y - 4.5}`
        : `${x},${y} ${x - 4.5},${y - 2.5} ${x - 4.5},${y + 2.5}`;
  return <polygon points={points} fill="#3D2E22" fillOpacity={0.35} />;
}

function PredictBanner() {
  return (
    <BannerFrame>
      <GWire d="M60 44 H98" />
      <GWire d="M142 44 H180" />
      <GBox x={26} y={33} w={34} h={22} />
      <GBar x={34} y={40} w={18} />
      <GBar x={34} y={46} w={12} />
      <GBox x={98} y={29} w={44} h={30} accent />
      <GBar x={110} y={43} w={20} />
      <GBox x={180} y={33} w={34} h={22} />
      <GBar x={188} y={40} w={18} />
      <GBar x={188} y={46} w={12} />
    </BannerFrame>
  );
}

function CotBanner() {
  return (
    <BannerFrame>
      <GWire d="M54 44 H94" />
      <GWire d="M146 44 H186" />
      <GBox x={22} y={33} w={32} h={22} />
      <GBar x={29} y={42} w={18} />
      <GBox x={94} y={16} w={52} h={56} accent />
      {/* The reasoning field, then the answer it leads to. */}
      <GBox x={102} y={23} w={36} h={20} />
      <GBar x={108} y={29} w={24} />
      <GBar x={108} y={35} w={16} />
      <GWire d="M120 43 V51" />
      <GBox x={102} y={51} w={36} h={14} />
      <GBar x={108} y={57} w={24} />
      <GBox x={186} y={33} w={32} h={22} />
      <GBar x={193} y={42} w={18} />
    </BannerFrame>
  );
}

function ReactBanner() {
  return (
    <BannerFrame>
      <GWire d="M50 34 H90" />
      <GWire d="M146 34 H190" />
      {/* The think/act loop: down into a tool, back up with its result. */}
      <GWire d="M104 46 V62" />
      <GWire d="M128 62 V46" />
      <GArrow x={104} y={62} dir="down" />
      <GArrow x={128} y={46} dir="up" />
      <GBox x={18} y={23} w={32} h={22} />
      <GBar x={25} y={32} w={18} />
      <GBox x={90} y={22} w={56} h={24} accent />
      <GBar x={104} y={32} w={28} />
      <GBox x={96} y={62} w={44} h={16} />
      <GBar x={106} y={69} w={24} />
      <GBox x={190} y={23} w={32} h={22} />
      <GBar x={197} y={32} w={18} />
    </BannerFrame>
  );
}

function FlexBanner() {
  return (
    <BannerFrame>
      {/* The program's own code, rewritten by the optimizer. */}
      <GWire d="M108 44 H128" />
      <GArrow x={132} y={44} dir="right" />
      <GBox x={26} y={16} w={78} h={56} />
      <GBar x={38} y={29} w={44} />
      <GBar x={38} y={37} w={54} />
      <GBar x={38} y={45} w={34} />
      <GBar x={38} y={53} w={46} />
      <GBox x={136} y={16} w={78} h={56} accent />
      <GBar x={148} y={29} w={52} />
      <GBar x={148} y={37} w={36} />
      <GBar x={148} y={45} w={48} />
      <GBar x={148} y={53} w={28} />
      <path
        d="M204 18 L206 23 L211 25 L206 27 L204 32 L202 27 L197 25 L202 23 Z"
        fill="#3D2E22"
        fillOpacity={0.35}
      />
    </BannerFrame>
  );
}

function WorkflowBanner() {
  return (
    <BannerFrame>
      <GWire d="M44 44 C68 44 70 19 92 19" />
      <GWire d="M44 44 C68 44 70 69 92 69" />
      <GWire d="M144 19 C166 19 172 44 196 44" />
      <GWire d="M144 69 C166 69 172 44 196 44" />
      <GBox x={14} y={33} w={30} h={22} />
      <GBar x={21} y={42} w={16} />
      <GBox x={92} y={8} w={52} h={22} accent />
      <GBar x={102} y={17} w={32} />
      <GBox x={92} y={58} w={52} h={22} accent />
      <GBar x={102} y={67} w={32} />
      <GBox x={196} y={33} w={30} h={22} />
      <GBar x={203} y={42} w={16} />
    </BannerFrame>
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
              <Repeat className="size-3" />
            </span>
          </button>
        )}
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
