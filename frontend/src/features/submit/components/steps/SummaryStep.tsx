"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { motion, AnimatePresence, useReducedMotion, type Variants } from "framer-motion";
import { Separator } from "@/shared/ui/primitives/separator";
import {
  CaretLeft,
  CaretRight,
  User,
  Code,
  Tag,
  Stack,
  Cube,
  Target,
  FileText,
  Columns,
  Shuffle,
  MagnifyingGlass,
  Database,
  Cpu,
  Gauge,
} from "@/shared/ui/icons";
import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";
import { moduleLabel } from "@/shared/lib/formatters";
import { TERMS } from "@/shared/lib/terms";
import { ModelChip } from "@/shared/ui/model-chip";
import { Skeleton } from "@/shared/ui/skeleton";
import { formatCredits } from "@/features/billing";
import { useUserPrefs } from "@/features/settings";
import { getActiveDir, getActiveIntlLocale } from "@/shared/lib/runtime-locale";

import { aggregateTokenSource, chargeableBracket } from "../../lib/cost-bracket";
import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={200} borderRadius={8} />,
});

const SUMMARY_SLIDES = perLocale(() => [
  {
    id: "general",
    label: msg("auto.features.submit.components.steps.summarystep.literal.1"),
    icon: <User className="size-3.5" />,
  },
  { id: "dataset", label: TERMS.dataset, icon: <Database className="size-3.5" /> },
  {
    id: "models",
    label: msg("auto.features.submit.components.steps.summarystep.literal.2"),
    icon: <Cpu className="size-3.5" />,
  },
  { id: "optimizer", label: TERMS.optimizer, icon: <Target className="size-3.5" /> },
  {
    id: "code",
    label: msg("auto.features.submit.components.steps.summarystep.literal.3"),
    icon: <Code className="size-3.5" />,
  },
]);

const SUMMARY_SLIDE_VARIANTS: Variants = {
  enter: (direction: 1 | -1) => ({ opacity: 0, x: direction * 36 }),
  center: { opacity: 1, x: 0 },
  exit: (direction: 1 | -1) => ({ opacity: 0, x: direction * -28 }),
};

const SUMMARY_SLIDE_TRANSITION = {
  duration: 0.24,
  ease: [0.2, 0.8, 0.2, 1] as const,
};

export function SummaryStep({ w }: { w: SubmitWizardContext }) {
  const { prefs } = useUserPrefs();
  const advanced = prefs.advancedMode;
  const {
    summaryTab,
    setSummaryTab,
    jobName,
    jobType,
    moduleName,
    isReact,
    reactConfig,
    datasetFileName,
    parsedDataset,
    columnRoles,
    split,
    shuffle,
    modelConfig,
    secondModelConfig,
    generationModels,
    reflectionModels,
    autoLevel,
    reflectionMinibatchSize,
    maxFullEvals,
    maxMetricCalls,
    useMerge,
    targetScore,
    signatureCode,
    metricCode,
    isWorkflow,
    workflowSpec,
    costBracket,
    maxCostCredits,
  } = w;

  // A workflow run has no single top-level signature — the graph carries the
  // per-node code, so the code tab shows only the metric plus a graph line.
  const displaySignatureCode = isWorkflow ? "" : signatureCode;

  // Read-only echo of the pre-run estimate the user set on the model step, so the
  // final review restates what this run is expected to cost (and any hard cap)
  // without making them step back. BYOK shows the platform fee, not the full
  // per-model cost the provider key absorbs — same chargeable bracket the cost
  // surface used, so the two never disagree.
  const locale = getActiveIntlLocale();
  const selectedConfigs =
    jobType === "run"
      ? [modelConfig, ...(secondModelConfig ? [secondModelConfig] : [])]
      : [...generationModels, ...reflectionModels];
  const tokenSource = aggregateTokenSource(selectedConfigs);
  const byok = tokenSource === "byok";
  const estimate = chargeableBracket(costBracket, tokenSource);
  const prefersReducedMotion = useReducedMotion();
  const isRtl = getActiveDir() === "rtl";
  const activeSlide = Math.max(0, Math.min(summaryTab, SUMMARY_SLIDES.length - 1));
  const [slideDirection, setSlideDirection] = React.useState<1 | -1>(isRtl ? -1 : 1);
  const touchStart = React.useRef<{ x: number; y: number } | null>(null);

  React.useEffect(() => {
    if (summaryTab !== activeSlide) setSummaryTab(activeSlide);
  }, [activeSlide, setSummaryTab, summaryTab]);

  const goToSlide = React.useCallback(
    (next: number) => {
      const clamped = Math.max(0, Math.min(SUMMARY_SLIDES.length - 1, next));
      if (clamped === activeSlide) return;
      const forward = clamped > activeSlide;
      setSlideDirection(forward === isRtl ? -1 : 1);
      setSummaryTab(clamped);
    },
    [activeSlide, isRtl, setSummaryTab],
  );

  const previousSlide = SUMMARY_SLIDES[activeSlide - 1];
  const currentSlide = SUMMARY_SLIDES[activeSlide] ?? SUMMARY_SLIDES[0]!;
  const nextSlide = SUMMARY_SLIDES[activeSlide + 1];
  const PreviousIcon = isRtl ? CaretRight : CaretLeft;
  const NextIcon = isRtl ? CaretLeft : CaretRight;

  return (
    <div className="space-y-4" data-tutorial="wizard-step-6">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: [0.2, 0.8, 0.2, 1] }}
        className="overflow-hidden rounded-2xl border border-border bg-card/80 shadow-lg backdrop-blur-xl"
        role="region"
        aria-label={currentSlide.label}
        tabIndex={0}
        onKeyDown={(event) => {
          if (event.target !== event.currentTarget) return;
          const forwardKey = isRtl ? "ArrowLeft" : "ArrowRight";
          const backKey = isRtl ? "ArrowRight" : "ArrowLeft";
          if (event.key === forwardKey) {
            event.preventDefault();
            goToSlide(activeSlide + 1);
          } else if (event.key === backKey) {
            event.preventDefault();
            goToSlide(activeSlide - 1);
          }
        }}
      >
        <div className="flex items-center justify-between gap-4 border-b border-border/60 bg-secondary/35 px-5 py-4 sm:px-8 sm:py-5">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex size-11 shrink-0 items-center justify-center rounded-xl bg-[#EDE7DD] text-[#3D2E22] shadow-[inset_0_1px_0_rgba(255,255,255,0.65)] [&_svg]:size-5">
              {currentSlide.icon}
            </span>
            <div className="min-w-0">
              <span
                className="font-mono text-[0.625rem] tabular-nums text-muted-foreground"
                dir="ltr"
              >
                {activeSlide + 1} / {SUMMARY_SLIDES.length}
              </span>
              <h3 className="truncate text-lg font-bold tracking-tight text-foreground sm:text-xl">
                {currentSlide.label}
              </h3>
            </div>
          </div>
          <div className="hidden items-center gap-1.5 sm:flex" aria-label={currentSlide.label}>
            {SUMMARY_SLIDES.map((slide, index) => (
              <button
                key={slide.id}
                type="button"
                onClick={() => goToSlide(index)}
                aria-label={slide.label}
                aria-current={activeSlide === index ? "step" : undefined}
                className="flex size-8 cursor-pointer items-center justify-center rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882] focus-visible:ring-offset-2"
              >
                <span
                  className={cn(
                    "h-1.5 rounded-full transition-[width,background-color] duration-200",
                    activeSlide === index
                      ? "w-6 bg-[#3D2E22]"
                      : "w-1.5 bg-[#3D2E22]/20 hover:bg-[#3D2E22]/40",
                  )}
                />
              </button>
            ))}
          </div>
        </div>

        <div
          className="relative overflow-hidden px-5 py-6 sm:px-8 sm:py-8"
          onTouchStart={(event) => {
            const touch = event.touches[0];
            touchStart.current = touch ? { x: touch.clientX, y: touch.clientY } : null;
          }}
          onTouchEnd={(event) => {
            const start = touchStart.current;
            const touch = event.changedTouches[0];
            touchStart.current = null;
            if (!start || !touch) return;
            const deltaX = touch.clientX - start.x;
            const deltaY = touch.clientY - start.y;
            if (Math.abs(deltaX) < 52 || Math.abs(deltaX) < Math.abs(deltaY) * 1.2) return;
            const forward = isRtl ? deltaX > 0 : deltaX < 0;
            goToSlide(activeSlide + (forward ? 1 : -1));
          }}
        >
          <AnimatePresence mode="wait" custom={slideDirection} initial={false}>
            <motion.div
              key={activeSlide}
              custom={slideDirection}
              variants={SUMMARY_SLIDE_VARIANTS}
              initial="enter"
              animate="center"
              exit="exit"
              transition={prefersReducedMotion ? { duration: 0 } : SUMMARY_SLIDE_TRANSITION}
              className="mx-auto min-h-[22rem] w-full max-w-3xl"
            >
              {activeSlide === 0 && (
                <div className="grid grid-cols-1 gap-x-10 sm:grid-cols-2">
                  <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                    <span className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Tag className="size-3.5" />
                      {msg("auto.features.submit.components.steps.summarystep.3")}
                      {TERMS.optimization}
                    </span>
                    <span className="text-sm font-medium">{jobName || "—"}</span>
                  </div>
                  <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                    <span className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Stack className="size-3.5" />
                      {msg("auto.features.submit.components.steps.summarystep.4")}
                      {TERMS.optimization}
                    </span>
                    <span className="text-sm font-medium">
                      {jobType === "run"
                        ? msg("auto.features.submit.components.steps.summarystep.literal.4")
                        : msg("auto.features.submit.components.steps.summarystep.literal.5")}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                    <span className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Cube className="size-3.5" />
                      {msg("auto.features.submit.components.steps.summarystep.5")}
                    </span>
                    <span className="text-sm font-medium font-mono" dir="ltr">
                      {moduleLabel(moduleName)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                    <span className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Target className="size-3.5" />
                      {TERMS.optimizer}
                    </span>
                    <span className="text-sm font-medium font-mono" dir="ltr">
                      {msg("auto.features.submit.components.steps.summarystep.6")}
                    </span>
                  </div>
                </div>
              )}

              {activeSlide === 1 && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                    <span className="flex items-center gap-2 text-xs text-muted-foreground">
                      <FileText className="size-3.5" />
                      {msg("auto.features.submit.components.steps.summarystep.7")}
                    </span>
                    <span
                      className="text-sm font-medium truncate max-w-[60%]"
                      title={datasetFileName ?? undefined}
                    >
                      {datasetFileName ?? "—"}
                    </span>
                  </div>
                  {parsedDataset && (
                    <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                      <span className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Database className="size-3.5" />
                        {msg("auto.features.submit.components.steps.summarystep.8")}
                      </span>
                      <span className="text-sm font-medium">
                        {parsedDataset.rowCount}
                        {msg("auto.features.submit.components.steps.summarystep.9")}
                        {parsedDataset.columns.length}
                        {msg("auto.features.submit.components.steps.summarystep.10")}
                      </span>
                    </div>
                  )}
                  {parsedDataset && parsedDataset.columns.length > 0 && (
                    <div className="space-y-2 pt-1">
                      <span className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Columns className="size-3.5" />
                        {msg("auto.features.submit.components.steps.summarystep.11")}
                      </span>
                      <div className="space-y-1.5">
                        {parsedDataset.columns.map((col) => {
                          const role = columnRoles[col];
                          if (role === "ignore") return null;
                          const roleLabel =
                            role === "input"
                              ? msg("auto.features.submit.components.steps.summarystep.literal.6")
                              : role === "output"
                                ? msg("auto.features.submit.components.steps.summarystep.literal.7")
                                : msg(
                                    "auto.features.submit.components.steps.summarystep.literal.8",
                                  );
                          const roleColor =
                            role === "input"
                              ? "text-[#3D2E22] bg-[#3D2E22]/10"
                              : role === "output"
                                ? "text-primary bg-primary/10"
                                : "text-muted-foreground bg-muted";
                          return (
                            <div key={col} className="flex items-center justify-between gap-2 py-1">
                              <span className="text-xs font-mono truncate" dir="ltr">
                                {col}
                              </span>
                              <span
                                className={cn(
                                  "text-[0.625rem] font-semibold px-2 py-0.5 rounded-full",
                                  roleColor,
                                )}
                              >
                                {roleLabel}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}
                  {/* Split breakdown is advanced-mode machinery. */}
                  {advanced && (
                    <>
                      <Separator />
                      <div className="space-y-3">
                        <span className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Stack className="size-3.5" />
                          {msg("auto.features.submit.components.steps.summarystep.12")}
                          {TERMS.dataset}
                        </span>
                        <div className="flex h-3 rounded-full overflow-hidden">
                          <div
                            className="bg-[#3D2E22]"
                            style={{ width: `${split.train * 100}%` }}
                          />
                          <div className="bg-[#C8A882]" style={{ width: `${split.val * 100}%` }} />
                          <div className="bg-[#8C7A6B]" style={{ width: `${split.test * 100}%` }} />
                        </div>
                        <div className="grid grid-cols-3 gap-4">
                          <div className="flex items-center gap-1.5 text-xs">
                            <span className="inline-block w-2 h-2 rounded-full bg-[#3D2E22]" />
                            {msg("auto.features.submit.components.steps.summarystep.13")}
                            {split.train}
                          </div>
                          <div className="flex items-center gap-1.5 text-xs">
                            <span className="inline-block w-2 h-2 rounded-full bg-[#C8A882]" />
                            {msg("auto.features.submit.components.steps.summarystep.14")}
                            {split.val}
                          </div>
                          <div className="flex items-center gap-1.5 text-xs">
                            <span className="inline-block w-2 h-2 rounded-full bg-[#8C7A6B]" />
                            {msg("auto.features.submit.components.steps.summarystep.15")}
                            {split.test}
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                  {advanced && (
                    <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                      <span className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Shuffle className="size-3.5" />
                        {msg("auto.features.submit.components.steps.summarystep.16")}
                      </span>
                      <span className="text-sm font-medium">
                        {shuffle
                          ? msg("auto.features.submit.components.steps.summarystep.literal.9")
                          : msg("auto.features.submit.components.steps.summarystep.literal.10")}
                      </span>
                    </div>
                  )}
                </div>
              )}

              {activeSlide === 2 && (
                <div className="space-y-2 pointer-events-none">
                  {jobType === "run" ? (
                    <div className="space-y-2">
                      <ModelChip
                        config={modelConfig}
                        roleLabel={msg("model.generation.label")}
                        onClick={() => {}}
                      />
                      {secondModelConfig?.name && (
                        <ModelChip
                          config={secondModelConfig}
                          roleLabel={TERMS.reflectionModel}
                          onClick={() => {}}
                        />
                      )}
                    </div>
                  ) : (
                    (() => {
                      const genCount = generationModels.filter((m) => m.name).length;
                      const refCount = reflectionModels.filter((m) => m.name).length;
                      const totalPairs = genCount * refCount;
                      return (
                        <div className="space-y-2">
                          <div className="flex items-center justify-between rounded-lg border border-border/40 bg-muted/20 px-3 py-2">
                            <span className="text-[0.625rem] uppercase tracking-wide text-muted-foreground">
                              {msg("auto.features.submit.components.steps.summarystep.17")}
                            </span>
                            <span className="font-mono text-sm text-foreground" dir="ltr">
                              {genCount} × {refCount} ={" "}
                              <span className="font-medium">{totalPairs}</span>
                            </span>
                          </div>
                          <span className="text-[0.625rem] uppercase tracking-wide text-muted-foreground">
                            {msg("model.generation.label_plural")}
                          </span>
                          <div className="space-y-1.5">
                            {generationModels
                              .filter((m) => m.name)
                              .map((m, i) => (
                                <ModelChip key={i} config={m} onClick={() => {}} />
                              ))}
                          </div>
                          <span className="text-[0.625rem] uppercase tracking-wide text-muted-foreground">
                            {msg("auto.features.submit.components.steps.summarystep.18")}
                          </span>
                          <div className="space-y-1.5">
                            {reflectionModels
                              .filter((m) => m.name)
                              .map((m, i) => (
                                <ModelChip key={i} config={m} onClick={() => {}} />
                              ))}
                          </div>
                        </div>
                      );
                    })()
                  )}
                </div>
              )}

              {activeSlide === 3 && (
                <div className="grid grid-cols-1 gap-x-10 sm:grid-cols-2">
                  {isReact && (
                    <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                      <span className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Stack className="size-3.5" />
                        {msg("submit.react.mcp_url_label")}
                      </span>
                      <span
                        className="text-sm font-medium font-mono truncate max-w-[60%]"
                        dir="ltr"
                        title={reactConfig.mcpUrl}
                      >
                        {reactConfig.mcpUrl || "—"}
                      </span>
                    </div>
                  )}
                  <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                    <span className="flex items-center gap-2 text-xs text-muted-foreground">
                      <MagnifyingGlass className="size-3.5" />
                      {msg("auto.features.submit.components.steps.summarystep.19")}
                    </span>
                    <span className="text-sm font-medium">
                      {autoLevel === "light"
                        ? msg("auto.features.submit.components.steps.summarystep.literal.11")
                        : autoLevel === "medium"
                          ? msg("auto.features.submit.components.steps.summarystep.literal.12")
                          : msg("auto.features.submit.components.steps.summarystep.literal.13")}
                    </span>
                  </div>
                  {advanced && (
                    <>
                      <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                        <span className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Target className="size-3.5" />
                          {msg("auto.features.submit.components.steps.summarystep.25")}
                        </span>
                        <span className="text-sm font-medium font-mono" dir="ltr">
                          {targetScore ? `${targetScore}%` : "—"}
                        </span>
                      </div>
                      <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                        <span className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Database className="size-3.5" />
                          {msg("auto.features.submit.components.steps.summarystep.20")}
                        </span>
                        <span className="text-sm font-medium font-mono">
                          {reflectionMinibatchSize || "—"}
                        </span>
                      </div>
                      <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                        <span className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Stack className="size-3.5" />
                          {msg("auto.features.submit.components.steps.summarystep.21")}
                        </span>
                        <span className="text-sm font-medium font-mono">{maxFullEvals || "—"}</span>
                      </div>
                      <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                        <span className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Gauge className="size-3.5" />
                          {msg("submit.metric_calls")}
                        </span>
                        <span className="text-sm font-medium font-mono">
                          {maxMetricCalls || "—"}
                        </span>
                      </div>
                      <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                        <span className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Shuffle className="size-3.5" />
                          {msg("auto.features.submit.components.steps.summarystep.22")}
                        </span>
                        <span className="text-sm font-medium">
                          {useMerge
                            ? msg("auto.features.submit.components.steps.summarystep.literal.14")
                            : msg("auto.features.submit.components.steps.summarystep.literal.15")}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              )}

              {activeSlide === 4 && (
                <div className="space-y-4">
                  {isWorkflow && workflowSpec && (
                    <p className="text-xs text-muted-foreground">
                      {formatMsg("workflow.summary.graph", {
                        p1: workflowSpec.nodes.length,
                        p2: workflowSpec.edges.length,
                      })}
                    </p>
                  )}
                  <div
                    className={cn(
                      "grid grid-cols-1 gap-6",
                      displaySignatureCode && metricCode && "lg:grid-cols-2",
                    )}
                  >
                    {displaySignatureCode && (
                      <section className="min-w-0 space-y-3">
                        <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          {msg("auto.features.submit.components.steps.summarystep.23")}
                        </h4>
                        <CodeEditor
                          value={displaySignatureCode}
                          onChange={() => {}}
                          height={`${Math.max(180, Math.min(displaySignatureCode.split("\n").length + 1, 14) * 19.6 + 8)}px`}
                          readOnly
                        />
                      </section>
                    )}
                    {metricCode && (
                      <section className="min-w-0 space-y-3">
                        <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          {msg("auto.features.submit.components.steps.summarystep.24")}
                        </h4>
                        <CodeEditor
                          value={metricCode}
                          onChange={() => {}}
                          height={`${Math.max(180, Math.min(metricCode.split("\n").length + 1, 14) * 19.6 + 8)}px`}
                          readOnly
                        />
                      </section>
                    )}
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>

        <div className="flex items-center justify-between gap-3 border-t border-border/60 bg-secondary/25 px-4 py-3 sm:px-6">
          <button
            type="button"
            onClick={() => goToSlide(activeSlide - 1)}
            disabled={!previousSlide}
            aria-label={msg("auto.features.agent.panel.components.toolscarousel.literal.14")}
            className="inline-flex min-h-11 min-w-11 cursor-pointer items-center gap-2 rounded-xl px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-background/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882] disabled:cursor-not-allowed disabled:opacity-30"
          >
            <PreviousIcon className="size-4" aria-hidden="true" />
            <span className="hidden sm:inline">{previousSlide?.label}</span>
          </button>

          <div className="flex items-center justify-center gap-1 sm:hidden">
            {SUMMARY_SLIDES.map((slide, index) => (
              <button
                key={slide.id}
                type="button"
                onClick={() => goToSlide(index)}
                aria-label={slide.label}
                aria-current={activeSlide === index ? "step" : undefined}
                className="flex size-8 cursor-pointer items-center justify-center rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]"
              >
                <span
                  className={cn(
                    "h-1.5 rounded-full transition-[width,background-color] duration-200",
                    activeSlide === index ? "w-5 bg-[#3D2E22]" : "w-1.5 bg-[#3D2E22]/20",
                  )}
                />
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => goToSlide(activeSlide + 1)}
            disabled={!nextSlide}
            aria-label={msg("auto.features.agent.panel.components.toolscarousel.literal.15")}
            className="inline-flex min-h-11 min-w-11 cursor-pointer items-center gap-2 rounded-xl px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-background/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882] disabled:cursor-not-allowed disabled:opacity-30"
          >
            <span className="hidden sm:inline">{nextSlide?.label}</span>
            <NextIcon className="size-4" aria-hidden="true" />
          </button>
        </div>
      </motion.div>

      <div className="rounded-xl border border-[#C8B9A8]/50 bg-[#FAF8F5] px-3.5 py-3 shadow-[0_1px_2px_rgba(61,46,34,0.04)]">
        <div className="flex items-center justify-between gap-3">
          <span className="flex items-center gap-2 text-[13px] font-semibold text-[#3D2E22]">
            <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-[#C8A882]/15 text-[#A8895E]">
              <Gauge className="h-3 w-3" />
            </span>
            {byok ? msg("submit.summary.estimate_fee") : msg("submit.summary.estimate_cost")}
          </span>
          <span className="text-[13px] font-medium text-[#3D2E22]" dir="auto">
            {/* Isolate "low–high" as one LTR run (U+2066…U+2069) so the en-dash
                between the two number groups doesn't flip them under RTL. */}
            {formatMsg("submit.summary.estimate_range", {
              low: `\u2066${formatCredits(estimate.lowCredits, locale)}`,
              high: `${formatCredits(estimate.highCredits, locale)}\u2069`,
            })}
          </span>
        </div>
        {maxCostCredits != null && (
          <p className="mt-1.5 text-[11px] text-[#8C7A6B]">
            {formatMsg("submit.summary.estimate_capped", {
              cap: formatCredits(maxCostCredits, locale),
            })}
          </p>
        )}
      </div>
    </div>
  );
}
