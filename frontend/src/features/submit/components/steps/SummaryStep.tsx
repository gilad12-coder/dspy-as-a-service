"use client";

import dynamic from "next/dynamic";
import { motion, AnimatePresence } from "framer-motion";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/shared/ui/primitives/tabs";
import { Separator } from "@/shared/ui/primitives/separator";
import {
  User,
  Code,
  Tag,
  Layers,
  Component,
  Target,
  FileText,
  Columns,
  Shuffle,
  Search,
  Database,
  Cpu,
  Boxes,
  Gauge,
} from "lucide-react";
import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";
import { moduleLabel } from "@/shared/lib/formatters";
import { TERMS } from "@/shared/lib/terms";
import { ModelChip } from "@/shared/ui/model-chip";
import { Skeleton } from "@/shared/ui/skeleton";
import { formatCredits, useCredits } from "@/features/billing";
import { getActiveIntlLocale } from "@/shared/lib/runtime-locale";

import { chargeableBracket } from "../../lib/cost-bracket";
import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={200} borderRadius={8} />,
});

const SUMMARY_TABS = perLocale(() => [
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

/**
 * Read-only row shown when a grid side is set to "all available models".
 *
 * Mirrors the sentinel chip in ModelStep but without interactions — the
 * summary tab is always pointer-events-none so this just has to carry the
 * same visual language (Boxes icon, primary tint, catalog count).
 */
function SummaryAllAvailableRow({ count }: { count: number }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border border-primary/30 bg-primary/5 px-3 py-2.5">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
        <Boxes className="size-4" />
      </span>
      <div className="flex min-w-0 flex-1 flex-col">
        <span className="text-sm font-medium text-foreground">
          {msg("auto.features.submit.components.steps.summarystep.1")}
        </span>
        <span className="text-[0.6875rem] text-muted-foreground">
          {count}
          {msg("auto.features.submit.components.steps.summarystep.2")}
        </span>
      </div>
    </div>
  );
}

export function SummaryStep({ w }: { w: SubmitWizardContext }) {
  const {
    summaryTab,
    setSummaryTab,
    summaryCodeTab,
    setSummaryCodeTab,
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
    useAllGenerationModels,
    useAllReflectionModels,
    catalog,
    autoLevel,
    reflectionMinibatchSize,
    maxFullEvals,
    useMerge,
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
  const { wallet } = useCredits();
  const locale = getActiveIntlLocale();
  const byok = wallet.mode === "byok";
  const estimate = chargeableBracket(costBracket, wallet.mode);

  return (
    <div className="space-y-4" data-tutorial="wizard-step-6">
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3, ease: [0.2, 0.8, 0.2, 1] }}
        className="rounded-2xl border border-border bg-card/80 backdrop-blur-xl shadow-lg overflow-hidden"
      >
        <div className="relative flex border-b border-border bg-secondary/50 p-1 gap-0.5">
          <div
            className="absolute top-1 bottom-1 rounded-lg bg-background shadow-sm transition-[inset-inline-start] duration-200 ease-out pointer-events-none"
            style={{
              width: `calc((100% - 12px) / ${SUMMARY_TABS.length})`,
              insetInlineStart: `calc(${summaryTab} * ${100 / SUMMARY_TABS.length}% + 4px)`,
            }}
          />
          {SUMMARY_TABS.map((tab, i) => (
            <button
              key={tab.id}
              type="button"
              onClick={() => setSummaryTab(i)}
              className={cn(
                "relative z-10 flex-1 flex items-center justify-center gap-1.5 rounded-lg py-2.5 text-xs font-medium transition-colors duration-150 cursor-pointer",
                summaryTab === i
                  ? "text-foreground"
                  : "text-muted-foreground hover:text-foreground/80",
              )}
            >
              {tab.icon}
              <span className="hidden sm:inline">{tab.label}</span>
            </button>
          ))}
        </div>

        <div className="p-5">
          <AnimatePresence mode="wait">
            <motion.div
              key={summaryTab}
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.15 }}
            >
              {summaryTab === 0 && (
                <div className="space-y-0">
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
                      <Layers className="size-3.5" />
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
                      <Component className="size-3.5" />
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

              {summaryTab === 1 && (
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
                  <Separator />
                  <div className="space-y-3">
                    <span className="flex items-center gap-2 text-xs text-muted-foreground">
                      <Layers className="size-3.5" />
                      {msg("auto.features.submit.components.steps.summarystep.12")}
                      {TERMS.dataset}
                    </span>
                    <div className="flex h-3 rounded-full overflow-hidden">
                      <div className="bg-[#3D2E22]" style={{ width: `${split.train * 100}%` }} />
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
                </div>
              )}

              {summaryTab === 2 && (
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
                      const availableCount = catalog?.models.length ?? 0;
                      const genCount = useAllGenerationModels
                        ? availableCount
                        : generationModels.filter((m) => m.name).length;
                      const refCount = useAllReflectionModels
                        ? availableCount
                        : reflectionModels.filter((m) => m.name).length;
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
                            {useAllGenerationModels ? (
                              <SummaryAllAvailableRow count={availableCount} />
                            ) : (
                              generationModels
                                .filter((m) => m.name)
                                .map((m, i) => <ModelChip key={i} config={m} onClick={() => {}} />)
                            )}
                          </div>
                          <span className="text-[0.625rem] uppercase tracking-wide text-muted-foreground">
                            {msg("auto.features.submit.components.steps.summarystep.18")}
                          </span>
                          <div className="space-y-1.5">
                            {useAllReflectionModels ? (
                              <SummaryAllAvailableRow count={availableCount} />
                            ) : (
                              reflectionModels
                                .filter((m) => m.name)
                                .map((m, i) => <ModelChip key={i} config={m} onClick={() => {}} />)
                            )}
                          </div>
                        </div>
                      );
                    })()
                  )}
                </div>
              )}

              {summaryTab === 3 && (
                <div className="space-y-0">
                  {isReact && (
                    <div className="flex items-center justify-between py-2.5 border-b border-border/40">
                      <span className="flex items-center gap-2 text-xs text-muted-foreground">
                        <Boxes className="size-3.5" />
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
                      <Search className="size-3.5" />
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
                      <Layers className="size-3.5" />
                      {msg("auto.features.submit.components.steps.summarystep.21")}
                    </span>
                    <span className="text-sm font-medium font-mono">{maxFullEvals || "—"}</span>
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
                </div>
              )}

              {summaryTab === 4 && (
                <Tabs
                  defaultValue={displaySignatureCode ? "signature" : "metric"}
                  dir="ltr"
                  onValueChange={setSummaryCodeTab}
                >
                  {isWorkflow && workflowSpec && (
                    <p className="mb-2 text-xs text-muted-foreground">
                      {formatMsg("workflow.summary.graph", {
                        p1: workflowSpec.nodes.length,
                        p2: workflowSpec.edges.length,
                      })}
                    </p>
                  )}
                  <TabsList className="relative inline-flex w-full rounded-lg bg-muted p-1 gap-1 border-none shadow-none h-auto">
                    {displaySignatureCode && metricCode && (
                      <div
                        className="absolute top-1 bottom-1 w-[calc(50%-6px)] rounded-md bg-[#3D2E22] shadow-sm transition-[inset-inline-start] duration-200 ease-out"
                        style={{
                          insetInlineStart: summaryCodeTab === "signature" ? 4 : "calc(50% + 2px)",
                        }}
                      />
                    )}
                    {displaySignatureCode && (
                      <TabsTrigger
                        value="signature"
                        className="relative z-10 rounded-md px-4 py-2 text-sm font-medium cursor-pointer border-none shadow-none bg-transparent data-[state=active]:bg-transparent data-[state=active]:text-white data-[state=active]:shadow-none data-[state=active]:border-none gap-1.5"
                      >
                        {msg("auto.features.submit.components.steps.summarystep.23")}
                      </TabsTrigger>
                    )}
                    {metricCode && (
                      <TabsTrigger
                        value="metric"
                        className="relative z-10 rounded-md px-4 py-2 text-sm font-medium cursor-pointer border-none shadow-none bg-transparent data-[state=active]:bg-transparent data-[state=active]:text-white data-[state=active]:shadow-none data-[state=active]:border-none gap-1.5"
                      >
                        {msg("auto.features.submit.components.steps.summarystep.24")}
                      </TabsTrigger>
                    )}
                  </TabsList>
                  {displaySignatureCode && (
                    <TabsContent value="signature">
                      <CodeEditor
                        value={displaySignatureCode}
                        onChange={() => {}}
                        height={`${Math.min(displaySignatureCode.split("\n").length + 1, 10) * 19.6 + 8}px`}
                        readOnly
                      />
                    </TabsContent>
                  )}
                  {metricCode && (
                    <TabsContent value="metric">
                      <CodeEditor
                        value={metricCode}
                        onChange={() => {}}
                        height={`${Math.min(metricCode.split("\n").length + 1, 10) * 19.6 + 8}px`}
                        readOnly
                      />
                    </TabsContent>
                  )}
                </Tabs>
              )}
            </motion.div>
          </AnimatePresence>
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
