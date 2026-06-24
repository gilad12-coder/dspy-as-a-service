"use client";

import * as React from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  ArrowRight,
  Check,
  ChevronDown,
  Loader2,
  Play,
  Search,
  Sparkles,
  Trash2,
  Trophy,
  XCircle,
} from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import { Button } from "@/shared/ui/primitives/button";
import { Label } from "@/shared/ui/primitives/label";
import { Input } from "@/shared/ui/primitives/input";
import { InlineErrorRow } from "@/shared/ui/inline-error-row";
import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";
import { probeModels, type ModelProbeRequest } from "@/shared/lib/api";
import type { CatalogModel, ColumnMapping } from "@/shared/types/api";

import type { SubmitWizardContext } from "../hooks/use-submit-wizard";
import {
  DEPRECATED_PATTERN,
  EXCLUDE_PATTERN,
  MAX_LOG_LINES,
  MIN_ROWS,
  groupByProvider,
  rowAsymptote,
  rowRankingScore,
  scoreModel,
  type ModelRow,
  type ProbeLogEntry,
} from "./model-probe-model";
import {
  ModelDetailPanel,
  PodiumCard,
  RaceRow,
  StatusIcon,
  TrajectoryCompareChart,
} from "./model-probe-panels";

interface ModelProbeDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  w: SubmitWizardContext;
  onSelect: (modelValue: string) => void;
  onRunningChange?: (running: boolean) => void;
  onHasResultsChange?: (has: boolean) => void;
}

export function ModelProbeDialog({
  open,
  onOpenChange,
  w,
  onSelect,
  onRunningChange,
  onHasResultsChange,
}: ModelProbeDialogProps) {
  const {
    signatureCode,
    metricCode,
    parsedDataset,
    columnRoles,
    moduleName,
    optimizerName,
    seed,
    shuffle,
    catalog,
  } = w;

  const catalogModels = React.useMemo(() => catalog?.models ?? [], [catalog]);
  const [search, setSearch] = React.useState("");
  const [selected, setSelected] = React.useState<Set<string>>(new Set());
  const [reflectionModel, setReflectionModel] = React.useState<string | null>(null);
  const [reflectionPickerOpen, setReflectionPickerOpen] = React.useState(false);
  const [reflectionSearch, setReflectionSearch] = React.useState("");
  const reflectionPickerRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!reflectionPickerOpen) return;
    const handleClick = (e: MouseEvent) => {
      if (!reflectionPickerRef.current?.contains(e.target as Node)) {
        setReflectionPickerOpen(false);
        setReflectionSearch("");
      }
    };
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, [reflectionPickerOpen]);

  React.useEffect(() => {
    if (reflectionModel !== null) return;
    const first = catalogModels[0];
    if (!first) return;
    const ranked = [...catalogModels]
      .filter((m) => !EXCLUDE_PATTERN.test(m.value) && !DEPRECATED_PATTERN.test(m.value))
      .sort((a, b) => scoreModel(b.value) - scoreModel(a.value));
    setReflectionModel((ranked[0] ?? first).value);
  }, [catalogModels, reflectionModel]);

  const selectedReflectionModel = React.useMemo(
    () => catalogModels.find((m) => m.value === reflectionModel) ?? null,
    [catalogModels, reflectionModel],
  );

  const filteredReflectionModels = React.useMemo(() => {
    const q = reflectionSearch.trim().toLowerCase();
    if (!q) return catalogModels;
    return catalogModels.filter(
      (m) => m.label.toLowerCase().includes(q) || m.value.toLowerCase().includes(q),
    );
  }, [catalogModels, reflectionSearch]);

  const groupedFilteredReflection = React.useMemo(
    () => groupByProvider(filteredReflectionModels),
    [filteredReflectionModels],
  );

  const [phase, setPhase] = React.useState<"idle" | "running" | "done" | "error">("idle");
  const [errorMessage, setErrorMessage] = React.useState<string | null>(null);
  const [total, setTotal] = React.useState<number>(0);
  const [modelRows, setModelRows] = React.useState<Record<string, ModelRow>>({});
  const [expandedModel, setExpandedModel] = React.useState<string | null>(null);
  const [now, setNow] = React.useState<number>(() => Date.now());
  const abortRef = React.useRef<AbortController | null>(null);

  React.useEffect(() => {
    if (phase !== "running") return;
    const id = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(id);
  }, [phase]);

  const columnMapping = React.useMemo<ColumnMapping>(() => {
    const inputs: Record<string, string> = {};
    const outputs: Record<string, string> = {};
    Object.entries(columnRoles).forEach(([col, role]) => {
      if (role === "input") inputs[col] = col;
      else if (role === "output") outputs[col] = col;
    });
    return { inputs, outputs };
  }, [columnRoles]);

  const dataset = parsedDataset?.rows as Array<Record<string, unknown>> | undefined;
  const datasetRows = dataset?.length ?? 0;

  const hasEnoughRows = datasetRows >= MIN_ROWS;
  const hasInputs = Object.keys(columnMapping.inputs).length > 0;
  const hasOutputs = Object.keys(columnMapping.outputs).length > 0;
  const hasSignature = signatureCode.trim().length > 0;
  const hasMetric = metricCode.trim().length > 0;
  const hasCatalog = catalogModels.length > 0;

  const modelsToRun = React.useMemo<CatalogModel[]>(
    () => catalogModels.filter((m) => selected.has(m.value)),
    [catalogModels, selected],
  );

  const canStart =
    phase !== "running" &&
    hasSignature &&
    hasMetric &&
    hasEnoughRows &&
    hasInputs &&
    hasOutputs &&
    hasCatalog &&
    !!reflectionModel &&
    modelsToRun.length > 0;

  const prerequisiteMessages: string[] = [];
  if (!hasSignature)
    prerequisiteMessages.push(msg("auto.features.submit.components.modelprobedialog.literal.1"));
  if (!hasMetric)
    prerequisiteMessages.push(
      formatMsg("auto.features.submit.components.modelprobedialog.template.1", {
        p1: TERMS.metric,
      }),
    );
  if (!hasInputs || !hasOutputs)
    prerequisiteMessages.push(msg("auto.features.submit.components.modelprobedialog.literal.2"));
  if (!hasEnoughRows) {
    prerequisiteMessages.push(
      formatMsg("auto.features.submit.components.modelprobedialog.template.2", {
        p1: MIN_ROWS,
        p2: datasetRows,
      }),
    );
  }
  if (!hasCatalog)
    prerequisiteMessages.push(msg("auto.features.submit.components.modelprobedialog.literal.3"));

  const reset = React.useCallback(() => {
    setPhase("idle");
    setErrorMessage(null);
    setTotal(0);
    setModelRows({});
    setExpandedModel(null);
  }, []);

  React.useEffect(() => {
    onRunningChange?.(phase === "running");
    onHasResultsChange?.(phase === "done" || phase === "error");
  }, [phase, onRunningChange, onHasResultsChange]);

  // When the dialog closes, only reset if we have no results to preserve.
  // Results (phase "done"/"error") persist so the user can reopen and see them.
  // Only the abort button (stop) resets everything.
  React.useEffect(() => {
    if (open) return;
    if (phase === "running") return;
    // If we have results, keep them — just clean up UI state
    if (phase === "done" || phase === "error") {
      setExpandedModel(null);
      return;
    }
    abortRef.current?.abort();
    abortRef.current = null;
    setSearch("");
    setSelected(new Set());
    setReflectionPickerOpen(false);
    setReflectionSearch("");
  }, [open, phase]);

  const toggleOne = (value: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(value)) next.delete(value);
      else next.add(value);
      return next;
    });
  };

  const filteredModels = React.useMemo(() => {
    const q = search.trim().toLowerCase();
    const filtered = catalogModels.filter((m) => {
      if (q && !m.label.toLowerCase().includes(q) && !m.value.toLowerCase().includes(q))
        return false;
      return true;
    });
    return filtered.sort((a, b) => {
      const aS = selected.has(a.value) ? 0 : 1;
      const bS = selected.has(b.value) ? 0 : 1;
      return aS - bS;
    });
  }, [catalogModels, search, selected]);

  const selectAllVisible = () => {
    setSelected((prev) => {
      const next = new Set(prev);
      for (const m of filteredModels) next.add(m.value);
      return next;
    });
  };
  const clearAllSelection = () => setSelected(new Set());

  const start = async () => {
    if (!dataset || !canStart) return;
    setExpandedModel(null);
    const initialRows: Record<string, ModelRow> = {};
    modelsToRun.forEach((m, idx) => {
      initialRows[m.value] = {
        position: idx,
        model: m.value,
        label: m.label,
        provider: m.provider,
        status: "pending",
        logs: [],
        score: null,
        scaling: null,
        trajectory: [],
        durationMs: null,
        errorMessage: null,
        startedAt: null,
      };
    });
    setPhase("running");
    setErrorMessage(null);
    setTotal(modelsToRun.length);
    setModelRows(initialRows);
    setNow(Date.now());

    const controller = new AbortController();
    abortRef.current = controller;

    const payload: ModelProbeRequest = {
      signature_code: signatureCode,
      metric_code: metricCode,
      module_name: moduleName,
      optimizer_name: optimizerName,
      dataset,
      column_mapping: columnMapping,
      train_count: 12,
      eval_count: 4,
      shuffle,
      seed: seed ?? null,
      model_ids: modelsToRun.map((m) => m.value),
      reflection_model_name: reflectionModel,
    };

    await probeModels(payload, {
      signal: controller.signal,
      onStart: (e) => setTotal(e.total),
      onModelStart: (e) =>
        setModelRows((prev) => {
          const existing = prev[e.model];
          const base: ModelRow = existing ?? {
            position: e.position,
            model: e.model,
            label: e.label,
            provider: e.provider,
            status: "running",
            logs: [],
            score: null,
            scaling: null,
            trajectory: [],
            durationMs: null,
            errorMessage: null,
            startedAt: Date.now(),
          };
          return {
            ...prev,
            [e.model]: {
              ...base,
              position: e.position,
              label: e.label,
              provider: e.provider,
              status: "running",
              startedAt: Date.now(),
            },
          };
        }),
      onLog: (e) =>
        setModelRows((prev) => {
          const entry = Object.values(prev).find((r) => r.position === e.position);
          if (!entry) return prev;
          const logEntry: ProbeLogEntry = {
            timestamp: e.timestamp,
            level: e.level,
            logger: e.logger,
            message: e.message,
          };
          const nextLogs = entry.logs.concat(logEntry);
          if (nextLogs.length > MAX_LOG_LINES) {
            nextLogs.splice(0, nextLogs.length - MAX_LOG_LINES);
          }
          return { ...prev, [entry.model]: { ...entry, logs: nextLogs } };
        }),
      onTrajectory: (e) =>
        setModelRows((prev) => {
          const entry = Object.values(prev).find((r) => r.position === e.position);
          if (!entry) return prev;
          const nextTrajectory = entry.trajectory.concat({
            step: e.point.step,
            score: e.point.score,
          });
          return {
            ...prev,
            [entry.model]: {
              ...entry,
              trajectory: nextTrajectory,
              scaling: e.scaling,
            },
          };
        }),
      onResult: (e) =>
        setModelRows((prev) => {
          const existing = prev[e.model];
          const base: ModelRow = existing ?? {
            position: e.position,
            model: e.model,
            label: e.label,
            provider: e.provider,
            status: "pending",
            logs: [],
            score: null,
            scaling: null,
            trajectory: [],
            durationMs: null,
            errorMessage: null,
            startedAt: null,
          };
          return {
            ...prev,
            [e.model]: {
              ...base,
              status: e.status === "ok" ? "done" : "error",
              score: e.score,
              scaling: e.scaling,
              durationMs: e.duration_ms,
              errorMessage: e.message ?? null,
            },
          };
        }),
      onComplete: () => setPhase("done"),
      onError: (message) => {
        setPhase("error");
        setErrorMessage(message);
      },
    });
  };

  const stop = () => {
    abortRef.current?.abort();
    abortRef.current = null;
    reset();
  };

  const allRows = React.useMemo(() => Object.values(modelRows), [modelRows]);
  const completedCount = React.useMemo(
    () => allRows.filter((r) => r.status === "done" || r.status === "error").length,
    [allRows],
  );
  const sortedRows = React.useMemo(() => {
    const rank = (r: ModelRow): number => {
      if (r.status === "running") return 0;
      if (r.status === "done") return 1;
      if (r.status === "pending") return 2;
      return 3;
    };
    return [...allRows].sort((a, b) => {
      const da = rank(a);
      const db = rank(b);
      if (da !== db) return da - db;
      if (a.status === "done" && b.status === "done") {
        return rowRankingScore(b) - rowRankingScore(a);
      }
      return a.position - b.position;
    });
  }, [allRows]);

  const podiumRows = React.useMemo(
    () =>
      [...allRows]
        .filter((r) => r.status === "done" && (r.score !== null || rowAsymptote(r) !== null))
        .sort((a, b) => rowRankingScore(b) - rowRankingScore(a))
        .slice(0, 3),
    [allRows],
  );

  const progressPct = total > 0 ? (completedCount / total) * 100 : 0;
  const expandedRow = expandedModel ? (modelRows[expandedModel] ?? null) : null;

  const tabCls =
    "relative px-2.5 sm:px-4 py-2.5 rounded-none border-b-2 border-transparent " +
    "data-[state=active]:border-transparent data-[state=active]:border-b-primary " +
    "data-[state=active]:text-foreground data-[state=active]:bg-transparent " +
    "data-[state=active]:shadow-none transition-all duration-200 text-xs sm:text-sm";

  const handleSelect = (value: string) => {
    onSelect(value);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[82vh] max-h-[82vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-[min(90vw,1060px)]">
        <DialogHeader className="sr-only">
          <DialogTitle>{msg("auto.features.submit.components.modelprobedialog.1")}</DialogTitle>
          <DialogDescription>
            {msg("auto.features.submit.components.modelprobedialog.2")}
          </DialogDescription>
        </DialogHeader>
        {errorMessage && (
          <InlineErrorRow
            title={msg("auto.features.submit.components.modelprobedialog.3")}
            message={errorMessage}
            onDismiss={() => setErrorMessage(null)}
            className="mx-6 mt-4"
          />
        )}
        <div className="flex-1 space-y-4 overflow-y-auto px-6 py-4">
          {phase === "idle" && hasCatalog && (
            <div className="space-y-2 pt-4">
              <div className="flex items-center">
                <Label className="text-xs font-semibold">
                  {TERMS.reflectionModel}
                  {msg("auto.features.submit.components.modelprobedialog.4")}
                </Label>
              </div>
              <div ref={reflectionPickerRef} className="relative">
                <button
                  type="button"
                  onClick={() => setReflectionPickerOpen((o) => !o)}
                  className={cn(
                    "flex w-full items-center gap-3 rounded-lg border border-border/50 bg-card/80 px-3 py-2 text-start transition-colors cursor-pointer",
                    "hover:border-primary/40 hover:shadow-sm",
                    reflectionPickerOpen && "border-primary/40 shadow-sm",
                  )}
                  aria-haspopup="listbox"
                  aria-expanded={reflectionPickerOpen}
                >
                  <span className="flex min-w-0 flex-1 flex-col items-start gap-0.5">
                    {selectedReflectionModel ? (
                      <span
                        className="truncate text-sm font-mono font-medium text-foreground"
                        dir="ltr"
                      >
                        {selectedReflectionModel.label}
                      </span>
                    ) : (
                      <span className="truncate text-sm text-muted-foreground" dir="ltr">
                        {msg("auto.features.submit.components.modelprobedialog.5")}
                      </span>
                    )}
                  </span>
                  <ChevronDown
                    className={cn(
                      "size-4 shrink-0 text-muted-foreground transition-transform duration-150",
                      reflectionPickerOpen && "rotate-180",
                    )}
                  />
                </button>
                {reflectionPickerOpen && (
                  <div
                    role="listbox"
                    className="absolute z-50 mt-1 w-full rounded-xl border border-border/70 bg-popover shadow-lg animate-in fade-in-0 zoom-in-95 slide-in-from-top-1"
                  >
                    <div className="sticky top-0 z-10 border-b border-border/40 bg-popover p-2">
                      <div className="relative">
                        <Search className="pointer-events-none absolute top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground start-2.5" />
                        <input
                          value={reflectionSearch}
                          onChange={(e) => setReflectionSearch(e.target.value)}
                          placeholder="Search models..."
                          dir="auto"
                          className="w-full rounded-md border border-border/60 bg-background py-1.5 ps-8 pe-3 text-xs outline-none placeholder:text-muted-foreground focus:border-primary/40"
                          autoFocus
                          onMouseDown={(e) => e.stopPropagation()}
                        />
                      </div>
                    </div>
                    <div className="max-h-[40vh] overflow-y-auto">
                      {filteredReflectionModels.length === 0 ? (
                        <div className="p-4 text-center text-xs text-muted-foreground">
                          {msg("auto.features.submit.components.modelprobedialog.6")}
                        </div>
                      ) : (
                        Array.from(groupedFilteredReflection.entries()).map(([provider, list]) => (
                          <div key={provider} className="border-b border-border/40 last:border-b-0">
                            <div className="bg-muted/40 px-3 py-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-muted-foreground">
                              {provider}
                            </div>
                            {list.map((m) => {
                              const isActive = m.value === reflectionModel;
                              return (
                                <button
                                  key={m.value}
                                  type="button"
                                  onClick={() => {
                                    setReflectionModel(m.value);
                                    setReflectionPickerOpen(false);
                                    setReflectionSearch("");
                                  }}
                                  role="option"
                                  aria-selected={isActive}
                                  className={cn(
                                    "flex w-full items-center gap-2 px-3 py-2 text-start transition-colors cursor-pointer",
                                    isActive ? "bg-primary/10" : "hover:bg-accent/60",
                                  )}
                                >
                                  <Check
                                    className={cn(
                                      "size-3.5 shrink-0 text-primary",
                                      isActive ? "opacity-100" : "opacity-0",
                                    )}
                                  />
                                  <span
                                    className="min-w-0 flex-1 truncate font-mono text-xs"
                                    dir="ltr"
                                  >
                                    {m.label}
                                  </span>
                                </button>
                              );
                            })}
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {phase === "idle" && (
            <div className="space-y-3">
              {prerequisiteMessages.length > 0 && (
                <ul className="space-y-1 rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-[0.75rem] text-amber-700 dark:text-amber-400">
                  {prerequisiteMessages.map((msg) => (
                    <li key={msg}>• {msg}</li>
                  ))}
                </ul>
              )}

              {hasCatalog && (
                <div className="space-y-3">
                  <div className="relative">
                    <Search className="pointer-events-none absolute top-1/2 -translate-y-1/2 size-3.5 text-muted-foreground start-3" />
                    <Input
                      value={search}
                      onChange={(e) => setSearch(e.target.value)}
                      placeholder="Search models..."
                      className="ps-9"
                      dir="ltr"
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <Label className="text-xs font-semibold">
                      {msg("auto.features.submit.components.modelprobedialog.7")}
                    </Label>
                    <div className="flex items-center gap-2 text-[0.6875rem]">
                      <span className="text-muted-foreground tabular-nums">
                        {selected.size}/{catalogModels.length}
                      </span>
                      <button
                        type="button"
                        onClick={selectAllVisible}
                        className="text-primary hover:underline cursor-pointer"
                      >
                        {msg("auto.features.submit.components.modelprobedialog.9")}
                      </button>
                      <span className="text-muted-foreground/50">·</span>
                      <button
                        type="button"
                        onClick={clearAllSelection}
                        className="text-muted-foreground hover:text-destructive hover:underline cursor-pointer"
                      >
                        {msg("auto.features.submit.components.modelprobedialog.10")}
                      </button>
                    </div>
                  </div>

                  <div className="max-h-[min(30vh,280px)] overflow-y-auto rounded-lg border border-border/60 bg-card/50">
                    {filteredModels.length === 0 ? (
                      <div className="p-6 text-center text-xs text-muted-foreground">
                        {msg("auto.features.submit.components.modelprobedialog.11")}
                      </div>
                    ) : (
                      <div className="divide-y divide-border/30">
                        {filteredModels.map((m) => {
                          const isSelected = selected.has(m.value);
                          return (
                            <label
                              key={m.value}
                              className={cn(
                                "flex items-center gap-3 px-3 py-2 text-sm cursor-pointer transition-colors",
                                isSelected ? "bg-primary/5" : "hover:bg-accent/40",
                              )}
                            >
                              <input
                                type="checkbox"
                                checked={isSelected}
                                onChange={() => toggleOne(m.value)}
                                className="size-4 shrink-0 cursor-pointer accent-primary"
                              />
                              <span className="min-w-0 flex-1 truncate font-mono text-xs" dir="ltr">
                                {m.label}
                              </span>
                            </label>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {phase !== "idle" && (
            <div className="space-y-4 pt-8">
              {expandedRow ? (
                <div className="space-y-3">
                  <motion.button
                    type="button"
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    onClick={() => setExpandedModel(null)}
                    className="flex items-center gap-1.5 text-[0.75rem] text-muted-foreground transition-colors hover:text-foreground cursor-pointer"
                  >
                    <ArrowRight className="size-3.5" />
                    {msg("auto.features.submit.components.modelprobedialog.12")}
                  </motion.button>

                  <div className="flex items-center gap-3 pb-1">
                    <StatusIcon status={expandedRow.status} />
                    <div className="min-w-0 flex-1">
                      <span className="truncate font-mono text-sm font-semibold" dir="ltr">
                        {expandedRow.label}
                      </span>
                    </div>
                  </div>

                  <ModelDetailPanel row={expandedRow} now={now} tabCls={tabCls} />
                </div>
              ) : (
                <>
                  <div className="flex items-center gap-3">
                    <span className="flex shrink-0 items-center gap-1.5 text-[0.6875rem] text-muted-foreground">
                      {phase === "running" ? (
                        <Loader2 className="size-3 animate-spin text-primary" />
                      ) : phase === "done" ? (
                        <Check className="size-3 text-primary" />
                      ) : phase === "error" ? (
                        <XCircle className="size-3 text-destructive" />
                      ) : null}
                      <span className="font-semibold tabular-nums text-foreground" dir="ltr">
                        {completedCount}/{total || "?"}
                      </span>
                      <span>{msg("auto.features.submit.components.modelprobedialog.13")}</span>
                    </span>
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted" dir="ltr">
                      <motion.div
                        className="h-full bg-gradient-to-r from-primary/60 to-primary"
                        initial={{ width: 0 }}
                        animate={{ width: `${progressPct}%` }}
                        transition={{ duration: 0.4, ease: [0.2, 0.8, 0.2, 1] }}
                      />
                    </div>
                  </div>

                  <AnimatePresence>
                    {phase === "done" && podiumRows.length > 0 && (
                      <motion.div
                        key="podium"
                        initial={{ opacity: 0, y: -12 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -8 }}
                        transition={{ duration: 0.4, ease: [0.2, 0.8, 0.2, 1] }}
                        className="space-y-2"
                      >
                        <Label className="flex items-center gap-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-muted-foreground">
                          <Trophy className="size-3 text-primary" />
                          {msg("auto.features.submit.components.modelprobedialog.14")}
                        </Label>
                        <div className="grid grid-cols-1 gap-2 lg:grid-cols-3">
                          {podiumRows.map((row, idx) => (
                            <PodiumCard
                              key={row.model}
                              row={row}
                              rank={idx + 1}
                              onSelect={handleSelect}
                            />
                          ))}
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>

                  <div className="max-h-[min(40vh,360px)] overflow-y-auto">
                    <motion.ul layout className="space-y-2">
                      <AnimatePresence initial={false}>
                        {sortedRows.map((row) => (
                          <RaceRow
                            key={row.model}
                            row={row}
                            now={now}
                            onNavigate={() => setExpandedModel(row.model)}
                            onSelect={handleSelect}
                          />
                        ))}
                      </AnimatePresence>
                    </motion.ul>
                  </div>

                  {allRows.filter((r) => r.trajectory.length > 0).length >= 2 && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <Label className="flex items-center gap-1.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-muted-foreground">
                          <Sparkles className="size-3 text-primary" />
                          {msg("auto.features.submit.components.modelprobedialog.15")}
                        </Label>
                      </div>
                      <div className="rounded-lg border border-border/60 bg-card/50 p-3">
                        <TrajectoryCompareChart rows={allRows} />
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        <div className="border-t border-border/60 bg-card/80 px-6 py-3">
          {phase === "running" ? (
            <Button variant="destructive" onClick={stop} className="w-full gap-1.5">
              <XCircle className="size-4" />
              {msg("auto.features.submit.components.modelprobedialog.16")}
            </Button>
          ) : phase === "done" || phase === "error" ? (
            <div className="flex items-center gap-2">
              <Button variant="outline" onClick={() => reset()} className="flex-1 gap-1.5">
                <Play className="size-4 fill-current" />
                {msg("auto.features.submit.components.modelprobedialog.17")}
              </Button>
              <Button
                variant="outline"
                onClick={() => {
                  reset();
                  onOpenChange(false);
                }}
                className="gap-1.5 text-muted-foreground"
              >
                <Trash2 className="size-3.5" />
                {msg("auto.features.submit.components.modelprobedialog.18")}
              </Button>
            </div>
          ) : (
            <Button onClick={start} disabled={!canStart} className="w-full gap-1.5">
              <Play className="size-4 fill-current" />
              {msg("auto.features.submit.components.modelprobedialog.19")}
            </Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
