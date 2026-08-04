"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  ArrowCounterClockwise,
  Binary,
  CaretRight,
  CircleNotch,
  CursorText,
  ListChecks,
  Plus,
  Trash,
} from "@/shared/ui/icons";
import { Button } from "@/shared/ui/primitives/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/shared/ui/primitives/card";
import { Input } from "@/shared/ui/primitives/input";
import { SubmitSplashOverlay, SUBMIT_SPLASH_HOLD_MS } from "@/shared/ui/submit-splash-overlay";
import {
  AgentThread,
  ChatTranscript,
  Composer,
  ComposerModelMenu,
  QuestionChoices,
  QuestionChoicesSkeleton,
} from "@/shared/ui/agent";
import type { AgentMessage, AgentThinking } from "@/shared/ui/agent";
import type { InterviewOption } from "@/shared/lib/api";
import { cachedCatalog, getModelCatalog } from "@/shared/lib/model-catalog";
import type { CatalogModel, ModelConfig } from "@/shared/types/api";
import { ModelConfigModal, useRecentModelConfigs } from "@/features/submit";
import { ModelChip } from "@/shared/ui/model-chip";
import { formatMsg, msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import type { AutotagEstimate } from "../hooks/use-tagger";
import {
  assistModelConfig,
  calibrationTarget,
  interviewComposerEffort,
  interviewComposerModel,
} from "../lib/assist";
import type { AnnotationMode, AssistState, Category, TaggerConfig } from "../lib/types";

/**
 * Focus a just-appended field without the native focus jump: ``focus()``
 * yanks the nearest scroll container to the element instantly, which reads
 * as the list "jumping" at the add button. Focus is taken without scrolling,
 * then the reveal happens as its own smooth glide (instant under
 * reduced-motion).
 */
function focusAppendedField(el: HTMLElement): void {
  el.focus({ preventScroll: true });
  el.scrollIntoView({
    block: "nearest",
    behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
  });
}

/** The confirmed task contract handed back from the labeling-guide card. */
export interface TaskContract {
  mode: AnnotationMode;
  question?: string;
  categories?: Category[];
}

interface Props {
  config: TaggerConfig;
  assist: AssistState;
  busy: boolean;
  streamText: string;
  thinking: AgentThinking | null;
  options: InterviewOption[];
  /** What's still generating after the reply: answer choices, or — on the
   *  final turn — the labeling-guide contract. */
  pending: "options" | "contract" | null;
  error: string | null;
  /** Row count and live credit estimate for the contract card's autopilot
   *  start buttons — cost shown before commitment. */
  rowCount: number;
  estimate: AutotagEstimate | null;
  onFetchEstimate: () => void;
  /** Persist the picked tagging model config on the session's assist state. */
  onSetModel: (config: ModelConfig) => void;
  /** Persist the interviewer's model (the composer's model menu). */
  onSetInterviewModel: (model: string | null) => void;
  /** Persist the interviewer's reasoning-effort level. */
  onSetInterviewEffort: (effort: string | null) => void;
  onSend: (content: string) => void;
  onEditResend: (index: number, content: string) => void;
  onStop: () => void;
  onRetry: () => void;
  onSkip: () => void;
  /** Discard the transcript and re-run the interview from its opening question. */
  onRestart: () => void;
  onConfirmRubric: (rubric: string[], task: TaskContract) => void;
}

/**
 * The dataset interview: the same live chat experience as the generalist
 * agent — streamed reply tokens, a collapsible thinking section, stop, and
 * edit-and-resend on any earlier answer — driving a purpose-built
 * interviewer. It ends in an editable task contract — the inferred answer
 * style, its question/categories, and the labeling guide — so the whole task
 * definition is confirmable in one place.
 */
export function TaggerInterview({
  config,
  assist,
  busy,
  streamText,
  thinking,
  options,
  pending,
  error,
  rowCount,
  estimate,
  onFetchEstimate,
  onSetModel,
  onSetInterviewModel,
  onSetInterviewEffort,
  onSend,
  onEditResend,
  onStop,
  onRetry,
  onSkip,
  onRestart,
  onConfirmRubric,
}: Props) {
  const [draft, setDraft] = useState("");
  const done = assist.interview.done;
  // Skipping asks the interviewer to finish with its best guess, so it only
  // makes sense once the agent has seen the data and said something.
  const canSkip =
    !done && !busy && assist.interview.turns.some((turn) => turn.role === "assistant");
  const messages: AgentMessage[] = assist.interview.turns.map((turn) => ({
    role: turn.role,
    content: turn.content,
    model: turn.model ?? null,
    servedModel: turn.servedModel ?? null,
  }));
  // The in-flight assistant reply streams into a trailing synthetic message,
  // exactly how the agent panel renders its live turn. The server filters
  // structured-output leakage from the stream; this is the client backstop —
  // JSON must never render as the reply (the parsed message lands at done).
  const liveText = /^\s*[{[`]/.test(streamText) ? "" : streamText;
  if (busy && (liveText || thinking)) {
    messages.push({ role: "assistant", content: liveText });
  }

  const submit = () => {
    const content = draft.trim();
    if (!content || busy || done) return;
    setDraft("");
    onSend(content);
  };

  return (
    // Height subtracts page padding plus the ~2.5rem back-to-sessions bar
    // rendered above, so the thread stays viewport-locked without page scroll.
    <div className="flex h-[calc(100dvh-var(--header-height,53px)-5.5rem)] w-full flex-col md:h-[calc(100dvh-var(--header-height,53px)-6.5rem)]">
      {done ? (
        <RubricCard
          config={config}
          assist={assist}
          rowCount={rowCount}
          estimate={estimate}
          onFetchEstimate={onFetchEstimate}
          onSetModel={onSetModel}
          onConfirm={onConfirmRubric}
        />
      ) : (
        <Card className="flex min-h-0 flex-1 flex-col overflow-hidden py-0 gap-0">
          <div className="flex items-start justify-between gap-3 border-b border-border/40 px-4 py-3 shrink-0">
            <div className="min-w-0">
              <h3 className="text-sm font-semibold text-foreground">
                {msg("tagger.assist.interview.title")}
              </h3>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {msg("tagger.assist.interview.subtitle")}
              </p>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onRestart}
              disabled={busy || assist.interview.turns.length === 0}
              className="shrink-0 gap-1.5 text-muted-foreground"
            >
              <ArrowCounterClockwise className="size-3.5" />
              {msg("tagger.assist.interview.restart")}
            </Button>
          </div>

          <AgentThread
            scrollDeps={[messages.length, streamText, thinking?.reasoning]}
            isEmpty={messages.length === 0}
            emptyState={
              <div className="flex h-full flex-col items-center justify-center gap-3 text-muted-foreground">
                <CircleNotch className="size-5 animate-spin" />
                <p className="text-sm">{msg("tagger.assist.interview.reading")}</p>
              </div>
            }
          >
            <ChatTranscript
              messages={messages}
              streaming={busy}
              editAndResend={onEditResend}
              thinking={thinking ?? undefined}
              animatePairs
            />
          </AgentThread>

          {error && (
            <div className="flex items-center justify-between gap-3 border-t border-border/40 px-4 py-2.5">
              <p className="text-sm text-destructive">{msg("tagger.assist.interview.error")}</p>
              <Button variant="outline" size="sm" onClick={onRetry} className="gap-1.5 shrink-0">
                <ArrowCounterClockwise className="size-3.5" />
                {msg("tagger.assist.retry")}
              </Button>
            </div>
          )}

          {!error && options.length > 0 && !busy && (
            <QuestionChoices
              options={options}
              onSelect={onSend}
              hint={msg("tagger.assist.interview.choices_hint")}
              ariaLabel={msg("tagger.assist.interview.choices_label")}
            />
          )}
          {!error && busy && pending === "options" && <QuestionChoicesSkeleton />}
          {!error && busy && pending === "contract" && <ContractPendingIndicator />}

          <Composer
            value={draft}
            onChange={setDraft}
            onSubmit={submit}
            onStop={onStop}
            disabled={!busy && messages.length === 0}
            streaming={busy}
            placeholder={msg("tagger.assist.interview.placeholder")}
            modelMenu={
              <ComposerModelMenu
                value={interviewComposerModel(assist)}
                onChange={onSetInterviewModel}
                effort={interviewComposerEffort(assist)}
                onEffortChange={onSetInterviewEffort}
              />
            }
          />

          {canSkip && (
            <div className="flex justify-center border-t border-border/40 py-1.5 shrink-0">
              <Button variant="ghost" size="sm" onClick={onSkip} className="text-muted-foreground">
                {msg("tagger.assist.interview.skip")}
              </Button>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}

/** Time constant of the pending bar's asymptotic fill (~65% at one tau). */
const CONTRACT_PROGRESS_TAU_MS = 9_000;
/** The fill never claims completion — arrival of the contract does that. */
const CONTRACT_PROGRESS_CAP = 97;

/**
 * Placeholder while the final turn's labeling guide and task contract are
 * still generating: numbered shimmer lines in the guide card's rule-row
 * shape, so the contract screen lands where the shimmer prefigured it —
 * distinct from `QuestionChoicesSkeleton`, which promises answer choices.
 *
 * The tqdm-style bar has no real signal to read, so it fills asymptotically,
 * calibrated to typical guide-generation time: fast early progress that keeps
 * slowing but never stalls, capped short of 100 so it cannot claim a finish
 * the stream hasn't delivered.
 */
function ContractPendingIndicator() {
  const [percent, setPercent] = useState(0);
  useEffect(() => {
    const startedAt = performance.now();
    const interval = window.setInterval(() => {
      const elapsed = performance.now() - startedAt;
      setPercent(CONTRACT_PROGRESS_CAP * (1 - Math.exp(-elapsed / CONTRACT_PROGRESS_TAU_MS)));
    }, 180);
    return () => window.clearInterval(interval);
  }, []);
  return (
    <div className="flex flex-col gap-2 border-t border-border/40 px-4 pb-2 pt-3 motion-safe:animate-in motion-safe:fade-in-0">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <CircleNotch className="size-3.5 animate-spin" />
        <p role="status" className="flex-1">
          {msg("tagger.assist.interview.contract_pending")}
        </p>
        {/* aria-hidden: the ticking number would spam the status live region. */}
        <span aria-hidden className="tabular-nums">
          {Math.round(percent)}%
        </span>
      </div>
      <div aria-hidden className="flex h-1 overflow-hidden rounded-full bg-muted">
        <div
          className="rounded-full bg-primary transition-[width] duration-200 ease-linear"
          style={{ width: `${percent}%` }}
        />
      </div>
      <div aria-hidden className="flex flex-col gap-2">
        {(["w-3/4", "w-4/5", "w-2/3"] as const).map((width, index) => (
          <div key={width} className="flex items-center gap-2.5">
            <span className="w-4 select-none text-end font-mono text-xs tabular-nums text-muted-foreground/50">
              {index + 1}
            </span>
            <span
              className={cn("h-3 rounded bg-muted motion-safe:animate-pulse", width)}
              style={{ animationDelay: `${index * 0.15}s` }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Mount-triggered fade-rise. The shared ``FadeIn`` animates on viewport
 * entry, which leaves content at opacity 0 until a scroll nudge when it
 * mounts right at the fold — on the contract screen the launch button is
 * the primary CTA and must be visible at first paint.
 */
function Rise({
  delay = 0,
  className,
  children,
}: {
  delay?: number;
  className?: string;
  children: ReactNode;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 24 }}
      animate={{ opacity: 1, y: 0 }}
      transition={reduce ? { duration: 0 } : { duration: 0.5, delay, ease: [0.2, 0.8, 0.2, 1] }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

/**
 * The task contract, editable in place before anything runs: the interview's
 * inferred answer style and its question/categories, alongside the distilled
 * labeling guide — every rule editable, removable and extendable in place,
 * mirroring the code interview's brief card. Confirming
 * is the moment the user takes ownership of what the AI believes. Fields for
 * a switched-to answer style start empty on purpose — the switch is a human
 * override, so the human fills it in.
 *
 * On autopilot the launch button commits to tagging every row — scope and
 * credit estimate on the button itself — and confirming goes straight into
 * the bulk job with no interstitial screen. On copilot it commits to the
 * opening batch: the AI tags it first and the human keeps or corrects each
 * label, so nobody hand-labels from a blank slate.
 */
function RubricCard({
  config,
  assist,
  rowCount,
  estimate,
  onFetchEstimate,
  onSetModel,
  onConfirm,
}: {
  config: TaggerConfig;
  assist: AssistState;
  rowCount: number;
  estimate: AutotagEstimate | null;
  onFetchEstimate: () => void;
  onSetModel: (config: ModelConfig) => void;
  onConfirm: (rubric: string[], task: TaskContract) => void;
}) {
  const autopilot = assist.mode === "autopilot";
  // Cost before commitment: the bulk button carries the live estimate, so it
  // is fetched the moment the contract card appears — and again whenever the
  // tagging model changes, since pricing is per model.
  useEffect(() => {
    if (autopilot) onFetchEstimate();
  }, [autopilot, onFetchEstimate, assist.model]);
  // Stable identity per assist snapshot: the model dialog resyncs its draft
  // whenever this prop changes, so an inline object would reset an open
  // dialog on any unrelated re-render (e.g. the estimate arriving).
  const modelConfig = useMemo(() => assistModelConfig(assist), [assist]);
  const [mode, setMode] = useState<AnnotationMode>(config.mode);
  // The inferred question isn't edited here — it rides through the confirm
  // untouched; the rubric rules are the editable surface of the task.
  const question = config.question ?? "";
  const [categories, setCategories] = useState<Category[]>(config.categories ?? []);
  // Which just-appended field should grab focus once it mounts.
  const focusAppended = useRef<"category" | "rule" | null>(null);
  // The guide is an editable draft, like the code interview's brief; a
  // re-run interview replaces it.
  const [rules, setRules] = useState<string[]>(assist.rubric);
  useEffect(() => setRules(assist.rubric), [assist.rubric]);

  const styleOptions: Array<{
    mode: AnnotationMode;
    label: string;
    desc: string;
    icon: typeof Binary;
  }> = [
    {
      mode: "binary",
      label: msg("auto.features.tagger.components.taggersetup.literal.4"),
      desc: msg("auto.features.tagger.components.taggersetup.literal.5"),
      icon: Binary,
    },
    {
      mode: "multiclass",
      label: msg("auto.features.tagger.components.taggersetup.literal.6"),
      desc: msg("auto.features.tagger.components.taggersetup.literal.7"),
      icon: ListChecks,
    },
    {
      mode: "freetext",
      label: msg("auto.features.tagger.components.taggersetup.literal.8"),
      desc: msg("auto.features.tagger.components.taggersetup.literal.9"),
      icon: CursorText,
    },
  ];

  const switchMode = (next: AnnotationMode) => {
    setMode(next);
    if (next === "multiclass" && categories.length === 0) {
      setCategories([
        { id: crypto.randomUUID(), label: "" },
        { id: crypto.randomUUID(), label: "" },
      ]);
    }
  };

  const updateCategory = (id: string, label: string) =>
    setCategories((prev) => prev.map((c) => (c.id === id ? { ...c, label } : c)));
  const removeCategory = (id: string) => setCategories((prev) => prev.filter((c) => c.id !== id));
  const addCategory = () => {
    focusAppended.current = "category";
    setCategories((prev) => [...prev, { id: crypto.randomUUID(), label: "" }]);
  };
  const cleanedCategories = categories.filter((c) => c.label.trim());

  const updateRule = (idx: number, value: string) =>
    setRules((prev) => prev.map((r, i) => (i === idx ? value : r)));
  const removeRule = (idx: number) => setRules((prev) => prev.filter((_, i) => i !== idx));
  const addRule = () => {
    focusAppended.current = "rule";
    setRules((prev) => [...prev, ""]);
  };

  const taskValid = mode !== "multiclass" || cleanedCategories.length >= 2;

  // Copilot's launch commits to the opening batch: the AI tags it and the
  // human audits. Sized from the live contract so switching the answer style
  // updates the promise on the button.
  const copilotBatch = Math.min(
    rowCount,
    calibrationTarget({ mode, categories: cleanedCategories, inputColumns: config.inputColumns }),
  );

  const confirm = () =>
    onConfirm(rules.map((r) => r.trim()).filter(Boolean), {
      mode,
      ...(mode === "binary" ? { question: question.trim() } : {}),
      ...(mode === "multiclass" ? { categories: cleanedCategories } : {}),
    });

  // Launch mirrors the wizard submit: the splash plays for the shared hold,
  // then the confirm flips the phase and the next screen is revealed under it.
  const [launching, setLaunching] = useState(false);
  const [modelDialogOpen, setModelDialogOpen] = useState(false);
  // Managed catalog for the model dialog's thinking detection and the chip's
  // vision badge — same source the submit wizard feeds it.
  const [catalogModels, setCatalogModels] = useState<CatalogModel[] | null>(
    cachedCatalog()?.models ?? null,
  );
  useEffect(() => {
    if (catalogModels) return;
    let cancelled = false;
    getModelCatalog()
      .then((c) => {
        if (!cancelled) setCatalogModels(c.models);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [catalogModels]);
  const launch = () => {
    if (launching) return;
    setLaunching(true);
    window.setTimeout(confirm, SUBMIT_SPLASH_HOLD_MS);
  };

  // The same recents the submit wizard's model dialog keeps — one shared
  // localStorage list across every model-config surface.
  const { recentConfigs, saveToRecent, removeRecentConfig } = useRecentModelConfigs();

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <Rise className="shrink-0">
        <h3 className="text-lg font-semibold text-foreground">
          {msg("tagger.assist.rubric.title")}
        </h3>
        <p className="mt-0.5 text-sm text-muted-foreground">
          {msg("tagger.assist.rubric.subtitle")}
        </p>
      </Rise>
      <div
        className={cn(
          "grid min-h-0 flex-1 gap-3 overflow-y-auto",
          // The launch column shrink-wraps the button (auto), so the cards
          // always absorb every pixel the button doesn't use, at any button
          // width.
          "lg:grid-cols-[minmax(0,1fr)_auto] lg:grid-rows-[minmax(0,1fr)] lg:overflow-visible",
        )}
      >
        <Rise
          delay={0.06}
          className={cn(
            "flex min-w-0 flex-col gap-3 overscroll-contain lg:min-h-0 lg:overflow-y-auto",
            "lg:[mask-image:linear-gradient(to_bottom,black,black_calc(100%-10px),transparent)]",
          )}
        >
          <Card className="shrink-0">
            <CardHeader>
              <CardTitle className="text-base">
                {msg("tagger.assist.rubric.answer_style")}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                {styleOptions.map((opt) => {
                  const selected = mode === opt.mode;
                  return (
                    <button
                      key={opt.mode}
                      type="button"
                      aria-pressed={selected}
                      onClick={() => switchMode(opt.mode)}
                      className={cn(
                        "flex items-start gap-3 rounded-xl border p-3.5 text-start transition-all cursor-pointer",
                        "hover:border-primary/40 hover:bg-primary/5",
                        "focus-visible:border-ring focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/40",
                        selected ? "border-primary bg-primary/10 shadow-sm" : "border-border/50",
                      )}
                    >
                      <opt.icon
                        className={cn(
                          "mt-0.5 size-5 shrink-0",
                          selected ? "text-primary" : "text-muted-foreground",
                        )}
                      />
                      <span className="flex min-w-0 flex-col gap-0.5">
                        <span
                          className={cn(
                            "text-sm font-medium",
                            selected ? "text-primary" : "text-foreground",
                          )}
                        >
                          {opt.label}
                        </span>
                        <span className="text-xs text-muted-foreground">{opt.desc}</span>
                      </span>
                    </button>
                  );
                })}
              </div>
              <AnimatePresence mode="wait" initial={false}>
                {mode === "multiclass" && (
                  <motion.div
                    key="multiclass"
                    initial={{ opacity: 0, y: -4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.15 }}
                    className="space-y-2"
                  >
                    <p className="text-xs text-muted-foreground">
                      {msg("auto.features.tagger.components.taggersetup.9")}
                    </p>
                    {categories.map((cat, idx) => (
                      <div key={cat.id} className="flex items-center gap-2">
                        <Input
                          ref={
                            idx === categories.length - 1
                              ? (el: HTMLInputElement | null) => {
                                  if (el && focusAppended.current === "category") {
                                    focusAppended.current = null;
                                    focusAppendedField(el);
                                  }
                                }
                              : undefined
                          }
                          value={cat.label}
                          onChange={(e) => updateCategory(cat.id, e.target.value)}
                          placeholder={msg(
                            "auto.features.tagger.components.taggersetup.literal.16",
                          )}
                          aria-label={msg("auto.features.tagger.components.taggersetup.literal.16")}
                        />
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          onClick={() => removeCategory(cat.id)}
                          disabled={categories.length <= 2}
                          aria-label={msg("auto.features.tagger.components.taggersetup.16")}
                        >
                          <Trash className="size-3.5 text-muted-foreground" />
                        </Button>
                      </div>
                    ))}
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={addCategory}
                      className="gap-1.5 self-start text-muted-foreground"
                    >
                      <Plus className="size-3.5" />
                      {msg("auto.features.tagger.components.taggersetup.literal.17")}
                    </Button>
                  </motion.div>
                )}
              </AnimatePresence>
            </CardContent>
          </Card>

          <Card className="shrink-0">
            <CardHeader>
              <CardTitle className="text-base">{msg("tagger.assist.model.title")}</CardTitle>
              <CardDescription>{msg("tagger.assist.model.hint")}</CardDescription>
            </CardHeader>
            <CardContent>
              <ModelChip
                config={modelConfig}
                emptyLabel={msg("tagger.assist.model.placeholder")}
                catalogModels={catalogModels ?? undefined}
                onClick={() => setModelDialogOpen(true)}
                onRemove={assist.model ? () => onSetModel({ name: "" }) : undefined}
              />
              <ModelConfigModal
                open={modelDialogOpen}
                onOpenChange={setModelDialogOpen}
                config={modelConfig}
                onSave={(cfg) => {
                  saveToRecent(cfg);
                  onSetModel(cfg);
                }}
                roleLabel={msg("tagger.assist.model.title")}
                catalogModels={catalogModels ?? undefined}
                recentConfigs={recentConfigs}
                onRemoveRecent={removeRecentConfig}
                showConnection={false}
              />
            </CardContent>
          </Card>

          {assist.rubric.length > 0 && (
            <Card className="shrink-0">
              <CardHeader>
                <CardTitle className="text-base">
                  {msg("tagger.assist.rubric.guide_title")}
                </CardTitle>
                <CardDescription>{msg("tagger.assist.rubric.guide_hint")}</CardDescription>
              </CardHeader>
              <CardContent className="flex flex-col gap-2.5">
                {rules.map((rule, idx) => (
                  <div key={idx} className="flex items-start gap-2.5">
                    <span className="w-4 shrink-0 select-none pt-2.5 text-end font-mono text-xs tabular-nums text-muted-foreground/60">
                      {idx + 1}
                    </span>
                    <textarea
                      ref={
                        idx === rules.length - 1
                          ? (el: HTMLTextAreaElement | null) => {
                              if (el && focusAppended.current === "rule") {
                                focusAppended.current = null;
                                focusAppendedField(el);
                              }
                            }
                          : undefined
                      }
                      value={rule}
                      onChange={(e) => updateRule(idx, e.target.value)}
                      rows={2}
                      aria-label={formatMsg("tagger.assist.rubric.rule_label", {
                        number: idx + 1,
                      })}
                      className={cn(
                        "flex-1 resize-none rounded-md border border-input bg-background px-3 py-2 text-sm",
                        "leading-relaxed outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50",
                      )}
                      dir="auto"
                    />
                    <Button
                      variant="ghost"
                      size="icon-xs"
                      onClick={() => removeRule(idx)}
                      aria-label={msg("tagger.assist.rubric.rule_remove")}
                      className="mt-1.5"
                    >
                      <Trash className="size-3.5 text-muted-foreground" />
                    </Button>
                  </div>
                ))}
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={addRule}
                  className="gap-1.5 self-start text-muted-foreground"
                >
                  <Plus className="size-3.5" />
                  {msg("tagger.assist.rubric.rule_add")}
                </Button>
              </CardContent>
            </Card>
          )}
        </Rise>

        {/* The launch column: the submit wizard's start button stood beside
            the guide, filling the row. On autopilot it commits to tagging
            every row, scope and cost on the button itself. The chevrons
            sweep in the reading direction — the container is authored LTR
            and mirrored in RTL so glyphs, order, and motion all flip
            together. */}
        <Rise delay={0.16} className="flex min-w-0 justify-end lg:min-h-0">
          <motion.button
            type="button"
            onClick={launch}
            disabled={!taskValid || launching}
            animate={{ scale: [1, 1.01, 1] }}
            transition={{ repeat: Infinity, duration: 3, ease: "easeInOut" }}
            className={cn(
              // overflow-hidden: on the narrow rail a long unbreakable word
              // (some locales) clips inside the button instead of spilling
              // over the cards.
              "group relative flex w-[100px] cursor-pointer flex-col items-center justify-center gap-4 overflow-hidden",
              "rounded-2xl bg-primary py-8 text-base font-semibold text-primary-foreground",
              "transition-all duration-300 hover:scale-[1.01] hover:shadow-[0_0_30px_rgba(61,46,34,0.35)]",
              "active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-60",
            )}
          >
            <span className="flex flex-col items-center gap-1 px-4">
              <span>
                {autopilot
                  ? formatMsg("tagger.assist.rubric.start_autotag", { rows: rowCount })
                  : formatMsg("tagger.assist.rubric.start_copilot_round", { rows: copilotBatch })}
              </span>
              {!taskValid ? (
                <span className="text-xs font-normal text-primary-foreground/75" dir="auto">
                  {msg("auto.features.tagger.components.taggersetup.9")}
                </span>
              ) : (
                autopilot &&
                estimate && (
                  <span className="text-xs font-normal tabular-nums text-primary-foreground/75">
                    {estimate.credits_low === estimate.credits_high
                      ? estimate.credits_low === 1
                        ? msg("tagger.assist.rubric.credits_estimate_one")
                        : formatMsg("tagger.assist.rubric.credits_estimate_flat", {
                            count: estimate.credits_low,
                          })
                      : formatMsg("tagger.assist.rubric.credits_estimate", {
                          low: estimate.credits_low,
                          high: estimate.credits_high,
                        })}
                  </span>
                )
              )}
            </span>
            <div
              dir="ltr"
              className="flex items-center -space-x-7 opacity-70 transition-opacity duration-200 group-hover:opacity-100 rtl:-scale-x-100 [&>svg]:animate-[cascadeDown_1s_ease-in-out_infinite] group-hover:[&>svg]:animate-[cascadeRightHyper_0.5s_ease-out_infinite]"
            >
              <CaretRight className="size-10 [animation-delay:0s] group-hover:[animation-delay:0s]" />
              <CaretRight className="size-10 [animation-delay:0.15s] group-hover:[animation-delay:0.08s]" />
              <CaretRight className="size-10 [animation-delay:0.3s] group-hover:[animation-delay:0.16s]" />
            </div>
          </motion.button>
        </Rise>
      </div>

      <SubmitSplashOverlay show={launching} />
    </div>
  );
}
