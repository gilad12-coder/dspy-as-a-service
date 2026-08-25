"use client";

import { useCallback, useRef, useState, type ReactNode } from "react";
import Link from "next/link";
import { AnimatePresence, motion, useReducedMotion, type Variants } from "framer-motion";
import {
  ArrowUpRight,
  Books,
  Brain,
  CaretLeft,
  CaretRight,
  Cpu,
  Cube,
  Database,
  DiceFive,
  Gauge,
  Gear,
  GearSix,
  GitMerge,
  Repeat,
  Ruler,
  Shuffle,
  Sparkle,
  Stack,
  Tag,
  Target,
  TextT,
  Thermometer,
  Wrench,
} from "@/shared/ui/icons";
import { FadeIn } from "@/shared/ui/motion";
import { useUserPrefs } from "@/features/settings";
import { HelpTip } from "@/shared/ui/help-tip";
import type {
  ModelConfig,
  OptimizationPayloadResponse,
  OptimizationStatusResponse,
  PairResult,
} from "@/shared/types/api";
import { tip } from "@/shared/lib/tooltips";
import { formatMsg, msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";
import { moduleLabel } from "@/shared/lib/formatters";
import { TERMS } from "@/shared/lib/terms";
import { cn } from "@/shared/lib/utils";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { ProviderLogo } from "@/shared/ui/provider-logo";
import { InfoCard, ReasoningPill } from "./ui-primitives";

const CONFIG_SLIDES = perLocale(() => [
  {
    id: "optimization",
    label: `${msg("auto.features.optimizations.components.configtab.5")}${TERMS.optimization}`,
    icon: <Gear className="size-5" />,
    tip: tip("config.section.summary"),
  },
  {
    id: "models",
    label: msg("auto.features.optimizations.components.configtab.6"),
    icon: <Cpu className="size-5" />,
    tip: tip("config.section.models"),
  },
  {
    id: "data",
    label: msg("auto.features.optimizations.components.configtab.8"),
    icon: <Database className="size-5" />,
    tip: tip("config.section.data"),
  },
]);

const CONFIG_SLIDE_VARIANTS: Variants = {
  enter: (direction: 1 | -1) => ({ opacity: 0, x: direction * 36 }),
  center: { opacity: 1, x: 0 },
  exit: (direction: 1 | -1) => ({ opacity: 0, x: direction * -28 }),
};

const CONFIG_SLIDE_TRANSITION = {
  duration: 0.24,
  ease: [0.2, 0.8, 0.2, 1] as const,
};

const OPT_PARAM_LABELS: Record<string, string> = perLocale(() => ({
  auto: msg("auto.features.optimizations.components.configtab.literal.1"),
  max_bootstrapped_demos: msg("auto.features.optimizations.components.configtab.literal.2"),
  max_labeled_demos: msg("auto.features.optimizations.components.configtab.literal.3"),
  minibatch: msg("auto.features.optimizations.components.configtab.literal.4"),
  minibatch_size: msg("auto.features.optimizations.components.configtab.literal.5"),
  reflection_minibatch_size: msg("auto.features.optimizations.components.configtab.literal.6"),
  max_full_evals: msg("auto.features.optimizations.components.configtab.literal.7"),
  max_metric_calls: msg("submit.metric_calls"),
  use_merge: msg("auto.features.optimizations.components.configtab.literal.8"),
  metric: TERMS.metric,
}));
const OPT_PARAM_TIPS: Record<string, string> = perLocale(() => ({
  auto: msg("auto.features.optimizations.components.configtab.literal.9"),
  max_bootstrapped_demos: msg("auto.features.optimizations.components.configtab.literal.10"),
  max_labeled_demos: formatMsg("auto.features.optimizations.components.configtab.template.1", {
    p1: TERMS.dataset,
    p2: TERMS.model,
  }),
  minibatch: formatMsg("auto.features.optimizations.components.configtab.template.2", {
    p1: TERMS.dataset,
  }),
  minibatch_size: msg("auto.features.optimizations.components.configtab.literal.11"),
  reflection_minibatch_size: formatMsg(
    "auto.features.optimizations.components.configtab.template.3",
    { p1: TERMS.model },
  ),
  max_full_evals: msg("auto.features.optimizations.components.configtab.literal.12"),
  max_metric_calls: msg("tooltip.submit.metric_calls"),
  use_merge: msg("auto.features.optimizations.components.configtab.literal.13"),
}));

function labelWithTip(key: string): ReactNode {
  const label = OPT_PARAM_LABELS[key] || key;
  const tipText = OPT_PARAM_TIPS[key];
  return tipText ? <HelpTip text={tipText}>{label}</HelpTip> : label;
}

const PARAM_ICONS: Record<string, ReactNode> = {
  auto: <Gauge className="size-3.5" />,
  max_bootstrapped_demos: <Sparkle className="size-3.5" />,
  max_labeled_demos: <Tag className="size-3.5" />,
  minibatch: <Stack className="size-3.5" />,
  minibatch_size: <Ruler className="size-3.5" />,
  reflection_minibatch_size: <Brain className="size-3.5" />,
  max_full_evals: <Repeat className="size-3.5" />,
  // Shares Gauge with `auto`: both are the run's (mutually exclusive) budget knob.
  max_metric_calls: <Gauge className="size-3.5" />,
  use_merge: <GitMerge className="size-3.5" />,
};

function paramIcon(key: string): ReactNode {
  return PARAM_ICONS[key] ?? <GearSix className="size-3.5" />;
}

// The GEPA budget level arrives as the raw "light" / "medium" / "heavy"
// string; translate it to the same Hebrew the submit summary shows so the
// value reads consistently across surfaces.
const AUTO_LEVEL_LABELS: Record<string, string> = perLocale(() => ({
  light: msg("auto.features.optimizations.components.configtab.literal.18"),
  medium: msg("auto.features.optimizations.components.configtab.literal.19"),
  heavy: msg("auto.features.optimizations.components.configtab.literal.20"),
}));

function formatParamValue(k: string, v: unknown): string {
  if (typeof v === "boolean")
    return v
      ? msg("auto.features.optimizations.components.configtab.literal.14")
      : msg("auto.features.optimizations.components.configtab.literal.15");
  if (k === "auto" && typeof v === "string" && AUTO_LEVEL_LABELS[v]) return AUTO_LEVEL_LABELS[v];
  return String(v);
}

// Tooltip copy keyed by the two named model-role labels. Grid cards use
// indexed short labels (no match here) since their columns are already tipped.
const MODEL_CARD_TIPS: Record<string, string> = perLocale(() => ({
  [msg("model.generation.label")]: tip("model.generation"),
  [TERMS.reflectionModel]: tip("model.reflection"),
}));

type ModelParameterKey = "temperature" | "max_tokens";

const MODEL_PARAMETER_DEFAULTS: Record<ModelParameterKey, number> = {
  temperature: 0.7,
  max_tokens: 1024,
};

function resolveModelParameter(cfg: Record<string, unknown>, key: ModelParameterKey): number {
  const rawExtra = cfg.extra;
  const extra =
    rawExtra && typeof rawExtra === "object" && !Array.isArray(rawExtra)
      ? (rawExtra as Record<string, unknown>)
      : undefined;
  const rawValue = cfg[key] ?? extra?.[key];
  const value =
    typeof rawValue === "number"
      ? rawValue
      : typeof rawValue === "string" && rawValue.trim()
        ? Number(rawValue)
        : Number.NaN;

  return Number.isFinite(value) ? value : MODEL_PARAMETER_DEFAULTS[key];
}

/** Resolve a routed model id to the provider mark used by the config card. */
function modelProviderSlug(id: string): string {
  const parts = id.split("/");
  const slug = (parts[0] === "openrouter" && parts.length > 2 ? parts[1] : parts[0]) ?? id;
  return slug === "x-ai" ? "xai" : slug;
}

/** Graphical model-config card for the Models carousel slide. */
function ModelCard({ label, cfg }: { label: string; cfg: Record<string, unknown> }) {
  const labelTip = MODEL_CARD_TIPS[label];
  const name = String(cfg.name || "—");
  const shortName = name.includes("/") ? name.split("/").pop()! : name;
  const temp = resolveModelParameter(cfg, "temperature");
  const maxTok = resolveModelParameter(cfg, "max_tokens");
  const extra = (cfg.extra ?? {}) as Record<string, unknown>;
  const reasoning = extra.reasoning_effort as string | undefined;
  const temperatureLabel = msg("auto.features.submit.components.modelconfigmodal.5");
  const maxTokensLabel = msg("auto.features.submit.components.modelconfigmodal.7");
  return (
    <article className="flex min-h-36 min-w-0 flex-col justify-between gap-5 rounded-2xl border border-border/60 bg-[#F8F4EE] p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.72)]">
      <div className="flex min-w-0 items-start gap-3.5">
        <span className="grid size-12 shrink-0 place-items-center rounded-2xl border border-border/45 bg-background shadow-sm">
          <ProviderLogo slug={modelProviderSlug(name)} size={30} />
        </span>
        <div className="flex min-w-0 flex-1 flex-col gap-1.5">
          <span className="text-[0.625rem] font-semibold uppercase tracking-[0.1em] text-[#8C7A6B]">
            {labelTip ? <HelpTip text={labelTip}>{label}</HelpTip> : label}
          </span>
          <span
            className="truncate font-mono text-base font-semibold tracking-tight text-foreground sm:text-lg"
            dir="ltr"
            title={name}
          >
            {shortName}
          </span>
        </div>
      </div>
      <div
        className="flex flex-wrap items-center gap-2 text-[0.6875rem] text-muted-foreground"
        dir="ltr"
      >
        <span
          className="inline-flex items-center gap-1.5 rounded-full border border-border/45 bg-background/75 px-2.5 py-1"
          aria-label={`${temperatureLabel}: ${temp.toFixed(1)}`}
          title={temperatureLabel}
        >
          <Thermometer className="size-3" aria-hidden="true" />
          {temp.toFixed(1)}
        </span>
        <span
          className="inline-flex items-center gap-1.5 rounded-full border border-border/45 bg-background/75 px-2.5 py-1"
          aria-label={`${maxTokensLabel}: ${maxTok}`}
          title={maxTokensLabel}
        >
          <TextT className="size-3" aria-hidden="true" />
          {maxTok}
        </span>
        {reasoning && <ReasoningPill value={reasoning} />}
      </div>
    </article>
  );
}

// The pair only carries the model name + reasoning effort; the richer
// ModelConfig (temperature, max_tokens, extra) lives on job.generation_models
// / job.reflection_models. Match by name first, narrowing on reasoning
// effort when multiple configs share the name. Fall back to a synthesized
// config so the ModelCard still renders when the grid lists are missing
// (older payloads, partial responses).
function pickPairModelConfig(
  configs: ModelConfig[] | undefined,
  name: string,
  reasoningEffort: string | null | undefined,
): Record<string, unknown> {
  const candidates = (configs ?? []).filter((c) => c.name === name);
  const matched =
    candidates.find(
      (c) =>
        ((c.extra?.reasoning_effort as string | undefined) ?? null) === (reasoningEffort ?? null),
    ) ?? candidates[0];
  if (matched) return matched as unknown as Record<string, unknown>;
  return reasoningEffort ? { name, extra: { reasoning_effort: reasoningEffort } } : { name };
}

export function ConfigTab({
  job,
  payload,
  activePair,
}: {
  job: OptimizationStatusResponse;
  payload: OptimizationPayloadResponse | null;
  activePair?: PairResult;
}) {
  const { prefs } = useUserPrefs();
  const advanced = prefs.advancedMode;
  const prefersReducedMotion = useReducedMotion() || prefs.liteMode;
  const isRtl = getActiveDir() === "rtl";
  const [activeSlide, setActiveSlide] = useState(0);
  const [slideDirection, setSlideDirection] = useState<1 | -1>(isRtl ? -1 : 1);
  const touchStart = useRef<{ x: number; y: number } | null>(null);

  const goToSlide = useCallback(
    (next: number) => {
      const clamped = Math.max(0, Math.min(CONFIG_SLIDES.length - 1, next));
      if (clamped === activeSlide) return;
      const forward = clamped > activeSlide;
      setSlideDirection(forward === isRtl ? -1 : 1);
      setActiveSlide(clamped);
    },
    [activeSlide, isRtl],
  );

  const previousSlide = CONFIG_SLIDES[activeSlide - 1];
  const currentSlide = CONFIG_SLIDES[activeSlide] ?? CONFIG_SLIDES[0]!;
  const nextSlide = CONFIG_SLIDES[activeSlide + 1];
  const PreviousIcon = isRtl ? CaretRight : CaretLeft;
  const NextIcon = isRtl ? CaretLeft : CaretRight;

  // Merge job-level data with full payload for richer config display
  const p = (payload?.payload ?? {}) as Record<string, unknown>;
  const splitFractions = (p.split_fractions ??
    job.split_fractions ?? { train: 0.7, val: 0.15, test: 0.15 }) as {
    train: number;
    val: number;
    test: number;
  };
  const shuffleVal =
    p.shuffle != null ? Boolean(p.shuffle) : job.shuffle != null ? job.shuffle : true;
  const seedVal = (p.seed ?? job.seed) as number | undefined;
  const optKw = (p.optimizer_kwargs ?? job.optimizer_kwargs ?? {}) as Record<string, unknown>;
  const compKw = (p.compile_kwargs ?? job.compile_kwargs ?? {}) as Record<string, unknown>;
  const modelCfg = (p.model_config ?? job.model_settings ?? null) as Record<string, unknown> | null;
  const reflCfg = (p.reflection_model_config ?? null) as Record<string, unknown> | null;
  const taskCfg = (p.task_model_config ?? null) as Record<string, unknown> | null;

  // React runs carry a tool-source config the generic rows don't cover.
  // Scoring lives in metric_code (shown in the code view), so there is no
  // reward preset to surface here.
  const toolSource = (p.tool_source ?? null) as Record<string, unknown> | null;
  const reactRows: Array<{ label: ReactNode; value: string; icon: ReactNode }> =
    (job.module_name ?? "").toLowerCase() === "react" && toolSource?.kind
      ? [
          {
            label: (
              <HelpTip text={tip("react.tool_source")}>
                {msg("submit.react.tool_source_label")}
              </HelpTip>
            ),
            value: String(toolSource.mcp_url || toolSource.kind),
            icon: <Wrench className="size-3.5" />,
          },
        ]
      : [];

  const items: Array<{ label: ReactNode; value: string; icon: ReactNode }> = [
    {
      label: (
        <HelpTip text={tip("module.choice")}>
          {msg("auto.features.optimizations.components.configtab.1")}
        </HelpTip>
      ),
      value: moduleLabel(job.module_name),
      icon: <Cube className="size-3.5" />,
    },
    {
      label: <HelpTip text={tip("optimizer.choice")}>{TERMS.optimizer}</HelpTip>,
      value: job.optimizer_name ?? "—",
      icon: <Target className="size-3.5" />,
    },
    ...reactRows,
    ...Object.entries(optKw)
      .filter(([k]) => k !== "metric" && (advanced || k === "auto"))
      .map(([k, v]) => ({
        label: labelWithTip(k),
        value: formatParamValue(k, v),
        icon: paramIcon(k),
      })),
    ...Object.entries(compKw).map(([k, v]) => ({
      label: labelWithTip(k),
      value: formatParamValue(k, v),
      icon: <Stack className="size-3.5" />,
    })),
  ];

  return (
    <>
      <FadeIn>
        <p className="mb-4 max-w-3xl text-sm text-muted-foreground">
          {msg("auto.features.optimizations.components.configtab.2")}
          {TERMS.optimization}
          {msg("auto.features.optimizations.components.configtab.3")}
          {TERMS.model}, {TERMS.optimizer}
          {msg("auto.features.optimizations.components.configtab.4")}
        </p>
      </FadeIn>
      <motion.section
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={prefersReducedMotion ? { duration: 0 } : { duration: 0.3 }}
        className="w-full overflow-hidden rounded-2xl border border-border bg-card/80 shadow-lg backdrop-blur-xl"
        data-tutorial="config-summary"
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
        <div className="flex items-center justify-between gap-5 border-b border-border/60 bg-secondary/35 px-5 py-4 sm:px-6 sm:py-5 lg:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex size-11 shrink-0 items-center justify-center rounded-xl bg-[#EDE7DD] text-[#3D2E22] shadow-[inset_0_1px_0_rgba(255,255,255,0.65)]">
              {currentSlide.icon}
            </span>
            <div className="min-w-0">
              <span
                className="font-mono text-[0.625rem] tabular-nums text-muted-foreground"
                dir="ltr"
              >
                {activeSlide + 1} / {CONFIG_SLIDES.length}
              </span>
              <h3 className="truncate text-lg font-bold tracking-tight text-foreground sm:text-xl">
                <HelpTip text={currentSlide.tip}>{currentSlide.label}</HelpTip>
              </h3>
            </div>
          </div>
          <div
            className="hidden min-w-0 flex-1 items-center justify-end gap-2 sm:flex"
            aria-label={currentSlide.label}
          >
            {CONFIG_SLIDES.map((slide, index) => (
              <button
                key={slide.id}
                type="button"
                onClick={() => goToSlide(index)}
                aria-label={slide.label}
                aria-current={activeSlide === index ? "step" : undefined}
                className={cn(
                  "flex min-w-0 cursor-pointer items-center gap-2 rounded-xl border px-2.5 py-2 text-start transition-[background-color,border-color,color,transform] duration-150",
                  "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882] focus-visible:ring-offset-2 active:scale-[0.98]",
                  activeSlide === index
                    ? "border-[#C8A882]/70 bg-background text-foreground shadow-sm"
                    : "border-transparent text-muted-foreground hover:border-border/60 hover:bg-background/55 hover:text-foreground",
                )}
              >
                <span
                  className={cn(
                    "grid size-8 shrink-0 place-items-center rounded-lg transition-colors [&_svg]:size-4",
                    activeSlide === index
                      ? "bg-[#3D2E22] text-[#FAF8F5]"
                      : "bg-[#EDE7DD] text-[#8C7A6B]",
                  )}
                >
                  {slide.icon}
                </span>
                <span className="hidden truncate text-xs font-semibold lg:block">
                  {slide.label}
                </span>
              </button>
            ))}
          </div>
        </div>

        <div
          className="relative overflow-hidden p-4 sm:p-6 lg:p-8"
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
              variants={CONFIG_SLIDE_VARIANTS}
              initial="enter"
              animate="center"
              exit="exit"
              transition={prefersReducedMotion ? { duration: 0 } : CONFIG_SLIDE_TRANSITION}
              className="min-h-[24rem] w-full"
            >
              {activeSlide === 0 && (
                <div className="flex min-h-[24rem] flex-col gap-5">
                  <div className="grid items-stretch gap-3 md:grid-cols-2">
                    {items.slice(0, 2).map((item, index) => (
                      <article
                        key={index}
                        className="flex min-h-40 min-w-0 flex-col justify-between gap-6 rounded-2xl border border-border/60 bg-[#F8F4EE] p-5 shadow-[inset_0_1px_0_rgba(255,255,255,0.72)] sm:p-6"
                      >
                        <div className="flex items-start justify-between gap-3">
                          <span className="grid size-12 shrink-0 place-items-center rounded-2xl bg-[#3D2E22] text-[#FAF8F5] [&_svg]:size-5">
                            {item.icon}
                          </span>
                          <span
                            className="font-mono text-[0.625rem] tabular-nums text-[#8C7A6B]/70"
                            dir="ltr"
                          >
                            0{index + 1}
                          </span>
                        </div>
                        <div className="min-w-0">
                          <div className="text-[0.6875rem] font-semibold uppercase tracking-[0.1em] text-[#8C7A6B]">
                            {item.label}
                          </div>
                          <div
                            className="mt-1 truncate font-mono text-xl font-semibold tracking-tight text-foreground sm:text-2xl"
                            dir="ltr"
                            title={item.value}
                          >
                            {item.value}
                          </div>
                        </div>
                      </article>
                    ))}
                  </div>

                  {items.length > 2 && (
                    <div
                      className="grid flex-1 gap-3"
                      style={{
                        gridTemplateColumns: "repeat(auto-fit, minmax(min(11rem, 100%), 1fr))",
                      }}
                    >
                      {items.slice(2).map((item, index) => (
                        <article
                          key={index}
                          className="flex min-h-28 min-w-0 flex-col justify-between gap-4 rounded-xl border border-border/45 bg-background/65 p-4"
                        >
                          <span className="grid size-9 place-items-center rounded-xl bg-[#EDE7DD] text-[#8C7A6B] [&_svg]:size-4">
                            {item.icon}
                          </span>
                          <div className="min-w-0">
                            <div className="truncate text-[0.6875rem] font-medium text-muted-foreground">
                              {item.label}
                            </div>
                            <div
                              className="mt-0.5 truncate font-mono text-base font-semibold text-foreground"
                              dir="ltr"
                              title={item.value}
                            >
                              {item.value}
                            </div>
                          </div>
                        </article>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {activeSlide === 1 && (
                <div className="min-h-[24rem]">
                  {job.optimization_type !== "grid_search" ? (
                    <div
                      className="grid gap-4"
                      style={{
                        gridTemplateColumns: "repeat(auto-fit, minmax(min(17rem, 100%), 1fr))",
                      }}
                    >
                      {modelCfg && (
                        <ModelCard label={msg("model.generation.label")} cfg={modelCfg} />
                      )}
                      {reflCfg && <ModelCard label={TERMS.reflectionModel} cfg={reflCfg} />}
                      {taskCfg && <ModelCard label={msg("model.generation.label")} cfg={taskCfg} />}
                      {!modelCfg && !reflCfg && !taskCfg && job.model_name && (
                        <>
                          <ModelCard
                            label={msg("model.generation.label")}
                            cfg={{ name: job.model_name, ...(job.model_settings || {}) }}
                          />
                          {job.reflection_model_name && (
                            <ModelCard
                              label={TERMS.reflectionModel}
                              cfg={{ name: job.reflection_model_name }}
                            />
                          )}
                        </>
                      )}
                    </div>
                  ) : activePair ? (
                    <div
                      className="grid gap-4"
                      style={{
                        gridTemplateColumns: "repeat(auto-fit, minmax(min(17rem, 100%), 1fr))",
                      }}
                    >
                      <ModelCard
                        label={msg("model.generation.label")}
                        cfg={pickPairModelConfig(
                          job.generation_models,
                          activePair.generation_model,
                          activePair.generation_reasoning_effort,
                        )}
                      />
                      <ModelCard
                        label={TERMS.reflectionModel}
                        cfg={pickPairModelConfig(
                          job.reflection_models,
                          activePair.reflection_model,
                          activePair.reflection_reasoning_effort,
                        )}
                      />
                    </div>
                  ) : job.generation_models && job.reflection_models ? (
                    <div className="grid grid-cols-1 gap-8 md:grid-cols-2">
                      <div className="space-y-3">
                        <p className="flex items-center gap-2 text-[0.625rem] font-semibold uppercase tracking-[0.1em] text-[#8C7A6B]">
                          <span className="grid size-7 place-items-center rounded-lg bg-[#3D2E22] text-[#FAF8F5]">
                            <Cpu className="size-3.5" aria-hidden="true" />
                          </span>
                          <HelpTip text={tip("grid.generation_models")}>
                            {msg("model.generation.label_plural")}
                          </HelpTip>
                        </p>
                        {job.generation_models.map((m, i) => (
                          <ModelCard
                            key={i}
                            label={`${msg("model.generation.label_short")} ${i + 1}`}
                            cfg={m as unknown as Record<string, unknown>}
                          />
                        ))}
                      </div>
                      <div className="space-y-3">
                        <p className="flex items-center gap-2 text-[0.625rem] font-semibold uppercase tracking-[0.1em] text-[#8C7A6B]">
                          <span className="grid size-7 place-items-center rounded-lg bg-[#C8A882] text-[#3D2E22]">
                            <Brain className="size-3.5" aria-hidden="true" />
                          </span>
                          <HelpTip text={tip("grid.reflection_models")}>
                            {msg("auto.features.optimizations.components.configtab.7")}
                          </HelpTip>
                        </p>
                        {job.reflection_models.map((m, i) => (
                          <ModelCard
                            key={i}
                            label={formatMsg(
                              "auto.features.optimizations.components.configtab.template.4",
                              { p1: i + 1 },
                            )}
                            cfg={m as unknown as Record<string, unknown>}
                          />
                        ))}
                      </div>
                    </div>
                  ) : (
                    <p className="py-12 text-center text-sm text-muted-foreground">—</p>
                  )}
                </div>
              )}

              {activeSlide === 2 && (
                <div className="flex min-h-[24rem] flex-col gap-6">
                  {job.source_dataset_id && (
                    <Link
                      href={`/datasets?open=${job.source_dataset_id}`}
                      className={cn(
                        "group/srclink flex min-h-28 items-center gap-4 rounded-2xl border border-border/60 bg-[#F8F4EE] p-5 transition-[background-color,border-color,transform] hover:border-[#C8A882]/70 hover:bg-[#F4EEE6] active:scale-[0.995] sm:p-6",
                        !advanced && "flex-1",
                      )}
                    >
                      <span className="grid size-14 shrink-0 place-items-center rounded-2xl bg-[#3D2E22] text-[#FAF8F5] shadow-sm">
                        <Books className="size-6" aria-hidden="true" />
                      </span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-[0.6875rem] font-semibold uppercase tracking-[0.1em] text-[#8C7A6B]">
                          {msg("optimizations.source_dataset.label")}
                        </span>
                        <span
                          className="mt-1 block truncate font-mono text-base font-semibold text-foreground sm:text-lg"
                          dir="ltr"
                        >
                          {job.source_dataset_id}
                        </span>
                      </span>
                      <span className="hidden shrink-0 text-xs font-semibold text-[#7C6350] sm:block">
                        {msg("optimizations.source_dataset.view")}
                      </span>
                      <ArrowUpRight className="size-4 shrink-0 text-muted-foreground/60 transition-colors group-hover/srclink:text-foreground" />
                    </Link>
                  )}
                  {advanced && (
                    <div className="flex flex-1 flex-col gap-3">
                      <div className="flex items-center gap-2.5">
                        <span className="grid size-9 place-items-center rounded-xl bg-[#EDE7DD] text-[#8C7A6B]">
                          <Database className="size-4" aria-hidden="true" />
                        </span>
                        <p className="text-[0.6875rem] font-semibold uppercase tracking-[0.1em] text-[#8C7A6B]">
                          <HelpTip text={tip("data.split_explanation")}>
                            {msg("auto.features.optimizations.components.configtab.9")}
                            {TERMS.dataset}
                          </HelpTip>
                        </p>
                      </div>
                      <div className="flex min-h-28 flex-1 overflow-hidden rounded-2xl border border-border/50 bg-muted/30 shadow-[inset_0_1px_0_rgba(255,255,255,0.42)]">
                        <div
                          className="flex min-w-0 flex-col items-center justify-center gap-3 bg-[#3D2E22] p-2 text-[#FAF8F5] transition-[width] sm:items-stretch sm:justify-between sm:p-5"
                          style={{ width: `${splitFractions.train * 100}%` }}
                        >
                          <span className="hidden truncate text-[0.625rem] font-semibold uppercase tracking-[0.08em] opacity-70 sm:block">
                            {msg("auto.features.optimizations.components.configtab.10")}
                          </span>
                          <span
                            className="font-mono text-sm font-semibold tabular-nums sm:text-2xl"
                            dir="ltr"
                          >
                            {Math.round(splitFractions.train * 100)}%
                          </span>
                        </div>
                        <div
                          className="flex min-w-0 flex-col items-center justify-center gap-3 bg-[#C8A882] p-2 text-[#3D2E22] transition-[width] sm:items-stretch sm:justify-between sm:p-5"
                          style={{ width: `${splitFractions.val * 100}%` }}
                        >
                          <span className="hidden truncate text-[0.625rem] font-semibold uppercase tracking-[0.08em] opacity-70 sm:block">
                            {msg("auto.features.optimizations.components.configtab.11")}
                          </span>
                          <span
                            className="font-mono text-sm font-semibold tabular-nums sm:text-2xl"
                            dir="ltr"
                          >
                            {Math.round(splitFractions.val * 100)}%
                          </span>
                        </div>
                        <div
                          className="flex min-w-0 flex-col items-center justify-center gap-3 bg-[#8C7A6B] p-2 text-[#FAF8F5] transition-[width] sm:items-stretch sm:justify-between sm:p-5"
                          style={{ width: `${splitFractions.test * 100}%` }}
                        >
                          <span className="hidden truncate text-[0.625rem] font-semibold uppercase tracking-[0.08em] opacity-70 sm:block">
                            {msg("auto.features.optimizations.components.configtab.12")}
                          </span>
                          <span
                            className="font-mono text-sm font-semibold tabular-nums sm:text-2xl"
                            dir="ltr"
                          >
                            {Math.round(splitFractions.test * 100)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                  {advanced && (
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                      <InfoCard
                        label={
                          <HelpTip text={tip("data.shuffle_explanation")}>
                            {msg("auto.features.optimizations.components.configtab.13")}
                          </HelpTip>
                        }
                        value={
                          shuffleVal
                            ? msg("auto.features.optimizations.components.configtab.literal.16")
                            : msg("auto.features.optimizations.components.configtab.literal.17")
                        }
                        icon={<Shuffle className="size-3.5" />}
                      />
                      {seedVal != null && (
                        <InfoCard
                          label={
                            <HelpTip text={tip("data.seed")}>
                              {msg("auto.features.optimizations.components.configtab.14")}
                            </HelpTip>
                          }
                          value={seedVal}
                          icon={<DiceFive className="size-3.5" />}
                        />
                      )}
                    </div>
                  )}
                  {!job.source_dataset_id && !advanced && (
                    <p className="py-12 text-center text-sm text-muted-foreground">—</p>
                  )}
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
            className="inline-flex min-h-[44px] min-w-[44px] cursor-pointer items-center gap-2 rounded-xl px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-background/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882] disabled:cursor-not-allowed disabled:opacity-30"
          >
            <PreviousIcon className="size-4" aria-hidden="true" />
            <span className="hidden sm:inline">{previousSlide?.label}</span>
          </button>

          <div className="flex items-center justify-center gap-1 sm:hidden">
            {CONFIG_SLIDES.map((slide, index) => (
              <button
                key={slide.id}
                type="button"
                onClick={() => goToSlide(index)}
                aria-label={slide.label}
                aria-current={activeSlide === index ? "step" : undefined}
                className="flex size-[44px] cursor-pointer items-center justify-center rounded-full focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]"
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
            className="inline-flex min-h-[44px] min-w-[44px] cursor-pointer items-center gap-2 rounded-xl px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-background/70 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882] disabled:cursor-not-allowed disabled:opacity-30"
          >
            <span className="hidden sm:inline">{nextSlide?.label}</span>
            <NextIcon className="size-4" aria-hidden="true" />
          </button>
        </div>
      </motion.section>
    </>
  );
}
