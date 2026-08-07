"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import {
  ArrowUpRight,
  Books,
  Brain,
  Cpu,
  Cube,
  Database,
  DiceFive,
  Gauge,
  Gear,
  GearSix,
  GitMerge,
  Hash,
  Repeat,
  Ruler,
  Shuffle,
  Sparkle,
  Stack,
  Tag,
  Target,
  Thermometer,
  Wrench,
} from "@/shared/ui/icons";
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/primitives/card";
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
import { InfoCard, ReasoningPill } from "./ui-primitives";

const OPT_PARAM_LABELS: Record<string, string> = perLocale(() => ({
  auto: msg("auto.features.optimizations.components.configtab.literal.1"),
  max_bootstrapped_demos: msg("auto.features.optimizations.components.configtab.literal.2"),
  max_labeled_demos: msg("auto.features.optimizations.components.configtab.literal.3"),
  minibatch: msg("auto.features.optimizations.components.configtab.literal.4"),
  minibatch_size: msg("auto.features.optimizations.components.configtab.literal.5"),
  reflection_minibatch_size: msg("auto.features.optimizations.components.configtab.literal.6"),
  max_full_evals: msg("auto.features.optimizations.components.configtab.literal.7"),
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

/** Inline model-config card — matches the ModelChip style. */
function ModelCard({ label, cfg }: { label: string; cfg: Record<string, unknown> }) {
  const labelTip = MODEL_CARD_TIPS[label];
  const name = String(cfg.name || "—");
  const shortName = name.includes("/") ? name.split("/").pop()! : name;
  const temp = cfg.temperature as number | undefined;
  const maxTok = cfg.max_tokens as number | undefined;
  const extra = (cfg.extra ?? {}) as Record<string, unknown>;
  const reasoning = extra.reasoning_effort as string | undefined;
  return (
    <div className="flex items-center gap-2.5 rounded-lg border border-border/50 bg-card/80 px-3 py-2">
      <div className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="text-[0.625rem] font-medium uppercase tracking-wide text-muted-foreground">
          {labelTip ? <HelpTip text={labelTip}>{label}</HelpTip> : label}
        </span>
        <span className="truncate text-sm text-foreground font-mono font-medium" dir="ltr">
          {shortName}
        </span>
        <div className="flex items-center gap-2.5 text-[0.625rem] text-muted-foreground" dir="ltr">
          {temp != null && (
            <span className="inline-flex items-center gap-0.5">
              <Thermometer className="size-2.5" />
              {temp.toFixed(1)}
            </span>
          )}
          {maxTok != null && (
            <span className="inline-flex items-center gap-0.5">
              <Hash className="size-2.5" />
              {maxTok}
            </span>
          )}
          {reasoning && <ReasoningPill value={reasoning} />}
        </div>
      </div>
    </div>
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
        <p className="text-sm text-muted-foreground mb-4">
          {msg("auto.features.optimizations.components.configtab.2")}
          {TERMS.optimization}
          {msg("auto.features.optimizations.components.configtab.3")}
          {TERMS.model}, {TERMS.optimizer}
          {msg("auto.features.optimizations.components.configtab.4")}
        </p>
      </FadeIn>
      <div className="space-y-4">
        <Card
          data-tutorial="config-summary"
          className="relative overflow-hidden shadow-[0_1px_3px_rgba(28,22,18,0.04),inset_0_1px_0_rgba(255,255,255,0.5)]"
        >
          <div
            className="absolute inset-x-0 top-0 h-px bg-gradient-to-l from-transparent via-[#C8A882]/40 to-transparent"
            aria-hidden="true"
          />
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Gear className="size-4 text-[#7C6350]" aria-hidden="true" />
              <HelpTip text={tip("config.section.summary")}>
                <span className="font-bold tracking-tight">
                  {msg("auto.features.optimizations.components.configtab.5")}
                  {TERMS.optimization}
                </span>
              </HelpTip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="divide-y divide-border/40">
              {items.map((item, i) => (
                <div key={i} className="flex items-center justify-between py-2.5 gap-3">
                  <span className="flex items-center gap-2 text-xs text-muted-foreground shrink-0">
                    <span className="text-[#A89680]">{item.icon}</span>
                    {item.label}
                  </span>
                  <span
                    className="text-sm font-semibold text-foreground font-mono truncate"
                    dir="ltr"
                  >
                    {item.value}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="relative overflow-hidden shadow-[0_1px_3px_rgba(28,22,18,0.04),inset_0_1px_0_rgba(255,255,255,0.5)]">
          <div
            className="absolute inset-x-0 top-0 h-px bg-gradient-to-l from-transparent via-[#C8A882]/40 to-transparent"
            aria-hidden="true"
          />
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Cpu className="size-4 text-[#7C6350]" aria-hidden="true" />
              <HelpTip text={tip("config.section.models")}>
                <span className="font-bold tracking-tight">
                  {msg("auto.features.optimizations.components.configtab.6")}
                </span>
              </HelpTip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {job.optimization_type !== "grid_search" ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {modelCfg && <ModelCard label={msg("model.generation.label")} cfg={modelCfg} />}
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
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
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
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <p className="text-[0.625rem] font-semibold tracking-[0.08em] uppercase text-[#A89680] mb-1">
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
                <div className="space-y-2">
                  <p className="text-[0.625rem] font-semibold tracking-[0.08em] uppercase text-[#A89680] mb-1">
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
            ) : null}
          </CardContent>
        </Card>

        <Card className="relative overflow-hidden shadow-[0_1px_3px_rgba(28,22,18,0.04),inset_0_1px_0_rgba(255,255,255,0.5)]">
          <div
            className="absolute inset-x-0 top-0 h-px bg-gradient-to-l from-transparent via-[#C8A882]/40 to-transparent"
            aria-hidden="true"
          />
          <CardHeader className="pb-3">
            <CardTitle className="text-base flex items-center gap-2">
              <Database className="size-4 text-[#7C6350]" aria-hidden="true" />
              <HelpTip text={tip("config.section.data")}>
                <span className="font-bold tracking-tight">
                  {msg("auto.features.optimizations.components.configtab.8")}
                </span>
              </HelpTip>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {job.source_dataset_id && (
              <Link
                href={`/datasets?open=${job.source_dataset_id}`}
                className="group/srclink flex items-center gap-2.5 rounded-lg border border-border/50 bg-card/80 px-3 py-2.5 transition-colors hover:border-[#C8A882]/60 hover:bg-accent/40"
              >
                <Books className="size-3.5 shrink-0 text-[#A89680]" />
                <span className="min-w-0 flex-1 text-xs text-foreground">
                  {msg("optimizations.source_dataset.label")}
                </span>
                <span className="shrink-0 text-xs font-medium text-[#7C6350]">
                  {msg("optimizations.source_dataset.view")}
                </span>
                <ArrowUpRight className="size-4 shrink-0 text-muted-foreground/60 transition-colors group-hover/srclink:text-foreground" />
              </Link>
            )}
            {/* The train/val/test breakdown is advanced-mode machinery. */}
            {advanced && (
              <div className="space-y-2">
                <p className="text-[0.625rem] font-semibold tracking-[0.08em] uppercase text-[#A89680]">
                  <HelpTip text={tip("data.split_explanation")}>
                    {msg("auto.features.optimizations.components.configtab.9")}
                    {TERMS.dataset}
                  </HelpTip>
                </p>
                <div className="flex h-2.5 rounded-full overflow-hidden">
                  <div
                    className="bg-[#3D2E22] transition-all"
                    style={{ width: `${splitFractions.train * 100}%` }}
                  />
                  <div
                    className="bg-[#C8A882] transition-all"
                    style={{ width: `${splitFractions.val * 100}%` }}
                  />
                  <div
                    className="bg-[#8C7A6B] transition-all"
                    style={{ width: `${splitFractions.test * 100}%` }}
                  />
                </div>
                <div
                  className="grid gap-1 text-xs"
                  style={{
                    gridTemplateColumns: `${splitFractions.train}fr ${splitFractions.val}fr ${splitFractions.test}fr`,
                  }}
                >
                  <span className="flex items-center gap-1.5 min-w-0">
                    <span className="inline-block w-2 h-2 rounded-full bg-[#3D2E22] shrink-0" />
                    <span className="truncate">
                      {msg("auto.features.optimizations.components.configtab.10")}{" "}
                      <span className="font-mono tabular-nums text-muted-foreground" dir="ltr">
                        {splitFractions.train}
                      </span>
                    </span>
                  </span>
                  <span className="flex items-center gap-1.5 min-w-0">
                    <span className="inline-block w-2 h-2 rounded-full bg-[#C8A882] shrink-0" />
                    <span className="truncate">
                      {msg("auto.features.optimizations.components.configtab.11")}{" "}
                      <span className="font-mono tabular-nums text-muted-foreground" dir="ltr">
                        {splitFractions.val}
                      </span>
                    </span>
                  </span>
                  <span className="flex items-center gap-1.5 min-w-0">
                    <span className="inline-block w-2 h-2 rounded-full bg-[#8C7A6B] shrink-0" />
                    <span className="truncate">
                      {msg("auto.features.optimizations.components.configtab.12")}{" "}
                      <span className="font-mono tabular-nums text-muted-foreground" dir="ltr">
                        {splitFractions.test}
                      </span>
                    </span>
                  </span>
                </div>
              </div>
            )}
            {advanced && (
              <div className="grid grid-cols-2 gap-2.5">
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
          </CardContent>
        </Card>
      </div>
    </>
  );
}
