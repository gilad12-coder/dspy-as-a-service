"use client";

import * as React from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Coins, Key, X } from "@/shared/ui/icons";
import { useByokKeys, litellmProviderForByok, type TokenSourceMode } from "@/features/billing";
import { useSettingsModal } from "@/features/settings";
import { getByokModelCatalog, cachedByokCatalog } from "@/shared/lib/model-catalog";
import { Dialog, DialogContent, DialogFooter } from "@/shared/ui/primitives/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/shared/ui/primitives/tooltip";
import { DialogTitleRow } from "@/shared/ui/dialog-title-row";
import { Button } from "@/shared/ui/primitives/button";
import { Label } from "@/shared/ui/primitives/label";
import { Switch } from "@/shared/ui/primitives/switch";
import { Separator } from "@/shared/ui/primitives/separator";
import { ModelPicker, modelSupportsThinking } from "./ModelPicker";
import { effortLabel, effortsFor } from "@/shared/lib/model-efforts";
import { NumberInput } from "@/shared/ui/number-input";
import { cn } from "@/shared/lib/utils";
import type { ModelConfig, CatalogModel } from "@/shared/types/api";
import { HelpTip } from "@/shared/ui/help-tip";
import { tip } from "@/shared/lib/tooltips";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";

interface ModelConfigModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  config: ModelConfig;
  onSave: (config: ModelConfig) => void;
  /** Label shown in the dialog header, e.g. the primary-model or reflection-model term. */
  roleLabel?: string;
  /** Catalog models for thinking detection */
  catalogModels?: CatalogModel[];
  /** Recently used configs — shown as quick-select at top */
  recentConfigs?: ModelConfig[];
  /** Remove a single recent config by its model name (rendered as a per-row X). */
  onRemoveRecent?: (name: string) => void;
}

const TOKEN_SOURCE_SEGMENTS: Array<{
  mode: TokenSourceMode;
  icon: typeof Coins;
  labelKey: "billing.mode.managed" | "billing.mode.byok";
}> = [
  { mode: "managed", icon: Coins, labelKey: "billing.mode.managed" },
  { mode: "byok", icon: Key, labelKey: "billing.mode.byok" },
];

const TOKEN_SOURCE_TRANSITION = {
  type: "tween",
  duration: 0.16,
  ease: [0.22, 1, 0.36, 1],
} as const;

function withoutInlineConnection(config: ModelConfig): ModelConfig {
  const { base_url: _baseUrl, ...rest } = config;
  const {
    api_key: _apiKey,
    api_base: _ApiBase,
    base_url: _ExtraBaseUrl,
    ...safeExtra
  } = rest.extra ?? {};
  const tokenSource = rest.token_source ?? "managed";
  return {
    ...rest,
    token_source: tokenSource,
    byok_provider: tokenSource === "byok" ? rest.byok_provider : undefined,
    extra: Object.keys(safeExtra).length > 0 ? safeExtra : undefined,
  };
}

export function ModelConfigModal({
  open,
  onOpenChange,
  config,
  onSave,
  roleLabel = msg("auto.features.submit.components.modelconfigmodal.literal.1"),
  catalogModels,
  recentConfigs,
  onRemoveRecent,
}: ModelConfigModalProps) {
  const { keys } = useByokKeys();
  const { openTo } = useSettingsModal();
  const prefersReducedMotion = useReducedMotion();
  // Two of these modals coexist (generation + reflection); the sliding-pill
  // layoutId must be unique per instance or Framer pairs them up.
  const effortPillId = React.useId();
  const tokenSourcePillId = React.useId();
  const [draft, setDraft] = React.useState<ModelConfig>(() => withoutInlineConnection(config));
  const mode = draft.token_source ?? "managed";

  // In BYOK mode the picker lists the BYOK catalog narrowed to the providers
  // the user has a *verified* key for (mapped to their LiteLLM prefix), so a
  // typo'd, revoked, or unverified key never offers models a run could only
  // fail to authenticate.
  const byokProviders = React.useMemo(
    () =>
      keys.filter((k) => k.status === "verified").map((k) => litellmProviderForByok(k.provider)),
    [keys],
  );
  const byokProviderKey = [...byokProviders].sort().join("\u0000");
  // BYOK catalog models also feed reasoning-toggle detection, since a BYOK model
  // won't appear in the managed `catalogModels`.
  const [byokModels, setByokModels] = React.useState<CatalogModel[] | null>(
    cachedByokCatalog()?.models ?? null,
  );
  React.useEffect(() => {
    if (mode !== "byok") return;
    let cancelled = false;
    getByokModelCatalog()
      .then((c) => {
        if (!cancelled) setByokModels(c.models);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [mode, byokProviderKey]);
  const detectionModels = mode === "byok" ? (byokModels ?? undefined) : catalogModels;

  // Sync draft when config changes externally (e.g. opening with different model)
  React.useEffect(() => {
    if (open) {
      setDraft(withoutInlineConnection(config));
    }
  }, [open, config]);

  // The effort ladder is model-specific (providers reject levels outside
  // their documented set). "none" is expressed by the switch itself, and an
  // empty ladder (always-on thinkers like MiniMax M3) leaves nothing to
  // configure, so the whole section disappears.
  const effortLadder = React.useMemo(
    () => effortsFor(draft.name).filter((level) => level !== "none"),
    [draft.name],
  );
  const defaultEffort = (ladder: readonly string[]) =>
    ladder.includes("medium") ? "medium" : (ladder[Math.floor(ladder.length / 2)] ?? "medium");
  const canThink = modelSupportsThinking(draft.name, detectionModels) && effortLadder.length > 0;
  const thinkingEnabled = !!draft.extra?.reasoning_effort;
  const reasoningEffort = (draft.extra?.reasoning_effort as string) ?? "medium";

  const setThinking = (on: boolean) => {
    setDraft((p) => ({
      ...p,
      extra: on
        ? { ...p.extra, reasoning_effort: defaultEffort(effortLadder) }
        : (() => {
            const rest = { ...p.extra };
            delete rest.reasoning_effort;
            return Object.keys(rest).length ? rest : undefined;
          })(),
    }));
  };

  const setEffort = (level: string) => {
    setDraft((p) => ({ ...p, extra: { ...p.extra, reasoning_effort: level } }));
  };

  const handleSave = () => {
    onSave(withoutInlineConnection(draft));
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex max-h-[calc(100dvh-1rem)] flex-col gap-0 overflow-hidden p-0 sm:max-h-[85vh] sm:max-w-2xl [&_[data-slot=dialog-close]]:size-[44px] lg:[&_[data-slot=dialog-close]]:size-8">
        <DialogTitleRow title={roleLabel} className="px-4 pt-4 sm:px-6 sm:pt-6" />

        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4 sm:px-6">
          {recentConfigs && recentConfigs.length > 0 && (
            <div className="space-y-1.5">
              <Label className="text-[0.625rem] uppercase tracking-wide text-muted-foreground">
                {msg("auto.features.submit.components.modelconfigmodal.2")}
              </Label>
              <div className="flex gap-1.5 overflow-x-auto pb-1.5 scrollbar-thin" dir="ltr">
                {recentConfigs.map((rc, i) => {
                  const isActive = draft.name === rc.name;
                  return (
                    <div
                      key={`${rc.name}-${i}`}
                      className={cn(
                        "group/recent flex shrink-0 items-center gap-1.5 rounded-md border ps-2 pe-1 py-1 text-[0.6875rem] font-mono transition-all",
                        isActive
                          ? "border-primary/50 bg-primary/5 text-foreground"
                          : "border-border/40 bg-muted/30 text-muted-foreground hover:border-primary/40 hover:text-foreground hover:bg-muted/50",
                      )}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          setDraft(withoutInlineConnection(rc));
                        }}
                        className="flex min-h-[44px] items-center gap-1.5 cursor-pointer outline-none lg:min-h-0"
                      >
                        <span className="truncate max-w-[120px]">{rc.name.split("/").pop()}</span>
                        <span className="text-[9px] opacity-60">{rc.temperature?.toFixed(1)}</span>
                      </button>
                      {onRemoveRecent && (
                        <button
                          type="button"
                          aria-label={formatMsg(
                            "auto.features.submit.components.modelconfigmodal.recent.remove",
                            { model: rc.name.split("/").pop() ?? rc.name },
                          )}
                          onClick={(e) => {
                            e.stopPropagation();
                            onRemoveRecent(rc.name);
                          }}
                          className="ms-0.5 inline-flex size-[44px] items-center justify-center rounded text-muted-foreground/60 transition-colors hover:bg-destructive/10 hover:text-destructive lg:size-4"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
              <Separator />
            </div>
          )}

          <div className="space-y-2">
            <Label className="text-[0.625rem] uppercase tracking-wide text-muted-foreground">
              {msg("billing.mode.label")}
            </Label>
            <div
              role="group"
              aria-label={msg("billing.mode.aria")}
              data-tutorial="model-billing-source"
              className="flex w-full rounded-lg bg-muted p-0.5 sm:w-fit"
            >
              {TOKEN_SOURCE_SEGMENTS.map(({ mode: value, icon: Icon, labelKey }) => (
                <button
                  key={value}
                  type="button"
                  onClick={() =>
                    setDraft((current) =>
                      withoutInlineConnection({
                        ...current,
                        name: "",
                        token_source: value,
                        byok_provider: undefined,
                      }),
                    )
                  }
                  aria-pressed={mode === value}
                  className={cn(
                    "relative flex min-h-[44px] flex-1 cursor-pointer items-center justify-center rounded-md px-3 py-1.5 text-xs font-medium transition-colors sm:flex-none lg:min-h-0",
                    mode === value
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {mode === value && (
                    <motion.span
                      layoutId={`token-source-pill-${tokenSourcePillId}`}
                      className="absolute inset-0 rounded-md bg-background shadow-[0_1px_2px_oklch(0.25_0.04_45/.12)]"
                      transition={prefersReducedMotion ? { duration: 0 } : TOKEN_SOURCE_TRANSITION}
                      aria-hidden="true"
                    />
                  )}
                  <span className="relative z-10 flex items-center gap-1.5">
                    <Icon className="size-3.5" aria-hidden="true" />
                    {msg(labelKey)}
                  </span>
                </button>
              ))}
            </div>
            {mode === "managed" && (
              <div className="flex items-center gap-2 rounded-md bg-muted/30 px-2.5 py-1.5 text-xs text-muted-foreground">
                <span className="min-w-0 flex-1">{msg("billing.mode.managed_hint")}</span>
              </div>
            )}
          </div>

          {mode === "byok" && (
            <div className="flex items-center gap-2 rounded-md bg-muted/30 px-2.5 py-1.5 text-xs text-muted-foreground">
              <span className="min-w-0 flex-1">{msg("billing.mode.byok_hint")}</span>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => {
                      onOpenChange(false);
                      openTo("providers");
                    }}
                    aria-label={msg("billing.mode.manage_keys")}
                    className="inline-flex size-[44px] shrink-0 cursor-pointer items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-background hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/60 lg:size-8"
                  >
                    <Key className="size-4" aria-hidden="true" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>{msg("billing.mode.manage_keys")}</TooltipContent>
              </Tooltip>
            </div>
          )}

          <div className="space-y-2">
            <Label>
              <HelpTip text={tip("model_config.model")}>
                {msg("auto.features.submit.components.modelconfigmodal.4")}
              </HelpTip>
            </Label>
            <ModelPicker
              value={draft.name}
              selectedByokProvider={draft.byok_provider}
              onChange={(next) => {
                setDraft((p) => {
                  const ladder = effortsFor(next).filter((level) => level !== "none");
                  const rest = { ...p.extra };
                  const effort = rest.reasoning_effort as string | undefined;
                  if (!modelSupportsThinking(next, detectionModels) || ladder.length === 0) {
                    delete rest.reasoning_effort;
                  } else if (effort && !ladder.includes(effort)) {
                    rest.reasoning_effort = defaultEffort(ladder);
                  }
                  return { ...p, name: next, extra: Object.keys(rest).length ? rest : undefined };
                });
              }}
              onSelect={(model) =>
                setDraft((current) => ({
                  ...current,
                  byok_provider: mode === "byok" ? (model.byok_provider ?? undefined) : undefined,
                }))
              }
              byokMode={mode === "byok"}
              byokProviders={byokProviders}
              placeholder={msg("auto.features.submit.components.modelconfigmodal.literal.3")}
            />
          </div>

          <Separator />

          <Label className="text-[0.625rem] uppercase tracking-wide text-muted-foreground">
            {msg("auto.features.submit.components.modelconfigmodal.section.parameters")}
          </Label>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>
                <HelpTip text={tip("model_config.temperature")}>
                  {msg("auto.features.submit.components.modelconfigmodal.5")}
                </HelpTip>
              </Label>
              <span className="text-xs font-mono text-muted-foreground">
                {draft.temperature?.toFixed(1) ?? "0.7"}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={draft.temperature ?? 0.7}
              onChange={(e) => setDraft((p) => ({ ...p, temperature: parseFloat(e.target.value) }))}
              className="h-[44px] w-full cursor-pointer appearance-none rounded-full bg-transparent accent-primary [&::-moz-range-track]:h-2 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-muted [&::-webkit-slider-runnable-track]:h-2 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-muted lg:h-2 lg:bg-muted"
              dir="auto"
            />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label>
                <HelpTip text={tip("model_config.top_p")}>
                  {msg("auto.features.submit.components.modelconfigmodal.6")}
                </HelpTip>
              </Label>
              <span className="text-xs font-mono text-muted-foreground">
                {draft.top_p?.toFixed(2) ?? "—"}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={draft.top_p ?? 1}
              onChange={(e) => setDraft((p) => ({ ...p, top_p: parseFloat(e.target.value) }))}
              className="h-[44px] w-full cursor-pointer appearance-none rounded-full bg-transparent accent-primary [&::-moz-range-track]:h-2 [&::-moz-range-track]:rounded-full [&::-moz-range-track]:bg-muted [&::-webkit-slider-runnable-track]:h-2 [&::-webkit-slider-runnable-track]:rounded-full [&::-webkit-slider-runnable-track]:bg-muted lg:h-2 lg:bg-muted"
              dir="auto"
            />
          </div>

          <div className="space-y-2">
            <Label>
              <HelpTip text={tip("model_config.max_tokens")}>
                {msg("auto.features.submit.components.modelconfigmodal.7")}
              </HelpTip>
            </Label>
            <NumberInput
              min={1}
              step={256}
              value={draft.max_tokens ?? ""}
              onChange={(v) => setDraft((p) => ({ ...p, max_tokens: v }))}
              className="h-[44px] [&_button]:size-[44px] [&_input]:text-base lg:h-9 lg:[&_button]:size-9 lg:[&_input]:text-sm"
            />
          </div>

          {canThink && (
            <>
              <Separator />
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <Label>{msg("auto.features.submit.components.modelconfigmodal.8")}</Label>
                  <Switch
                    checked={thinkingEnabled}
                    onCheckedChange={setThinking}
                    className="relative before:absolute before:-inset-3 before:content-[''] lg:before:hidden"
                  />
                </div>
                {thinkingEnabled && (
                  <div className="space-y-2 p-3 border rounded-lg bg-muted/30">
                    <Label>{msg("auto.features.submit.components.modelconfigmodal.9")}</Label>
                    <div className="flex rounded-lg bg-muted p-0.5 w-full">
                      {effortLadder.map((val) => (
                        <button
                          key={val}
                          type="button"
                          onClick={() => setEffort(val)}
                          className={cn(
                            "relative min-h-[44px] flex-1 cursor-pointer rounded-md px-2 py-1.5 text-center text-xs font-medium transition-colors sm:px-3 lg:min-h-0",
                            reasoningEffort === val
                              ? "text-foreground"
                              : "text-muted-foreground hover:text-foreground",
                          )}
                        >
                          {reasoningEffort === val && (
                            <motion.span
                              layoutId={effortPillId}
                              transition={
                                prefersReducedMotion
                                  ? { duration: 0 }
                                  : {
                                      type: "tween",
                                      duration: 0.2,
                                      ease: [0.2, 0.8, 0.2, 1],
                                    }
                              }
                              className="absolute inset-0 rounded-md bg-background shadow-sm"
                              aria-hidden="true"
                            />
                          )}
                          <span className="relative">{effortLabel(val)}</span>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        <DialogFooter className="border-t border-border/40 px-4 pb-4 pt-4 sm:px-6 sm:pb-6">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            className="min-h-[44px] flex-1 lg:min-h-0"
          >
            {msg("auto.features.submit.components.modelconfigmodal.10")}
          </Button>
          <Button
            onClick={handleSave}
            disabled={!draft.name.trim()}
            className="min-h-[44px] flex-1 lg:min-h-0"
          >
            {msg("auto.features.submit.components.modelconfigmodal.11")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
