"use client";

import { ChevronDown } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/shared/ui/primitives/card";
import { Label } from "@/shared/ui/primitives/label";
import { Badge } from "@/shared/ui/primitives/badge";
import { Separator } from "@/shared/ui/primitives/separator";
import { Switch } from "@/shared/ui/primitives/switch";
import { NumberInput } from "@/shared/ui/number-input";
import { HelpTip } from "@/shared/ui/help-tip";
import { cn } from "@/shared/lib/utils";
import { tip } from "@/shared/lib/tooltips";
import { TERMS } from "@/shared/lib/terms";
import { msg } from "@/shared/lib/messages";
import { useUserPrefs } from "@/features/settings";

import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";
import { SplitRecommendationCard } from "../SplitRecommendationCard";

export function ParamsStep({ w }: { w: SubmitWizardContext }) {
  const { prefs } = useUserPrefs();
  const advanced = prefs.advancedMode;
  const {
    split,
    updateSplit,
    splitSum,
    splitMode,
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
    optimizerSettingsOpen,
    setOptimizerSettingsOpen,
  } = w;

  return (
    <Card
      className=" border-border/50 bg-card/80 backdrop-blur-xl shadow-lg"
      data-tutorial="wizard-step-5"
    >
      <CardHeader>
        <CardTitle className="text-lg">
          {msg("auto.features.submit.components.steps.paramsstep.1")}
        </CardTitle>
        <CardDescription>
          {msg("auto.features.submit.components.steps.paramsstep.2")}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Simple mode leaves the split to the server recommendation (splitMode
            defaults to auto) and hides the whole train/val/test surface. */}
        {advanced && (
        <div className="space-y-3" data-tutorial="data-splits">
          <div className="flex items-center justify-between">
            <Label className="font-semibold">
              <HelpTip text={tip("data.split_explanation")}>
                {msg("auto.features.submit.components.steps.paramsstep.4")}
                {TERMS.dataset}
              </HelpTip>
            </Label>
            {splitSum !== 1 && (
              <Badge variant="destructive" className="text-xs">
                {msg("auto.features.submit.components.steps.paramsstep.5")}
                {splitSum}
              </Badge>
            )}
          </div>
          <SplitRecommendationCard w={w} />
          {splitMode === "manual" && (
            <div className="space-y-3">
              <div className="flex h-3 rounded-full overflow-hidden">
                <div
                  className="bg-[#3D2E22] transition-all"
                  style={{ width: `${split.train * 100}%` }}
                />
                <div
                  className="bg-[#C8A882] transition-all"
                  style={{ width: `${split.val * 100}%` }}
                />
                <div
                  className="bg-[#8C7A6B] transition-all"
                  style={{ width: `${split.test * 100}%` }}
                />
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="space-y-1">
                  <Label htmlFor="split-train" className="flex items-center gap-1.5 text-xs">
                    <span className="inline-block w-2 h-2 rounded-full bg-[#3D2E22]" />
                    <HelpTip text={tip("data.split.train")}>
                      {msg("auto.features.submit.components.steps.paramsstep.6")}
                    </HelpTip>
                  </Label>
                  <NumberInput
                    id="split-train"
                    step={0.05}
                    min={0}
                    max={1}
                    value={split.train}
                    onChange={(v) => updateSplit("train", String(v))}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="split-val" className="flex items-center gap-1.5 text-xs">
                    <span className="inline-block w-2 h-2 rounded-full bg-[#C8A882]" />
                    <HelpTip text={tip("data.split.val")}>
                      {msg("auto.features.submit.components.steps.paramsstep.7")}
                    </HelpTip>
                  </Label>
                  <NumberInput
                    id="split-val"
                    step={0.05}
                    min={0}
                    max={1}
                    value={split.val}
                    onChange={(v) => updateSplit("val", String(v))}
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="split-test" className="flex items-center gap-1.5 text-xs">
                    <span className="inline-block w-2 h-2 rounded-full bg-[#8C7A6B]" />
                    <HelpTip text={tip("data.split.test")}>
                      {msg("auto.features.submit.components.steps.paramsstep.8")}
                    </HelpTip>
                  </Label>
                  <NumberInput
                    id="split-test"
                    step={0.05}
                    min={0}
                    max={1}
                    value={split.test}
                    onChange={(v) => updateSplit("test", String(v))}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
        )}

        {advanced && <Separator />}

        <div className="space-y-4">
          <Label className="font-semibold">
            {msg("auto.features.submit.components.steps.paramsstep.9")}
          </Label>
          <div className="flex items-center justify-between">
            <Label htmlFor="shuffle" className="cursor-pointer text-sm">
              <HelpTip text={tip("data.shuffle_explanation")}>
                {msg("auto.features.submit.components.steps.paramsstep.10")}
              </HelpTip>
            </Label>
            <Switch id="shuffle" checked={shuffle} onCheckedChange={setShuffle} />
          </div>
          <div className="space-y-2" data-tutorial="auto-level">
            <Label className="text-sm">
              <HelpTip text={tip("submit.depth")}>
                {msg("auto.features.submit.components.steps.paramsstep.12")}
              </HelpTip>
            </Label>
            <div className="relative inline-flex w-full rounded-lg bg-muted p-1 gap-1">
              {autoLevel && (
                <div
                  className="absolute top-1 bottom-1 rounded-md bg-background shadow-sm transition-[inset-inline-start] duration-100 ease-out pointer-events-none"
                  style={{
                    width: "calc((100% - 8px) / 3)",
                    insetInlineStart: `calc(${(["light", "medium", "heavy"] as string[]).indexOf(autoLevel)} * (100% / 3) + 4px)`,
                  }}
                />
              )}
              {(
                [
                  ["light", msg("auto.features.submit.components.steps.paramsstep.literal.1")],
                  ["medium", msg("auto.features.submit.components.steps.paramsstep.literal.2")],
                  ["heavy", msg("auto.features.submit.components.steps.paramsstep.literal.3")],
                ] as const
              ).map(([val, label]) => (
                <button
                  key={val}
                  type="button"
                  onClick={() => setAutoLevel(autoLevel === val ? "" : val)}
                  className={cn(
                    "relative z-[1] flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors text-center cursor-pointer",
                    autoLevel === val
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <Separator />
          <div className="space-y-3">
            <button
              type="button"
              onClick={() => setOptimizerSettingsOpen(!optimizerSettingsOpen)}
              aria-expanded={optimizerSettingsOpen}
              className="flex w-full cursor-pointer items-center justify-between gap-2"
            >
              <span className="text-sm leading-none font-medium">
                {msg("auto.features.submit.components.steps.paramsstep.11")}
                {TERMS.optimizer}
              </span>
              <ChevronDown
                className={cn(
                  "size-4 shrink-0 text-muted-foreground transition-transform duration-150",
                  optimizerSettingsOpen && "rotate-180",
                )}
              />
            </button>
            {optimizerSettingsOpen && (
              <div
                className="grid grid-cols-1 gap-3 sm:grid-cols-2 motion-safe:animate-in motion-safe:fade-in motion-safe:slide-in-from-top-1 motion-safe:duration-200"
                data-tutorial="gepa-params"
              >
                <div className="space-y-1.5">
                  <Label className="text-xs">
                    <HelpTip text={tip("submit.reflection_minibatch")}>
                      {msg("auto.features.submit.components.steps.paramsstep.13")}
                    </HelpTip>
                  </Label>
                  <NumberInput
                    min={1}
                    max={20}
                    step={1}
                    value={reflectionMinibatchSize ? parseInt(reflectionMinibatchSize, 10) : ""}
                    onChange={(v) => setReflectionMinibatchSize(String(v))}
                  />
                </div>
                <div className="space-y-1.5">
                  <Label className={cn("text-xs", autoLevel && "text-muted-foreground/50")}>
                    <HelpTip text={tip("submit.eval_rounds")}>
                      {msg("auto.features.submit.components.steps.paramsstep.14")}
                    </HelpTip>
                  </Label>
                  <NumberInput
                    min={1}
                    max={50}
                    step={1}
                    value={maxFullEvals ? parseInt(maxFullEvals, 10) : ""}
                    onChange={(v) => setMaxFullEvals(String(v))}
                    disabled={!!autoLevel}
                  />
                </div>
                <div className="col-span-2 flex items-center justify-between">
                  <Label className="text-sm cursor-pointer">
                    <HelpTip text={tip("submit.merge")}>
                      {msg("auto.features.submit.components.steps.paramsstep.15")}
                    </HelpTip>
                  </Label>
                  <Switch checked={useMerge} onCheckedChange={setUseMerge} />
                </div>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
