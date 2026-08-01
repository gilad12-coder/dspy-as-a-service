"use client";

import * as React from "react";
import { AlertTriangle } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/shared/ui/primitives/card";
import { Label } from "@/shared/ui/primitives/label";
import { Separator } from "@/shared/ui/primitives/separator";
import { formatMsg, msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";
import { ModelChip, AddModelButton } from "@/shared/ui/model-chip";
import { TokenSourceToggle, useCredits } from "@/features/billing";
import { useUserPrefs } from "@/features/settings";
import { CostCeilingCard } from "../CostCeilingCard";

import { emptyModelConfig } from "../../constants";
import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";

/**
 * Two-line primer distinguishing the two model roles this step configures:
 * the generation model does the task, the reflection model rewrites the
 * prompt between rounds. The distinction trips up even people who have used
 * DSPy before, so it's stated inline here rather than hidden behind a hover.
 */
function ModelRolesExplainer() {
  return (
    <div className="space-y-2 rounded-lg border border-border/50 bg-muted/20 px-3.5 py-3">
      <p className="text-[0.75rem] leading-relaxed text-muted-foreground">
        <span className="font-semibold text-foreground">{msg("model.generation.label")}</span>
        {" — "}
        {msg("model.generation.explainer")}
      </p>
      <p className="text-[0.75rem] leading-relaxed text-muted-foreground">
        <span className="font-semibold text-foreground">{TERMS.reflectionModel}</span>
        {" — "}
        {msg("model.reflection.explainer")}
      </p>
    </div>
  );
}
export function ModelStep({ w }: { w: SubmitWizardContext }) {
  const { prefs } = useUserPrefs();
  const advanced = prefs.advancedMode;
  const {
    jobType,
    modelConfig,
    setModelConfig,
    secondModelConfig,
    setSecondModelConfig,
    generationModels,
    setGenerationModels,
    reflectionModels,
    setReflectionModels,
    setEditingModel,
    catalog,
  } = w;

  const { wallet } = useCredits();
  const availableCount = catalog?.models.length ?? 0;
  const catalogEmpty = catalog != null && availableCount === 0;

  return (
    <Card
      className="border-border/50 bg-card/80 backdrop-blur-xl shadow-lg"
      data-tutorial="wizard-step-3"
    >
      <CardHeader>
        <CardTitle className="text-lg">
          {msg("auto.features.submit.components.steps.modelstep.6")}
        </CardTitle>
        <CardDescription>
          {msg("auto.features.submit.components.steps.modelstep.7")}
          {TERMS.optimization}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <TokenSourceToggle />
        <ModelRolesExplainer />
        {jobType === "run" || !advanced ? (
          <div className="space-y-3" data-tutorial="model-catalog">
            <Label className="text-sm font-semibold">
              {msg("auto.features.submit.components.steps.modelstep.13")}
            </Label>
            <div className="space-y-2">
              <ModelChip
                config={modelConfig}
                roleLabel={msg("model.generation.label")}
                required
                catalogModels={catalog?.models}
                onClick={() =>
                  setEditingModel({
                    config: modelConfig,
                    onSave: setModelConfig,
                    label: msg("model.generation.label"),
                  })
                }
                onRemove={modelConfig.name ? () => setModelConfig(emptyModelConfig()) : undefined}
              />
              <ModelChip
                config={secondModelConfig ?? emptyModelConfig()}
                roleLabel={TERMS.reflectionModel}
                required
                catalogModels={catalog?.models}
                onClick={() =>
                  setEditingModel({
                    config: secondModelConfig ?? emptyModelConfig(),
                    onSave: setSecondModelConfig,
                    label: TERMS.reflectionModel,
                  })
                }
                onRemove={secondModelConfig?.name ? () => setSecondModelConfig(null) : undefined}
              />
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            {catalogEmpty && (
              <div className="flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/5 px-3 py-2 text-[0.75rem] text-amber-700 dark:text-amber-400">
                <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
                <span>{msg("auto.features.submit.components.steps.modelstep.14")}</span>
              </div>
            )}
            <div className="space-y-2">
              <Label className="text-sm font-semibold">
                {msg("model.generation.label_plural")}
              </Label>
              <div className="flex flex-wrap gap-2">
                {generationModels.map((m, i) => (
                  <ModelChip
                    key={i}
                    config={m}
                    catalogModels={catalog?.models}
                    onClick={() =>
                      setEditingModel({
                        config: m,
                        onSave: (c) => {
                          const u = [...generationModels];
                          u[i] = c;
                          setGenerationModels(u);
                        },
                        label: `${msg("model.generation.label")} ${i + 1}`,
                      })
                    }
                    onRemove={() => {
                      const next = generationModels.filter((_, j) => j !== i);
                      setGenerationModels(next.length ? next : [emptyModelConfig()]);
                    }}
                  />
                ))}
                {generationModels.every((m) => m.name.trim()) && (
                  <AddModelButton
                    label={msg("auto.features.submit.components.steps.modelstep.literal.7")}
                    onClick={() =>
                      setEditingModel({
                        config: generationModels.length
                          ? { ...generationModels[generationModels.length - 1], name: "" }
                          : emptyModelConfig(),
                        onSave: (c) =>
                          setGenerationModels([
                            ...generationModels.filter((m) => m.name.trim()),
                            c,
                          ]),
                        label: msg("model.generation.new"),
                      })
                    }
                  />
                )}
              </div>
            </div>
            <Separator />
            <div className="space-y-2">
              <Label className="text-sm font-semibold">
                {msg("auto.features.submit.components.steps.modelstep.15")}
              </Label>
              <div className="flex flex-wrap gap-2">
                {reflectionModels.map((m, i) => (
                  <ModelChip
                    key={i}
                    config={m}
                    catalogModels={catalog?.models}
                    onClick={() =>
                      setEditingModel({
                        config: m,
                        onSave: (c) => {
                          const u = [...reflectionModels];
                          u[i] = c;
                          setReflectionModels(u);
                        },
                        label: `${TERMS.reflectionModel} ${i + 1}`,
                      })
                    }
                    onRemove={() => {
                      const next = reflectionModels.filter((_, j) => j !== i);
                      setReflectionModels(next.length ? next : [emptyModelConfig()]);
                    }}
                  />
                ))}
                {reflectionModels.every((m) => m.name.trim()) && (
                  <AddModelButton
                    label={msg("auto.features.submit.components.steps.modelstep.literal.8")}
                    onClick={() =>
                      setEditingModel({
                        config: reflectionModels.length
                          ? { ...reflectionModels[reflectionModels.length - 1], name: "" }
                          : emptyModelConfig(),
                        onSave: (c) =>
                          setReflectionModels([
                            ...reflectionModels.filter((m) => m.name.trim()),
                            c,
                          ]),
                        label: formatMsg(
                          "auto.features.submit.components.steps.modelstep.template.2",
                          { p1: TERMS.reflectionModel },
                        ),
                      })
                    }
                  />
                )}
              </div>
            </div>
          </div>
        )}
        {/* Pre-run cost bracket + Max Cost Ceiling [FG-1]. Shown in both modes:
            managed displays the full per-model credit cost, BYOK the platform fee
            (the provider key absorbs the model cost, but credits still meter it). */}
        <CostCeilingCard w={w} mode={wallet.mode} />
      </CardContent>
    </Card>
  );
}
