"use client";

import * as React from "react";
import { AlertTriangle, Boxes, Trash2 } from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/shared/ui/primitives/card";
import { Label } from "@/shared/ui/primitives/label";
import { Separator } from "@/shared/ui/primitives/separator";
import { cn } from "@/shared/lib/utils";
import { formatMsg, msg } from "@/shared/lib/messages";
import { TERMS } from "@/shared/lib/terms";
import { ModelChip, AddModelButton } from "@/shared/ui/model-chip";
import { TokenSourceToggle, useCredits } from "@/features/billing";
import { ModelConfigModal } from "../ModelConfigModal";
import { CostCeilingCard } from "../CostCeilingCard";

import { emptyModelConfig } from "../../constants";
import type { SubmitWizardContext } from "../../hooks/use-submit-wizard";

interface AllAvailableChipProps {
  availableCount: number;
  onReset: () => void;
}

/**
 * Wide chip shown in place of the manual list once "all available" is on.
 *
 * Uses the same visual language as ModelChip but reads as a single pinned
 * selection that sweeps every catalog model. The reset button flips the
 * sentinel flag back off, returning the side to manual mode.
 */
function AllAvailableChip({ availableCount, onReset }: AllAvailableChipProps) {
  return (
    <div
      className={cn(
        "group relative flex w-full items-center gap-3 rounded-lg border px-3 py-2.5",
        "border-primary/30 bg-primary/5",
      )}
    >
      <span className="flex size-9 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
        <Boxes className="size-4" />
      </span>
      <div className="flex min-w-0 flex-1 flex-col">
        <span className="text-sm font-medium text-foreground">
          {msg("auto.features.submit.components.steps.modelstep.1")}
        </span>
        <span className="text-[0.6875rem] text-muted-foreground">
          {msg("auto.features.submit.components.steps.modelstep.2")}
          {TERMS.optimization}
          {msg("auto.features.submit.components.steps.modelstep.3")}
          {TERMS.model}
          {msg("auto.features.submit.components.steps.modelstep.4")}
          {availableCount}
          {msg("auto.features.submit.components.steps.modelstep.5")}
        </span>
      </div>
      <button
        type="button"
        onClick={onReset}
        className="cursor-pointer rounded-md p-1 text-muted-foreground opacity-60 transition-all hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100"
        title={msg("auto.features.submit.components.steps.modelstep.literal.1")}
      >
        <Trash2 className="size-3.5" />
      </button>
    </div>
  );
}

export function ModelStep({ w }: { w: SubmitWizardContext }) {
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
    useAllGenerationModels,
    setUseAllGenerationModels,
    useAllReflectionModels,
    setUseAllReflectionModels,
    editingModel,
    setEditingModel,
    saveToRecent,
    recentConfigs,
    removeRecentConfig,
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
        {jobType === "run" ? (
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
              {useAllGenerationModels ? (
                <AllAvailableChip
                  availableCount={availableCount}
                  onReset={() => setUseAllGenerationModels(false)}
                />
              ) : (
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
                          onSelectAllAvailable: () => setUseAllGenerationModels(true),
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
                          onSelectAllAvailable: () => setUseAllGenerationModels(true),
                        })
                      }
                    />
                  )}
                </div>
              )}
            </div>
            <Separator />
            <div className="space-y-2">
              <Label className="text-sm font-semibold">
                {msg("auto.features.submit.components.steps.modelstep.15")}
              </Label>
              {useAllReflectionModels ? (
                <AllAvailableChip
                  availableCount={availableCount}
                  onReset={() => setUseAllReflectionModels(false)}
                />
              ) : (
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
                          onSelectAllAvailable: () => setUseAllReflectionModels(true),
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
                          onSelectAllAvailable: () => setUseAllReflectionModels(true),
                        })
                      }
                    />
                  )}
                </div>
              )}
            </div>
          </div>
        )}
        {/* Pre-run cost bracket + Max Cost Ceiling [FG-1]. Shown in both modes:
            managed displays the full per-model credit cost, BYOK the platform fee
            (the provider key absorbs the model cost, but credits still meter it). */}
        <CostCeilingCard w={w} mode={wallet.mode} />
        {/* Model config modal — shared across all model chips */}
        <ModelConfigModal
          open={!!editingModel}
          onOpenChange={(open) => {
            if (!open) setEditingModel(null);
          }}
          config={editingModel?.config ?? emptyModelConfig()}
          onSave={(c) => {
            editingModel?.onSave(c);
            saveToRecent(c);
            setEditingModel(null);
          }}
          roleLabel={
            editingModel?.label ?? msg("auto.features.submit.components.steps.modelstep.literal.9")
          }
          catalogModels={catalog?.models}
          recentConfigs={recentConfigs}
          onRemoveRecent={removeRecentConfig}
          onSelectAllAvailable={
            editingModel?.onSelectAllAvailable
              ? () => {
                  editingModel.onSelectAllAvailable?.();
                  setEditingModel(null);
                }
              : undefined
          }
        />
      </CardContent>
    </Card>
  );
}
