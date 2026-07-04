"use client";

import { motion, AnimatePresence } from "framer-motion";

import { msg } from "@/shared/lib/messages";

import { useSubmitWizard } from "../hooks/use-submit-wizard";
import { slideVariants, emptyModelConfig } from "../constants";
import { SubmitStepper } from "./SubmitStepper";
import { SubmitNav } from "./SubmitNav";
import { SubmitSplash } from "./SubmitSplash";
import { ModelConfigModal } from "./ModelConfigModal";
import { BasicsStep } from "./steps/BasicsStep";
import { DatasetStep } from "./steps/DatasetStep";
import { ModelStep } from "./steps/ModelStep";
import { CodeStep } from "./steps/CodeStep";
import { ParamsStep } from "./steps/ParamsStep";
import { SummaryStep } from "./steps/SummaryStep";

export function SubmitWizard() {
  const w = useSubmitWizard();

  const steps = [
    <BasicsStep key="basics" w={w} />,
    <DatasetStep key="data" w={w} />,
    <ParamsStep key="params" w={w} />,
    <CodeStep key="code" w={w} />,
    <ModelStep key="model" w={w} />,
    <SummaryStep key="review" w={w} />,
  ];

  // Code step (index 3) renders a two-pane layout with an agent side-panel
  // in auto mode, so it needs more horizontal room than the other steps.
  const isCodeStep = w.step === 3;
  const containerWidthClass = isCodeStep && w.codeAssistMode === "auto" ? "max-w-5xl" : "max-w-2xl";

  return (
    <div
      className={`space-y-6 ${containerWidthClass} mx-auto pb-8 -mt-2 md:-mt-4 transition-[max-width] duration-300`}
    >
      <SubmitStepper w={w} />

      <div className="relative overflow-hidden pt-[10px]" data-tutorial="submit-wizard">
        <AnimatePresence mode="wait" custom={w.direction}>
          <motion.div
            key={w.step}
            custom={w.direction}
            variants={slideVariants}
            initial="enter"
            animate="center"
            exit="exit"
            transition={{ duration: 0.1 }}
          >
            {steps[w.step]}
          </motion.div>
        </AnimatePresence>
      </div>

      <SubmitNav w={w} />

      {/* Model config modal — shared by every model chip AND the workflow
          canvas's dry-run "pick a model in place" flow, so it mounts at the
          wizard root rather than inside the model step. */}
      <ModelConfigModal
        open={!!w.editingModel}
        onOpenChange={(open) => {
          if (!open) w.setEditingModel(null);
        }}
        config={w.editingModel?.config ?? emptyModelConfig()}
        onSave={(c) => {
          w.editingModel?.onSave(c);
          w.saveToRecent(c);
          w.setEditingModel(null);
        }}
        roleLabel={w.editingModel?.label ?? msg("model.generation.label")}
        catalogModels={w.catalog?.models}
        recentConfigs={w.recentConfigs}
        onRemoveRecent={w.removeRecentConfig}
        onSelectAllAvailable={
          w.editingModel?.onSelectAllAvailable
            ? () => {
                w.editingModel?.onSelectAllAvailable?.();
                w.setEditingModel(null);
              }
            : undefined
        }
      />

      {/* Submit splash overlay — portal to body so it covers sidebar + header */}
      <SubmitSplash w={w} />
    </div>
  );
}
