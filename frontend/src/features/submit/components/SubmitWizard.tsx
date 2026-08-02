"use client";

import { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ArrowCounterClockwise } from "@/shared/ui/icons";

import { msg } from "@/shared/lib/messages";
import { Button } from "@/shared/ui/primitives/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";

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
  // "Start over" is a full remount: bumping this key drops every field in the
  // wizard hook back to its defaults. The unmounting instance discards its
  // parked draft (see startOver) so the fresh mount doesn't restore it.
  const [resetKey, setResetKey] = useState(0);
  return <SubmitWizardInner key={resetKey} onStartOver={() => setResetKey((k) => k + 1)} />;
}

function SubmitWizardInner({ onStartOver }: { onStartOver: () => void }) {
  const w = useSubmitWizard();
  const [confirmStartOver, setConfirmStartOver] = useState(false);

  const handleStartOver = () => {
    // Mark this instance so its unmount cleanup discards the draft rather than
    // parking it, then trigger the remount that re-seeds every field. The
    // confirming click unmounts this whole subtree, so the dialog goes with it.
    w.startOver();
    onStartOver();
  };

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
      <div>
        {w.canStartOver && (
          <div className="mb-2 flex justify-end">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setConfirmStartOver(true)}
              className="gap-1.5 text-muted-foreground"
            >
              <ArrowCounterClockwise className="size-3.5" />
              {msg("submit.start_over")}
            </Button>
          </div>
        )}
        <SubmitStepper w={w} />
      </div>

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
      />

      {/* Submit splash overlay — portal to body so it covers sidebar + header */}
      <SubmitSplash w={w} />

      {/* "Start over" wipes an uploaded dataset and every step, so guard it. */}
      <Dialog open={confirmStartOver} onOpenChange={setConfirmStartOver}>
        <DialogContent className="sm:max-w-md" showCloseButton={false}>
          <DialogHeader>
            <DialogTitle>{msg("submit.start_over_confirm.title")}</DialogTitle>
            <DialogDescription>{msg("submit.start_over_confirm.body")}</DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-2 gap-3">
            <Button
              variant="outline"
              onClick={() => setConfirmStartOver(false)}
              className="w-full justify-center"
            >
              {msg("submit.start_over_confirm.cancel")}
            </Button>
            <Button
              variant="destructive"
              onClick={handleStartOver}
              className="w-full justify-center"
            >
              {msg("submit.start_over_confirm.discard")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
