"use client";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/primitives/dialog";
import type { OptimizationStatusResponse } from "@/shared/types/api";
import { TERMS } from "@/shared/lib/terms";
import { PIPELINE_STAGES, STAGE_INFO, type PipelineStage } from "../constants";
import { msg } from "@/shared/lib/messages";

export function StageInfoModal({
  stage,
  job,
  onClose,
}: {
  stage: PipelineStage | null;
  job: OptimizationStatusResponse | null;
  onClose: () => void;
}) {
  return (
    <Dialog
      open={stage !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent className="w-[min(28rem,92vw)] max-w-[min(28rem,92vw)] sm:max-w-md">
        {stage &&
          (() => {
            const info = STAGE_INFO[stage];
            if (!info) return null;
            const stageIndex = PIPELINE_STAGES.findIndex((s) => s.key === stage);
            return (
              <>
                <DialogHeader>
                  <div className="flex items-center gap-3 mb-1">
                    <div className="size-9 rounded-full bg-[#3D2E22] text-white flex items-center justify-center text-sm font-bold">
                      {stageIndex + 1}
                    </div>
                    <div>
                      <DialogTitle className="text-base">{info.title}</DialogTitle>
                      <DialogDescription className="text-[0.8125rem] mt-0.5">
                        {info.description}
                      </DialogDescription>
                    </div>
                  </div>
                </DialogHeader>
                <div className="space-y-4 text-sm">
                  <p className="text-muted-foreground leading-relaxed">{info.details}</p>

                  {stage === "baseline" && job?.baseline_test_metric != null && (
                    <div className="rounded-xl bg-muted/40 p-3">
                      <div className="text-[0.6875rem] font-semibold text-[#3D2E22] mb-1">
                        {msg("auto.features.optimizations.components.stageinfomodal.1")}
                      </div>
                      <div className="flex items-baseline gap-1">
                        <span className="text-2xl font-bold tabular-nums">
                          {job.baseline_test_metric.toFixed(1)}%
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {TERMS.baselineScore}
                          {msg("auto.features.optimizations.components.stageinfomodal.2")}
                        </span>
                      </div>
                    </div>
                  )}

                  {stage === "evaluating" && job?.optimized_test_metric != null && (
                    <div className="rounded-xl bg-muted/40 p-3">
                      <div className="text-[0.6875rem] font-semibold text-[#3D2E22] mb-1">
                        {msg("auto.features.optimizations.components.stageinfomodal.3")}
                      </div>
                      <div className="flex items-baseline gap-3">
                        <div>
                          <span className="text-2xl font-bold tabular-nums text-[#3D2E22]">
                            {job.optimized_test_metric.toFixed(1)}%
                          </span>
                          <span className="text-[0.625rem] text-muted-foreground ms-1">
                            {msg("auto.features.optimizations.components.stageinfomodal.4")}
                          </span>
                        </div>
                        {job.baseline_test_metric != null && (
                          <div className="text-xs text-muted-foreground">
                            {msg("auto.features.optimizations.components.stageinfomodal.5")}
                            {job.baseline_test_metric.toFixed(1)}%
                            <span
                              className={`ms-1 font-semibold ${(job.metric_improvement ?? 0) >= 0 ? "text-[#3D2E22]" : "text-[#B04030]"}`}
                            >
                              ({(job.metric_improvement ?? 0) >= 0 ? "+" : ""}
                              {(job.metric_improvement ?? 0).toFixed(1)}%)
                            </span>
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </>
            );
          })()}
      </DialogContent>
    </Dialog>
  );
}
