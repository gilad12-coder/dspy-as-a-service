import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";

export type PipelineStage =
  | "validating"
  | "splitting"
  | "baseline"
  | "optimizing"
  | "evaluating"
  | "done";

export const PIPELINE_STAGES: Array<{ key: PipelineStage; label: string }> = [
  { key: "validating", label: msg("auto.features.optimizations.constants.literal.1") },
  { key: "splitting", label: msg("auto.features.optimizations.constants.literal.2") },
  { key: "baseline", label: TERMS.baselineScore },
  { key: "optimizing", label: TERMS.optimization },
  { key: "evaluating", label: msg("auto.features.optimizations.constants.literal.3") },
];

export const STAGE_INFO: Record<string, { title: string; description: string; details: string }> = {
  validating: {
    title: msg("auto.features.optimizations.constants.literal.4"),
    description: formatMsg("auto.features.optimizations.constants.template.1", {
      p1: TERMS.optimization,
    }),
    details: formatMsg("auto.features.optimizations.constants.template.2", {
      p1: TERMS.signature,
      p2: TERMS.dataset,
      p3: TERMS.metric,
      p4: TERMS.module,
      p5: TERMS.optimizer,
      p6: TERMS.optimization,
    }),
  },
  splitting: {
    title: msg("auto.features.optimizations.constants.literal.5"),
    description: formatMsg("auto.features.optimizations.constants.template.3", {
      p1: TERMS.dataset,
      p2: TERMS.splitTrain,
      p3: TERMS.splitVal,
      p4: TERMS.splitTest,
    }),
    details: formatMsg("auto.features.optimizations.constants.template.4", {
      p1: TERMS.optimizationTypeRun,
      p2: TERMS.optimization,
    }),
  },
  baseline: {
    title: formatMsg("auto.features.optimizations.constants.template.5", {
      p1: TERMS.baselineScore,
    }),
    description: formatMsg("auto.features.optimizations.constants.template.6", {
      p1: TERMS.program,
      p2: TERMS.optimization,
    }),
    details: formatMsg("auto.features.optimizations.constants.template.7", {
      p1: TERMS.program,
      p2: TERMS.example,
      p3: TERMS.metric,
      p4: TERMS.score,
      p5: TERMS.example,
      p6: TERMS.baselineScore,
      p7: TERMS.scoreImprovement,
      p8: TERMS.optimization,
    }),
  },
  optimizing: {
    title: TERMS.optimization,
    description: formatMsg("auto.features.optimizations.constants.template.8", {
      p1: TERMS.optimizer,
      p2: TERMS.program,
    }),
    details: formatMsg("auto.features.optimizations.constants.template.9", {
      p1: TERMS.optimizer,
      p2: TERMS.program,
    }),
  },
  evaluating: {
    title: msg("auto.features.optimizations.constants.literal.6"),
    description: formatMsg("auto.features.optimizations.constants.template.10", {
      p1: TERMS.program,
    }),
    details: formatMsg("auto.features.optimizations.constants.template.11", {
      p1: TERMS.program,
      p2: TERMS.baseline,
      p3: TERMS.scoreImprovement,
      p4: TERMS.program,
      p5: TERMS.program,
    }),
  },
};
