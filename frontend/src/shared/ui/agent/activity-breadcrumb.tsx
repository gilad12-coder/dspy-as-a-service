"use client";

import * as React from "react";
import { Check, CircleNotch } from "@/shared/ui/icons";
import { formatMsg, msg } from "@/shared/lib/messages";

import { cn } from "@/shared/lib/utils";
import { TERMS } from "@/shared/lib/terms";

export type ArtifactStatus = "idle" | "waiting" | "writing" | "done";
type StepState = "pending" | "active" | "done";

/**
 * The code agent's reading→signature→metric progress stepper. Shared by the
 * submit wizard's code step and the generalist panel's code-authoring card so
 * both render the identical timeline. Driven purely by the two artifact
 * statuses, so any caller (the wizard's useCodeAgent or the panel's slim
 * useCodeAuthoring) can feed it without sharing a hook.
 */
export function ActivityBreadcrumb({
  signatureStatus,
  metricStatus,
}: {
  signatureStatus: ArtifactStatus;
  metricStatus: ArtifactStatus;
}) {
  const steps = React.useMemo<Array<{ label: string; state: StepState }>>(() => {
    const readingState: StepState =
      signatureStatus === "waiting" && metricStatus === "waiting" ? "active" : "done";
    const sigState: StepState =
      signatureStatus === "writing" ? "active" : signatureStatus === "done" ? "done" : "pending";
    const metState: StepState =
      metricStatus === "writing" ? "active" : metricStatus === "done" ? "done" : "pending";
    return [
      {
        label: formatMsg("auto.features.submit.components.steps.codeagentpanel.template.1", {
          p1: TERMS.dataset,
        }),
        state: readingState,
      },
      {
        label: msg("auto.features.submit.components.steps.codeagentpanel.literal.5"),
        state: sigState,
      },
      { label: TERMS.metric, state: metState },
    ];
  }, [signatureStatus, metricStatus]);

  const lastReachedIdx = steps.reduce((acc, s, i) => (s.state === "pending" ? acc : i), -1);
  const fillPct = lastReachedIdx < 0 ? 0 : (lastReachedIdx / (steps.length - 1)) * 100;

  return (
    <div
      className="relative flex w-full max-w-[340px] items-start justify-between"
      aria-live="polite"
    >
      <div className="absolute top-[9px] start-[9px] end-[9px] h-px bg-border/70" />
      <div
        className="absolute top-[9px] start-[9px] h-px bg-[#3D2E22]/55 transition-[width] duration-500 ease-out"
        style={{ width: `calc(${fillPct}% - 18px)` }}
        aria-hidden
      />
      {steps.map((step) => (
        <div key={step.label} className="relative z-[1] flex min-w-0 flex-col items-center gap-1.5">
          <StepNode state={step.state} />
          <span
            className={cn(
              "whitespace-nowrap text-[0.625rem] leading-none tracking-wide transition-colors duration-200",
              step.state === "pending" && "text-muted-foreground/45",
              step.state === "active" && "text-[#3D2E22] font-semibold",
              step.state === "done" && "text-[#3D2E22]/75",
            )}
          >
            {step.label}
          </span>
        </div>
      ))}
    </div>
  );
}

function StepNode({ state }: { state: StepState }) {
  if (state === "done") {
    return (
      <span className="inline-flex size-[18px] items-center justify-center rounded-full bg-[#3D2E22] text-white transition-colors">
        <Check className="size-2.5" />
      </span>
    );
  }
  if (state === "active") {
    return (
      <span className="inline-flex size-[18px] items-center justify-center rounded-full bg-[#3D2E22] text-white shadow-[0_0_0_3px_rgba(61,46,34,0.12)] transition-colors motion-reduce:shadow-none">
        <CircleNotch className="size-2.5 animate-spin motion-reduce:animate-none" />
      </span>
    );
  }
  return (
    <span className="inline-flex size-[18px] items-center justify-center rounded-full bg-[#E5DDD4] transition-colors">
      <span className="size-1 rounded-full bg-[#8C7A6B]/70" />
    </span>
  );
}
