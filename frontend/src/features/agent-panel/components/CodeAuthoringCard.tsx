"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { Sparkles, XCircle } from "lucide-react";
import { msg } from "@/shared/lib/messages";

import { cn } from "@/shared/lib/utils";
import { TERMS } from "@/shared/lib/terms";
import { Skeleton } from "@/shared/ui/skeleton";
import { ActivityBreadcrumb } from "@/shared/ui/agent/activity-breadcrumb";
import { ThinkingSection } from "@/shared/ui/agent/thinking-section";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import type { ValidateCodeResponse } from "@/shared/types/api";

import type { CodeAuthoringAgentState } from "../hooks/use-code-authoring-agent";

const CodeEditor = dynamic(() => import("@/shared/ui/code-editor").then((m) => m.CodeEditor), {
  ssr: false,
  loading: () => <Skeleton height={150} borderRadius={8} />,
});

const NOOP = () => {};

interface CodeAuthoringCardProps {
  /** Lifted code-agent state, owned by the panel so it survives collapse. */
  agent: CodeAuthoringAgentState;
}

/**
 * Inline mirror of the wizard's code agent, shown in the generalist chat when
 * the agent calls ``request_code_authoring``. It renders the lifted code-agent
 * state with the same shared pieces the wizard uses — the thinking timer, the
 * reading→signature→metric breadcrumb, and the streaming Signature + Metric
 * editors — so it reflects exactly what the code agent is doing. It drives
 * nothing itself: the panel hosts the agent and writes the result into the
 * shared wizard state on completion.
 */
export function CodeAuthoringCard({ agent }: CodeAuthoringCardProps) {
  const streaming = agent.status === "streaming";
  const hasOutput = streaming || !!agent.signatureCode || !!agent.metricCode || !!agent.reasoning;

  // Before the seed starts (or on a reopened historical conversation where the
  // run state was cleared) there is nothing to mirror — show the neutral hint.
  if (!hasOutput) {
    return (
      <div className="rounded-2xl border border-border/50 bg-card/70 px-4 py-3 text-[0.75rem] text-muted-foreground">
        {msg("auto.features.submit.components.steps.codeagentpanel.literal.16")}
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-[#C8A882]/30 bg-[#FAF8F5] shadow-sm">
      <div className="flex items-center gap-2 border-b border-[#3D2E22]/10 px-4 py-2.5 text-[0.8125rem] font-medium text-[#3D2E22]">
        <Sparkles className="size-3.5 text-[#3D2E22]" aria-hidden="true" />
        {msg("auto.features.submit.components.steps.codestep.1")}
      </div>

      <ThinkingSection
        thinking={{
          reasoning: agent.reasoning,
          startedAt: agent.reasoningStartedAt,
          endedAt: agent.reasoningEndedAt,
          streaming,
        }}
      />

      {streaming && agent.mode === "seed" && (
        <div className="flex justify-center px-4 py-3">
          <ActivityBreadcrumb
            signatureStatus={agent.signatureStatus}
            metricStatus={agent.metricStatus}
          />
        </div>
      )}

      <div dir="ltr" className="space-y-3 px-4 pb-4 pt-3">
        <ArtifactBlock
          label={TERMS.signature}
          code={agent.signatureCode}
          streaming={agent.signatureStatus === "writing"}
          validationResult={agent.signatureValidation}
          flashLines={agent.signatureFlashLines}
        />
        <ArtifactBlock
          label={TERMS.metric}
          code={agent.metricCode}
          streaming={agent.metricStatus === "writing"}
          validationResult={agent.metricValidation}
          flashLines={agent.metricFlashLines}
        />
      </div>

      {agent.error && (
        <div
          className="flex items-start gap-1.5 border-t border-[#9B2C1F]/20 bg-[#FCEFEB]/60 px-4 py-2 text-xs text-[#7A1E13]"
        >
          <XCircle className="mt-0.5 size-3 shrink-0 text-[#9B2C1F]" aria-hidden="true" />
          <span className="min-w-0 flex-1 break-words" dir="auto">
            {agent.error}
          </span>
        </div>
      )}
    </div>
  );
}

function ArtifactBlock({
  label,
  code,
  streaming,
  validationResult,
  flashLines,
}: {
  label: string;
  code: string;
  streaming: boolean;
  validationResult: ValidateCodeResponse | null;
  flashLines: number[];
}) {
  return (
    <div className="space-y-1.5">
      <span
        className={cn(
          "text-xs font-semibold uppercase tracking-wide text-muted-foreground",
          streaming && "text-[#3D2E22]",
        )}
        dir={getActiveDir()}
      >
        {label}
      </span>
      <CodeEditor
        value={code}
        onChange={NOOP}
        height="150px"
        readOnly
        streaming={streaming}
        validationResult={validationResult}
        flashLines={flashLines}
      />
    </div>
  );
}
