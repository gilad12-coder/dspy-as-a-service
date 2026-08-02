"use client";

import * as React from "react";
import dynamic from "next/dynamic";
import { Sparkle, XCircle } from "@/shared/ui/icons";
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

// The same react-flow canvas the /submit wizard uses, mounted inline so a
// workflow (multi-module) run can be authored and edited without leaving chat.
const WorkflowCanvas = dynamic(
  () => import("@/features/submit/workflow/WorkflowCanvas").then((m) => m.WorkflowCanvas),
  { ssr: false, loading: () => <Skeleton height={420} borderRadius={8} /> },
);

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
  const authored = agent.isWorkflow ? !!agent.workflowSpec : !!agent.signatureCode;
  const hasOutput = streaming || authored || !!agent.metricCode || !!agent.reasoning;
  const hasError = agent.status === "error" && !!agent.error;

  // A run that fails before emitting any reasoning or code — e.g. an upstream
  // rate-limit on the first token — would otherwise fall through to the neutral
  // hint below and swallow the reason. Surface the error itself so the user can
  // act (retry, or pick a different model in the composer).
  if (hasError && !hasOutput) {
    return (
      <div className="flex items-start gap-1.5 rounded-2xl border border-[#9B2C1F]/20 bg-[#FCEFEB]/60 px-4 py-3 text-xs text-[#7A1E13]">
        <XCircle className="mt-0.5 size-3 shrink-0 text-[#9B2C1F]" aria-hidden="true" />
        <span className="min-w-0 flex-1 break-words" dir="auto">
          {agent.error}
        </span>
      </div>
    );
  }

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
        <Sparkle className="size-3.5 text-[#3D2E22]" aria-hidden="true" />
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

      {streaming && agent.mode === "seed" && !agent.isWorkflow && (
        <div className="flex justify-center px-4 py-3">
          <ActivityBreadcrumb
            signatureStatus={agent.signatureStatus}
            metricStatus={agent.metricStatus}
          />
        </div>
      )}

      <div dir="ltr" className="space-y-3 px-4 pb-4 pt-3">
        {agent.isWorkflow ? (
          agent.workflowSpec && (
            <div className="h-[440px] overflow-hidden rounded-lg border border-border/40">
              <WorkflowCanvas
                spec={agent.workflowSpec}
                specRevision={agent.workflowRevision}
                onSpecChange={agent.updateWorkflowSpec}
                pulseNodeId={agent.agentPulseNodeId}
              />
            </div>
          )
        ) : (
          <ArtifactBlock
            label={TERMS.signature}
            code={agent.signatureCode}
            streaming={agent.signatureStatus === "writing"}
            validationResult={agent.signatureValidation}
            flashLines={agent.signatureFlashLines}
          />
        )}
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
