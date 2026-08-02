"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowSquareOut, CheckCircle, WarningCircle } from "@/shared/ui/icons";
import { formatMsg, msg } from "@/shared/lib/messages";

import { cn } from "@/shared/lib/utils";
import { TERMS } from "@/shared/lib/terms";

import type { AgentToolCall } from "@/shared/ui/agent/types";

interface SubmitSummaryCardProps {
  call: AgentToolCall;
  className?: string;
}

interface SubmitResult {
  id?: string;
  job_name?: string;
  status?: string;
  detail?: string;
}

function extractResult(call: AgentToolCall): SubmitResult | null {
  const payload = (call.payload ?? {}) as Record<string, unknown>;
  // The submit endpoints return ``OptimizationSubmissionResponse``
  // (``optimization_id`` + ``name``); error/legacy payloads may instead carry
  // ``id`` / ``job_name`` / ``detail``. Read from ``payload.result`` when the
  // tool nests its output there, else from the payload top level, and accept
  // either field name so the link and title always resolve.
  const raw = (
    payload.result && typeof payload.result === "object" ? payload.result : payload
  ) as Record<string, unknown>;
  const id = raw.optimization_id ?? raw.id;
  const jobName = raw.name ?? raw.job_name;
  const { status, detail } = raw;
  if ([id, jobName, status, detail].every((v) => v === undefined)) return null;
  return {
    id: typeof id === "string" ? id : undefined,
    job_name: typeof jobName === "string" ? jobName : undefined,
    status: typeof status === "string" ? status : undefined,
    detail: typeof detail === "string" ? detail : undefined,
  };
}

/**
 * Summary card rendered after a successful `submit_optimization` call.
 * Replaces the generic tool chip so the user immediately sees the
 * resulting job and can jump to it.
 */
export function SubmitSummaryCard({ call, className }: SubmitSummaryCardProps) {
  const result = extractResult(call);
  const isError = call.status === "error";
  const jobId = result?.id;
  const jobName = result?.job_name ?? TERMS.notificationNewOpt;

  return (
    <div
      className={cn(
        "rounded-2xl border shadow-sm overflow-hidden",
        isError ? "border-red-200 bg-red-50" : "border-[#5E7A5E]/25 bg-[#F0F4EC]",
        className,
      )}
    >
      <div
        className={cn(
          "flex items-start gap-2.5 px-4 py-3 border-b",
          isError ? "border-red-200" : "border-[#5E7A5E]/15",
        )}
      >
        <span
          className={cn(
            "inline-flex size-7 shrink-0 items-center justify-center rounded-full",
            isError ? "bg-red-100 text-red-700" : "bg-[#5E7A5E]/20 text-[#3E5240]",
          )}
        >
          {isError ? (
            <WarningCircle className="size-3.5" aria-hidden="true" />
          ) : (
            <CheckCircle className="size-3.5" aria-hidden="true" />
          )}
        </span>
        <div className="min-w-0 flex-1">
          <div
            className={cn(
              "text-[0.8125rem] font-medium leading-tight",
              isError ? "text-red-800" : "text-[#2F3E32]",
            )}
          >
            {isError
              ? msg("auto.features.agent.panel.components.submitsummarycard.literal.1")
              : formatMsg("auto.features.agent.panel.components.submitsummarycard.template.1", {
                  p1: TERMS.optimization,
                })}
          </div>
          <div className="text-[0.75rem] text-foreground/80 mt-0.5 truncate">{jobName}</div>
        </div>
      </div>

      {!isError && jobId && (
        <Link
          href={`/optimizations/${jobId}`}
          className={cn(
            "flex items-center justify-between gap-2 px-4 py-2.5",
            "text-[0.75rem] text-[#3D2E22] hover:bg-[#DFE7D9]/50 transition-colors",
          )}
        >
          <span className="font-mono truncate text-muted-foreground">{jobId}</span>
          <span className="inline-flex items-center gap-1 text-[#3D2E22] shrink-0">
            {msg("auto.features.agent.panel.components.submitsummarycard.1")}
            <ArrowSquareOut className="size-3" aria-hidden="true" />
          </span>
        </Link>
      )}

      {isError && result?.detail && (
        <div className="px-4 py-2.5 text-[0.75rem] text-red-800 leading-relaxed">
          {result.detail}
        </div>
      )}
    </div>
  );
}
