"use client";

import { Sparkles, AlertTriangle, Info } from "lucide-react";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/shared/ui/primitives/tooltip";
import { cn } from "@/shared/lib/utils";
import { msg } from "@/shared/lib/messages";

import type { SubmitWizardContext } from "../hooks/use-submit-wizard";

const percent = (value: number): string => `${Math.round(value * 100)}%`;

export function SplitRecommendationCard({ w }: { w: SubmitWizardContext }) {
  const { splitPlan, datasetProfile, splitMode, setSplitMode, profileLoading } = w;

  if (!splitPlan) {
    if (profileLoading) {
      return (
        <div
          className="flex items-center gap-2 rounded-xl border border-[#DDD6CC]/60 bg-[#FAF8F5]/70 px-3.5 py-2.5 text-xs text-[#8C7A6B]"
        >
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-[#C8A882] motion-safe:animate-pulse" />
          {msg("submit.split.recommended_title")}…
        </div>
      );
    }
    return null;
  }

  const { fractions, counts, rationale } = splitPlan;
  const warnings = datasetProfile?.warnings ?? [];
  const hasRationale = rationale.length > 0;
  const hasWarnings = warnings.length > 0;
  const hasInfo = hasRationale || hasWarnings;

  return (
    <div
      className="rounded-xl border border-[#C8B9A8]/50 bg-[#FAF8F5] shadow-[0_1px_2px_rgba(61,46,34,0.04)] overflow-hidden"
    >
      <div className="px-3.5 pt-3 pb-2.5">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-[#3D2E22]">
            <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-[#C8A882]/15 text-[#A8895E]">
              <Sparkles className="h-3 w-3" />
            </span>
            <span className="text-[13px] font-semibold tracking-tight">
              {msg("submit.split.recommended_title")}
            </span>
            {hasInfo && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    aria-label={msg("submit.split.rationale_aria")}
                    className="inline-flex h-5 w-5 items-center justify-center rounded-full text-[#8C7A6B] hover:bg-[#EFE7DC] hover:text-[#3D2E22] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/60 transition-colors cursor-default"
                  >
                    <Info className="h-3.5 w-3.5" />
                  </button>
                </TooltipTrigger>
                <TooltipContent
                  side="bottom"
                  sideOffset={8}
                  className="max-w-[min(320px,92vw)] rounded-xl border border-[#C8B9A8]/60 bg-[#FAF8F5] px-4 py-3 text-end text-[#3D2E22] shadow-[0_8px_24px_-8px_rgba(61,46,34,0.2)] [&>span]:hidden"
                >
                  {hasRationale && (
                    <>
                      <div className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-[#8C7A6B]">
                        <Sparkles className="h-3 w-3 text-[#C8A882]" />
                        {msg("submit.split.rationale_title")}
                      </div>
                      <ul className="space-y-1.5 text-[12px] leading-relaxed text-[#3D2E22]">
                        {rationale.map((line, idx) => (
                          <li key={idx} className="flex gap-2">
                            <span className="mt-[7px] inline-block h-1 w-1 shrink-0 rounded-full bg-[#C8A882]" />
                            <span>{line}</span>
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
                  {hasWarnings && (
                    <>
                      <div
                        className={cn(
                          "flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-[#8C7A6B]",
                          hasRationale ? "mt-3 mb-2 border-t border-[#DDD6CC]/60 pt-2.5" : "mb-2",
                        )}
                      >
                        <AlertTriangle className="h-3 w-3 text-[#C8924A]" />
                        {msg("submit.split.warnings_title")}
                      </div>
                      <ul className="space-y-1.5 text-[12px] leading-relaxed text-[#7A5A38]">
                        {warnings.map((warning) => (
                          <li key={warning.code} className="flex gap-2">
                            <span className="mt-[7px] inline-block h-1 w-1 shrink-0 rounded-full bg-[#C8924A]" />
                            <span>{warning.message}</span>
                          </li>
                        ))}
                      </ul>
                    </>
                  )}
                </TooltipContent>
              </Tooltip>
            )}
          </div>
          <ModeToggle value={splitMode} onChange={setSplitMode} />
        </div>
      </div>

      <div
        className={cn(
          "grid transition-[grid-template-rows,opacity] duration-200 ease-out",
          splitMode === "auto"
            ? "grid-rows-[1fr] opacity-100"
            : "grid-rows-[0fr] opacity-0",
        )}
      >
        <div className="overflow-hidden">
          <div className="px-3.5 pb-3 space-y-2.5">
            <div className="flex h-2.5 overflow-hidden rounded-full bg-[#EFE7DC]">
              <div
                className="bg-[#3D2E22] transition-[width] duration-300 ease-out"
                style={{ width: `${fractions.train * 100}%` }}
              />
              <div
                className="bg-[#C8A882] transition-[width] duration-300 ease-out"
                style={{ width: `${fractions.val * 100}%` }}
              />
              <div
                className="bg-[#8C7A6B] transition-[width] duration-300 ease-out"
                style={{ width: `${fractions.test * 100}%` }}
              />
            </div>
            <div className="grid grid-cols-3 gap-2">
              <PlanChip
                color="#3D2E22"
                label={msg("submit.split.label_train")}
                percent={percent(fractions.train)}
                count={counts.train}
              />
              <PlanChip
                color="#C8A882"
                label={msg("submit.split.label_val")}
                percent={percent(fractions.val)}
                count={counts.val}
              />
              <PlanChip
                color="#8C7A6B"
                label={msg("submit.split.label_test")}
                percent={percent(fractions.test)}
                count={counts.test}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ModeToggle({
  value,
  onChange,
}: {
  value: "auto" | "manual";
  onChange: (mode: "auto" | "manual") => void;
}) {
  return (
    <div className="relative inline-grid grid-cols-2 rounded-lg bg-[#EFE7DC]/70 p-0.5 gap-0.5">
      <div
        aria-hidden
        className="absolute top-0.5 bottom-0.5 w-[calc(50%-4px)] rounded-md bg-white shadow-[0_1px_2px_rgba(61,46,34,0.08)] transition-[inset-inline-start] duration-200 ease-out pointer-events-none"
        style={{ insetInlineStart: value === "auto" ? 2 : "calc(50% + 2px)" }}
      />
      {(
        [
          ["auto", msg("submit.split.mode_auto")],
          ["manual", msg("submit.split.mode_manual")],
        ] as const
      ).map(([mode, label]) => (
        <button
          key={mode}
          type="button"
          onClick={() => onChange(mode)}
          aria-pressed={value === mode}
          className={cn(
            "relative z-[1] rounded-md px-3 py-1 text-[11px] font-medium leading-none text-center transition-colors cursor-pointer",
            value === mode ? "text-[#3D2E22]" : "text-[#8C7A6B] hover:text-[#3D2E22]",
          )}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function PlanChip({
  color,
  label,
  percent,
  count,
}: {
  color: string;
  label: string;
  percent: string;
  count: number;
}) {
  return (
    <div className="rounded-lg bg-white/70 px-2.5 py-1.5">
      <div className="flex items-center gap-1.5">
        <span
          className="inline-block w-1.5 h-1.5 rounded-full"
          style={{ backgroundColor: color }}
        />
        <span className="text-[10.5px] font-medium uppercase tracking-wide text-[#8C7A6B]">
          {label}
        </span>
      </div>
      <div className="mt-1 flex items-baseline gap-1.5 text-[#3D2E22]">
        <span
          className="font-semibold tabular-nums tracking-tight text-[17px] leading-none"
          dir="ltr"
        >
          {percent}
        </span>
        <span className="text-[10.5px] tabular-nums text-[#8C7A6B]" dir="ltr">
          {count.toLocaleString("he-IL")}
        </span>
      </div>
    </div>
  );
}
