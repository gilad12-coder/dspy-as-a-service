"use client";

import { Sparkle, Info } from "@/shared/ui/icons";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/shared/ui/primitives/tooltip";
import { cn } from "@/shared/lib/utils";
import { msg, type MessageKey } from "@/shared/lib/messages";
import { getActiveIntlLocale } from "@/shared/lib/runtime-locale";
import { useLocale } from "@/shared/providers";
import { dirForLocale } from "@/shared/lib/locale";

import type { SubmitWizardContext } from "../hooks/use-submit-wizard";

const percent = (value: number): string => `${Math.round(value * 100)}%`;

// Tier thresholds mirror backend planner.py (recommend_split / _recommend_fractions):
// the split fractions are chosen there purely by total row count, so we recover the
// matching rationale tier from the same total and localize it through the UI catalog.
// The backend only ships Hebrew copy for these, so rendering them client-side is what
// gives every locale its own translation (and correct direction).
const TIER_TINY = 30;
const TIER_SMALL = 80;
const TIER_MEDIUM = 300;

function rationaleKey(total: number): MessageKey {
  if (total < TIER_TINY) return "submit.split.rationale.tiny";
  if (total < TIER_SMALL) return "submit.split.rationale.small";
  if (total < TIER_MEDIUM) return "submit.split.rationale.medium";
  return "submit.split.rationale.large";
}

export function SplitRecommendationCard({ w }: { w: SubmitWizardContext }) {
  const { splitPlan, splitMode, setSplitMode, profileLoading } = w;
  // The rationale/warning copy is portaled into a Radix tooltip, where the `rtl:`
  // variant doesn't fire — drive direction off the locale explicitly instead.
  const { locale } = useLocale();
  const dir = dirForLocale(locale);

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

  const { fractions, counts } = splitPlan;
  const total = counts.train + counts.val + counts.test;
  const rationaleText = msg(rationaleKey(total), {
    total,
    val_count: counts.val,
    test_count: counts.test,
  });
  const hasRationale = total > 0;

  return (
    <div
      className="rounded-xl border border-[#C8B9A8]/50 bg-[#FAF8F5] shadow-[0_1px_2px_rgba(61,46,34,0.04)] overflow-hidden"
    >
      <div className="px-3.5 pt-3 pb-2.5">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-[#3D2E22]">
            <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-[#C8A882]/15 text-[#A8895E]">
              <Sparkle className="h-3 w-3" />
            </span>
            <span className="text-[13px] font-semibold tracking-tight">
              {msg("submit.split.recommended_title")}
            </span>
            {hasRationale && (
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
                  dir={dir}
                  className="max-w-[min(320px,92vw)] rounded-xl border border-[#C8B9A8]/60 bg-[#FAF8F5] px-4 py-3 text-start text-[#3D2E22] shadow-[0_8px_24px_-8px_rgba(61,46,34,0.2)] [&>span]:hidden"
                >
                  <div className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide text-[#8C7A6B]">
                    <Sparkle className="h-3 w-3 text-[#C8A882]" />
                    {msg("submit.split.rationale_title")}
                  </div>
                  <ul className="space-y-1.5 text-[12px] leading-relaxed text-[#3D2E22]">
                    <li className="flex gap-2">
                      <span className="mt-[7px] inline-block h-1 w-1 shrink-0 rounded-full bg-[#C8A882]" />
                      <span>{rationaleText}</span>
                    </li>
                  </ul>
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
          {count.toLocaleString(getActiveIntlLocale())}
        </span>
      </div>
    </div>
  );
}
