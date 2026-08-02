"use client";

import * as React from "react";
import { Gauge } from "@/shared/ui/icons";
import { Switch } from "@/shared/ui/primitives/switch";
import { NumberInput } from "@/shared/ui/number-input";
import { formatCredits, type TokenSourceMode } from "@/features/billing";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveIntlLocale } from "@/shared/lib/runtime-locale";

import { chargeableBracket } from "../lib/cost-bracket";
import type { SubmitWizardContext } from "../hooks/use-submit-wizard";

/**
 * Pre-run cost surface [FG-1]: a projected credit *bracket* (never a tight,
 * false-precision number) plus an optional user-set Max Cost Ceiling. Enabling
 * the cap threads `max_cost_credits` into the submit payload; the backend
 * hard-stops the run once spend would exceed it. The run button carries the cap
 * separately (see SubmitNav).
 *
 * Mode-aware: a managed run shows the full per-model credit cost; a BYOK run
 * shows only the platform fee (the provider tokens are paid on the user's own
 * key), so the surface stays honest in both modes. Warm, calm, factual.
 */
export function CostCeilingCard({ w, mode }: { w: SubmitWizardContext; mode: TokenSourceMode }) {
  const { costBracket, suggestedCeiling, maxCostCredits, setMaxCostCredits } = w;
  const locale = getActiveIntlLocale();
  const capped = maxCostCredits != null;
  const byok = mode === "byok";
  // In BYOK the user is charged only Skynet's platform fee, so the headline range
  // shows the fee, not the full per-model cost the provider key absorbs.
  const displayBracket = chargeableBracket(costBracket, mode);
  const bracketKey = byok ? "submit.cost_ceiling.bracket_byok" : "submit.cost_ceiling.bracket";

  const toggleCap = (on: boolean) => {
    setMaxCostCredits(on ? suggestedCeiling : null);
  };

  return (
    <div className="rounded-xl border border-[#C8B9A8]/50 bg-[#FAF8F5] shadow-[0_1px_2px_rgba(61,46,34,0.04)] overflow-hidden">
      <div className="px-3.5 pt-3 pb-2.5">
        <div className="flex items-center justify-between gap-3">
          {/* Title + estimate form one column; the enable switch is the row's other
              child, so `items-center` drops it onto the block's vertical midline. */}
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-[#3D2E22]">
              <span className="inline-flex h-5 w-5 items-center justify-center rounded-full bg-[#C8A882]/15 text-[#A8895E]">
                <Gauge className="h-3 w-3" />
              </span>
              <span className="text-[13px] font-semibold tracking-tight">
                {msg("submit.cost_ceiling.label")}
              </span>
            </div>

            <p className="mt-2 text-[12px] leading-relaxed text-[#3D2E22]" dir="auto">
              {/* Wrap the range in an LTR isolate (U+2066…U+2069): in an RTL line the
              en-dash between two Latin number groups is a neutral that resolves to
              the paragraph's RTL direction and visually swaps the numbers to
              "high–low". Isolating "low–high" as one LTR run keeps the order. */}
              {formatMsg(bracketKey, {
                low: `\u2066${formatCredits(displayBracket.lowCredits, locale)}`,
                high: `${formatCredits(displayBracket.highCredits, locale)}\u2069`,
              })}
            </p>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <span className="text-[11px] text-[#8C7A6B]">{msg("submit.cost_ceiling.enable")}</span>
            <Switch checked={capped} onCheckedChange={toggleCap} />
          </div>
        </div>
      </div>

      <div className={cnGrid(capped)}>
        <div className="overflow-hidden">
          <div className="border-t border-[#DDD6CC]/60 px-3.5 py-3 space-y-2">
            <div className="flex items-center justify-between gap-3">
              <label
                htmlFor="costCeilingInput"
                className="text-[11px] font-medium uppercase tracking-wide text-[#8C7A6B]"
              >
                {msg("submit.cost_ceiling.cap_label")}
              </label>
              <div className="flex items-center gap-2">
                <div className="w-28">
                  <NumberInput
                    id="costCeilingInput"
                    min={1}
                    step={10}
                    value={maxCostCredits ?? ""}
                    onChange={(v) => setMaxCostCredits(Math.max(1, v))}
                  />
                </div>
                <span className="text-[11px] text-[#8C7A6B]">
                  {msg("submit.cost_ceiling.cap_unit")}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Collapse/expand the ceiling input with the same grid-rows transition the split card uses. */
function cnGrid(open: boolean): string {
  return [
    "grid transition-[grid-template-rows,opacity] duration-200 ease-out",
    open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
  ].join(" ");
}
