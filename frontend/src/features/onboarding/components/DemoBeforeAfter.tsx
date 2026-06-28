"use client";

import { ArrowRight } from "lucide-react";
import { formatImprovement } from "@/shared/lib";
import { msg } from "@/shared/lib/messages";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { ONBOARDING_DEMO } from "../constants";

/** One labelled score in the before/after pair. */
function DemoScore({ label, value, emphasis }: { label: string; value: number; emphasis?: boolean }) {
  return (
    <div className="flex flex-1 flex-col gap-1">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      <span
        dir="ltr"
        className={
          emphasis
            ? "text-3xl font-semibold tabular-nums text-[#3D2E22]"
            : "text-3xl font-semibold tabular-nums text-foreground/70"
        }
      >
        {value.toFixed(1)}
      </span>
    </div>
  );
}

/**
 * The pre-baked demo before/after — the first onboarding moment.
 *
 * Renders a static sample run (baseline vs optimized on a held-out test split)
 * so a brand-new user feels the result's shape in seconds, before uploading
 * anything. The arrow flips with the active text direction so the "before →
 * after" reading order holds in RTL.
 */
export function DemoBeforeAfter() {
  const rtl = getActiveDir() === "rtl";
  const lift = ONBOARDING_DEMO.optimized - ONBOARDING_DEMO.baseline;

  return (
    <div className="flex flex-col gap-4 rounded-xl border border-border/60 bg-card p-5">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {msg("onboarding.demo.task_label")}
        </span>
        <span className="text-xs font-semibold text-foreground" dir="auto">
          {msg("onboarding.demo.task_value")}
        </span>
      </div>

      <div className="flex items-center gap-4">
        <DemoScore label={msg("onboarding.demo.before_label")} value={ONBOARDING_DEMO.baseline} />
        <ArrowRight
          className={`size-5 shrink-0 text-muted-foreground/60 ${rtl ? "rotate-180" : ""}`}
          aria-hidden="true"
        />
        <DemoScore
          label={msg("onboarding.demo.after_label")}
          value={ONBOARDING_DEMO.optimized}
          emphasis
        />
        <span
          dir="ltr"
          className="rounded-full bg-[#C8A882]/15 px-2.5 py-1 text-sm font-semibold tabular-nums text-[#8a6d44]"
        >
          {formatImprovement(lift)}
        </span>
      </div>

      <p className="text-xs text-muted-foreground">{msg("onboarding.demo.caption")}</p>
    </div>
  );
}
