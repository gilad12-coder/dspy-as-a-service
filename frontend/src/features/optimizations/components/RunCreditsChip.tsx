"use client";

import { Coins, Undo2 } from "lucide-react";
import { useLocale } from "@/shared/providers";
import { useSettingsModal } from "@/features/settings";
import { msg } from "@/shared/lib/messages";
import { formatCredits } from "@/features/billing";
import { readBilling } from "../lib/proof-banner";

/**
 * Compact per-run credits affordance for the detail header.
 *
 * Reads the worker's billing stamp and shows what the run cost: the credits it
 * consumed, or "Free" when the guarantee refunded. It's a quiet button — clicking
 * opens the wallet, the same surface that holds the full usage history. Renders
 * nothing until the run settles and a billing outcome is stamped (so it stays
 * hidden on active runs and pairs, which carry no billing of their own).
 */
export function RunCreditsChip({ details }: { details?: Record<string, unknown> }) {
  const { locale } = useLocale();
  const { openTo } = useSettingsModal();
  const billing = readBilling(details);
  if (billing == null) return null;

  return (
    <button
      type="button"
      onClick={() => openTo("billing")}
      title={msg("billing.action.view_wallet")}
      aria-label={msg("billing.action.view_wallet")}
      dir="ltr"
      className="flex items-center gap-1.5 tabular-nums transition-colors hover:text-foreground"
    >
      {billing.outcome === "refunded" ? (
        <>
          <Undo2 className="size-3.5 rtl:-scale-x-100" aria-hidden="true" />
          {msg("billing.plans.price_free")}
        </>
      ) : (
        <>
          <Coins className="size-3.5" aria-hidden="true" />
          {formatCredits(billing.credits, locale)}
        </>
      )}
    </button>
  );
}
