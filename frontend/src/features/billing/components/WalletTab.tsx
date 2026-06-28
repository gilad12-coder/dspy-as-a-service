"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowDownLeft, Crown, Gift, Plus, RefreshCw, Sparkles } from "lucide-react";
import { toast } from "react-toastify";
import { msg, formatMsg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { SettingsRow } from "@/shared/ui/settings-row";
import { Switch } from "@/shared/ui/primitives/switch";
import { createSubscriptionCheckout, openBillingPortal } from "@/shared/lib/api";
import { useCredits } from "../providers/credit-provider";
import {
  formatCredits,
  formatResetDate,
  type LedgerKind,
  type UsageEntry,
} from "../lib/credit";

const KIND_ICON: Record<LedgerKind, React.ComponentType<{ className?: string }>> = {
  run: Sparkles,
  topup: Plus,
  grant: Gift,
};

/** One ledger row. Amounts and model ids are numerals/identifiers — always LTR-islanded. */
function LedgerRow({ entry }: { entry: UsageEntry }) {
  const { locale } = useLocale();
  const Icon = KIND_ICON[entry.kind];
  const isByokRun = entry.kind === "run" && entry.mode === "byok";
  const credited = entry.credits > 0;

  return (
    <li className="flex items-center gap-3 py-2.5 border-b border-border/30 last:border-b-0">
      <span className="grid size-7 shrink-0 place-items-center rounded-full bg-muted text-muted-foreground">
        {credited ? (
          <Icon className="size-3.5" aria-hidden="true" />
        ) : (
          <ArrowDownLeft className="size-3.5 rtl:-scale-x-100" aria-hidden="true" />
        )}
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <span dir="auto" className="truncate text-sm text-foreground">
          {entry.label}
        </span>
        {entry.model && (
          <span dir="ltr" className="truncate text-[0.6875rem] text-muted-foreground">
            {entry.model}
          </span>
        )}
      </span>
      <span className="flex shrink-0 flex-col items-end">
        {isByokRun ? (
          <span className="text-xs font-medium text-muted-foreground">
            {msg("billing.history.byok_tag")}
          </span>
        ) : (
          <span
            dir="ltr"
            className={cn(
              "text-sm font-medium tabular-nums",
              credited ? "text-foreground" : "text-muted-foreground",
            )}
          >
            {credited ? "+" : "−"}
            {formatCredits(Math.abs(entry.credits), locale)}
          </span>
        )}
        <span dir="ltr" className="text-[0.6875rem] text-muted-foreground/70">
          {formatResetDate(entry.at, locale)}
        </span>
      </span>
    </li>
  );
}

/**
 * Wallet — the `billing` settings tab.
 *
 * A calm, left-aligned balance block (not a centered hero metric), the opt-in
 * auto-reload control, and the usage ledger. Balances read in credits only — no
 * dollar equivalent is shown.
 */
export function WalletTab() {
  const { wallet, totalCredits, setAutoReload, status, syncing } = useCredits();
  const { locale } = useLocale();

  // Premium subscribe / manage both redirect to Stripe-hosted pages; the backend
  // creates the session and returns its URL. A failure (e.g. billing unconfigured)
  // surfaces as one toast rather than a dead button.
  const goToStripe = React.useCallback(async (start: () => Promise<{ url: string }>) => {
    try {
      const { url } = await start();
      window.location.assign(url);
    } catch {
      toast.error(msg("billing.checkout.error"));
    }
  }, []);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {msg("billing.popover.title")}
          </span>
          <div className="flex items-baseline gap-2" aria-busy={syncing || undefined}>
            {/* Post-checkout sync [FG-3]: shimmer over the prior balance until the
                webhook lands, mirroring the header chip so the two never disagree. */}
            <span
              dir="ltr"
              className={cn(
                "text-3xl font-semibold text-foreground tabular-nums",
                syncing && "animate-pulse text-muted-foreground",
              )}
            >
              {formatCredits(totalCredits, locale)}
            </span>
          </div>
          <span className="text-xs text-muted-foreground">
            {formatMsg("billing.wallet.breakdown", {
              p1: formatCredits(wallet.paidBalanceCredits, locale),
              p2: formatCredits(wallet.freeGrant.creditsRemaining, locale),
              p3: formatCredits(wallet.freeGrant.creditsTotal, locale),
            })}
          </span>
          {/* Low balance stays an operational metric here too — same calm line as
              the header chip, no red, no urgency. */}
          {status === "low" && (
            <span className="text-xs text-muted-foreground/80">{msg("billing.chip.low_note")}</span>
          )}
        </div>
        <Link
          href="/upgrade"
          className="inline-flex items-center gap-1.5 rounded-lg bg-[#3D2E22] px-3.5 py-2 text-sm font-semibold text-[#FAF8F5] transition-colors duration-200 hover:bg-[#2A1F17] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
        >
          <Plus className="size-4" aria-hidden="true" />
          {msg("billing.action.add_credits")}
        </Link>
      </div>

      <div>
        <SettingsRow
          icon={RefreshCw}
          label={msg("billing.wallet.autoreload_label")}
          description={formatMsg("billing.wallet.autoreload_desc", {
            p1: formatCredits(wallet.autoReload.topUpCredits, locale),
            p2: formatCredits(wallet.autoReload.thresholdCredits, locale),
          })}
        >
          <Switch
            checked={wallet.autoReload.enabled}
            onCheckedChange={(enabled) => setAutoReload({ enabled })}
            aria-label={msg("billing.wallet.autoreload_label")}
          />
        </SettingsRow>

        <SettingsRow
          icon={Crown}
          label={msg("billing.premium.title")}
          description={
            wallet.premiumActive ? msg("billing.premium.active") : msg("billing.premium.desc")
          }
        >
          {wallet.premiumActive ? (
            <button
              type="button"
              onClick={() => goToStripe(openBillingPortal)}
              className="inline-flex items-center text-sm font-semibold text-[#3D2E22] underline-offset-4 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 rounded"
            >
              {msg("billing.premium.manage")}
            </button>
          ) : (
            <button
              type="button"
              onClick={() => goToStripe(createSubscriptionCheckout)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#C8A882]/60 px-3 py-1.5 text-xs font-semibold text-[#8a6d44] transition-colors duration-200 hover:bg-[#C8A882]/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              {msg("billing.premium.subscribe")}
            </button>
          )}
        </SettingsRow>
      </div>

      <div className="flex flex-col gap-1">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {msg("billing.wallet.history_title")}
        </span>
        {wallet.usage.length === 0 ? (
          <div className="flex flex-col items-center gap-1 rounded-lg border border-dashed border-border/60 px-4 py-8 text-center">
            <span className="text-sm font-medium text-foreground">
              {msg("billing.wallet.history_empty_title")}
            </span>
            <span className="max-w-[42ch] text-xs text-muted-foreground">
              {msg("billing.wallet.history_empty_desc")}
            </span>
          </div>
        ) : (
          <ul className="flex flex-col">
            {wallet.usage.map((entry) => (
              <LedgerRow key={entry.id} entry={entry} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
