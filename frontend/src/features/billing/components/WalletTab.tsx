"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { Coins, Loader2, RefreshCw, Sparkles } from "lucide-react";
import { toast } from "react-toastify";
import { formatMsg, msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { SettingsRow } from "@/shared/ui/settings-row";
import { Switch } from "@/shared/ui/primitives/switch";
import { Input } from "@/shared/ui/primitives/input";
import { Button } from "@/shared/ui/primitives/button";
import { createCheckoutSession } from "@/shared/lib/api";
import { useCredits } from "../providers/credit-provider";
import {
  CREDIT_PACKS,
  CREDIT_USD_VALUE,
  formatCredits,
  formatUsd,
  type CreditPack,
} from "../lib/credit";

// Slide transition for the credit-pack selector's shared-layout pill — matches the
// runs-source segmented control in explore/SearchBar so the two read identically.
const PILL_TRANSITION = { type: "tween", duration: 0.18, ease: [0.22, 1, 0.36, 1] } as const;

/** Whole-dollar USD (no cents) for the buy button — "$20", not "$20.00". */
function formatUsdWhole(usd: number, locale: string): string {
  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(usd);
}

/**
 * Inline credit-pack purchase — pick a pack, buy through Stripe. Lives directly
 * in the wallet tab now that the standalone /upgrade page is gone: prepaid
 * packs are the only plan, so buying is a settings row, not a pricing page.
 */
function AddCreditsControls() {
  const { locale } = useLocale();
  const [pack, setPack] = React.useState<CreditPack>(
    () => CREDIT_PACKS.find((p) => p.popular) ?? CREDIT_PACKS[0]!,
  );
  const [buying, setBuying] = React.useState(false);

  const onBuy = async () => {
    setBuying(true);
    try {
      const { url } = await createCheckoutSession(pack.id);
      window.location.assign(url);
    } catch {
      setBuying(false);
      toast.error(msg("billing.checkout.error"));
    }
  };

  return (
    <>
      <div
        role="group"
        aria-label={msg("billing.plans.credits.pack_aria")}
        className="relative flex shrink-0 gap-0.5 rounded-full border border-border/50 bg-muted/40 p-0.5"
      >
        {CREDIT_PACKS.map((p) => {
          const active = p.id === pack.id;
          return (
            <button
              key={p.id}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => setPack(p)}
              className={cn(
                "relative rounded-full px-2 py-0.5 text-[0.6875rem] font-semibold tabular-nums transition-colors duration-150 cursor-pointer",
                active ? "text-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {/* Shared-layout pill slides between packs instead of the selected
                  background snapping — mirrors the runs-source segmented control. */}
              {active && (
                <motion.span
                  layoutId="credit-pack-pill"
                  className="absolute inset-0 rounded-full bg-background shadow-sm"
                  transition={PILL_TRANSITION}
                  aria-hidden="true"
                />
              )}
              <span dir="ltr" className="relative z-10">
                {formatCredits(p.credits, locale)}
              </span>
            </button>
          );
        })}
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={onBuy}
        disabled={buying}
        className="border-[#C8A882]/70 text-[#8a6d44] hover:bg-[#C8A882]/10 hover:text-[#8a6d44]"
      >
        {buying ? <Loader2 className="animate-spin" /> : <Sparkles aria-hidden="true" />}
        {formatMsg("billing.upgrade.buy", { p1: formatUsdWhole(pack.usd, locale) })}
      </Button>
    </>
  );
}

/**
 * Wallet — the `billing` settings tab.
 *
 * A calm, left-aligned balance block (not a centered hero metric), the opt-in
 * auto-reload control, and the add-credits row. Spend history lives in its own
 * Usage tab. Balances read in credits only — no dollar equivalent is shown.
 */
export function WalletTab() {
  const { wallet, totalCredits, setAutoReload, status, syncing } = useCredits();
  const { locale } = useLocale();

  // Auto-reload amount is a free-type field — any credit amount works. `customDraft`
  // holds the raw input string so the field can be cleared mid-edit without snapping
  // to 0; a committed value is mirrored into the wallet's `topUpCredits`.
  const topUpCredits = wallet.autoReload.topUpCredits;
  const [customDraft, setCustomDraft] = React.useState(String(topUpCredits));

  const onCustomDraftChange = React.useCallback(
    (raw: string) => {
      const digits = raw.replace(/\D/g, "");
      setCustomDraft(digits);
      if (digits !== "") setAutoReload({ topUpCredits: Number(digits) });
    },
    [setAutoReload],
  );

  // Restore the last committed amount if the field is left empty, so auto-reload
  // never sits on a zero top-up.
  const onCustomBlur = React.useCallback(() => {
    if (customDraft === "") setCustomDraft(String(topUpCredits));
  }, [customDraft, topUpCredits]);

  // The amount commits silently on every keystroke, so Enter gets an explicit
  // toast — the only confirmation the value actually landed. Blur (fired by the
  // trailing `.blur()`) restores an empty field, so toast the effective amount.
  const onCustomKeyDown = React.useCallback(
    (event: React.KeyboardEvent<HTMLInputElement>) => {
      if (event.key !== "Enter") return;
      event.preventDefault();
      const amount = customDraft === "" ? topUpCredits : Number(customDraft);
      toast.success(
        formatMsg("billing.wallet.autoreload_saved", {
          p1: formatCredits(amount, locale),
        }),
      );
      event.currentTarget.blur();
    },
    [customDraft, topUpCredits, locale],
  );

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex flex-col gap-1">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {msg("billing.popover.title")}
          </span>
          <div className="flex items-center gap-2" aria-busy={syncing || undefined}>
            {/* Same gold coin as the header chip so the balance reads as credits at a glance. */}
            <Coins
              className={cn(
                "size-6 shrink-0",
                syncing ? "text-muted-foreground" : "text-[#C8A882]",
              )}
              aria-hidden="true"
            />
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
          {/* Low balance stays an operational metric here too — same calm line as
              the header chip, no red, no urgency. */}
          {status === "low" && (
            <span className="text-xs text-muted-foreground/80">{msg("billing.chip.low_note")}</span>
          )}
        </div>
      </div>

      <div>
        <SettingsRow
          icon={RefreshCw}
          label={msg("billing.wallet.autoreload_label")}
          description={formatMsg("billing.wallet.autoreload_rate", {
            p1: formatUsd(CREDIT_USD_VALUE, locale),
          })}
        >
          {wallet.autoReload.enabled && (
            <Input
              value={customDraft}
              onChange={(event) => onCustomDraftChange(event.target.value)}
              onBlur={onCustomBlur}
              onKeyDown={onCustomKeyDown}
              inputMode="numeric"
              dir="ltr"
              aria-label={msg("billing.wallet.autoreload_amount_label")}
              className="h-8 w-24 text-center tabular-nums"
            />
          )}
          <Switch
            checked={wallet.autoReload.enabled}
            onCheckedChange={(enabled) => setAutoReload({ enabled })}
            aria-label={msg("billing.wallet.autoreload_label")}
          />
        </SettingsRow>

        <SettingsRow
          icon={Sparkles}
          label={msg("billing.action.add_credits")}
          description={msg("billing.plans.credits.summary")}
        >
          <AddCreditsControls />
        </SettingsRow>
      </div>
    </div>
  );
}
