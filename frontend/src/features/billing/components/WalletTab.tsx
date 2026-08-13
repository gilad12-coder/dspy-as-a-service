"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { CircleNotch, Coins, Sparkle } from "@/shared/ui/icons";
import { toast } from "react-toastify";
import { formatMsg, msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { SettingsRow } from "@/shared/ui/settings-row";
import { Button } from "@/shared/ui/primitives/button";
import { createCheckoutSession } from "@/shared/lib/api";
import { useCredits } from "../providers/credit-provider";
import {
  CREDIT_PACKS,
  CUSTOM_CREDITS_MAX,
  CUSTOM_CREDITS_MIN,
  creditsToUsd,
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
 * Inline credit purchase — one pill control holding the pack segments, a
 * free-type custom amount, and the buy action, all in the segmented-control
 * style. Lives directly in the wallet tab now that the standalone /upgrade
 * page is gone: prepaid credits are the only plan, so buying is a settings
 * row, not a pricing page.
 */
function AddCreditsControls() {
  const { locale } = useLocale();
  const [selection, setSelection] = React.useState<string>(
    () => (CREDIT_PACKS.find((p) => p.popular) ?? CREDIT_PACKS[0]!).id,
  );
  const [customDraft, setCustomDraft] = React.useState("");
  const [buying, setBuying] = React.useState(false);

  const pack: CreditPack | undefined = CREDIT_PACKS.find((p) => p.id === selection);
  const customCredits = Number(customDraft || "0");
  const customValid = customCredits >= CUSTOM_CREDITS_MIN && customCredits <= CUSTOM_CREDITS_MAX;
  const usd = pack ? pack.usd : creditsToUsd(customCredits);
  const priceLabel = Number.isInteger(usd) ? formatUsdWhole(usd, locale) : formatUsd(usd, locale);

  const onBuy = async () => {
    setBuying(true);
    try {
      const { url } = await createCheckoutSession(
        pack ? { packId: pack.id } : { credits: customCredits },
      );
      window.location.assign(url);
    } catch {
      setBuying(false);
      toast.error(msg("billing.checkout.error"));
    }
  };

  const customActive = pack === undefined;
  return (
    <div
      role="group"
      aria-label={msg("billing.plans.credits.pack_aria")}
      className="relative flex shrink-0 items-center gap-0.5 rounded-full border border-border/50 bg-muted/40 p-0.5"
    >
      {CREDIT_PACKS.map((p) => {
        const active = p.id === selection;
        return (
          <button
            key={p.id}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => setSelection(p.id)}
            className={cn(
              "relative rounded-full px-2 py-0.5 text-[0.6875rem] font-semibold tabular-nums transition-colors duration-150 cursor-pointer",
              active ? "text-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {/* Shared-layout pill slides between segments instead of the selected
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
      {/* Custom amount is a fourth, free-type segment: focusing or typing makes
          it the active selection and the pill slides behind it. */}
      <span className="relative">
        {customActive && (
          <motion.span
            layoutId="credit-pack-pill"
            className="absolute inset-0 rounded-full bg-background shadow-sm"
            transition={PILL_TRANSITION}
            aria-hidden="true"
          />
        )}
        <input
          value={customDraft}
          onChange={(event) => {
            setCustomDraft(event.target.value.replace(/\D/g, ""));
            setSelection("custom");
          }}
          onFocus={() => setSelection("custom")}
          inputMode="numeric"
          maxLength={6}
          dir="ltr"
          placeholder={msg("billing.plans.credits.custom")}
          aria-label={msg("billing.plans.credits.custom_amount_aria")}
          className={cn(
            "relative z-10 w-16 rounded-full bg-transparent px-2 py-0.5 text-center text-[0.6875rem] font-semibold tabular-nums outline-none transition-colors duration-150 placeholder:font-normal placeholder:text-muted-foreground/70",
            customActive ? "text-foreground" : "text-muted-foreground",
          )}
        />
      </span>
      <span aria-hidden="true" className="mx-0.5 h-3.5 w-px shrink-0 bg-border/70" />
      <Button
        variant="outline"
        size="sm"
        onClick={onBuy}
        disabled={buying || (!pack && !customValid)}
        className="h-6 rounded-full px-2.5 text-[0.6875rem] font-semibold border-[#C8A882]/70 text-[#8a6d44] hover:bg-[#C8A882]/10 hover:text-[#8a6d44] [&_svg:not([class*='size-'])]:size-3"
      >
        {buying ? <CircleNotch className="animate-spin" /> : <Sparkle aria-hidden="true" />}
        {formatMsg("billing.upgrade.buy", { p1: priceLabel })}
      </Button>
    </div>
  );
}

/**
 * Wallet — the `billing` settings tab.
 *
 * A calm, left-aligned balance block (not a centered hero metric) and the
 * add-credits row. Spend history lives in its own
 * Usage tab. Balances read in credits only — no dollar equivalent is shown.
 */
export function WalletTab() {
  const { totalCredits, status, syncing, loading, available, loadError, refresh } = useCredits();
  const { locale } = useLocale();

  return (
    <div className="flex flex-col gap-5">
      {loadError && (
        <div
          role="alert"
          className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2"
        >
          <span className="text-xs text-destructive">{msg("billing.wallet.load_error")}</span>
          <Button variant="outline" size="sm" disabled={loading} onClick={refresh}>
            {loading && <CircleNotch className="size-3.5 animate-spin" aria-hidden="true" />}
            {msg("billing.wallet.retry")}
          </Button>
        </div>
      )}
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
              {available ? formatCredits(totalCredits, locale) : "—"}
            </span>
          </div>
          {/* Low balance stays an operational metric here too — same calm line as
              the header chip, no red, no urgency. */}
          {available && status === "low" && (
            <span className="text-xs text-muted-foreground/80">{msg("billing.chip.low_note")}</span>
          )}
        </div>
      </div>

      <div>
        <SettingsRow icon={Sparkle} label={msg("billing.action.add_credits")}>
          <AddCreditsControls />
        </SettingsRow>
      </div>
    </div>
  );
}
