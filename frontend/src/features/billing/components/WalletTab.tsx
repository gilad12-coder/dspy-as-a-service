"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { ArrowsClockwise, CircleNotch, Coins, Sparkle } from "@/shared/ui/icons";
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
          icon={ArrowsClockwise}
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

        <SettingsRow icon={Sparkle} label={msg("billing.action.add_credits")}>
          <AddCreditsControls />
        </SettingsRow>
      </div>
    </div>
  );
}
