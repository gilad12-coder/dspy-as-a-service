"use client";

import * as React from "react";
import Link from "next/link";
import { Coins, RefreshCw, Sparkles } from "lucide-react";
import { toast } from "react-toastify";
import { formatMsg, msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { useSettingsModal } from "@/features/settings";
import { SettingsRow } from "@/shared/ui/settings-row";
import { Switch } from "@/shared/ui/primitives/switch";
import { Input } from "@/shared/ui/primitives/input";
import { Button } from "@/shared/ui/primitives/button";
import { useCredits } from "../providers/credit-provider";
import { CREDIT_USD_VALUE, formatCredits, formatUsd } from "../lib/credit";

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
  const { setOpen } = useSettingsModal();

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
          <Button
            asChild
            variant="outline"
            size="sm"
            className="border-[#C8A882]/70 text-[#8a6d44] hover:bg-[#C8A882]/10 hover:text-[#8a6d44]"
          >
            <Link href="/upgrade" onClick={() => setOpen(false)}>
              <Sparkles aria-hidden="true" />
              {msg("billing.action.add_credits")}
            </Link>
          </Button>
        </SettingsRow>
      </div>
    </div>
  );
}
