"use client";

import * as React from "react";
import Link from "next/link";
import { ArrowDownLeft, Coins, Crown, Gift, Plus, RefreshCw, Sparkles } from "lucide-react";
import { toast } from "react-toastify";
import { formatMsg, msg, type MessageKey } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { useSettingsModal } from "@/features/settings";
import { SettingsRow } from "@/shared/ui/settings-row";
import { Switch } from "@/shared/ui/primitives/switch";
import { Input } from "@/shared/ui/primitives/input";
import { openBillingPortal } from "@/shared/lib/api";
import { useCredits } from "../providers/credit-provider";
import {
  CREDIT_USD_VALUE,
  formatCredits,
  formatResetDate,
  formatUsd,
  type LedgerKind,
  type UsageEntry,
} from "../lib/credit";

const KIND_ICON: Record<LedgerKind, React.ComponentType<{ className?: string }>> = {
  run: Sparkles,
  topup: Plus,
  grant: Gift,
};

/** Which slice of the ledger the usage list shows. */
type LedgerFilter = "all" | "refunds" | "costs";

/** Segmented filter over the usage ledger, split by the sign of the credit delta. */
const LEDGER_FILTERS: ReadonlyArray<{ value: LedgerFilter; labelKey: MessageKey }> = [
  { value: "all", labelKey: "billing.wallet.filter_all" },
  { value: "refunds", labelKey: "billing.wallet.filter_refunds" },
  { value: "costs", labelKey: "billing.wallet.filter_costs" },
];

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

  // Usage ledger filter: refunds are credited rows (positive delta), costs are
  // debited rows (negative delta). BYOK runs carry a zero delta, so they're
  // neither — they surface only under "All".
  const [usageFilter, setUsageFilter] = React.useState<LedgerFilter>("all");

  const visibleUsage = React.useMemo(() => {
    if (usageFilter === "refunds") return wallet.usage.filter((entry) => entry.credits > 0);
    if (usageFilter === "costs") return wallet.usage.filter((entry) => entry.credits < 0);
    return wallet.usage;
  }, [usageFilter, wallet.usage]);

  // Picking a filter locks the list to that slice and confirms with a localized
  // toast — once the rows re-render there's no other label for the active slice.
  const applyUsageFilter = React.useCallback(
    (next: LedgerFilter, labelKey: MessageKey) => {
      if (next === usageFilter) return;
      setUsageFilter(next);
      toast.success(formatMsg("billing.wallet.filter_applied", { p1: msg(labelKey) }));
    },
    [usageFilter],
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
            <Link
              href="/upgrade"
              onClick={() => setOpen(false)}
              className="inline-flex items-center gap-1.5 rounded-lg border border-[#C8A882]/60 px-3 py-1.5 text-xs font-semibold text-[#8a6d44] transition-colors duration-200 hover:bg-[#C8A882]/10 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              {msg("billing.premium.subscribe")}
            </Link>
          )}
        </SettingsRow>
      </div>

      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-3">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {msg("billing.wallet.history_title")}
          </span>
          {wallet.usage.length > 0 && (
            <div
              role="group"
              aria-label={msg("billing.wallet.history_title")}
              className="flex items-center gap-0.5 rounded-lg bg-muted/60 p-0.5"
            >
              {LEDGER_FILTERS.map(({ value, labelKey }) => {
                const active = usageFilter === value;
                return (
                  <button
                    key={value}
                    type="button"
                    aria-pressed={active}
                    onClick={() => applyUsageFilter(value, labelKey)}
                    className={cn(
                      "rounded-md px-2.5 py-1 text-xs font-medium transition-colors duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45",
                      active
                        ? "bg-background text-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    {msg(labelKey)}
                  </button>
                );
              })}
            </div>
          )}
        </div>
        {wallet.usage.length === 0 ? (
          <div className="flex flex-col items-center gap-1 rounded-lg border border-dashed border-border/60 px-4 py-8 text-center">
            <span className="text-sm font-medium text-foreground">
              {msg("billing.wallet.history_empty_title")}
            </span>
            <span className="max-w-[42ch] text-xs text-muted-foreground">
              {msg("billing.wallet.history_empty_desc")}
            </span>
          </div>
        ) : visibleUsage.length === 0 ? (
          <p className="px-1 py-6 text-center text-xs text-muted-foreground">
            {msg("billing.wallet.filter_empty")}
          </p>
        ) : (
          <ul className="flex max-h-64 flex-col overflow-y-auto pe-1">
            {visibleUsage.map((entry) => (
              <LedgerRow key={entry.id} entry={entry} />
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
