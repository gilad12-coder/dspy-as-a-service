"use client";

import * as React from "react";
import { motion } from "framer-motion";
import {
  ArrowSquareOut,
  CircleNotch,
  Coins,
  CreditCard,
  Plus,
  Sparkle,
} from "@/shared/ui/icons";
import { toast } from "react-toastify";
import { formatMsg, msg, type MessageKey } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { SettingsRow } from "@/shared/ui/settings-row";
import { Button } from "@/shared/ui/primitives/button";
import { RetryIconButton } from "@/shared/ui/retry-icon-button";
import {
  createBillingPortalSession,
  createCheckoutSession,
  getBillingProfile,
  getBillingTransactions,
  type BillingAddressResponse,
  type BillingProfileResponse,
  type BillingTransaction,
  type BillingTransactionsResponse,
} from "@/shared/lib/api";
import { useCredits } from "../providers/credit-provider";
import {
  CREDIT_PACKS,
  CUSTOM_CREDITS_MAX,
  CUSTOM_CREDITS_MIN,
  creditsToUsd,
  formatCredits,
  formatResetDate,
  formatUsd,
  type CreditPack,
} from "../lib/credit";

// Slide transition for the credit-pack selector's shared-layout pill — matches the
// runs-source segmented control in explore/SearchBar so the two read identically.
const PILL_TRANSITION = { type: "tween", duration: 0.18, ease: [0.22, 1, 0.36, 1] } as const;

const TRANSACTION_STATUS_LABEL: Record<BillingTransaction["status"], MessageKey> = {
  paid: "billing.transactions.status.paid",
  processing: "billing.transactions.status.processing",
  refunded: "billing.transactions.status.refunded",
  partially_refunded: "billing.transactions.status.partially_refunded",
  disputed: "billing.transactions.status.disputed",
};

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

/** Collapse a Stripe billing address into a compact, locale-safe display line. */
function formatAddress(address: BillingAddressResponse): string {
  return [
    address.line1,
    address.line2,
    [address.city, address.state, address.postal_code].filter(Boolean).join(" "),
    address.country,
  ]
    .filter(Boolean)
    .join(", ");
}

/** Format a Stripe minor-unit amount in its declared currency. */
function formatTransactionAmount(amount: number, currency: string, locale: string): string {
  try {
    return new Intl.NumberFormat(locale, {
      style: "currency",
      currency: currency.toUpperCase(),
    }).format(amount / 100);
  } catch {
    return `${(amount / 100).toFixed(2)} ${currency.toUpperCase()}`;
  }
}

/** Render Stripe purchase history within the billing settings tab. */
function TransactionHistory() {
  const { locale } = useLocale();
  const [data, setData] = React.useState<BillingTransactionsResponse | null>(null);
  const [loading, setLoading] = React.useState(true);
  const [failed, setFailed] = React.useState(false);

  const load = React.useCallback(() => {
    setLoading(true);
    setFailed(false);
    getBillingTransactions()
      .then(setData)
      .catch(() => setFailed(true))
      .finally(() => setLoading(false));
  }, []);

  React.useEffect(() => {
    load();
  }, [load]);

  return (
    <section
      className="flex flex-col gap-2 border-t border-border/40 pt-5"
      aria-labelledby="transaction-history-heading"
    >
      <div className="flex items-center justify-between gap-3">
        <h3 id="transaction-history-heading" className="text-sm font-semibold text-foreground">
          {msg("billing.transactions.title")}
        </h3>
        {loading && data != null && (
          <CircleNotch className="size-3.5 animate-spin text-muted-foreground" aria-hidden="true" />
        )}
      </div>
      {failed ? (
        <div className="flex items-center justify-between gap-3 border-y border-border/35 py-3">
          <span className="text-xs text-muted-foreground">
            {msg("billing.transactions.load_error")}
          </span>
          <RetryIconButton label={msg("billing.wallet.retry")} onClick={load} />
        </div>
      ) : data == null ? (
        <div
          className="flex h-20 items-center justify-center border-y border-border/35"
          aria-busy="true"
        >
          <CircleNotch className="size-4 animate-spin text-muted-foreground" aria-hidden="true" />
        </div>
      ) : data.entries.length === 0 ? (
        <div className="flex items-center gap-2 border-y border-border/35 py-4 text-xs text-muted-foreground">
          <CreditCard className="size-4 shrink-0" aria-hidden="true" />
          {data.available
            ? msg("billing.transactions.empty")
            : msg("billing.transactions.unavailable")}
        </div>
      ) : (
        <ul className="divide-y divide-border/35 border-y border-border/35">
          {data.entries.map((transaction) => {
            const statusTone =
              transaction.status === "paid"
                ? "bg-emerald-700/10 text-emerald-800"
                : transaction.status === "processing"
                  ? "bg-amber-700/10 text-amber-800"
                  : "bg-destructive/10 text-destructive";
            return (
              <li key={transaction.id} className="flex flex-wrap items-center gap-3 py-3">
                <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground">
                  <CreditCard className="size-4" aria-hidden="true" />
                </span>
                <span className="flex min-w-36 flex-1 flex-col gap-0.5">
                  <span className="text-sm font-medium text-foreground">
                    {transaction.credits == null
                      ? msg("billing.transactions.purchase")
                      : formatMsg("billing.transactions.credits", {
                          p1: formatCredits(transaction.credits, locale),
                        })}
                  </span>
                  <span dir="ltr" className="text-xs text-muted-foreground">
                    {formatResetDate(transaction.at, locale)}
                  </span>
                </span>
                <span className="ms-auto flex shrink-0 items-center gap-2">
                  <span
                    className={cn(
                      "rounded-full px-2 py-0.5 text-[0.625rem] font-semibold",
                      statusTone,
                    )}
                  >
                    {msg(TRANSACTION_STATUS_LABEL[transaction.status])}
                  </span>
                  <span dir="ltr" className="text-sm font-semibold tabular-nums text-foreground">
                    {formatTransactionAmount(transaction.amount, transaction.currency, locale)}
                  </span>
                  {transaction.document_url && (
                    <a
                      href={transaction.document_url}
                      target="_blank"
                      rel="noreferrer"
                      aria-label={msg("billing.transactions.receipt")}
                      title={msg("billing.transactions.receipt")}
                      className="grid size-8 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
                    >
                      <ArrowSquareOut className="size-3.5" aria-hidden="true" />
                    </a>
                  )}
                </span>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

/** Stripe-backed billing identity and masked saved payment methods. */
function BillingDetails() {
  const [profile, setProfile] = React.useState<BillingProfileResponse | null>(null);
  const [loadError, setLoadError] = React.useState(false);
  const [portalFlow, setPortalFlow] = React.useState<"manage" | "payment_method" | null>(null);

  const loadProfile = React.useCallback(() => {
    setLoadError(false);
    getBillingProfile()
      .then(setProfile)
      .catch(() => setLoadError(true));
  }, []);

  React.useEffect(() => {
    loadProfile();
  }, [loadProfile]);

  const openPortal = React.useCallback(async (flow: "manage" | "payment_method") => {
    setPortalFlow(flow);
    try {
      const { url } = await createBillingPortalSession(flow);
      window.location.assign(url);
    } catch {
      setPortalFlow(null);
      toast.error(msg("billing.portal.error"));
    }
  }, []);

  if (loadError) {
    return (
      <div className="flex items-center justify-between gap-3 border-t border-border/40 pt-4">
        <span className="text-xs text-muted-foreground">{msg("billing.profile.load_error")}</span>
        <RetryIconButton label={msg("billing.wallet.retry")} onClick={loadProfile} />
      </div>
    );
  }

  if (profile == null) {
    return (
      <div className="flex h-28 items-center justify-center border-t border-border/40" aria-busy="true">
        <CircleNotch className="size-4 animate-spin text-muted-foreground" aria-hidden="true" />
      </div>
    );
  }

  const address = formatAddress(profile.address);
  const unavailable = !profile.available;

  return (
    <div className="flex flex-col gap-6 border-t border-border/40 pt-5">
      <section className="flex flex-col gap-2" aria-labelledby="billing-profile-heading">
        <div className="flex items-center justify-between gap-3">
          <h3 id="billing-profile-heading" className="text-sm font-semibold text-foreground">
            {msg("billing.profile.title")}
          </h3>
          <Button
            variant="ghost"
            size="sm"
            disabled={unavailable || portalFlow !== null}
            onClick={() => void openPortal("manage")}
          >
            {portalFlow === "manage" && <CircleNotch className="animate-spin" aria-hidden="true" />}
            {msg("billing.profile.edit")}
          </Button>
        </div>
        <dl className="divide-y divide-border/35 border-y border-border/35">
          {[
            [msg("billing.profile.email"), profile.email],
            [msg("billing.profile.name"), profile.name],
            [msg("billing.profile.address"), address],
            [msg("billing.profile.phone"), profile.phone],
          ].map(([label, value]) => (
            <div key={label} className="grid grid-cols-[minmax(7rem,0.4fr)_1fr] gap-4 py-2.5 text-xs">
              <dt className="text-muted-foreground">{label}</dt>
              <dd dir="auto" className="min-w-0 break-words text-foreground">
                {value || msg("billing.profile.empty")}
              </dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="flex flex-col gap-2" aria-labelledby="payment-methods-heading">
        <div className="flex items-center justify-between gap-3">
          <h3 id="payment-methods-heading" className="text-sm font-semibold text-foreground">
            {msg("billing.payment_methods.title")}
          </h3>
          <Button
            variant="outline"
            size="sm"
            disabled={unavailable || portalFlow !== null}
            onClick={() => void openPortal("payment_method")}
          >
            {portalFlow === "payment_method" ? (
              <CircleNotch className="animate-spin" aria-hidden="true" />
            ) : (
              <Plus aria-hidden="true" />
            )}
            {msg("billing.payment_methods.add")}
          </Button>
        </div>
        {profile.payment_methods.length === 0 ? (
          <div className="flex items-center gap-2 border-y border-border/35 py-4 text-xs text-muted-foreground">
            <CreditCard className="size-4 shrink-0" aria-hidden="true" />
            {unavailable
              ? msg("billing.profile.unavailable")
              : msg("billing.payment_methods.empty")}
          </div>
        ) : (
          <ul className="divide-y divide-border/35 border-y border-border/35">
            {profile.payment_methods.map((method) => (
              <li key={method.id} className="flex items-center gap-3 py-3">
                <span className="grid size-8 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground">
                  <CreditCard className="size-4" aria-hidden="true" />
                </span>
                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="flex flex-wrap items-center gap-2 text-sm font-medium text-foreground">
                    <span className="capitalize">{method.brand || method.type.replaceAll("_", " ")}</span>
                    {method.last4 && <span dir="ltr">•••• {method.last4}</span>}
                    {method.is_default && (
                      <span className="rounded-full bg-muted px-1.5 py-0.5 text-[0.625rem] font-semibold text-muted-foreground">
                        {msg("billing.payment_methods.default")}
                      </span>
                    )}
                  </span>
                  {method.exp_month != null && method.exp_year != null && (
                    <span dir="ltr" className="text-xs text-muted-foreground">
                      {formatMsg("billing.payment_methods.expires", {
                        p1: `${String(method.exp_month).padStart(2, "0")}/${String(method.exp_year).slice(-2)}`,
                      })}
                    </span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
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
          <RetryIconButton
            label={msg("billing.wallet.retry")}
            loading={loading}
            onClick={refresh}
          />
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

      <BillingDetails />
      <TransactionHistory />
    </div>
  );
}
