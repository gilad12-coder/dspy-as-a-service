"use client";

import * as React from "react";
import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
import { Check, KeyRound, Loader2, Sparkles, Zap } from "lucide-react";
import { toast } from "react-toastify";
import { msg, formatMsg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { useSettingsModal } from "@/features/settings";
import {
  createCheckoutSession,
  createFoundersCheckout,
  getFoundersRate,
  openBillingPortal,
  type FoundersRateResponse,
} from "@/shared/lib/api";
import { useCredits } from "../providers/credit-provider";
import { CREDIT_PACKS, formatCredits, formatResetDate, type CreditPack } from "../lib/credit";

const EASE_OUT = [0.16, 1, 0.3, 1] as const;

// Monthly price of the Founder's Rate, in USD. Mirrors the Stripe price; the
// Premium column shows it as the big number above "/ month".
const FOUNDERS_USD_PER_MONTH = 20;

/** Whole-dollar USD (no cents) for the big price number — "$20", not "$20.00". */
function formatUsdWhole(usd: number, locale: string): string {
  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(usd);
}

interface Feature {
  label: string;
  /** A leading "Everything in Free, plus:" header row, rendered without a check. */
  header?: boolean;
}

/**
 * The shared plan-card shell — one column of the pricing grid.
 *
 * Anatomy mirrors a pricing-modal column: a label row (name + optional badge or
 * an in-card control), a headline + one-line summary, the price block, the CTA,
 * the check-feature list, and a disclaimer footer pinned to the bottom so the
 * three cards' CTAs and footers align regardless of feature count.
 */
function PlanCard({
  name,
  badge,
  control,
  headline,
  summary,
  price,
  cta,
  features,
  footer,
  highlight,
  index,
}: {
  name: string;
  badge?: string;
  control?: React.ReactNode;
  headline: string;
  summary: string;
  price: React.ReactNode;
  cta: React.ReactNode;
  features: Feature[];
  footer?: React.ReactNode;
  highlight?: boolean;
  index: number;
}) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: EASE_OUT, delay: 0.07 * index }}
      className={cn(
        "relative flex flex-col rounded-2xl border bg-card p-6",
        highlight
          ? "border-[#C8A882]/70 ring-1 ring-[#C8A882]/40 shadow-[0_8px_30px_rgba(200,168,130,0.18)]"
          : "border-border/60 shadow-[0_4px_8px_rgba(0,0,0,0.04)]",
      )}
    >
      {/* Soft gold halo behind the featured column — static (no animation), so it
          reads as emphasis, not decoration, and respects reduced motion by design. */}
      {highlight && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute -inset-px -z-10 rounded-2xl bg-[radial-gradient(120%_60%_at_50%_0%,rgba(200,168,130,0.16),transparent_70%)]"
        />
      )}

      <div className="flex min-h-8 items-start justify-between gap-2">
        <h3 className="text-lg font-semibold tracking-tight text-foreground">{name}</h3>
        {badge ? (
          <span className="rounded-full bg-[#C8A882]/15 px-2 py-0.5 text-[0.6875rem] font-semibold uppercase tracking-wide text-[#8a6d44]">
            {badge}
          </span>
        ) : (
          control
        )}
      </div>

      <div className="mt-5 flex flex-col gap-1.5">
        <p className="text-xl font-semibold text-foreground">{headline}</p>
        <p className="min-h-[2.5rem] text-sm leading-snug text-muted-foreground">{summary}</p>
      </div>

      <div className="mt-4 min-h-[3.25rem]">{price}</div>

      <div className="mt-4">{cta}</div>

      <ul className="mt-6 flex flex-col gap-3.5">
        {features.map((f, i) =>
          f.header ? (
            <li key={i} className="text-[0.9375rem] font-medium text-foreground">
              {f.label}
            </li>
          ) : (
            <li key={i} className="flex items-start gap-3 text-[0.9375rem] leading-5 text-foreground">
              <Check
                className={cn(
                  "mt-0.5 size-4 shrink-0",
                  highlight ? "text-[#C8A882]" : "text-muted-foreground",
                )}
                aria-hidden="true"
              />
              <span>{f.label}</span>
            </li>
          ),
        )}
      </ul>

      {footer && (
        <div className="mt-auto pt-6 text-xs leading-relaxed text-muted-foreground/80">{footer}</div>
      )}
    </motion.div>
  );
}

/** Big price number + suffix. The amount is LTR-islanded; the suffix follows locale flow. */
function Price({ amount, suffix }: { amount: string; suffix?: string }) {
  return (
    <div className="flex items-end gap-1.5">
      <span dir="ltr" className="text-[2.75rem] font-semibold leading-none tracking-tight text-foreground tabular-nums">
        {amount}
      </span>
      {suffix && <span className="pb-1 text-sm text-muted-foreground">{suffix}</span>}
    </div>
  );
}

const PRIMARY_CTA =
  "inline-flex w-full items-center justify-center gap-1.5 rounded-lg bg-[#3D2E22] px-4 py-2.5 text-sm font-semibold text-[#FAF8F5] transition-colors duration-200 cursor-pointer hover:bg-[#2A1F17] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 disabled:cursor-wait disabled:opacity-70";
const SECONDARY_CTA =
  "inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-border/70 px-4 py-2.5 text-sm font-semibold text-foreground transition-colors duration-200 cursor-pointer hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 disabled:cursor-wait disabled:opacity-70";
const DISABLED_CTA =
  "inline-flex w-full cursor-not-allowed items-center justify-center rounded-lg border border-border/50 px-4 py-2.5 text-sm font-medium text-muted-foreground";

/** Free column — the always-on baseline. Its CTA is the inert "current plan" marker. */
function FreeCard({ premiumActive, index }: { premiumActive: boolean; index: number }) {
  const features: Feature[] = [
    { label: msg("billing.plans.free.f1") },
    { label: msg("billing.plans.free.f2") },
    { label: msg("billing.plans.free.f3") },
    { label: msg("billing.plans.free.f4") },
  ];
  return (
    <PlanCard
      index={index}
      name={msg("billing.plans.free.name")}
      headline={msg("billing.plans.free.headline")}
      summary={msg("billing.plans.free.summary")}
      price={<Price amount={msg("billing.plans.price_free")} />}
      cta={
        <button type="button" disabled className={DISABLED_CTA}>
          {premiumActive ? msg("billing.plans.free.included") : msg("billing.plans.free.cta")}
        </button>
      }
      features={features}
      footer={msg("billing.plans.free.note")}
    />
  );
}

/**
 * Premium column — the featured plan and the home of the Founder's Rate logic.
 *
 * Carries the deadline gate (config-driven on the backend): an active subscriber
 * sees a manage action, an open offer the gold CTA into the Stripe subscription,
 * and a closed offer an inert button. The no-lift-no-charge guarantee leads the
 * feature list — it's the reason to subscribe.
 */
function PremiumCard({
  founders,
  premiumActive,
  index,
}: {
  founders: FoundersRateResponse | null;
  premiumActive: boolean;
  index: number;
}) {
  const { locale } = useLocale();
  const [working, setWorking] = React.useState(false);
  const open = founders ? founders.open : true;

  const goToStripe = React.useCallback(async (start: () => Promise<{ url: string }>) => {
    setWorking(true);
    try {
      const { url } = await start();
      window.location.assign(url);
    } catch {
      setWorking(false);
      toast.error(msg("billing.checkout.error"));
    }
  }, []);

  const features: Feature[] = [
    { label: msg("billing.plans.premium.everything_free"), header: true },
    { label: msg("billing.plans.premium.f1") },
    { label: msg("billing.founders.stack_guarantee") },
    { label: msg("billing.founders.stack_serving") },
    { label: msg("billing.founders.stack_credits") },
    { label: msg("billing.founders.stack_lock") },
  ];

  let cta: React.ReactNode;
  if (premiumActive) {
    cta = (
      <button
        type="button"
        onClick={() => goToStripe(openBillingPortal)}
        disabled={working}
        className={SECONDARY_CTA}
      >
        {working ? <Loader2 className="size-4 animate-spin" /> : msg("billing.founders.manage")}
      </button>
    );
  } else if (open) {
    cta = (
      <button
        type="button"
        onClick={() => goToStripe(createFoundersCheckout)}
        disabled={working}
        className={PRIMARY_CTA}
      >
        {working ? <Loader2 className="size-4 animate-spin" /> : <Zap className="size-4" aria-hidden="true" />}
        {msg("billing.founders.cta")}
      </button>
    );
  } else {
    cta = (
      <button type="button" disabled className={DISABLED_CTA}>
        {msg("billing.founders.closed")}
      </button>
    );
  }

  const note =
    !premiumActive && open && founders
      ? formatMsg("billing.founders.deadline", { p1: formatResetDate(founders.closes_at, locale) })
      : msg("billing.founders.price_note");

  return (
    <PlanCard
      index={index}
      highlight
      name={msg("billing.plans.premium.name")}
      badge={msg("billing.plans.premium.badge")}
      headline={msg("billing.plans.premium.headline")}
      summary={msg("billing.plans.premium.summary")}
      price={
        premiumActive ? (
          <Price amount={msg("billing.founders.active_short")} />
        ) : (
          <Price amount={formatUsdWhole(FOUNDERS_USD_PER_MONTH, locale)} suffix={msg("billing.plans.per_month")} />
        )
      }
      cta={cta}
      features={features}
      footer={note}
    />
  );
}

/**
 * Pay-as-you-go column — prepaid credit packs with an in-card pack selector that
 * mirrors the reference's in-column usage toggle: pick a pack, the price and the
 * "≈ N runs" line update, and the CTA buys exactly that pack through Stripe.
 */
function CreditsCard({ index }: { index: number }) {
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

  const control = (
    <div
      role="group"
      aria-label={msg("billing.plans.credits.pack_aria")}
      className="flex shrink-0 gap-0.5 rounded-full border border-border/50 bg-muted/40 p-0.5"
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
              "rounded-full px-2 py-0.5 text-[0.6875rem] font-semibold tabular-nums transition-colors duration-150 cursor-pointer",
              active
                ? "bg-background text-foreground shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <span dir="ltr">{formatCredits(p.credits, locale)}</span>
          </button>
        );
      })}
    </div>
  );

  const features: Feature[] = [
    { label: msg("billing.plans.credits.f1") },
    { label: msg("billing.plans.credits.f2") },
    { label: msg("billing.plans.credits.f3") },
  ];

  return (
    <PlanCard
      index={index}
      name={msg("billing.plans.credits.name")}
      control={control}
      headline={msg("billing.plans.credits.headline")}
      summary={msg("billing.plans.credits.summary")}
      price={
        <Price amount={formatUsdWhole(pack.usd, locale)} suffix={msg("billing.plans.price_onetime")} />
      }
      cta={
        <button type="button" onClick={onBuy} disabled={buying} className={SECONDARY_CTA}>
          {buying ? <Loader2 className="size-4 animate-spin" /> : null}
          {msg("billing.action.add_credits")}
        </button>
      }
      features={features}
      footer={msg("billing.upgrade.reassurance")}
    />
  );
}

/**
 * Upgrade page — the in-app pricing surface, laid out as a plan-comparison grid.
 *
 * A centered title, then three columns:
 * Free (the baseline), Premium (the featured Founder's Rate, guarantee-led), and
 * Pay-as-you-go (prepaid credit packs with an in-card pack selector). Below the
 * grid, a quiet BYOK exit. All Stripe wiring — founders rate gate, founders /
 * pack / portal checkouts, and the post-checkout `?status=success` balance sync
 * [FG-3] — is preserved from the prior layout.
 */
export function UpgradeView() {
  const { openTo } = useSettingsModal();
  const { beginSync, wallet } = useCredits();
  const [founders, setFounders] = React.useState<FoundersRateResponse | null>(null);

  React.useEffect(() => {
    getFoundersRate()
      .then(setFounders)
      .catch(() => {
        /* keep the optimistic-open default — no backend or signed-out */
      });
  }, []);

  // Stripe redirects back to /upgrade?status=success|cancel. On success, enter
  // the post-checkout `syncing` state [FG-3] — the chip shimmers over the prior
  // balance until the webhook lands, never a false zero — and clear the param so
  // a reload doesn't re-toast.
  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const status = new URLSearchParams(window.location.search).get("status");
    if (!status) return;
    if (status === "success") {
      toast.success(msg("billing.upgrade.success_toast"));
      beginSync();
    }
    window.history.replaceState(null, "", window.location.pathname);
  }, [beginSync]);

  return (
    <div className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 py-10 sm:py-14">
      <header className="flex flex-col items-center gap-2 text-center">
        <h1 className="text-[clamp(1.75rem,4vw,2.25rem)] font-semibold tracking-tight text-foreground">
          {msg("billing.plans.heading")}
        </h1>
        <p className="max-w-[46ch] text-sm text-muted-foreground">{msg("billing.plans.subheading")}</p>
      </header>

      <div className="grid items-stretch gap-5 md:grid-cols-2 min-[60rem]:grid-cols-3">
        <FreeCard premiumActive={wallet.premiumActive} index={0} />
        <PremiumCard founders={founders} premiumActive={wallet.premiumActive} index={1} />
        <CreditsCard index={2} />
      </div>

      <footer className="flex flex-col items-center gap-2 border-t border-border/40 pt-6 text-center">
        <span className="inline-flex items-center gap-2 text-sm text-muted-foreground">
          <KeyRound className="size-4" aria-hidden="true" />
          {msg("billing.plans.footer.byok_q")}
        </span>
        <button
          type="button"
          onClick={() => openTo("api")}
          className="inline-flex items-center gap-1 text-sm font-semibold text-[#3D2E22] underline-offset-4 hover:underline cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 rounded"
        >
          <Sparkles className="size-3.5" aria-hidden="true" />
          {msg("billing.upgrade.byok_action")}
        </button>
      </footer>
    </div>
  );
}
