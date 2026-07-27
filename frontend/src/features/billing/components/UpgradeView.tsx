"use client";

import * as React from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Check, Loader2 } from "lucide-react";
import { toast } from "react-toastify";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { createCheckoutSession } from "@/shared/lib/api";
import { useCredits } from "../providers/credit-provider";
import { CREDIT_PACKS, formatCredits, type CreditPack } from "../lib/credit";

const EASE_OUT = [0.16, 1, 0.3, 1] as const;

// Slide transition for the credit-pack selector's shared-layout pill — matches the
// runs-source segmented control in explore/SearchBar so the two read identically.
const PILL_TRANSITION = { type: "tween", duration: 0.18, ease: [0.22, 1, 0.36, 1] } as const;

/** Whole-dollar USD (no cents) for the big price number — "$20", not "$20.00". */
function formatUsdWhole(usd: number, locale: string): string {
  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(usd);
}

/** Big price number + suffix. The amount is LTR-islanded; the suffix follows locale flow. */
function Price({ amount, suffix }: { amount: string; suffix?: string }) {
  return (
    <div className="flex items-end gap-1.5">
      <span
        dir="ltr"
        className="text-[2.75rem] font-semibold leading-none tracking-tight text-foreground tabular-nums"
      >
        {amount}
      </span>
      {suffix && <span className="pb-1 text-sm text-muted-foreground">{suffix}</span>}
    </div>
  );
}

const PRIMARY_CTA =
  "inline-flex w-full items-center justify-center gap-1.5 rounded-lg bg-[#3D2E22] px-4 py-2.5 text-sm font-semibold text-[#FAF8F5] transition-colors duration-200 cursor-pointer hover:bg-[#2A1F17] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 disabled:cursor-wait disabled:opacity-70";

/**
 * Pay-as-you-go card — the whole pricing surface, since prepaid credits are the
 * only plan. An in-card pack selector: pick a pack, the price updates, and the
 * CTA buys exactly that pack through Stripe.
 */
function CreditsCard() {
  const { locale } = useLocale();
  const reduce = useReducedMotion();
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
  );

  const features = [msg("billing.plans.credits.f1"), msg("billing.plans.credits.f2")];

  return (
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: EASE_OUT }}
      className="flex w-full max-w-96 flex-col rounded-2xl border border-border/60 bg-card p-6 shadow-[0_4px_8px_rgba(0,0,0,0.04)]"
    >
      <div className="flex min-h-8 items-start justify-between gap-2">
        <h3 className="text-lg font-semibold tracking-tight text-foreground">
          {msg("billing.plans.credits.name")}
        </h3>
        {control}
      </div>

      <div className="mt-5 flex flex-col gap-1.5">
        <p className="text-xl font-semibold text-foreground">
          {msg("billing.plans.credits.headline")}
        </p>
        <p className="text-sm leading-snug text-muted-foreground">
          {msg("billing.plans.credits.summary")}
        </p>
      </div>

      <div className="mt-4">
        <Price
          amount={formatUsdWhole(pack.usd, locale)}
          suffix={msg("billing.plans.price_onetime")}
        />
      </div>

      <div className="mt-4">
        <button type="button" onClick={onBuy} disabled={buying} className={PRIMARY_CTA}>
          {buying ? <Loader2 className="size-4 animate-spin" /> : null}
          {msg("billing.action.add_credits")}
        </button>
      </div>

      <ul className="mt-6 flex flex-col gap-3.5">
        {features.map((label, i) => (
          <li key={i} className="flex items-start gap-3 text-[0.9375rem] leading-5 text-foreground">
            <Check className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
            <span>{label}</span>
          </li>
        ))}
      </ul>
    </motion.div>
  );
}

/**
 * Upgrade page — the in-app add-credits surface.
 *
 * A centered title over the single pay-as-you-go card (prepaid credit packs
 * with an in-card pack selector). The Stripe wiring — pack checkout and the
 * post-checkout `?status=success` balance sync [FG-3] — is preserved from the
 * prior plan-grid layout.
 */
export function UpgradeView() {
  const { beginSync } = useCredits();

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
    <div className="mx-auto flex w-full max-w-[78rem] flex-col items-center gap-8 px-4 py-10 sm:py-14">
      <header className="text-center">
        <h1 className="text-[clamp(1.75rem,4vw,2.25rem)] font-semibold tracking-tight text-foreground">
          {msg("billing.upgrade.title")}
        </h1>
      </header>

      <CreditsCard />
    </div>
  );
}
