"use client";

import * as React from "react";
import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
import { Check, KeyRound, Plus, ShieldCheck, Sparkles } from "lucide-react";
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
import {
  CREDIT_PACKS,
  formatCredits,
  formatResetDate,
  formatUsd,
  type CreditPack,
} from "../lib/credit";

const EASE_OUT_HERO = [0.16, 1, 0.3, 1] as const;

// Founding-membership stack — credible itemization (no fake dollar anchors).
// Each row is a message key; order matters (guarantee-adjacent value first).
const FOUNDERS_STACK = [
  "billing.founders.stack_managed",
  "billing.founders.stack_proof",
  "billing.founders.stack_guarantee",
  "billing.founders.stack_serving",
  "billing.founders.stack_history",
  "billing.founders.stack_credits",
  "billing.founders.stack_lock",
] as const;

// Monthly price of the Founder's Rate, in USD. Mirrors the Stripe price; shown
// alongside "/mo locked 12 months" so the lock reads as the promise, not a trick.
const FOUNDERS_USD_PER_MONTH = 20;

// Rough credit cost of a single run, used only to translate a pack size into a
// "what does this buy me" line. Display heuristics, not billing truth.
const APPROX_FRONTIER_CREDITS = 300;
const APPROX_MINI_CREDITS = 15;

const EASE_OUT = [0.16, 1, 0.3, 1] as const;

function PackOption({ pack, index }: { pack: CreditPack; index: number }) {
  const { locale } = useLocale();
  const reduce = useReducedMotion();
  const frontierRuns = Math.round(pack.credits / APPROX_FRONTIER_CREDITS);
  const miniRuns = Math.round(pack.credits / APPROX_MINI_CREDITS);
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
    <motion.div
      initial={reduce ? false : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: EASE_OUT, delay: 0.06 * index }}
      className={cn(
        "flex flex-col gap-4 rounded-xl border bg-card p-5",
        pack.popular ? "border-[#C8A882]/70 ring-1 ring-[#C8A882]/40" : "border-border/60",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {msg("billing.unit.credits")}
        </span>
        {pack.popular && (
          <span className="rounded-full bg-[#C8A882]/15 px-2 py-0.5 text-[0.6875rem] font-semibold text-[#8a6d44]">
            {msg("billing.upgrade.pack_popular")}
          </span>
        )}
      </div>

      <div className="flex items-baseline gap-2">
        <span dir="ltr" className="text-3xl font-semibold text-foreground tabular-nums">
          {formatCredits(pack.credits, locale)}
        </span>
        <span dir="ltr" className="text-sm text-muted-foreground tabular-nums">
          {formatUsd(pack.usd, locale)}
        </span>
      </div>

      <p className="text-xs text-muted-foreground">
        {formatMsg("billing.upgrade.pack_estimate", {
          p1: formatCredits(frontierRuns, locale),
          p2: formatCredits(miniRuns, locale),
        })}
      </p>

      <button
        type="button"
        onClick={onBuy}
        disabled={buying}
        className={cn(
          "mt-auto inline-flex items-center justify-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-semibold transition-colors duration-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 disabled:cursor-wait disabled:opacity-70",
          pack.popular
            ? "bg-[#3D2E22] text-[#FAF8F5] hover:bg-[#2A1F17]"
            : "border border-border/70 text-foreground hover:bg-accent",
        )}
      >
        <Plus className="size-4" aria-hidden="true" />
        {formatMsg("billing.upgrade.buy", { p1: formatUsd(pack.usd, locale) })}
      </button>
    </motion.div>
  );
}

/** A secondary "other way to run" option — mini models or BYOK. */
function AltOption({
  icon: Icon,
  title,
  description,
  action,
  onAction,
  href,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  action: string;
  onAction?: () => void;
  href?: string;
}) {
  const actionClass =
    "inline-flex items-center gap-1 self-start text-sm font-semibold text-[#3D2E22] underline-offset-4 hover:underline cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 rounded";
  return (
    <div className="flex flex-col gap-2 rounded-xl border border-border/60 p-5">
      <Icon className="size-5 text-muted-foreground" aria-hidden="true" />
      <span className="text-sm font-semibold text-foreground">{title}</span>
      <p className="text-xs text-muted-foreground">{description}</p>
      {href ? (
        <Link href={href} className={actionClass}>
          {action}
        </Link>
      ) : (
        <button type="button" onClick={onAction} className={actionClass}>
          {action}
        </button>
      )}
    </div>
  );
}

/**
 * The Founder's Rate hero — the guarantee on top, a credible-itemized stack, a
 * calm deadline line, and the one gold primary CTA into the Stripe subscription.
 *
 * The deadline gate is config-driven on the backend; when the offer has closed
 * the CTA is replaced by a quiet closed line. An active subscriber sees a
 * founding-member confirmation and a manage link instead of the CTA. Microcopy
 * stays factual — one honest deadline, no countdown theatre.
 */
function FoundersHero({
  founders,
  premiumActive,
}: {
  founders: FoundersRateResponse | null;
  premiumActive: boolean;
}) {
  const reduce = useReducedMotion();
  const { locale } = useLocale();
  const [working, setWorking] = React.useState(false);

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

  const open = founders ? founders.open : true;
  const priceLabel = formatMsg("billing.founders.price", {
    p1: formatUsd(FOUNDERS_USD_PER_MONTH, locale),
  });

  return (
    <motion.section
      initial={reduce ? false : { opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.55, ease: EASE_OUT_HERO }}
      aria-labelledby="founders-heading"
      className="flex flex-col gap-6 rounded-2xl border border-border/60 bg-card p-6 sm:p-8"
    >
      <div className="flex flex-col gap-3">
        <span className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-[0.14em] text-[#8a6d44]">
          <ShieldCheck className="size-3.5" aria-hidden="true" />
          {msg("billing.founders.eyebrow")}
        </span>
        <h1
          id="founders-heading"
          className="text-[clamp(1.75rem,4vw,2.5rem)] font-semibold leading-tight text-foreground"
        >
          {msg("billing.founders.title")}
        </h1>
        <p className="max-w-[58ch] text-sm leading-relaxed text-muted-foreground">
          {msg("billing.founders.lead")}
        </p>
      </div>

      <div className="flex flex-col gap-3">
        <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          {msg("billing.founders.stack_title")}
        </h2>
        <ul className="grid gap-x-6 gap-y-2.5 sm:grid-cols-2">
          {FOUNDERS_STACK.map((key) => (
            <li key={key} className="flex items-start gap-2 text-sm text-foreground">
              <Check className="mt-0.5 size-4 shrink-0 text-[#C8A882]" aria-hidden="true" />
              <span>{msg(key)}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="flex flex-col gap-3 border-t border-border/40 pt-5">
        {premiumActive ? (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-foreground">
              <ShieldCheck className="size-4 text-[#C8A882]" aria-hidden="true" />
              {msg("billing.founders.active")}
            </span>
            <button
              type="button"
              onClick={() => goToStripe(openBillingPortal)}
              disabled={working}
              className="rounded-lg border border-border/70 px-3.5 py-2 text-sm font-medium text-foreground transition-colors duration-200 cursor-pointer hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 disabled:cursor-wait disabled:opacity-70"
            >
              {msg("billing.founders.manage")}
            </button>
          </div>
        ) : open ? (
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex flex-col gap-1">
              <span dir="ltr" className="text-2xl font-semibold text-foreground tabular-nums">
                {priceLabel}
              </span>
              <span className="text-xs text-muted-foreground">
                {founders
                  ? formatMsg("billing.founders.deadline", {
                      p1: formatResetDate(founders.closes_at, locale),
                    })
                  : msg("billing.founders.price_note")}
              </span>
            </div>
            <button
              type="button"
              onClick={() => goToStripe(createFoundersCheckout)}
              disabled={working}
              className="inline-flex items-center justify-center gap-1.5 rounded-lg bg-[#3D2E22] px-4 py-2.5 text-sm font-semibold text-[#FAF8F5] transition-colors duration-200 cursor-pointer hover:bg-[#2A1F17] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45 disabled:cursor-wait disabled:opacity-70"
            >
              <ShieldCheck className="size-4" aria-hidden="true" />
              {msg("billing.founders.cta")}
            </button>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">{msg("billing.founders.closed")}</p>
        )}
      </div>
    </motion.section>
  );
}

/**
 * Upgrade page — the Founder's Rate offer and the only in-app pricing explainer.
 *
 * Leads with the Founder's Rate (the guarantee on top, the value stack, a calm
 * deadline line, the $20/mo locked CTA). Credit packs are demoted to a "need
 * more credits?" section below, and the two free exits — stay on mini models or
 * bring your own key — sit beneath that, so the page reads as "here's the offer
 * and how else to proceed", never a hard wall.
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
    <div className="mx-auto flex w-full max-w-4xl flex-col gap-12 px-4 py-10 sm:px-6 sm:py-14">
      <FoundersHero founders={founders} premiumActive={wallet.premiumActive} />

      <section className="flex flex-col gap-3" aria-labelledby="packs-heading">
        <div className="flex flex-col gap-1">
          <h2
            id="packs-heading"
            className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
          >
            {msg("billing.founders.packs_title")}
          </h2>
          <p className="text-xs text-muted-foreground/80">{msg("billing.founders.packs_lead")}</p>
        </div>
        <div className="grid gap-4 sm:grid-cols-3">
          {CREDIT_PACKS.map((pack, i) => (
            <PackOption key={pack.id} pack={pack} index={i} />
          ))}
        </div>
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground/80">
          <Check className="size-3.5 text-[#C8A882]" aria-hidden="true" />
          {msg("billing.upgrade.reassurance")}
        </p>
      </section>

      <section className="flex flex-col gap-3" aria-labelledby="alt-heading">
        <h2
          id="alt-heading"
          className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
        >
          {msg("billing.upgrade.other_title")}
        </h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <AltOption
            icon={Sparkles}
            title={msg("billing.upgrade.mini_title")}
            description={msg("billing.upgrade.mini_desc")}
            action={msg("billing.upgrade.mini_action")}
            href="/"
          />
          <AltOption
            icon={KeyRound}
            title={msg("billing.upgrade.byok_title")}
            description={msg("billing.upgrade.byok_desc")}
            action={msg("billing.upgrade.byok_action")}
            onAction={() => openTo("api")}
          />
        </div>
      </section>
    </div>
  );
}
