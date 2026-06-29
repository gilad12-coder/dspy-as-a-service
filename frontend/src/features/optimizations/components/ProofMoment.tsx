"use client";

import { BadgeCheck, Undo2 } from "lucide-react";
import type { GuaranteeBasis } from "@/shared/types/api";
import { FadeIn } from "@/shared/ui/motion";
import { formatCredits } from "@/features/billing";
import { useLocale } from "@/shared/providers";
import { formatImprovement } from "@/shared/lib";
import { formatMsg, msg } from "@/shared/lib/messages";
import { proofBannerVariant, readBilling } from "../lib/proof-banner";

/**
 * The proof moment — the trust climax on a finished run.
 *
 * Turns billing into the evidence the product worked. A billed run reads as the
 * receipt that the lift was real: "we beat your baseline by +X% on data the
 * optimizer never saw — that's why this run was billed." A refunded run reads as
 * the guarantee holding: "no lift, so the run was free — N credits refunded."
 *
 * Renders only once the billing outcome has been stamped on the result (after
 * the worker debits/adjudicates), so it never claims a charge before the ledger
 * settles. The basis line distinguishes the unbiased test split from the valset
 * fallback, keeping "we beat your baseline" honest about which slice was scored.
 */
export function ProofMoment({
  improvement,
  guarantee,
  details,
}: {
  improvement: number | undefined;
  guarantee?: GuaranteeBasis | null;
  details?: Record<string, unknown>;
}) {
  const { locale } = useLocale();
  const billing = readBilling(details);
  const variant = proofBannerVariant(billing, improvement);
  if (billing == null || variant == null) return null;

  const basisLabel =
    guarantee?.basis === "val"
      ? msg("optimization.proof.basis.val")
      : msg("optimization.proof.basis.test");
  const credits = formatCredits(billing.credits, locale);

  if (variant === "refunded") {
    return (
      <FadeIn>
        <div className="rounded-xl border border-[#E3DCD0] bg-[#FBF9F4] px-5 py-4">
          <div className="flex items-start gap-3">
            <span
              className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-full bg-[#EFEAE0] text-[#7C6350]"
              aria-hidden="true"
            >
              <Undo2 className="size-4 rtl:-scale-x-100" />
            </span>
            <div className="flex min-w-0 flex-col gap-1">
              <p className="text-sm font-semibold text-[#1C1612]">
                {msg("optimization.proof.refunded.title")}
              </p>
              <p className="text-sm leading-relaxed text-[#5C4F42]">
                {formatMsg("optimization.proof.refunded.line", {
                  p1: basisLabel,
                  p2: credits,
                })}
              </p>
            </div>
          </div>
        </div>
      </FadeIn>
    );
  }

  return (
    <FadeIn>
      <div className="rounded-xl border border-[#C8A882]/45 bg-[#FBF9F4] px-5 py-4">
        <div className="flex items-start gap-3">
          <span
            className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-full bg-[#C8A882]/15 text-[#8a6d44]"
            aria-hidden="true"
          >
            <BadgeCheck className="size-4" />
          </span>
          <div className="flex min-w-0 flex-col gap-1">
            <p className="text-sm font-semibold text-[#1C1612]">
              {variant === "billed-lift"
                ? msg("optimization.proof.billed.title")
                : msg("optimization.proof.billed.neutral.title")}
            </p>
            <p className="text-sm leading-relaxed text-[#5C4F42]">
              {variant === "billed-lift"
                ? formatMsg("optimization.proof.billed.line", {
                    p1: formatImprovement(improvement),
                    p2: basisLabel,
                    p3: credits,
                  })
                : formatMsg("optimization.proof.billed.neutral", { p1: credits })}
            </p>
          </div>
        </div>
      </div>
    </FadeIn>
  );
}
