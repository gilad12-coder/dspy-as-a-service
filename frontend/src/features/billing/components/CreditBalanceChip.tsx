"use client";

import * as React from "react";
import Link from "next/link";
import { Coins, KeyRound, Plus } from "lucide-react";
import { msg, formatMsg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useLocale } from "@/shared/providers";
import { useSettingsModal } from "@/features/settings";
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/ui/primitives/popover";
import { useCredits } from "../providers/credit-provider";
import { formatCredits } from "../lib/credit";

/**
 * Header credit-balance chip — the spine of the billing UI.
 *
 * Sits inline-end beside the language switcher. Shows total spendable credits
 * (free grant + purchased). Theming is calm by design: gold `#C8A882` marks a healthy balance and the
 * primary "Add credits" affordance only; a low balance reads as quiet taupe, not
 * an alarm. When the account is running on its own provider key (BYOK), the chip
 * switches to a key glyph instead of a number, since managed credits aren't spent.
 */
export function CreditBalanceChip({ className }: { className?: string }) {
  const { wallet, status, totalCredits, loading, syncing } = useCredits();
  const { locale } = useLocale();
  const { openTo } = useSettingsModal();
  const [open, setOpen] = React.useState(false);

  if (loading) {
    return (
      <div
        className={cn(
          "h-[26px] w-16 animate-pulse rounded-lg border border-border/60 bg-muted/60",
          className,
        )}
        aria-hidden="true"
      />
    );
  }

  const isByok = wallet.mode === "byok";

  // One trigger, three visual registers. Healthy spends the gold accent; low and
  // empty stay in the warm neutrals so the chip never shouts. Empty reframes the
  // chip itself as the "add credits" call to action.
  const triggerTone =
    isByok || status === "healthy"
      ? "border-border/70 text-foreground hover:bg-accent"
      : status === "low"
        ? "border-border/70 text-muted-foreground hover:bg-accent"
        : "border-[#C8A882]/60 text-foreground hover:bg-[#C8A882]/10";

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={formatMsg("billing.chip.aria", {
            p1: isByok ? msg("billing.chip.byok") : formatCredits(totalCredits, locale),
          })}
          aria-busy={syncing || undefined}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 text-xs font-semibold transition-colors duration-200 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45",
            triggerTone,
            className,
          )}
        >
          {isByok ? (
            <>
              <KeyRound className="size-3.5 text-muted-foreground" aria-hidden="true" />
              <span>{msg("billing.chip.byok")}</span>
            </>
          ) : status === "empty" ? (
            <>
              <Plus className="size-3.5 text-[#C8A882]" aria-hidden="true" />
              <span>{msg("billing.chip.empty")}</span>
            </>
          ) : (
            <>
              <Coins
                className={cn(
                  "size-3.5",
                  status === "healthy" ? "text-[#C8A882]" : "text-muted-foreground",
                )}
                aria-hidden="true"
              />
              {/* Post-checkout sync [FG-3]: a quiet shimmer over the *prior* balance
                  until the webhook lands, so the chip never flashes a false zero. */}
              <span
                dir="ltr"
                className={cn("tabular-nums", syncing && "animate-pulse text-muted-foreground")}
              >
                {formatCredits(totalCredits, locale)}
              </span>
            </>
          )}
        </button>
      </PopoverTrigger>

      <PopoverContent align="end" className="w-72 p-0">
        <div className="flex flex-col">
          <div className="flex items-baseline justify-between gap-3 px-4 pt-4 pb-3">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {msg("billing.popover.title")}
            </span>
          </div>

          {isByok ? (
            <div className="flex flex-col gap-1 px-4 pb-4">
              <span className="text-sm font-medium text-foreground">
                {msg("billing.popover.byok_active")}
              </span>
              <span className="text-xs text-muted-foreground">
                {msg("billing.popover.byok_hint")}
              </span>
            </div>
          ) : (
            <>
              <div className="px-4 pb-3">
                <span
                  dir="ltr"
                  className="block text-right text-2xl font-semibold text-foreground tabular-nums"
                >
                  {formatCredits(totalCredits, locale)}
                </span>
              </div>
              <dl className="flex flex-col gap-2 border-t border-border/40 px-4 py-3 text-xs">
                <div className="flex items-center justify-between gap-3">
                  <dt className="text-muted-foreground">{msg("billing.popover.paid")}</dt>
                  <dd dir="ltr" className="font-medium text-foreground tabular-nums">
                    {formatCredits(wallet.paidBalanceCredits, locale)}
                  </dd>
                </div>
                <div className="flex items-center justify-between gap-3">
                  <dt className="text-muted-foreground">{msg("billing.popover.free_grant")}</dt>
                  <dd dir="ltr" className="font-medium text-foreground tabular-nums">
                    {formatCredits(wallet.freeGrant.creditsRemaining, locale)}
                    <span className="text-muted-foreground">
                      {" / "}
                      {formatCredits(wallet.freeGrant.creditsTotal, locale)}
                    </span>
                  </dd>
                </div>
                {/* Low balance reads as an operational metric, not an alarm: one
                    calm factual line in the warm neutrals, no red, no urgency. */}
                {status === "low" && (
                  <p className="text-muted-foreground/80">{msg("billing.chip.low_note")}</p>
                )}
              </dl>
            </>
          )}

          <div className="flex items-center gap-2 border-t border-border/40 p-3">
            <Link
              href="/upgrade"
              onClick={() => setOpen(false)}
              className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-[#3D2E22] px-3 py-1.5 text-xs font-semibold text-[#FAF8F5] transition-colors duration-200 hover:bg-[#2A1F17] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              <Plus className="size-3.5" aria-hidden="true" />
              {msg("billing.action.add_credits")}
            </Link>
            <button
              type="button"
              onClick={() => {
                setOpen(false);
                openTo("billing");
              }}
              className="rounded-lg border border-border/70 px-3 py-1.5 text-xs font-medium text-foreground transition-colors duration-200 cursor-pointer hover:bg-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#C8A882]/45"
            >
              {msg("billing.action.view_wallet")}
            </button>
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
