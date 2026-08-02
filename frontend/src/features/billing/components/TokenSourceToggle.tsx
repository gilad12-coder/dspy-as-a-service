"use client";

import * as React from "react";
import { motion } from "framer-motion";
import { Coins, Key } from "@/shared/ui/icons";
import { msg } from "@/shared/lib/messages";
import { cn } from "@/shared/lib/utils";
import { useCredits } from "../providers/credit-provider";
import type { TokenSourceMode } from "../lib/credit";

const SEGMENTS: Array<{
  mode: TokenSourceMode;
  icon: typeof Coins;
  labelKey: "billing.mode.managed" | "billing.mode.byok";
}> = [
  { mode: "managed", icon: Coins, labelKey: "billing.mode.managed" },
  { mode: "byok", icon: Key, labelKey: "billing.mode.byok" },
];

// Mirrors the logs verbosity pill so the billing-mode toggle slides alike.
const PILL_TRANSITION = { type: "tween", duration: 0.16, ease: [0.22, 1, 0.36, 1] } as const;

/**
 * Managed-credits ↔ your-own-key toggle for the model step.
 *
 * Writes the global wallet mode, so flipping it here also flips the header
 * balance chip into its BYOK state. The hint underneath spells out who gets
 * billed, so the choice never reads as a hidden setting.
 */
export function TokenSourceToggle() {
  const { wallet, setMode } = useCredits();
  const mode = wallet.mode;

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border/50 bg-muted/20 px-3 py-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-semibold text-foreground">{msg("billing.mode.label")}</span>
        <div
          role="group"
          aria-label={msg("billing.mode.aria")}
          className="flex rounded-lg bg-muted p-0.5"
        >
          {SEGMENTS.map(({ mode: value, icon: Icon, labelKey }) => (
            <button
              key={value}
              type="button"
              onClick={() => setMode(value)}
              aria-pressed={mode === value}
              className={cn(
                "relative flex items-center justify-center rounded-md px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer",
                mode === value ? "text-foreground" : "text-muted-foreground hover:text-foreground",
              )}
            >
              {mode === value && (
                <motion.span
                  layoutId="token-source-pill"
                  className="absolute inset-0 rounded-md bg-background shadow-[0_1px_2px_oklch(0.25_0.04_45/.12)]"
                  transition={PILL_TRANSITION}
                  aria-hidden="true"
                />
              )}
              <span className="relative z-10 flex items-center gap-1.5">
                <Icon className="size-3.5" aria-hidden="true" />
                {msg(labelKey)}
              </span>
            </button>
          ))}
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        {mode === "managed" ? msg("billing.mode.managed_hint") : msg("billing.mode.byok_hint")}
      </p>
    </div>
  );
}
