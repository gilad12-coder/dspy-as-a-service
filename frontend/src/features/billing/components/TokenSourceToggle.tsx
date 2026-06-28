"use client";

import * as React from "react";
import { Coins, KeyRound } from "lucide-react";
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
  { mode: "byok", icon: KeyRound, labelKey: "billing.mode.byok" },
];

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
                "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors cursor-pointer",
                mode === value
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Icon className="size-3.5" aria-hidden="true" />
              {msg(labelKey)}
            </button>
          ))}
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        {mode === "managed" ? msg("billing.mode.managed_hint") : msg("billing.mode.byok_hint")}
      </p>
      {/* BYOK concurrency note [FG-2]: a your-key run shares Skynet's self-hosted
          LiteLLM compute and may queue at peak. Stating the reason keeps it from
          reading as a pay-to-skip penalty for not buying credits. */}
      {mode === "byok" && (
        <p className="text-xs text-muted-foreground/80">{msg("billing.mode.byok_queue_note")}</p>
      )}
    </div>
  );
}
