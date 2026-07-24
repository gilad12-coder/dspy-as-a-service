"use client";

import * as React from "react";
import { formatMsg, msg } from "@/shared/lib/messages";

import { getActiveIntlLocale } from "@/shared/lib/runtime-locale";

import type { AgentToolCall } from "@/shared/ui/agent/types";

import { ToolCallRow } from "./ToolCallRow";
import { StatTile } from "./result-card-atoms";

interface WalletCardProps {
  call: AgentToolCall;
}

interface FreeGrant {
  credits_remaining?: number;
  credits_total?: number;
  resets_at?: string | null;
}

interface UsageEntry {
  id?: string;
  label?: string;
  model?: string | null;
  credits?: number;
  kind?: string;
}

interface WalletResult {
  paid_balance_credits?: number;
  free_grant?: FreeGrant;
  premium_active?: boolean;
  usage?: UsageEntry[];
}

/** One credit == $0.01 (mirrors the billing slice's CREDIT_USD_VALUE). */
const CREDIT_USD = 0.01;

function extractWallet(call: AgentToolCall): WalletResult | null {
  const payload = (call.payload ?? {}) as Record<string, unknown>;
  const result = payload.result;
  if (!result || typeof result !== "object" || Array.isArray(result)) return null;
  const r = result as WalletResult;
  if (typeof r.paid_balance_credits !== "number" || typeof r.free_grant !== "object") return null;
  return r;
}

function totalCredits(w: WalletResult): number {
  return (w.free_grant?.credits_remaining ?? 0) + (w.paid_balance_credits ?? 0);
}

function fmtCredits(n: number): string {
  return n.toLocaleString(getActiveIntlLocale());
}

function fmtUsd(credits: number): string {
  return `$${(credits * CREDIT_USD).toFixed(2)}`;
}

function planLabel(premium: boolean): string {
  return premium
    ? msg("auto.features.agent.panel.components.walletcard.plan_premium")
    : msg("auto.features.agent.panel.components.walletcard.plan_free");
}

function buildSummary(w: WalletResult | null, isRunning: boolean): string | null {
  if (isRunning || !w) return null;
  return formatMsg("auto.features.agent.panel.components.walletcard.summary", {
    p1: fmtCredits(totalCredits(w)),
    p2: planLabel(Boolean(w.premium_active)),
  });
}

/**
 * Result card for ``get_wallet_for_agent`` — the caller's spendable credits
 * (free grant + paid balance) as a headline with its USD value, the plan, a
 * free-grant / paid-balance split, and the most recent ledger entries.
 */
export function WalletCard({ call }: WalletCardProps) {
  const wallet = extractWallet(call);
  const summary = buildSummary(wallet, call.status === "running");

  if (!wallet) {
    return <ToolCallRow call={call} summary={summary} />;
  }

  const total = totalCredits(wallet);
  const premium = Boolean(wallet.premium_active);
  const grant = wallet.free_grant ?? {};
  const usage = wallet.usage ?? [];

  const customBody = (
    <div className="space-y-3">
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        <span dir="ltr" className="text-[1.25rem] font-semibold tabular-nums text-foreground">
          {fmtCredits(total)}
        </span>
        <span className="text-[0.6875rem] text-muted-foreground/70">
          {msg("auto.features.agent.panel.components.walletcard.title")}
        </span>
        <span dir="ltr" className="text-[0.625rem] text-muted-foreground/55">
          ≈ {fmtUsd(total)}
        </span>
        <span className="ms-auto inline-flex items-center gap-1.5">
          <span className="text-[0.625rem] text-muted-foreground/60">
            {msg("auto.features.agent.panel.components.walletcard.plan_label")}
          </span>
          <PlanChip premium={premium} />
        </span>
      </div>

      <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5">
        <StatTile
          label={msg("auto.features.agent.panel.components.walletcard.free_grant")}
          value={
            grant.credits_remaining == null
              ? null
              : `${fmtCredits(grant.credits_remaining)} / ${fmtCredits(grant.credits_total ?? 0)}`
          }
          valueDir="ltr"
        />
        <StatTile
          label={msg("auto.features.agent.panel.components.walletcard.paid_balance")}
          value={fmtCredits(wallet.paid_balance_credits ?? 0)}
          valueDir="ltr"
        />
      </dl>

      {usage.length > 0 && (
        <div className="space-y-1">
          <div className="text-[0.625rem] uppercase tracking-wide text-muted-foreground/60">
            {msg("auto.features.agent.panel.components.walletcard.recent")}
          </div>
          <ul className="divide-y divide-border/40">
            {usage.slice(0, 5).map((row, idx) => (
              <li key={row.id ?? idx} className="flex items-center gap-2 py-1 text-[0.6875rem]">
                <span dir="auto" className="min-w-0 flex-1 truncate text-foreground/75">
                  {row.label ?? "—"}
                </span>
                <CreditDelta credits={row.credits ?? 0} />
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );

  return <ToolCallRow call={call} summary={summary} customBody={customBody} />;
}

function PlanChip({ premium }: { premium: boolean }) {
  const label = planLabel(premium);
  return (
    <span
      className="inline-flex items-center rounded-full px-2 py-0.5 text-[0.625rem] font-medium leading-none"
      style={
        premium
          ? { backgroundColor: "var(--success-dim)", color: "var(--success)" }
          : undefined
      }
    >
      <span className={premium ? undefined : "text-muted-foreground/70"}>{label}</span>
    </span>
  );
}

function CreditDelta({ credits }: { credits: number }) {
  const positive = credits > 0;
  const text = positive ? `+${credits.toLocaleString(getActiveIntlLocale())}` : credits.toLocaleString(getActiveIntlLocale());
  return (
    <span
      dir="ltr"
      className="shrink-0 font-mono tabular-nums"
      style={{ color: positive ? "var(--success)" : undefined }}
    >
      {text}
    </span>
  );
}
