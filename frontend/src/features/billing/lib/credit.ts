/**
 * Credit-wallet domain model shared by the billing UI surfaces.
 *
 * Skynet runs on pay-as-you-go prepaid credits — the only plan: users buy
 * credit packs (at par, one credit per cent) and every run spends against
 * them; there is no free allowance. Credits are spendable on any model.
 * Credits are the unit of account; the dollar value is always shown alongside
 * (`CREDIT_USD_VALUE`). Pricing is strictly at-cost (MARKUP 1.20: payment
 * fees plus the CPU/storage share; BYOK runs pay just that infra share): the
 * platform neither subsidizes a run nor takes profit.
 *
 * Everything here is framework-agnostic (no React / `next/*`) so it imports from
 * server components, client components, and the provider alike. Wallet values
 * come from the billing API; the empty value below is only a truthful loading /
 * unavailable seed and never contains demo balances or activity.
 */

/** Where a job's tokens are billed: through Skynet's credits, or the user's own key. */
export type TokenSourceMode = "managed" | "byok";

/** Coarse health of the wallet, used to theme the balance chip. */
export type WalletStatus = "healthy" | "low" | "empty";

/** What a usage-ledger row represents. */
export type LedgerKind = "run" | "topup" | "grant";

/** Platform value of one credit, in USD. The user-facing "$ equivalent" of a credit balance. */
export const CREDIT_USD_VALUE = 0.01;

/**
 * Bounds for a custom (user-chosen) top-up, mirroring the backend's
 * CUSTOM_CREDITS_MIN/MAX. The floor clears Stripe's $0.50 charge minimum;
 * the ceiling keeps a typo'd amount from becoming a four-figure charge.
 */
export const CUSTOM_CREDITS_MIN = 50;
export const CUSTOM_CREDITS_MAX = 100_000;

/** Below this much spendable value the wallet reads as "running low" (calm, not alarming). */
export const LOW_BALANCE_USD = 0.5;

/** The one-time free grant that lets a new account try the platform. */
export interface FreeGrant {
  creditsRemaining: number;
  creditsTotal: number;
}

/** One row of the usage ledger. `label`/`model` are backend-supplied, not translated. */
export interface UsageEntry {
  id: string;
  /** ISO-8601 instant the entry was recorded. */
  at: string;
  /** Human label for the row (a run name, "Top-up", "Monthly grant") — dynamic, LTR-islanded. */
  label: string;
  /** Model id involved, or null for non-run entries. Always rendered LTR. */
  model: string | null;
  /** Signed credit delta: negative for a run (spend), positive for a top-up/grant. */
  credits: number;
  mode: TokenSourceMode;
  kind: LedgerKind;
}

/** A purchasable prepaid bundle. `usd` is what the user pays; `credits` is what they can spend. */
export interface CreditPack {
  id: string;
  credits: number;
  usd: number;
  /** Flagged as the recommended option in the pack grid. */
  popular?: boolean;
}

/** The whole wallet as the UI needs it. */
export interface CreditWallet {
  /** Purchased credits, on top of the free grant. */
  paidBalanceCredits: number;
  freeGrant: FreeGrant;
  /** Which token source the account is actively running jobs on. */
  mode: TokenSourceMode;
  /** Most-recent-first ledger rows. */
  usage: UsageEntry[];
}

/** Convert a credit count to its USD platform value. */
export function creditsToUsd(credits: number): number {
  return credits * CREDIT_USD_VALUE;
}

/** Total spendable credits = free grant remaining + purchased balance. */
export function totalCredits(wallet: CreditWallet): number {
  return wallet.freeGrant.creditsRemaining + wallet.paidBalanceCredits;
}

/** Derive the chip's health bucket from spendable value. */
export function walletStatus(wallet: CreditWallet): WalletStatus {
  const total = totalCredits(wallet);
  if (total <= 0) return "empty";
  if (creditsToUsd(total) < LOW_BALANCE_USD) return "low";
  return "healthy";
}

/** Locale-aware integer credit formatting (e.g. `1,240`). */
export function formatCredits(credits: number, locale: string): string {
  return new Intl.NumberFormat(locale, { maximumFractionDigits: 0 }).format(credits);
}

/**
 * Locale-aware USD formatting. Sub-cent values (a single mini-model run can cost
 * fractions of a cent) keep more precision so "$0.003" doesn't collapse to "$0.00".
 */
export function formatUsd(usd: number, locale: string): string {
  const fractionDigits = usd !== 0 && Math.abs(usd) < 0.01 ? 4 : 2;
  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: fractionDigits,
  }).format(usd);
}

/** Locale-aware medium date (e.g. `Jul 1, 2026`) for ledger/settings date lines. */
export function formatResetDate(iso: string, locale: string): string {
  return new Intl.DateTimeFormat(locale, { dateStyle: "medium" }).format(new Date(iso));
}

/** Prepaid packs offered on the wallet settings tab. At par — one credit per cent, no bonus subsidy. */
export const CREDIT_PACKS: CreditPack[] = [
  { id: "starter", credits: 500, usd: 5 },
  { id: "plus", credits: 2000, usd: 20, popular: true },
  { id: "pro", credits: 5000, usd: 50 },
];

/** Truthful zero-value seed used until the billing API returns real data. */
export const EMPTY_WALLET: CreditWallet = {
  paidBalanceCredits: 0,
  freeGrant: { creditsRemaining: 0, creditsTotal: 0 },
  mode: "managed",
  usage: [],
};
