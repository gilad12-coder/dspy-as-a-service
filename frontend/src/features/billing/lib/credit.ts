/**
 * Credit-wallet domain model shared by the billing UI surfaces.
 *
 * Skynet runs on pay-as-you-go prepaid credits: every new account gets a one-time
 * free grant, and beyond it users buy credit packs or start Premium. Credits are
 * spendable on any model — the free/paid line is about balance, not catalog
 * access. Credits are the unit of account; the dollar value is always shown
 * alongside (`CREDIT_USD_VALUE`). Pricing is break-even (MARKUP 1.09 covering
 * payment-processing fees only, zero BYOK fee): the platform takes no margin.
 *
 * Everything here is framework-agnostic (no React / `next/*`) so it imports from
 * server components, client components, and the provider alike. The values are
 * currently served by a STUB (`STUB_WALLET`) until the billing backend (credit
 * ledger + Stripe + OpenRouter metering) lands; the shapes are the contract that
 * backend will fill.
 */

/** Where a job's tokens are billed: through Skynet's credits, or the user's own key. */
export type TokenSourceMode = "managed" | "byok";

/** Coarse health of the wallet, used to theme the balance chip. */
export type WalletStatus = "healthy" | "low" | "empty";

/** What a usage-ledger row represents. */
export type LedgerKind = "run" | "topup" | "grant";

/** Platform value of one credit, in USD. The user-facing "$ equivalent" of a credit balance. */
export const CREDIT_USD_VALUE = 0.01;

/** Below this much spendable value the wallet reads as "running low" (calm, not alarming). */
export const LOW_BALANCE_USD = 0.5;

/** The one-time free grant that lets a new account try the platform. */
export interface FreeGrant {
  creditsRemaining: number;
  creditsTotal: number;
  /** ISO-8601 instant a Premium allotment renews; `null` for the one-time free grant. */
  resetsAt: string | null;
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

/** Opt-in automatic top-up so long-running work isn't interrupted by an empty balance. */
export interface AutoReload {
  enabled: boolean;
  thresholdCredits: number;
  topUpCredits: number;
}

/** The whole wallet as the UI needs it. */
export interface CreditWallet {
  /** Purchased credits, on top of the free grant. */
  paidBalanceCredits: number;
  freeGrant: FreeGrant;
  /** Which token source the account is actively running jobs on. */
  mode: TokenSourceMode;
  /** Whether an active Premium subscription is in effect. */
  premiumActive: boolean;
  autoReload: AutoReload;
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

/** Locale-aware medium date (e.g. `Jul 1, 2026`) for the grant reset line. */
export function formatResetDate(iso: string, locale: string): string {
  return new Intl.DateTimeFormat(locale, { dateStyle: "medium" }).format(new Date(iso));
}

/** Prepaid packs offered on the wallet + upgrade surfaces. Bigger packs carry more bonus credits. */
export const CREDIT_PACKS: CreditPack[] = [
  { id: "starter", credits: 500, usd: 5 },
  { id: "plus", credits: 2200, usd: 20, popular: true },
  { id: "pro", credits: 6500, usd: 50 },
];

/**
 * Placeholder wallet until the billing backend exists. A realistic "has some
 * paid balance + a partly-spent grant + recent activity" account so every UI
 * state renders against real-shaped data, not Lorem Ipsum. Dates are fixed
 * strings (Date.now() is intentionally avoided in shared modules).
 */
export const STUB_WALLET: CreditWallet = {
  paidBalanceCredits: 1240,
  freeGrant: { creditsRemaining: 480, creditsTotal: 500, resetsAt: null },
  mode: "managed",
  premiumActive: false,
  autoReload: { enabled: false, thresholdCredits: 200, topUpCredits: 2200 },
  usage: [
    {
      id: "u1",
      at: "2026-06-26T08:42:00Z",
      label: "sentiment-classifier v3",
      model: "anthropic/claude-opus-4-8",
      credits: -312,
      mode: "managed",
      kind: "run",
    },
    {
      id: "u2",
      at: "2026-06-25T19:10:00Z",
      label: "rag-reranker sweep",
      model: "openai/gpt-5.5",
      credits: -88,
      mode: "managed",
      kind: "run",
    },
    {
      id: "u3",
      at: "2026-06-24T14:03:00Z",
      label: "Top-up",
      model: null,
      credits: 2200,
      mode: "managed",
      kind: "topup",
    },
    {
      id: "u4",
      at: "2026-06-23T11:27:00Z",
      label: "entity-extractor tune",
      model: "openai/gpt-5.5-mini",
      credits: -14,
      mode: "managed",
      kind: "run",
    },
    {
      id: "u5",
      at: "2026-06-22T09:00:00Z",
      label: "intent-router (your key)",
      model: "google/gemini-3-pro",
      credits: 0,
      mode: "byok",
      kind: "run",
    },
    {
      id: "u6",
      at: "2026-06-01T00:00:00Z",
      label: "Monthly grant",
      model: null,
      credits: 200,
      mode: "managed",
      kind: "grant",
    },
  ],
};
