"use client";

import * as React from "react";
import { toast } from "react-toastify";
import { msg } from "@/shared/lib/messages";
import { getWallet, type BillingWalletResponse } from "@/shared/lib/api";
import {
  EMPTY_WALLET,
  totalCredits,
  walletStatus,
  type CreditWallet,
  type LedgerKind,
  type TokenSourceMode,
  type WalletStatus,
} from "../lib/credit";

interface CreditContextValue {
  wallet: CreditWallet;
  /** True while the wallet is being fetched — drives the chip's loading shimmer. */
  loading: boolean;
  /**
   * True while waiting for a post-checkout Stripe webhook to land [FG-3]. The
   * chip shows a quiet shimmer over the *prior* balance rather than flashing a
   * stale/empty number while the FastAPI→Postgres seam catches up.
   */
  syncing: boolean;
  /** Whether at least one real wallet response has loaded in this session. */
  available: boolean;
  /** Whether the latest wallet request failed. */
  loadError: boolean;
  status: WalletStatus;
  totalCredits: number;
  /** Switch the active token source (managed credits vs the user's own key). */
  setMode: (mode: TokenSourceMode) => void;
  /** Re-fetch the wallet from the backend — call after a checkout/portal return. */
  refresh: () => void;
  /**
   * Enter the post-checkout `syncing` state and poll the wallet until the webhook
   * lands (the balance changes) or a short budget elapses. Call on a Stripe
   * `?billing=success` return so the chip never shows a false zero.
   */
  beginSync: () => void;
}

/**
 * Overlay a backend wallet response onto the current wallet, preserving the
 * client-only active token-source `mode` toggle.
 * Backend ledger rows are all managed-mode credits (BYOK runs aren't billed),
 * so each maps to `mode: "managed"`.
 */
function applyWalletResponse(prev: CreditWallet, r: BillingWalletResponse): CreditWallet {
  return {
    ...prev,
    paidBalanceCredits: r.paid_balance_credits,
    freeGrant: {
      creditsRemaining: r.free_grant.credits_remaining,
      creditsTotal: r.free_grant.credits_total,
    },
    usage: r.usage.map((u) => ({
      id: u.id,
      at: u.at,
      label: u.label,
      model: u.model,
      credits: u.credits,
      mode: "managed",
      kind: u.kind as LedgerKind,
    })),
  };
}

const CreditContext = React.createContext<CreditContextValue | null>(null);

/** Read the credit wallet and its mutators from the nearest CreditProvider. */
export function useCredits(): CreditContextValue {
  const ctx = React.useContext(CreditContext);
  if (!ctx) {
    throw new Error("useCredits must be used within a CreditProvider");
  }
  return ctx;
}

/**
 * Provide the credit wallet to the client tree.
 *
 * Fetches the real wallet (paid balance, free grant, ledger)
 * from the billing backend on mount and after every `refresh()` — e.g. when a
 * Stripe Checkout returns. The provider starts with a truthful empty value and
 * exposes an explicit unavailable state when the request fails; it never shows
 * invented balances or ledger rows. The client-only `mode` toggle survives
 * each refresh.
 *
 * Args:
 *   children: App subtree.
 */
export function CreditProvider({ children }: { children: React.ReactNode }) {
  const [wallet, setWallet] = React.useState<CreditWallet>(EMPTY_WALLET);
  const [loading, setLoading] = React.useState(true);
  const [syncing, setSyncing] = React.useState(false);
  const [available, setAvailable] = React.useState(false);
  const [loadError, setLoadError] = React.useState(false);

  const refresh = React.useCallback(() => {
    setLoading(true);
    setLoadError(false);
    getWallet()
      .then((r) => {
        setWallet((prev) => applyWalletResponse(prev, r));
        setAvailable(true);
      })
      .catch(() => {
        setLoadError(true);
      })
      .finally(() => setLoading(false));
  }, []);

  React.useEffect(() => {
    refresh();
  }, [refresh]);

  // Post-checkout sync [FG-3]: Stripe webhooks aren't instant, so on a success
  // return we poll the wallet (without toggling `loading`, to avoid wiping the
  // prior balance) until it changes or a short budget elapses. The chip reads
  // `syncing` and shimmers over the prior balance instead of flashing a zero.
  const beginSync = React.useCallback(() => {
    setSyncing(true);
    let attempt = 0;
    const baseline = wallet.paidBalanceCredits;
    const poll = () => {
      attempt += 1;
      void getWallet()
        .then((r) => {
          setWallet((prev) => applyWalletResponse(prev, r));
          setAvailable(true);
          setLoadError(false);
          return r.paid_balance_credits !== baseline;
        })
        .catch(() => {
          setLoadError(true);
          return false;
        })
        .then((landed) => {
          // Stop once the webhook's effect is visible, or after ~12s of polling
          // so the shimmer is never permanent if nothing changes (e.g. cancel).
          if (landed || attempt >= 8) {
            setSyncing(false);
            return;
          }
          window.setTimeout(poll, 1500);
        });
    };
    poll();
  }, [wallet.paidBalanceCredits]);

  // Stripe Checkout returns to `/?billing=success|cancel` (there is no
  // standalone add-credits page). On success, toast and enter the syncing
  // state; either way strip the param so a reload doesn't re-toast. beginSync's
  // identity changes as the wallet loads, but re-runs bail on the cleared param.
  React.useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const status = params.get("billing");
    if (!status) return;
    if (status === "success") {
      toast.success(msg("billing.upgrade.success_toast"));
      beginSync();
    }
    params.delete("billing");
    const query = params.toString();
    window.history.replaceState(null, "", window.location.pathname + (query ? `?${query}` : ""));
  }, [beginSync]);

  const setMode = React.useCallback((mode: TokenSourceMode) => {
    setWallet((w) => ({ ...w, mode }));
  }, []);

  const value = React.useMemo<CreditContextValue>(
    () => ({
      wallet,
      loading,
      syncing,
      available,
      loadError,
      status: walletStatus(wallet),
      totalCredits: totalCredits(wallet),
      setMode,
      refresh,
      beginSync,
    }),
    [wallet, loading, syncing, available, loadError, setMode, refresh, beginSync],
  );

  return <CreditContext.Provider value={value}>{children}</CreditContext.Provider>;
}
