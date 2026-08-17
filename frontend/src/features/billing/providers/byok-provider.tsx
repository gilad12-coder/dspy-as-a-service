"use client";

import * as React from "react";
import {
  getProviderKeys,
  saveProviderKey,
  verifyProviderKey,
  removeProviderKey,
  type ProviderKeyResponse,
  type SaveProviderKeyOptions,
} from "@/shared/lib/api";
import { type KeyStatus, type ProviderKey } from "../lib/byok";
import { invalidateByokModelCatalog } from "@/shared/lib/model-catalog";
import { track, TelemetryEvent } from "@/shared/lib/telemetry";

interface ByokContextValue {
  /** Saved keys, keyed by provider slug via `keyFor`. */
  keys: ProviderKey[];
  /** True while the saved keys are being fetched on mount. */
  loading: boolean;
  /** The saved key for a provider slug, or null. */
  keyFor: (provider: string) => ProviderKey | null;
  /**
   * Save (or rotate) a provider's key. The plaintext is sent once to the vault,
   * encrypted at rest, and verified on entry; only the masked tail + verdict
   * come back. `opts` carries an optional custom endpoint / label. Resolves to
   * the saved key's status.
   */
  saveKey: (provider: string, secret: string, opts?: SaveProviderKeyOptions) => Promise<KeyStatus>;
  /** Re-run the verify probe against a stored key and persist the fresh verdict. */
  verifyKey: (provider: string) => Promise<KeyStatus>;
  /** Forget a provider's key. */
  removeKey: (provider: string) => Promise<void>;
}

const ByokContext = React.createContext<ByokContextValue | null>(null);

/** Map a backend masked-connection response onto the UI's ProviderKey shape. */
function toProviderKey(r: ProviderKeyResponse): ProviderKey {
  return {
    id: r.id,
    provider: r.provider,
    label: r.label ?? null,
    last4: r.last4,
    apiBase: r.api_base ?? null,
    status: r.status,
    addedAt: r.added_at,
  };
}

/** Read the BYOK key store from the nearest ByokKeysProvider. */
export function useByokKeys(): ByokContextValue {
  const ctx = React.useContext(ByokContext);
  if (!ctx) {
    throw new Error("useByokKeys must be used within a ByokKeysProvider");
  }
  return ctx;
}

/**
 * Provide the BYOK provider-key store to the client tree.
 *
 * Backed by the real encrypt-at-rest vault (`/billing/byok/keys`): the saved
 * keys are fetched on mount, and every mutation round-trips to the backend so a
 * secret is never held only in memory — the plaintext is sent once on save,
 * encrypted server-side, and never returned. The context exposes only the
 * masked tail + verification state. If the fetch fails (no backend, signed-out)
 * the store stays empty so the settings UI still renders its add-a-key rows.
 *
 * Args:
 *   initialKeys: Override the seed keys (tests / story scenarios).
 *   children: App subtree.
 */
export function ByokKeysProvider({
  initialKeys = [],
  children,
}: {
  initialKeys?: ProviderKey[];
  children: React.ReactNode;
}) {
  const [keys, setKeys] = React.useState<ProviderKey[]>(initialKeys);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    let active = true;
    getProviderKeys()
      .then((r) => {
        if (active) setKeys(r.keys.map(toProviderKey));
      })
      .catch(() => {
        /* keep the current keys — no backend or signed-out */
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const keyFor = React.useCallback(
    (provider: string) => keys.find((k) => k.provider === provider) ?? null,
    [keys],
  );

  const upsert = React.useCallback((next: ProviderKey) => {
    invalidateByokModelCatalog();
    setKeys((prev) => [...prev.filter((k) => k.provider !== next.provider), next]);
  }, []);

  const saveKey = React.useCallback(
    async (provider: string, secret: string, opts?: SaveProviderKeyOptions): Promise<KeyStatus> => {
      const saved = await saveProviderKey(provider, secret, opts);
      upsert(toProviderKey(saved));
      track(TelemetryEvent.ByokKeyAdded, { provider, status: saved.status });
      return saved.status;
    },
    [upsert],
  );

  const verifyKey = React.useCallback(
    async (provider: string): Promise<KeyStatus> => {
      const verified = await verifyProviderKey(provider);
      upsert(toProviderKey(verified));
      return verified.status;
    },
    [upsert],
  );

  const removeKey = React.useCallback(async (provider: string): Promise<void> => {
    const remaining = await removeProviderKey(provider);
    invalidateByokModelCatalog();
    setKeys(remaining.keys.map(toProviderKey));
  }, []);

  const value = React.useMemo<ByokContextValue>(
    () => ({ keys, loading, keyFor, saveKey, verifyKey, removeKey }),
    [keys, loading, keyFor, saveKey, verifyKey, removeKey],
  );

  return <ByokContext.Provider value={value}>{children}</ByokContext.Provider>;
}
