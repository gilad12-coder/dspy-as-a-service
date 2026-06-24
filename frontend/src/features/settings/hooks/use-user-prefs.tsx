"use client";

import * as React from "react";
import { toast } from "react-toastify";
import { msg } from "@/shared/lib/messages";
// Import from the leaf bridge module, not the @/features/tutorial barrel: the
// barrel re-exports demo-data, which pulls in @/features/trajectory and loops
// back through the settings barrel — a cycle that makes Turbopack fail to
// resolve useLiteMode's re-export. The bridge module is a pure leaf.
// eslint-disable-next-line no-restricted-imports -- deliberate leaf import; see above
import { registerTutorialHook } from "@/features/tutorial/lib/bridge";
import {
  DEFAULT_PREFS,
  PREF_KEYS,
  readPref,
  writePref,
  type UserPrefs,
} from "../lib/prefs";

interface UserPrefsContextValue {
  prefs: UserPrefs;
  setPref: <K extends keyof UserPrefs>(key: K, value: UserPrefs[K]) => void;
  resetAll: () => void;
}

const UserPrefsContext = React.createContext<UserPrefsContextValue | null>(null);

export function UserPrefsProvider({ children }: { children: React.ReactNode }) {
  const [prefs, setPrefs] = React.useState<UserPrefs>(DEFAULT_PREFS);

  React.useEffect(() => {
    const next: UserPrefs = { ...DEFAULT_PREFS };
    (Object.keys(DEFAULT_PREFS) as Array<keyof UserPrefs>).forEach((k) => {
      (next[k] as UserPrefs[typeof k]) = readPref(k);
    });
    setPrefs(next);
  }, []);

  React.useEffect(() => {
    const handler = (e: StorageEvent) => {
      if (!e.key) return;
      const match = (Object.entries(PREF_KEYS) as Array<[keyof UserPrefs, string]>).find(
        ([, key]) => key === e.key,
      );
      if (!match) return;
      const [prefKey] = match;
      setPrefs((prev) => ({ ...prev, [prefKey]: readPref(prefKey) }));
    };
    window.addEventListener("storage", handler);
    return () => window.removeEventListener("storage", handler);
  }, []);

  const setPref = React.useCallback(
    <K extends keyof UserPrefs>(key: K, value: UserPrefs[K]) => {
      setPrefs((prev) => ({ ...prev, [key]: value }));
      writePref(key, value);
      toast.success(msg("settings.saved"), { autoClose: 1500, toastId: "settings-saved" });
    },
    [],
  );

  // Tutorial bridge — flips advancedMode silently (no settings-saved toast)
  // so the deep-dive tour can reveal /explore without leaking a "settings
  // saved" affordance the user never asked for.
  React.useEffect(
    () =>
      registerTutorialHook("setAdvancedMode", (enabled) => {
        setPrefs((prev) => ({ ...prev, advancedMode: enabled }));
        writePref("advancedMode", enabled);
      }),
    [],
  );

  const resetAll = React.useCallback(() => {
    setPrefs(DEFAULT_PREFS);
    if (typeof window === "undefined") return;
    Object.values(PREF_KEYS).forEach((key) => {
      try {
        window.localStorage.removeItem(key);
      } catch {
        /* noop */
      }
    });
  }, []);

  const value = React.useMemo<UserPrefsContextValue>(
    () => ({ prefs, setPref, resetAll }),
    [prefs, setPref, resetAll],
  );

  return <UserPrefsContext.Provider value={value}>{children}</UserPrefsContext.Provider>;
}

export function useUserPrefs(): UserPrefsContextValue {
  const ctx = React.useContext(UserPrefsContext);
  if (!ctx) throw new Error("useUserPrefs must be used within UserPrefsProvider");
  return ctx;
}
