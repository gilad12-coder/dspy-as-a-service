"use client";

import * as React from "react";

interface SettingsModalContextValue {
  open: boolean;
  setOpen: (open: boolean) => void;
  /** Tab to jump to on the next open, or null to keep the last/default tab. */
  targetTab: string | null;
  /** Open the modal focused on a specific tab (e.g. the credit chip → wallet). */
  openTo: (tab: string) => void;
  /** Consume the pending target tab so it doesn't re-apply on the next manual open. */
  clearTarget: () => void;
}

const SettingsModalContext = React.createContext<SettingsModalContextValue | null>(null);

export function SettingsModalProvider({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = React.useState(false);
  const [targetTab, setTargetTab] = React.useState<string | null>(null);

  const openTo = React.useCallback((tab: string) => {
    setTargetTab(tab);
    setOpen(true);
  }, []);

  const clearTarget = React.useCallback(() => setTargetTab(null), []);

  const value = React.useMemo(
    () => ({ open, setOpen, targetTab, openTo, clearTarget }),
    [open, targetTab, openTo, clearTarget],
  );
  return <SettingsModalContext.Provider value={value}>{children}</SettingsModalContext.Provider>;
}

export function useSettingsModal(): SettingsModalContextValue {
  const ctx = React.useContext(SettingsModalContext);
  if (!ctx) throw new Error("useSettingsModal must be used within SettingsModalProvider");
  return ctx;
}
