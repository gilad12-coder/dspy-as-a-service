"use client";

import * as React from "react";
import { msg } from "@/shared/lib/messages";
import { useUserPrefs } from "@/features/settings";

import type { TrustMode } from "../lib/types";

const MODES: TrustMode[] = ["ask", "auto_safe", "yolo"];

export function useTrustMode(): {
  mode: TrustMode;
  setMode: (m: TrustMode) => void;
  next: () => void;
} {
  const { prefs, setPref } = useUserPrefs();
  const mode = prefs.agentTrustMode;

  const setMode = React.useCallback(
    (m: TrustMode) => setPref("agentTrustMode", m),
    [setPref],
  );

  const next = React.useCallback(() => {
    const idx = MODES.indexOf(mode);
    const nextMode = MODES[(idx + 1) % MODES.length] ?? "ask";
    setPref("agentTrustMode", nextMode);
  }, [mode, setPref]);

  return { mode, setMode, next };
}

// The hue map is locale-independent and stays concrete.
export const TRUST_MODE_HUE: Record<TrustMode, string> = {
  ask: "#3D2E22",
  auto_safe: "#5E7A5E",
  yolo: "#A85A1A",
};

// Labels/descriptions are thunks, not resolved strings: `msg()` reads the active
// locale's out-of-band catalog, which is empty at module-eval on the server
// (frozen to the raw key, process-wide) but populated in the browser — resolving
// eagerly here would hydrate-mismatch. Callers invoke them per render.
export const TRUST_MODE_LABEL: Record<TrustMode, () => string> = {
  ask: () => msg("auto.features.agent.panel.hooks.use.trust.mode.literal.1"),
  auto_safe: () => msg("auto.features.agent.panel.hooks.use.trust.mode.literal.2"),
  yolo: () => msg("auto.features.agent.panel.hooks.use.trust.mode.literal.3"),
};

export const TRUST_MODE_DESCRIPTION: Record<TrustMode, () => string> = {
  ask: () => msg("auto.features.agent.panel.hooks.use.trust.mode.literal.4"),
  auto_safe: () => msg("auto.features.agent.panel.hooks.use.trust.mode.literal.5"),
  yolo: () => msg("auto.features.agent.panel.hooks.use.trust.mode.literal.6"),
};
