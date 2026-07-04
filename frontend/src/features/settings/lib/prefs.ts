export type CodeAssistDefault = "auto" | "manual";
export type SplitModeDefault = "auto" | "manual";
export type TrustModeDefault = "ask" | "auto_safe" | "yolo";

export interface AgentShortcut {
  key: string;
  ctrl: boolean;
  alt: boolean;
  shift: boolean;
  meta: boolean;
}

export interface UserPrefs {
  // Layout preference, not a capability gate: advanced sections are always
  // reachable; this only pre-expands them everywhere.
  expandAdvanced: boolean;
  // Lightweight mode for low-resource machines: kills motion/blur and swaps the
  // heavy visualizations (charts, SVG trajectory tree, code editor) for static
  // equivalents. See LiteModeProvider.
  liteMode: boolean;
  wizardCodeAssist: CodeAssistDefault;
  wizardSplitMode: SplitModeDefault;
  agentTrustMode: TrustModeDefault;
  agentShortcut: AgentShortcut;
}

export const PREF_KEYS: Record<keyof UserPrefs, string> = {
  expandAdvanced: "skynet.prefs.expand-advanced",
  liteMode: "skynet.prefs.lite-mode",
  wizardCodeAssist: "skynet.prefs.wizard.code-assist",
  wizardSplitMode: "skynet.prefs.wizard.split-mode",
  agentTrustMode: "skynet.prefs.agent.trust-mode",
  agentShortcut: "skynet.prefs.agent.shortcut",
};

export const DEFAULT_AGENT_SHORTCUT: AgentShortcut = {
  key: "j",
  ctrl: true,
  alt: false,
  shift: false,
  meta: false,
};

export const DEFAULT_PREFS: UserPrefs = {
  expandAdvanced: false,
  liteMode: false,
  wizardCodeAssist: "auto",
  wizardSplitMode: "auto",
  agentTrustMode: "ask",
  agentShortcut: DEFAULT_AGENT_SHORTCUT,
};

// The retired global "advanced mode" toggle. Users who had it on expect the
// advanced sections open, so it seeds expandAdvanced once and is then removed.
const LEGACY_ADVANCED_MODE_KEY = "skynet.prefs.advanced-mode";

export function migrateLegacyPrefs(): void {
  if (typeof window === "undefined") return;
  try {
    const legacy = window.localStorage.getItem(LEGACY_ADVANCED_MODE_KEY);
    if (legacy === null) return;
    if (legacy === "true" && window.localStorage.getItem(PREF_KEYS.expandAdvanced) === null) {
      window.localStorage.setItem(PREF_KEYS.expandAdvanced, "true");
    }
    window.localStorage.removeItem(LEGACY_ADVANCED_MODE_KEY);
  } catch {
    /* noop */
  }
}

export function readPref<K extends keyof UserPrefs>(key: K): UserPrefs[K] {
  if (typeof window === "undefined") return DEFAULT_PREFS[key];
  try {
    const raw = window.localStorage.getItem(PREF_KEYS[key]);
    if (raw == null) return DEFAULT_PREFS[key];
    return JSON.parse(raw) as UserPrefs[K];
  } catch {
    return DEFAULT_PREFS[key];
  }
}

export function writePref<K extends keyof UserPrefs>(key: K, value: UserPrefs[K]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(PREF_KEYS[key], JSON.stringify(value));
  } catch {
    /* noop */
  }
}
