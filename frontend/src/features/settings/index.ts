export { UserPrefsProvider, useUserPrefs } from "./hooks/use-user-prefs";
export { LiteModeProvider, useLiteMode } from "./hooks/use-lite-mode";
export { SettingsModalProvider, useSettingsModal } from "./hooks/use-settings-modal";
export { SettingsModal } from "./components/SettingsModal.lazy";
export { LiteModeHint } from "./components/LiteModeHint";
export { readPref } from "./lib/prefs";
export type { AgentShortcut } from "./lib/prefs";
export { formatShortcut } from "./lib/shortcuts";
