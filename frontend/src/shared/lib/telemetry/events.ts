/**
 * The canonical set of telemetry event names.
 *
 * Centralised so call sites can't drift on spelling and the backend's
 * top-events leaderboard stays legible. Names are short snake_case identifiers
 * (the ingest table caps them at 80 chars). `page_view` and `element_click` are
 * emitted by the autocapture layer; the rest are explicit flow milestones.
 */

export const TelemetryEvent = {
  PageView: "page_view",
  ElementClick: "element_click",
  LoginSucceeded: "login_succeeded",
  LoginFailed: "login_failed",
  SignupStarted: "signup_started",
  SignupSucceeded: "signup_succeeded",
  RunSubmitted: "run_submitted",
  GridSearchSubmitted: "grid_search_submitted",
  SettingsOpened: "settings_opened",
  SettingsTabChanged: "settings_tab_changed",
} as const;
