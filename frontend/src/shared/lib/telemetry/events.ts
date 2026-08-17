/**
 * The canonical set of telemetry event names.
 *
 * Centralised so call sites can't drift on spelling and the backend's
 * top-events leaderboard stays legible. Names are short snake_case identifiers
 * (the ingest table caps them at 80 chars). `page_view` and `element_click` are
 * emitted by the autocapture layer; the rest are explicit flow milestones.
 * The server emits `purchase_completed` and `run_completed` / `run_failed` /
 * `run_cancelled` itself (Stripe webhook, worker) — those never fire from here.
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
  CheckoutStarted: "checkout_started",
  ResultsViewed: "results_viewed",
  ArtifactDownloaded: "artifact_downloaded",
  DatasetCreated: "dataset_created",
  ByokKeyAdded: "byok_key_added",
  TutorialStarted: "tutorial_started",
  TutorialCompleted: "tutorial_completed",
  ShareCreated: "share_created",
} as const;
