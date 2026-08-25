/**
 * The canonical set of telemetry event names.
 *
 * Centralised so call sites can't drift on spelling and the backend's
 * top-events leaderboard stays legible. Names are short snake_case identifiers
 * (the ingest table caps them at 80 chars). `page_view` and `element_click` are
 * emitted by the autocapture layer; the rest are explicit flow milestones.
 * Run outcome events are emitted by the worker and browser failures by the
 * local error-reporting facade.
 */

export const TelemetryEvent = {
  PageView: "page_view",
  ElementClick: "element_click",
  LoginSucceeded: "login_succeeded",
  LoginFailed: "login_failed",
  ClientError: "client_error",
  RunSubmitted: "run_submitted",
  GridSearchSubmitted: "grid_search_submitted",
  SettingsOpened: "settings_opened",
  SettingsTabChanged: "settings_tab_changed",
  ResultsViewed: "results_viewed",
  ArtifactDownloaded: "artifact_downloaded",
  DatasetCreated: "dataset_created",
  ByokKeyAdded: "byok_key_added",
  TutorialStarted: "tutorial_started",
  TutorialCompleted: "tutorial_completed",
  ShareCreated: "share_created",
} as const;
