/**
 * Tutorial System — Step Definitions
 * Skynet prompt optimization platform
 *
 * Focused, replayable walkthroughs for the product's main user workflows.
 * Works even for users with zero optimizations.
 */

import {
  resetDemoSimulation,
  DEMO_METRIC_CODE,
  DEMO_OPTIMIZATION_ID,
  DEMO_SIGNATURE_CODE,
  getCachedDemoDashboardAnalytics,
  getCachedDemoDashboardJobs,
  getCachedDemoExplorePoints,
} from "./demo-data";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";

/**
 * The short end-to-end path plus three focused workflow guides.
 *
 * Keeping each guide narrow makes the tutorial useful after onboarding too:
 * users can replay only the part they need instead of stepping through the
 * entire application again.
 */
export type TutorialTrack = "quick" | "data" | "results" | "workspace";

const QUICK_ONLY: readonly TutorialTrack[] = ["quick"];
const QUICK_AND_RESULTS: readonly TutorialTrack[] = ["quick", "results"];
const DATA_ONLY: readonly TutorialTrack[] = ["data"];
const RESULTS_ONLY: readonly TutorialTrack[] = ["results"];
const WORKSPACE_ONLY: readonly TutorialTrack[] = ["workspace"];

export interface TutorialStep {
  id: string;
  title: string;
  description: string;
  target: string;
  placement?: "top" | "bottom" | "left" | "right" | "auto";
  /** Vertical nudge in pixels — positive moves the card down, negative up. Used to de-overlap dense sections. */
  offsetY?: number;
  /** Override the default popover height (260px) for steps that need more breathing room. */
  popoverHeight?: number;
  /** Override spotlight padding (default 8) for this step. */
  highlightPadding?: number;
  /** Override spotlight border radius (default 12) for this step. */
  highlightRadius?: number;
  beforeShow?: () => void | Promise<void>;
  /**
   * Best-effort UI cleanup fired when the step is left (PREV / NEXT / exit).
   * Use to undo sticky state the step set (e.g. selected rows, query strings,
   * optimizer choice) so traversal doesn't accumulate. Fire-and-forget —
   * the return value is not awaited.
   */
  afterHide?: () => void | Promise<void>;
  /** Focused guides that include this step. */
  tracks: readonly TutorialTrack[];
  readingTimeSec: number;
}

export interface TutorialTrackDefinition {
  id: TutorialTrack;
  name: string;
  description: string;
  icon: string;
  stepCount: number;
  /** Rounded wall-clock estimate, for setting expectations before starting. */
  estimatedMinutes: number;
  steps: TutorialStep[];
}

import {
  callTutorialHook,
  hasTutorialHook,
  setTutorialNavigating,
  queryTutorialHook,
  waitForHook,
} from "./bridge";
import { isGeneralistAgentEnabled } from "@/features/agent-panel";

function navigateTo(path: string) {
  // Prefer in-app client navigation via the tutorial-overlay hook.
  // Fall back to a full reload if no overlay is mounted (e.g. tests).
  if (hasTutorialHook("routerPush")) {
    callTutorialHook("routerPush", path);
  } else {
    setTutorialNavigating(true);
    window.location.href = path;
  }
}

/**
 * Wait for a selector to appear and have layout (non-zero rect) — handles
 * route + wizard + AnimatePresence transitions where the element exists
 * in the DOM but is still at 0×0 during the enter animation.
 */
function isElementVisible(selector: string): boolean {
  const el = document.querySelector(selector) as HTMLElement | null;
  if (!el) return false;
  const r = el.getBoundingClientRect();
  return r.width > 0 && r.height > 0;
}

function waitForElement(selector: string, timeoutMs = 5000): Promise<boolean> {
  return new Promise((resolve) => {
    if (isElementVisible(selector)) {
      resolve(true);
      return;
    }
    const start = Date.now();
    const check = () => {
      if (isElementVisible(selector)) {
        resolve(true);
        return;
      }
      if (Date.now() - start > timeoutMs) {
        resolve(false);
        return;
      }
      requestAnimationFrame(check);
    };
    requestAnimationFrame(check);
  });
}

/** Show a splash screen identical to the real submit animation */
function showSubmitSplash(): Promise<void> {
  callTutorialHook("showTutorialSplash");
  return new Promise((resolve) => setTimeout(resolve, 1500));
}

export function resetTutorialOneShotState(): void {
  // Reserved for future per-tour ephemeral flags. Currently a no-op:
  // the submit splash now keys off path transition (not a one-shot flag),
  // so nothing needs resetting between tour runs.
}

async function ensureDashboard() {
  if (window.location.pathname !== "/") {
    navigateTo("/");
    await waitForElement("[data-tutorial='dashboard-kpis']");
  }
  await waitForHook("setTab");
  // One commit tick so backward arrivals measure after React tab commit
  await new Promise<void>((r) => requestAnimationFrame(() => r()));
}

/** Inject demo jobs + analytics into dashboard for the tutorial — cached “real” data path */
function injectDemoDashboardData() {
  callTutorialHook("setDemoJobs", getCachedDemoDashboardJobs());
  callTutorialHook("setDemoAnalytics", getCachedDemoDashboardAnalytics());
}

async function ensureSubmit() {
  if (!window.location.pathname.startsWith("/submit")) {
    navigateTo("/submit");
    await waitForElement("[data-tutorial='wizard-stepper']");
  }
  await waitForHook("setWizardStep");
  await new Promise<void>((r) => requestAnimationFrame(() => r()));
}

async function ensureDemoDetail() {
  const path = `/optimizations/${DEMO_OPTIMIZATION_ID}`;
  if (window.location.pathname === path) {
    await waitForHook("setDetailTab");
    await new Promise<void>((r) => requestAnimationFrame(() => r()));
    return;
  }
  navigateTo(path);
  await waitForElement("[data-tutorial='detail-header']");
  await waitForHook("setDetailTab");
  await new Promise<void>((r) => requestAnimationFrame(() => r()));
}

function setTab(tab: string) {
  callTutorialHook("setTab", tab);
}

function setWizardStep(step: number) {
  callTutorialHook("setWizardStep", step);
}

function setDetailTab(tab: string) {
  callTutorialHook("setDetailTab", tab);
}

function setOptimizerName(name: string) {
  callTutorialHook("setOptimizerName", name);
}

async function ensureTagger() {
  if (!window.location.pathname.startsWith("/tagger")) {
    navigateTo("/tagger");
    await waitForHook("setTaggerStartingNew");
    // Force the sessions chooser into setup; no-op if already in setup
    callTutorialHook("setTaggerStartingNew", true);
    await waitForElement("[data-tutorial='tagger-setup']");
  } else if (!hasTutorialHook("setTaggerStep")) {
    // Already on /tagger but still on the sessions panel (startingNew false)
    await waitForHook("setTaggerStartingNew");
    callTutorialHook("setTaggerStartingNew", true);
    await waitForElement("[data-tutorial='tagger-setup']");
  }
  await waitForHook("setTaggerStep");
}

async function ensureExplore() {
  if (window.location.pathname !== "/explore") {
    navigateTo("/explore");
    await waitForElement("[data-tutorial='explore-search']");
  }
  await waitForHook("setDemoExplorePoints");
  callTutorialHook("setDemoExplorePoints", getCachedDemoExplorePoints());
}

async function ensureDatasets() {
  if (window.location.pathname !== "/datasets") {
    navigateTo("/datasets");
    await waitForElement("[data-tutorial='datasets-library']");
  }
}

async function openSettingsTab(tab: string) {
  await ensureDashboard();
  await waitForHook("setSettingsTab");
  callTutorialHook("setSettingsTab", tab);
  await waitForElement(`[data-tutorial='settings-${tab}']`);
}

function closeSettings() {
  callTutorialHook("setSettingsTab", null);
}

function setGeneralistPanelOpen(open: boolean) {
  callTutorialHook("setGeneralistPanelOpen", open);
}

/** Inject demo data into tagger setup when empty and advance to the requested step */
function injectDemoTaggerData(targetStep: number) {
  if (!queryTutorialHook("hasTaggerData")) {
    const rows = [
      { id: 1, text: msg("auto.features.tutorial.lib.steps.literal.1") },
      { id: 2, text: msg("auto.features.tutorial.lib.steps.literal.2") },
      { id: 3, text: msg("auto.features.tutorial.lib.steps.literal.3") },
      { id: 4, text: msg("auto.features.tutorial.lib.steps.literal.4") },
      { id: 5, text: msg("auto.features.tutorial.lib.steps.literal.5") },
    ];
    callTutorialHook("setTaggerDemoData", {
      rows,
      cols: ["text"],
      textCol: "text",
    });
  }
  callTutorialHook("setTaggerStep", targetStep);
}

/** Inject sample dataset + code into the wizard for the tutorial */
function injectSampleDataset() {
  const rows = [
    { email_text: "Click here to win $1000 now!", category: "spam" },
    { email_text: "Meeting moved to 3pm tomorrow", category: "important" },
    { email_text: "50% off all items this weekend only", category: "promotional" },
    { email_text: "Your quarterly report is ready for review", category: "important" },
    { email_text: "Free gift card waiting for you", category: "spam" },
    { email_text: "Team standup notes from Monday", category: "important" },
  ];
  callTutorialHook("setParsedDataset", {
    columns: ["email_text", "category"],
    rows,
    rowCount: rows.length,
  });
  callTutorialHook("setColumnRoles", {
    email_text: "input",
    category: "output",
  });
  callTutorialHook("setDatasetFileName", "emails_sample.csv");
  callTutorialHook("setSignatureCode", DEMO_SIGNATURE_CODE);
  callTutorialHook("setMetricCode", DEMO_METRIC_CODE);
}

const tutorialSteps: TutorialStep[] = perLocale(() => [
  {
    id: "dd-dataset-library",
    title: msg("tutorial.step.dataset_library.title"),
    description: msg("tutorial.step.dataset_library.body"),
    target: "[data-tutorial='datasets-library']",
    placement: "bottom",
    beforeShow: ensureDatasets,
    tracks: DATA_ONLY,
    readingTimeSec: 10,
  },
  {
    id: "dd-tagger-setup",
    title: msg("auto.features.tutorial.lib.steps.literal.29"),
    description: msg("auto.features.tutorial.lib.steps.literal.30"),
    target: "[data-tutorial='tagger-setup']",
    placement: "auto",
    beforeShow: async () => {
      await ensureTagger();
      injectDemoTaggerData(0);
    },
    tracks: DATA_ONLY,
    readingTimeSec: 8,
  },
  {
    id: "dd-tagger-modes",
    title: msg("auto.features.tutorial.lib.steps.literal.31"),
    description: msg("auto.features.tutorial.lib.steps.literal.32"),
    target: "[data-tutorial='tagger-modes']",
    placement: "auto",
    beforeShow: async () => {
      await ensureTagger();
      injectDemoTaggerData(1);
      await waitForElement("[data-tutorial='tagger-modes']");
    },
    tracks: DATA_ONLY,
    readingTimeSec: 9,
  },
  {
    id: "dd-data-upload",
    title: formatMsg("auto.features.tutorial.lib.steps.template.16", { p1: TERMS.dataset }),
    description: `${formatMsg("auto.features.tutorial.lib.steps.template.17", {
      p1: TERMS.examplePlural,
      p2: TERMS.optimization,
    })} ${formatMsg("auto.features.tutorial.lib.steps.template.18", { p1: TERMS.model })}`,
    target: "[data-tutorial='wizard-step-2']",
    placement: "left",
    beforeShow: async () => {
      await ensureSubmit();
      injectSampleDataset();
      setWizardStep(1);
    },
    tracks: QUICK_ONLY,
    readingTimeSec: 7,
  },
  {
    id: "dd-code-setup",
    title: `${msg("auto.features.tutorial.lib.steps.literal.20")} + ${TERMS.metric}`,
    description: `${formatMsg("auto.features.tutorial.lib.steps.template.22", {
      p1: TERMS.model,
    })} ${formatMsg("auto.features.tutorial.lib.steps.template.23", {
      p1: TERMS.score,
      p2: TERMS.optimizer,
      p3: TERMS.score,
    })}`,
    target: "[data-tutorial='signature-editor']",
    placement: "top",
    beforeShow: async () => {
      await ensureSubmit();
      injectSampleDataset();
      setWizardStep(3);
      callTutorialHook("setCodeAssistMode", "manual");
      callTutorialHook("chooseModule", "predict");
      callTutorialHook("setSignatureCode", DEMO_SIGNATURE_CODE);
      callTutorialHook("setMetricCode", DEMO_METRIC_CODE);
      await waitForElement("[data-tutorial='signature-editor']");
    },
    tracks: QUICK_ONLY,
    readingTimeSec: 12,
  },

  {
    id: "dd-models",
    title: msg("auto.features.tutorial.lib.steps.template.24"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.25", {
      p1: TERMS.generationModel,
      p2: TERMS.reflectionModel,
      p3: TERMS.modelPlural,
    }),
    target: "[data-tutorial='model-catalog']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(4);
    },
    tracks: QUICK_ONLY,
    readingTimeSec: 7,
  },
  {
    id: "dd-review",
    title: msg("auto.features.tutorial.lib.steps.literal.21"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.26", {
      p1: TERMS.dataset,
      p2: TERMS.modelPlural,
      p3: TERMS.optimizer,
    }),
    target: "[data-tutorial='wizard-step-6']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureSubmit();
      setOptimizerName("gepa");
      setWizardStep(5);
    },
    tracks: QUICK_ONLY,
    readingTimeSec: 5,
  },
  {
    id: "dd-result-actions",
    title: msg("tutorial.step.result_actions.title"),
    description: msg("tutorial.step.result_actions.body"),
    target: "[data-tutorial='result-actions']",
    placement: "left",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("overview");
      await waitForElement("[data-tutorial='result-actions']");
    },
    tracks: RESULTS_ONLY,
    readingTimeSec: 9,
  },
  {
    id: "dd-pipeline",
    title: msg("auto.features.tutorial.lib.steps.literal.23"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.30", {
      p1: TERMS.optimization,
      p2: TERMS.baselineScore,
      p3: TERMS.optimization,
      p4: TERMS.optimization,
    }),
    target: "[data-tutorial='pipeline-stages']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("overview");
    },
    tracks: RESULTS_ONLY,
    readingTimeSec: 8,
  },
  {
    id: "dd-scores",
    title: formatMsg("auto.features.tutorial.lib.steps.template.31", { p1: TERMS.scorePlural }),
    description: formatMsg("auto.features.tutorial.lib.steps.template.32", {
      p1: TERMS.baselineScore,
      p2: TERMS.optimization,
      p3: TERMS.optimizedScore,
    }),
    target: "[data-tutorial='score-cards']",
    placement: "bottom",
    beforeShow: async () => {
      const onDetail = window.location.pathname === `/optimizations/${DEMO_OPTIMIZATION_ID}`;
      if (!onDetail) {
        resetDemoSimulation();
        await showSubmitSplash();
      }
      await ensureDemoDetail();
      setDetailTab("overview");
    },
    tracks: QUICK_ONLY,
    readingTimeSec: 5,
  },
  {
    id: "dd-trajectory",
    title: msg("auto.features.tutorial.lib.steps.literal.46"),
    description: msg("auto.features.tutorial.lib.steps.literal.48"),
    target: "[data-tutorial='trajectory-panel']",
    placement: "top",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("overview");
      // Re-stream the candidates so the user sees the tree grow instead of
      // landing on a completed graph with no explanation of its branches.
      callTutorialHook("replayDemoSimulation");
      await waitForElement("[data-tutorial='trajectory-panel']");
    },
    tracks: RESULTS_ONLY,
    readingTimeSec: 12,
  },
  {
    id: "dd-playground",
    title: msg("auto.features.tutorial.lib.steps.literal.25"),
    description: `${formatMsg("auto.features.tutorial.lib.steps.template.36", { p1: TERMS.model })} ${msg("auto.features.tutorial.lib.steps.literal.41")}`,
    target: "[data-tutorial='serve-playground']",
    placement: "bottom",
    offsetY: 0,
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("playground");
      await waitForElement("[data-tutorial='serve-playground']");
    },
    tracks: QUICK_ONLY,
    readingTimeSec: 12,
  },
  {
    id: "dd-code",
    title: msg("tutorial.step.code.title"),
    description: msg("tutorial.step.code.body"),
    target: "[data-tutorial='code-sources']",
    placement: "top",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("code");
      await waitForElement("[data-tutorial='code-sources']");
    },
    tracks: RESULTS_ONLY,
    readingTimeSec: 9,
  },
  {
    id: "dd-artifact",
    title: msg("tutorial.step.artifact.title"),
    description: msg("tutorial.step.artifact.body"),
    target: "[data-tutorial='artifact-output']",
    placement: "top",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("artifact");
      await waitForElement("[data-tutorial='artifact-output']");
    },
    tracks: QUICK_AND_RESULTS,
    readingTimeSec: 12,
  },
  {
    id: "dd-data-tab",
    title: msg("auto.features.tutorial.lib.steps.literal.24"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.35", {
      p1: TERMS.dataset,
      p2: TERMS.score,
      p3: TERMS.model,
      p4: TERMS.splitTrain,
      p5: TERMS.splitVal,
      p6: TERMS.splitTest,
      p7: TERMS.score,
    }),
    target: "[data-tutorial='data-table']",
    placement: "top",
    offsetY: 0,
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("data");
      await waitForElement("[data-tutorial='data-table']");
    },
    tracks: RESULTS_ONLY,
    readingTimeSec: 9,
  },
  {
    id: "dd-logs",
    title: msg("auto.features.tutorial.lib.steps.literal.26"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.37", { p1: TERMS.optimizer }),
    target: "[data-tutorial='live-logs']",
    placement: "top",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("logs");
      await waitForElement("[data-tutorial='live-logs']");
    },
    tracks: RESULTS_ONLY,
    readingTimeSec: 6,
  },
  {
    id: "dd-kpis",
    title: msg("auto.features.tutorial.lib.steps.literal.6"),
    description: msg("auto.features.tutorial.lib.steps.literal.7"),
    target: "[data-tutorial='dashboard-kpis']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureDashboard();
      injectDemoDashboardData();
      setTab("jobs");
    },
    tracks: WORKSPACE_ONLY,
    readingTimeSec: 4,
  },
  {
    id: "dd-table",
    title: formatMsg("auto.features.tutorial.lib.steps.template.1", {
      p1: TERMS.optimizationPlural,
    }),
    description: formatMsg("auto.features.tutorial.lib.steps.template.2", {
      p1: TERMS.optimization,
    }),
    target: "[data-tutorial='dashboard-table']",
    placement: "top",
    offsetY: 0,
    beforeShow: async () => {
      await ensureDashboard();
      injectDemoDashboardData();
      setTab("jobs");
      await waitForElement("[data-tutorial='dashboard-table']");
    },
    tracks: WORKSPACE_ONLY,
    readingTimeSec: 6,
  },
  {
    id: "dd-analytics",
    title: msg("auto.features.tutorial.lib.steps.literal.9"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.47", {
      p1: TERMS.scorePlural,
      p2: TERMS.optimization,
      p3: TERMS.optimization,
    }),
    target: "[data-tutorial='dashboard-stats']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureDashboard();
      injectDemoDashboardData();
      setTab("analytics");
      await waitForElement("[data-tutorial='dashboard-stats']");
      await new Promise((r) => setTimeout(r, 250));
    },
    tracks: WORKSPACE_ONLY,
    readingTimeSec: 5,
  },
  {
    id: "dd-explore",
    title: msg("auto.features.tutorial.lib.steps.literal.38"),
    description: msg("auto.features.tutorial.lib.steps.literal.49"),
    target: "[data-tutorial='explore-search']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureExplore();
    },
    tracks: WORKSPACE_ONLY,
    readingTimeSec: 14,
  },
  {
    id: "dd-agent-panel",
    title: msg("auto.features.tutorial.lib.steps.literal.44"),
    description: msg("auto.features.tutorial.lib.steps.literal.50"),
    target: "[data-tutorial='agent-panel']",
    placement: "left",
    beforeShow: async () => {
      await ensureDashboard();
      setGeneralistPanelOpen(true);
      await waitForElement("[data-tutorial='agent-panel']");
    },
    afterHide: () => {
      setGeneralistPanelOpen(false);
    },
    tracks: WORKSPACE_ONLY,
    readingTimeSec: 10,
  },
  {
    id: "dd-settings-billing",
    title: msg("tutorial.step.settings_billing.title"),
    description: msg("tutorial.step.settings_billing.body"),
    target: "[data-tutorial='settings-billing']",
    placement: "left",
    beforeShow: () => openSettingsTab("billing"),
    afterHide: closeSettings,
    tracks: WORKSPACE_ONLY,
    readingTimeSec: 11,
  },
  {
    id: "dd-settings-providers",
    title: msg("tutorial.step.settings_providers.title"),
    description: msg("tutorial.step.settings_providers.body"),
    target: "[data-tutorial='settings-providers']",
    placement: "left",
    beforeShow: () => openSettingsTab("providers"),
    afterHide: closeSettings,
    tracks: WORKSPACE_ONLY,
    readingTimeSec: 9,
  },
]);

const AGENT_PANEL_STEP_IDS = new Set(["dd-agent-panel"]);

function getVisibleSteps(): TutorialStep[] {
  const generalist = isGeneralistAgentEnabled();
  return tutorialSteps.filter((s) => {
    if (!generalist && AGENT_PANEL_STEP_IDS.has(s.id)) return false;
    return true;
  });
}

// Reading time alone undersells a step: the tour also navigates, waits for
// the target to paint, and gives the user a moment to look at it. Padding
// each step keeps the menu's estimate from reading as optimistic.
const STEP_OVERHEAD_SEC = 6;

export function getTrack(trackId: TutorialTrack): TutorialTrackDefinition | undefined {
  const steps = getVisibleSteps().filter((s) => s.tracks.includes(trackId));
  if (steps.length === 0) return undefined;
  const seconds = steps.reduce((sum, s) => sum + s.readingTimeSec + STEP_OVERHEAD_SEC, 0);
  const metadata: Record<TutorialTrack, { name: string; description: string }> = {
    quick: {
      name: msg("tutorial.track.quick.name"),
      description: msg("tutorial.track.quick.desc"),
    },
    data: {
      name: msg("tutorial.track.data.name"),
      description: msg("tutorial.track.data.desc"),
    },
    results: {
      name: msg("tutorial.track.results.name"),
      description: msg("tutorial.track.results.desc"),
    },
    workspace: {
      name: msg("tutorial.track.workspace.name"),
      description: msg("tutorial.track.workspace.desc"),
    },
  };
  return {
    id: trackId,
    name: metadata[trackId].name,
    description: metadata[trackId].description,
    icon: trackId,
    stepCount: steps.length,
    estimatedMinutes: Math.max(1, Math.round(seconds / 60)),
    steps,
  };
}
