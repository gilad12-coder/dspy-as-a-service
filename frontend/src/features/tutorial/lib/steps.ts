/**
 * Tutorial System — Step Definitions
 * Skynet prompt optimization platform
 *
 * One ordered walkthrough of every feature, published as two tracks: the
 * essentials, and the essentials plus everything else.
 * Works even for users with zero optimizations.
 */

import {
  resetDemoSimulation,
  DEMO_GRID_OPTIMIZATION_ID,
  DEMO_OPTIMIZATION_ID,
  getCachedDemoDashboardAnalytics,
  getCachedDemoDashboardJobs,
  getCachedDemoExplorePoints,
} from "./demo-data";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";

/**
 * The two ways through the tour.
 *
 * Users abandon a walkthrough at wildly different points, so the steps are
 * ordered by necessity rather than by screen: whenever someone quits, what
 * they already saw is the part that mattered most. "quick" is that ordering
 * cut down to the single path that gets a first result; "deep-dive" is the
 * same path with every remaining feature slotted in where it belongs.
 */
export type TutorialTrack = "quick" | "deep-dive";

/** Steps on the shortest path to a first result — shown in both tracks. */
const ESSENTIAL: readonly TutorialTrack[] = ["quick", "deep-dive"];
/** Steps that deepen or widen that path — the full tour only. */
const FULL_ONLY: readonly TutorialTrack[] = ["deep-dive"];

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
  /** Tracks this step belongs to — {@link ESSENTIAL} or {@link FULL_ONLY}. */
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

async function ensureGridDemo() {
  const path = `/optimizations/${DEMO_GRID_OPTIMIZATION_ID}`;
  if (window.location.pathname === path && !window.location.search.includes("pair=")) {
    await waitForHook("setDetailTab");
    await new Promise<void>((r) => requestAnimationFrame(() => r()));
    return;
  }
  navigateTo(path);
  await waitForElement("[data-tutorial='grid-search']");
  await waitForHook("setDetailTab");
  await new Promise<void>((r) => requestAnimationFrame(() => r()));
}

async function ensureGridPairDetail() {
  const path = `/optimizations/${DEMO_GRID_OPTIMIZATION_ID}`;
  const wantSearch = "?pair=0";
  const alreadyThere = window.location.pathname === path && window.location.search === wantSearch;
  if (alreadyThere) return;
  navigateTo(`${path}${wantSearch}`);
  await waitForElement("[data-tutorial='pair-detail']");
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

function setGeneralistPanelOpen(open: boolean) {
  callTutorialHook("setGeneralistPanelOpen", open);
}

/**
 * Slide the off-canvas sidebar drawer into view before a step spotlights it.
 * Below 768px the sidebar is translated off-screen, so its target rect would
 * sit off the viewport edge and the spotlight would highlight nothing. Open
 * the drawer and let its 300ms transform settle before the overlay measures.
 * On desktop the sidebar is permanently docked, so we skip the work entirely.
 */
async function revealSidebarDrawer() {
  if (typeof window === "undefined") return;
  if (!window.matchMedia("(max-width: 767.98px)").matches) return;
  callTutorialHook("setSidebarOpen", true);
  await new Promise((r) => setTimeout(r, 340));
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
  callTutorialHook(
    "setSignatureCode",
    `class EmailClassifier(dspy.Signature):
    """Classify an email into a category: spam, important, or promotional."""

    # inputs
    email_text: str = dspy.InputField(desc="The email content to classify")

    # outputs
    category: str = dspy.OutputField(desc="One of: spam, important, promotional")
`,
  );
  callTutorialHook(
    "setMetricCode",
    `def metric(example: dspy.Example, prediction: dspy.Prediction, trace: bool = None) -> float:
    return float(example.category.strip().lower() == prediction.category.strip().lower())
`,
  );
}

const tutorialSteps: TutorialStep[] = perLocale(() => [
  // 1 — Where a dataset comes from. The tour opens here because nothing
  // downstream is reachable without labelled data.

  {
    id: "dd-tagger-intro",
    title: msg("auto.features.tutorial.lib.steps.literal.28"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.43", { p1: TERMS.dataset }),
    target: "[data-tutorial='sidebar-data']",
    placement: "right",
    highlightPadding: 6,
    highlightRadius: 8,
    beforeShow: async () => {
      await ensureDashboard();
      await revealSidebarDrawer();
    },
    afterHide: () => {
      callTutorialHook("setSidebarOpen", false);
    },
    tracks: ESSENTIAL,
    readingTimeSec: 7,
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
    tracks: FULL_ONLY,
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
    tracks: FULL_ONLY,
    readingTimeSec: 9,
  },
  // 2 — Running an optimization. Follows the wizard's own step order, so
  // the tour never jumps forward and back through the stepper.

  {
    id: "dd-stepper",
    title: msg("auto.features.tutorial.lib.steps.literal.13"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.14", {
      p1: TERMS.dataset,
      p2: TERMS.model,
    }),
    target: "[data-tutorial='wizard-stepper']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(0);
    },
    tracks: ESSENTIAL,
    readingTimeSec: 4,
  },

  {
    id: "dd-basics",
    title: msg("auto.features.tutorial.lib.steps.literal.14"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.15", {
      p1: TERMS.optimization,
      p2: TERMS.optimizationTypeRun,
      p3: TERMS.optimizationTypeGrid,
    }),
    target: "[data-tutorial='wizard-step-1']",
    placement: "left",
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(0);
    },
    tracks: FULL_ONLY,
    readingTimeSec: 5,
  },

  {
    id: "dd-data-upload",
    title: formatMsg("auto.features.tutorial.lib.steps.template.16", { p1: TERMS.dataset }),
    description: formatMsg("auto.features.tutorial.lib.steps.template.17", {
      p1: TERMS.examplePlural,
      p2: TERMS.optimization,
    }),
    target: "[data-tutorial='wizard-step-2']",
    placement: "left",
    beforeShow: async () => {
      await ensureSubmit();
      injectSampleDataset();
      setWizardStep(1);
    },
    tracks: ESSENTIAL,
    readingTimeSec: 7,
  },
  {
    id: "dd-columns",
    title: msg("auto.features.tutorial.lib.steps.literal.15"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.18", { p1: TERMS.model }),
    target: "[data-tutorial='column-mapping']",
    placement: "top",
    offsetY: 0,
    beforeShow: async () => {
      await ensureSubmit();
      injectSampleDataset();
      setWizardStep(1);
    },
    tracks: ESSENTIAL,
    readingTimeSec: 7,
  },
  {
    id: "dd-splits",
    title: msg("auto.features.tutorial.lib.steps.literal.16"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.20", {
      p1: TERMS.splitTrain,
      p2: TERMS.optimizer,
      p3: TERMS.examplePlural,
      p4: TERMS.splitVal,
      p5: TERMS.examplePlural,
      p6: TERMS.splitTest,
      p7: TERMS.examplePlural,
      p8: TERMS.optimizer,
    }),
    target: "[data-tutorial='data-splits']",
    placement: "top",
    beforeShow: async () => {
      await ensureSubmit();
      callTutorialHook("setAdvancedMode", true);
      callTutorialHook("setAdvancedSectionsOpen", true);
      setWizardStep(2);
      await waitForElement("[data-tutorial='data-splits']");
    },
    tracks: FULL_ONLY,
    readingTimeSec: 10,
  },
  {
    id: "dd-auto-level",
    title: msg("auto.features.tutorial.lib.steps.literal.17"),
    description: msg("auto.features.tutorial.lib.steps.literal.18"),
    target: "[data-tutorial='auto-level']",
    placement: "top",
    offsetY: 0,
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(2);
    },
    tracks: FULL_ONLY,
    readingTimeSec: 8,
  },
  {
    id: "dd-gepa",
    title: msg("auto.features.tutorial.lib.steps.literal.19"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.21", {
      p1: TERMS.examplePlural,
      p2: TERMS.model,
    }),
    target: "[data-tutorial='gepa-params']",
    placement: "top",
    offsetY: 24,
    beforeShow: async () => {
      await ensureSubmit();
      callTutorialHook("setAdvancedMode", true);
      setOptimizerName("gepa");
      // The GEPA grid lives inside the collapsed optimizer disclosure.
      callTutorialHook("setAdvancedSectionsOpen", true);
      setWizardStep(2);
      await waitForElement("[data-tutorial='gepa-params']");
    },
    tracks: FULL_ONLY,
    readingTimeSec: 16,
  },

  {
    id: "dd-module",
    title: TERMS.module,
    description: formatMsg("auto.features.tutorial.lib.steps.template.45", {
      p1: TERMS.model,
      p2: TERMS.model,
    }),
    target: "[data-tutorial='module-selector']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureSubmit();
      injectSampleDataset();
      setWizardStep(3);
      // Walking back into this step from the next one would otherwise find
      // the picker already answered and the carousel gone.
      callTutorialHook("reopenModulePicker");
      await waitForElement("[data-tutorial='module-selector']");
    },
    tracks: ESSENTIAL,
    readingTimeSec: 9,
  },

  {
    id: "dd-signature",
    title: msg("auto.features.tutorial.lib.steps.literal.20"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.22", { p1: TERMS.model }),
    target: "[data-tutorial='signature-editor']",
    placement: "top",
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(3);
      // The editors only exist once a module is committed, so the tour makes
      // the pick the previous step just demonstrated.
      callTutorialHook("chooseModule", "predict");
      await waitForElement("[data-tutorial='signature-editor']");
    },
    tracks: ESSENTIAL,
    readingTimeSec: 11,
  },
  {
    id: "dd-metric",
    title: TERMS.metric,
    description: formatMsg("auto.features.tutorial.lib.steps.template.23", {
      p1: TERMS.score,
      p2: TERMS.optimizer,
      p3: TERMS.score,
    }),
    target: "[data-tutorial='metric-editor']",
    placement: "top",
    offsetY: 0,
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(3);
      callTutorialHook("chooseModule", "predict");
      await waitForElement("[data-tutorial='metric-editor']");
    },
    tracks: ESSENTIAL,
    readingTimeSec: 8,
  },

  {
    id: "dd-models",
    title: formatMsg("auto.features.tutorial.lib.steps.template.24", { p1: TERMS.modelPlural }),
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
    tracks: ESSENTIAL,
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
    tracks: FULL_ONLY,
    readingTimeSec: 5,
  },
  {
    id: "dd-submit",
    title: formatMsg("auto.features.tutorial.lib.steps.template.27", { p1: TERMS.optimization }),
    description: formatMsg("auto.features.tutorial.lib.steps.template.28", {
      p1: TERMS.baselineScore,
      p2: TERMS.optimizer,
    }),
    target: "[data-tutorial='submit-button']",
    placement: "top",
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(5);
    },
    tracks: ESSENTIAL,
    readingTimeSec: 8,
  },
  // 3 — Reading the result. Score first: it is the payoff the previous
  // twelve steps were building toward.

  {
    id: "dd-detail-header",
    title: msg("auto.features.tutorial.lib.steps.literal.22"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.29", {
      p1: TERMS.optimization,
      p2: TERMS.optimization,
      p3: TERMS.optimization,
    }),
    target: "[data-tutorial='detail-header']",
    placement: "bottom",
    offsetY: 0,
    beforeShow: async () => {
      const onDetail = window.location.pathname === `/optimizations/${DEMO_OPTIMIZATION_ID}`;
      // Splash plays whenever we cross into /detail from another route
      // (typically /submit). When the user is already on /detail (e.g.
      // PREV-then-NEXT cycling within the detail tabs), the splash is
      // suppressed so the morph isn't gratuitously replayed.
      if (!onDetail) {
        resetDemoSimulation();
        await showSubmitSplash();
      }
      await ensureDemoDetail();
      setDetailTab("overview");
      await waitForElement("[data-tutorial='detail-header']");
    },
    tracks: ESSENTIAL,
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
    tracks: FULL_ONLY,
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
      await ensureDemoDetail();
      setDetailTab("overview");
    },
    tracks: ESSENTIAL,
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
      // Re-stream the candidates so the user sees the tree grow live (with
      // the GEPA TQDM bar visible in the pipeline) instead of jumping to
      // the completed graph. Shown before the score chart because the tree
      // sits vertically above the chart in the layout — narrative follows
      // visual order.
      callTutorialHook("replayDemoSimulation");
      await waitForElement("[data-tutorial='trajectory-panel']");
    },
    tracks: ESSENTIAL,
    readingTimeSec: 12,
  },
  {
    id: "dd-score-chart",
    title: formatMsg("auto.features.tutorial.lib.steps.template.33", { p1: TERMS.scorePlural }),
    description: formatMsg("auto.features.tutorial.lib.steps.template.34", {
      p1: TERMS.score,
      p2: TERMS.optimizer,
      p3: TERMS.score,
    }),
    target: "[data-tutorial='score-chart']",
    placement: "top",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("overview");
    },
    tracks: FULL_ONLY,
    readingTimeSec: 7,
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
    tracks: ESSENTIAL,
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
    tracks: FULL_ONLY,
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
    tracks: FULL_ONLY,
    readingTimeSec: 6,
  },
  {
    id: "dd-lm-activity",
    title: msg("auto.features.tutorial.lib.steps.literal.52"),
    description: msg("auto.features.tutorial.lib.steps.literal.53"),
    target: "[data-tutorial='lm-activity']",
    placement: "auto",
    offsetY: 0,
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("lm-activity");
      await waitForElement("[data-tutorial='lm-activity']");
    },
    tracks: FULL_ONLY,
    readingTimeSec: 11,
  },
  {
    id: "dd-config",
    title: msg("auto.features.tutorial.lib.steps.literal.27"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.38", {
      p1: TERMS.modelPlural,
    }),
    target: "[data-tutorial='config-summary']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("config");
      await waitForElement("[data-tutorial='config-summary']");
    },
    tracks: FULL_ONLY,
    readingTimeSec: 7,
  },
  // 4 — Grid search: many runs at once.

  {
    id: "dd-grid-overview",
    title: formatMsg("auto.features.tutorial.lib.steps.template.39", { p1: TERMS.modelPlural }),
    description: formatMsg("auto.features.tutorial.lib.steps.template.40", {
      p1: TERMS.model,
      p2: TERMS.pairPlural,
      p3: TERMS.generationModel,
      p4: TERMS.reflectionModel,
      p5: TERMS.task,
      p6: TERMS.pair,
      p7: TERMS.score,
    }),
    target: "[data-tutorial='grid-pair-list']",
    placement: "top",
    offsetY: 0,
    beforeShow: async () => {
      await ensureGridDemo();
      setDetailTab("overview");
      await waitForElement("[data-tutorial='grid-pair-list']");
    },
    tracks: FULL_ONLY,
    readingTimeSec: 14,
  },
  {
    id: "dd-grid-pair",
    title: formatMsg("auto.features.tutorial.lib.steps.template.41", {
      p1: TERMS.pair,
      p2: TERMS.modelPlural,
    }),
    description: formatMsg("auto.features.tutorial.lib.steps.template.42", {
      p1: TERMS.pair,
      p2: TERMS.scorePlural,
      p3: TERMS.modelPlural,
      p4: TERMS.pair,
      p5: TERMS.model,
    }),
    target: "[data-tutorial='pair-detail-summary']",
    placement: "auto",
    beforeShow: async () => {
      await ensureGridPairDetail();
      await waitForElement("[data-tutorial='pair-detail-summary']");
    },
    tracks: FULL_ONLY,
    readingTimeSec: 11,
  },
  // 6 — The dashboard, which is where all of the above accumulates.
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
    tracks: FULL_ONLY,
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
    tracks: FULL_ONLY,
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
    tracks: FULL_ONLY,
    readingTimeSec: 5,
  },
  {
    id: "dd-sidebar",
    title: msg("auto.features.tutorial.lib.steps.literal.8"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.3", {
      p1: TERMS.optimization,
      p2: TERMS.optimization,
    }),
    target: "[data-tutorial='sidebar-full']",
    placement: "auto",
    beforeShow: async () => {
      await ensureDashboard();
      injectDemoDashboardData();
      setTab("jobs");
      await revealSidebarDrawer();
    },
    afterHide: () => {
      callTutorialHook("setSidebarOpen", false);
    },
    tracks: FULL_ONLY,
    readingTimeSec: 7,
  },
  // 7 — Side tools, then the sign-off.
  {
    id: "dd-explore",
    title: msg("auto.features.tutorial.lib.steps.literal.38"),
    description: msg("auto.features.tutorial.lib.steps.literal.49"),
    target: "[data-tutorial='explore-search']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureExplore();
    },
    tracks: FULL_ONLY,
    readingTimeSec: 14,
  },
  // Agent panel: only the chat window. The pill alone is just the
  // floating button and is now covered by the chat step's intro.
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
    tracks: FULL_ONLY,
    readingTimeSec: 10,
  },
  {
    id: "dd-done",
    title: msg("auto.features.tutorial.lib.steps.literal.33"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.44", {
      p1: TERMS.optimization,
      p2: TERMS.scorePlural,
    }),
    target: "[data-tutorial='sidebar-full']",
    placement: "auto",
    beforeShow: async () => {
      await ensureDashboard();
      await revealSidebarDrawer();
    },
    afterHide: () => {
      callTutorialHook("setSidebarOpen", false);
    },
    tracks: ESSENTIAL,
    readingTimeSec: 5,
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
  const quick = trackId === "quick";
  return {
    id: trackId,
    name: quick ? msg("tutorial.track.quick.name") : msg("tutorial.track.full.name"),
    description: quick ? msg("tutorial.track.quick.desc") : msg("tutorial.track.full.desc"),
    icon: trackId,
    stepCount: steps.length,
    estimatedMinutes: Math.max(1, Math.round(seconds / 60)),
    steps,
  };
}
