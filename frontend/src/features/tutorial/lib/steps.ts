/**
 * Tutorial System — Step Definitions
 * Skynet prompt optimization platform
 *
 * Single comprehensive tutorial covering every feature.
 * Works even for users with zero optimizations.
 */

import {
  resetDemoSimulation,
  DEMO_DASHBOARD_JOBS,
  DEMO_DASHBOARD_ANALYTICS,
  DEMO_COMPARE_JOBS,
  DEMO_COMPARE_IDS,
  DEMO_COMPARE_EXAMPLES,
  DEMO_COMPARE_DATASET,
  DEMO_EXPLORE_POINTS,
  DEMO_GRID_OPTIMIZATION_ID,
  DEMO_OPTIMIZATION_ID,
} from "./demo-data";
import { TERMS } from "@/shared/lib/terms";
import { formatMsg, msg } from "@/shared/lib/messages";
import { perLocale } from "@/shared/lib/per-locale";

export type TutorialTrack = "deep-dive";

export interface TutorialStep {
  id: string;
  title: string;
  description: string;
  target: string;
  placement?: "top" | "bottom" | "left" | "right" | "auto";
  beforeShow?: () => void | Promise<void>;
  /**
   * Best-effort UI cleanup fired when the step is left (PREV / NEXT / exit).
   * Use to undo sticky state the step set (e.g. selected rows, query strings,
   * optimizer choice) so traversal doesn't accumulate. Fire-and-forget —
   * the return value is not awaited.
   */
  afterHide?: () => void | Promise<void>;
  track: TutorialTrack;
  readingTimeSec: number;
}

export interface TutorialTrackDefinition {
  id: TutorialTrack;
  name: string;
  description: string;
  icon: string;
  stepCount: number;
  steps: TutorialStep[];
}

import {
  callTutorialHook,
  hasTutorialHook,
  setTutorialNavigating,
  queryTutorialHook,
  setPendingCompareDemo,
  setPendingCompareExamples,
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
 * Wait for a selector to appear in the DOM (up to timeoutMs).
 * Resolves true when the element appears, false on timeout — so callers
 * can branch on "element really arrived" vs "we gave up waiting".
 */
function waitForElement(selector: string, timeoutMs = 5000): Promise<boolean> {
  return new Promise((resolve) => {
    if (document.querySelector(selector)) {
      resolve(true);
      return;
    }
    const start = Date.now();
    const check = () => {
      if (document.querySelector(selector)) {
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
  // The dashboard's tab/demo hooks live in a useEffect that fires AFTER
  // the JSX commits — waitForElement can resolve before they're registered.
  await waitForHook("setTab");
}

/** Inject demo jobs + analytics into dashboard for the tutorial */
function injectDemoDashboardData() {
  callTutorialHook("setDemoJobs", DEMO_DASHBOARD_JOBS);
  callTutorialHook("setDemoAnalytics", DEMO_DASHBOARD_ANALYTICS);
}

async function ensureSubmit() {
  if (!window.location.pathname.startsWith("/submit")) {
    navigateTo("/submit");
    await waitForElement("[data-tutorial='wizard-stepper']");
  }
  // SubmitWizard registers setWizardStep in a useEffect that runs after
  // the stepper paints — waitForElement can resolve before that effect.
  await waitForHook("setWizardStep");
}

async function ensureDemoDetail() {
  const path = `/optimizations/${DEMO_OPTIMIZATION_ID}`;
  if (window.location.pathname === path) {
    await waitForHook("setDetailTab");
    return;
  }
  navigateTo(path);
  await waitForElement("[data-tutorial='detail-header']");
  await waitForHook("setDetailTab");
}

async function ensureCompareDemo() {
  const query = `?jobs=${DEMO_COMPARE_IDS.join(",")}`;
  const alreadyThere = window.location.pathname === "/compare" && window.location.search === query;
  if (alreadyThere) {
    await waitForHook("setCompareTab");
    return;
  }
  setPendingCompareDemo(DEMO_COMPARE_JOBS);
  setPendingCompareExamples({
    byJobId: DEMO_COMPARE_EXAMPLES,
    dataset: DEMO_COMPARE_DATASET,
  });
  navigateTo(`/compare${query}`);
  await waitForElement("[data-tutorial='compare-verdict']");
  await waitForHook("setCompareTab");
}

async function ensureGridDemo() {
  const path = `/optimizations/${DEMO_GRID_OPTIMIZATION_ID}`;
  if (window.location.pathname === path && !window.location.search.includes("pair=")) {
    await waitForHook("setDetailTab");
    return;
  }
  navigateTo(path);
  await waitForElement("[data-tutorial='grid-search']");
  await waitForHook("setDetailTab");
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

function setCompareTab(tab: string) {
  callTutorialHook("setCompareTab", tab);
}

function setOptimizerName(name: string) {
  callTutorialHook("setOptimizerName", name);
}

async function ensureTagger() {
  if (!window.location.pathname.startsWith("/tagger")) {
    navigateTo("/tagger");
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
  callTutorialHook("setDemoExplorePoints", DEMO_EXPLORE_POINTS);
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
    track: "deep-dive",
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
    beforeShow: async () => {
      await ensureDashboard();
      injectDemoDashboardData();
      setTab("jobs");
    },
    track: "deep-dive",
    readingTimeSec: 6,
  },
  {
    id: "dd-sidebar",
    title: msg("auto.features.tutorial.lib.steps.literal.8"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.3", {
      p1: TERMS.optimization,
      p2: TERMS.optimization,
    }),
    target: "[data-tutorial='sidebar-full']",
    placement: "left",
    beforeShow: async () => {
      await ensureDashboard();
      injectDemoDashboardData();
      setTab("jobs");
      await revealSidebarDrawer();
    },
    afterHide: () => {
      callTutorialHook("setSidebarOpen", false);
    },
    track: "deep-dive",
    readingTimeSec: 7,
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
    placement: "top",
    beforeShow: async () => {
      await ensureDashboard();
      injectDemoDashboardData();
      setTab("analytics");
      await waitForElement("[data-tutorial='dashboard-stats']");
      await new Promise((r) => setTimeout(r, 250));
    },
    track: "deep-dive",
    readingTimeSec: 5,
  },

  {
    id: "dd-compare-trigger",
    title: formatMsg("auto.features.tutorial.lib.steps.template.5", {
      p1: TERMS.optimizationPlural,
    }),
    description: msg("auto.features.tutorial.lib.steps.literal.10"),
    target: "[data-tutorial='compare-button']",
    placement: "top",
    beforeShow: async () => {
      await ensureDashboard();
      injectDemoDashboardData();
      setTab("jobs");
      callTutorialHook("setSelectedJobIds", ["demo-001", "demo-002"]);
      await waitForElement("[data-tutorial='compare-button']");
      await new Promise((r) => setTimeout(r, 450));
    },
    afterHide: () => {
      // Drop the demo selection so dashboard steps revisited via PREV
      // don't show the compare-button still hovering with rows pre-checked.
      callTutorialHook("setSelectedJobIds", []);
    },
    track: "deep-dive",
    readingTimeSec: 8,
  },
  {
    id: "dd-compare-verdict",
    title: formatMsg("auto.features.tutorial.lib.steps.template.6", {
      p1: TERMS.optimizationPlural,
    }),
    description: formatMsg("auto.features.tutorial.lib.steps.template.7", {
      p1: TERMS.score,
      p2: TERMS.model,
    }),
    target: "[data-tutorial='compare-verdict']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureCompareDemo();
      setCompareTab("overview");
    },
    track: "deep-dive",
    readingTimeSec: 8,
  },
  {
    id: "dd-compare-scores",
    title: formatMsg("auto.features.tutorial.lib.steps.template.8", { p1: TERMS.scorePlural }),
    description: formatMsg("auto.features.tutorial.lib.steps.template.46", {
      p1: TERMS.finalScore,
      p2: TERMS.baselineScore,
      p3: TERMS.finalScore,
    }),
    target: "[data-tutorial='compare-scores']",
    placement: "top",
    beforeShow: async () => {
      await ensureCompareDemo();
      setCompareTab("overview");
      await waitForElement("[data-tutorial='compare-scores']");
    },
    track: "deep-dive",
    readingTimeSec: 8,
  },
  {
    id: "dd-compare-config",
    title: msg("auto.features.tutorial.lib.steps.literal.11"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.10", {
      p1: TERMS.module,
      p2: TERMS.optimizer,
      p3: TERMS.modelPlural,
      p4: TERMS.dataset,
    }),
    target: "[data-tutorial='compare-config']",
    placement: "top",
    beforeShow: async () => {
      await ensureCompareDemo();
      setCompareTab("config");
      await waitForElement("[data-tutorial='compare-config']");
    },
    track: "deep-dive",
    readingTimeSec: 9,
  },
  {
    id: "dd-compare-prompts",
    title: msg("auto.features.tutorial.lib.steps.literal.12"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.11", {
      p1: TERMS.examplePlural,
      p2: TERMS.model,
    }),
    target: "[data-tutorial='compare-prompts']",
    placement: "top",
    beforeShow: async () => {
      await ensureCompareDemo();
      setCompareTab("prompts");
      await waitForElement("[data-tutorial='compare-prompts']");
    },
    track: "deep-dive",
    readingTimeSec: 9,
  },
  {
    id: "dd-compare-examples",
    title: formatMsg("auto.features.tutorial.lib.steps.template.12", { p1: TERMS.examplePlural }),
    description: formatMsg("auto.features.tutorial.lib.steps.template.13", {
      p1: TERMS.examplePlural,
      p2: TERMS.score,
      p3: TERMS.examplePlural,
    }),
    target: "[data-tutorial='compare-examples']",
    placement: "top",
    beforeShow: async () => {
      await ensureCompareDemo();
      setCompareTab("examples");
      await waitForElement("[data-tutorial='compare-examples']");
    },
    track: "deep-dive",
    readingTimeSec: 9,
  },

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
    track: "deep-dive",
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
    placement: "right",
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(0);
    },
    track: "deep-dive",
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
    placement: "right",
    beforeShow: async () => {
      await ensureSubmit();
      injectSampleDataset();
      setWizardStep(1);
    },
    track: "deep-dive",
    readingTimeSec: 7,
  },
  {
    id: "dd-columns",
    title: msg("auto.features.tutorial.lib.steps.literal.15"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.18", { p1: TERMS.model }),
    target: "[data-tutorial='column-mapping']",
    placement: "top",
    beforeShow: async () => {
      await ensureSubmit();
      injectSampleDataset();
      setWizardStep(1);
    },
    track: "deep-dive",
    readingTimeSec: 7,
  },

  {
    id: "dd-module",
    title: TERMS.module,
    description: formatMsg("auto.features.tutorial.lib.steps.template.45", {
      p1: TERMS.model,
      p2: TERMS.model,
    }),
    target: "[data-tutorial='module-selector']",
    placement: "auto",
    beforeShow: async () => {
      await ensureSubmit();
      injectSampleDataset();
      setWizardStep(3);
    },
    track: "deep-dive",
    readingTimeSec: 9,
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
      setWizardStep(2);
    },
    track: "deep-dive",
    readingTimeSec: 10,
  },
  {
    id: "dd-auto-level",
    title: msg("auto.features.tutorial.lib.steps.literal.17"),
    description: msg("auto.features.tutorial.lib.steps.literal.18"),
    target: "[data-tutorial='auto-level']",
    placement: "top",
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(2);
    },
    track: "deep-dive",
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
    beforeShow: async () => {
      await ensureSubmit();
      setOptimizerName("gepa");
      // The GEPA grid lives inside the collapsed optimizer disclosure.
      callTutorialHook("setAdvancedSectionsOpen", true);
      setWizardStep(2);
    },
    track: "deep-dive",
    readingTimeSec: 16,
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
    },
    track: "deep-dive",
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
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(3);
    },
    track: "deep-dive",
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
    placement: "top",
    beforeShow: async () => {
      await ensureSubmit();
      setWizardStep(4);
    },
    track: "deep-dive",
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
    placement: "right",
    beforeShow: async () => {
      await ensureSubmit();
      setOptimizerName("gepa");
      setWizardStep(5);
    },
    track: "deep-dive",
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
    track: "deep-dive",
    readingTimeSec: 8,
  },

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
    },
    track: "deep-dive",
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
    placement: "top",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("overview");
    },
    track: "deep-dive",
    readingTimeSec: 8,
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
      // the completed graph. Placed right after dd-pipeline so the
      // optimization stage is showcased end-to-end before we move on to
      // the baseline-vs-final score comparison.
      callTutorialHook("replayDemoSimulation");
      await waitForElement("[data-tutorial='trajectory-panel']");
    },
    track: "deep-dive",
    readingTimeSec: 12,
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
    placement: "top",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("overview");
    },
    track: "deep-dive",
    readingTimeSec: 5,
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
    track: "deep-dive",
    readingTimeSec: 7,
  },
  {
    id: "dd-playground",
    title: msg("auto.features.tutorial.lib.steps.literal.25"),
    description: `${formatMsg("auto.features.tutorial.lib.steps.template.36", { p1: TERMS.model })} ${msg("auto.features.tutorial.lib.steps.literal.41")}`,
    target: "[data-tutorial='serve-playground']",
    placement: "top",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("playground");
      await waitForElement("[data-tutorial='serve-playground']");
    },
    track: "deep-dive",
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
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("data");
      await waitForElement("[data-tutorial='data-table']");
    },
    track: "deep-dive",
    readingTimeSec: 9,
  },
  {
    id: "dd-logs",
    title: msg("auto.features.tutorial.lib.steps.literal.26"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.37", { p1: TERMS.optimizer }),
    target: "[data-tutorial='live-logs']",
    placement: "bottom",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("logs");
      await waitForElement("[data-tutorial='live-logs']");
    },
    track: "deep-dive",
    readingTimeSec: 6,
  },
  {
    id: "dd-lm-activity",
    title: msg("auto.features.tutorial.lib.steps.literal.52"),
    description: msg("auto.features.tutorial.lib.steps.literal.53"),
    target: "[data-tutorial='lm-activity']",
    placement: "top",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("lm-activity");
      await waitForElement("[data-tutorial='lm-activity']");
    },
    track: "deep-dive",
    readingTimeSec: 11,
  },
  {
    id: "dd-config",
    title: msg("auto.features.tutorial.lib.steps.literal.27"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.38", {
      p1: TERMS.modelPlural,
    }),
    target: "[data-tutorial='config-summary']",
    placement: "top",
    beforeShow: async () => {
      await ensureDemoDetail();
      setDetailTab("config");
      await waitForElement("[data-tutorial='config-summary']");
    },
    track: "deep-dive",
    readingTimeSec: 7,
  },

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
    beforeShow: async () => {
      await ensureGridDemo();
      setDetailTab("overview");
      await waitForElement("[data-tutorial='grid-pair-list']");
    },
    track: "deep-dive",
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
    track: "deep-dive",
    readingTimeSec: 11,
  },

  {
    id: "dd-tagger-intro",
    title: msg("auto.features.tutorial.lib.steps.literal.28"),
    description: formatMsg("auto.features.tutorial.lib.steps.template.43", { p1: TERMS.dataset }),
    target: "[data-tutorial='sidebar-data']",
    placement: "left",
    beforeShow: async () => {
      await ensureDashboard();
    },
    track: "deep-dive",
    readingTimeSec: 7,
  },
  {
    id: "dd-tagger-setup",
    title: msg("auto.features.tutorial.lib.steps.literal.29"),
    description: msg("auto.features.tutorial.lib.steps.literal.30"),
    target: "[data-tutorial='tagger-setup']",
    placement: "right",
    beforeShow: async () => {
      await ensureTagger();
      injectDemoTaggerData(0);
    },
    track: "deep-dive",
    readingTimeSec: 8,
  },
  {
    id: "dd-tagger-modes",
    title: msg("auto.features.tutorial.lib.steps.literal.31"),
    description: msg("auto.features.tutorial.lib.steps.literal.32"),
    target: "[data-tutorial='tagger-modes']",
    placement: "right",
    beforeShow: async () => {
      await ensureTagger();
      injectDemoTaggerData(1);
      await waitForElement("[data-tutorial='tagger-modes']");
    },
    track: "deep-dive",
    readingTimeSec: 9,
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
    track: "deep-dive",
    readingTimeSec: 14,
  },
  // Agent panel steps. Filtered out below when the generalist agent
  // feature flag is off so prod users without the panel don't get
  // dead spotlights.
  {
    id: "dd-agent-pill",
    title: msg("auto.features.tutorial.lib.steps.literal.42"),
    description: msg("auto.features.tutorial.lib.steps.literal.43"),
    target: "[data-tutorial='agent-pill']",
    placement: "top",
    beforeShow: async () => {
      await ensureDashboard();
      setGeneralistPanelOpen(false);
      await waitForElement("[data-tutorial='agent-pill']");
    },
    track: "deep-dive",
    readingTimeSec: 7,
  },
  {
    id: "dd-agent-panel",
    title: msg("auto.features.tutorial.lib.steps.literal.44"),
    description: msg("auto.features.tutorial.lib.steps.literal.50"),
    target: "[data-tutorial='agent-panel']",
    placement: "right",
    beforeShow: async () => {
      await ensureDashboard();
      setGeneralistPanelOpen(true);
      await waitForElement("[data-tutorial='agent-panel']");
    },
    afterHide: () => {
      setGeneralistPanelOpen(false);
    },
    track: "deep-dive",
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
    placement: "left",
    beforeShow: async () => {
      await ensureDashboard();
      await revealSidebarDrawer();
    },
    afterHide: () => {
      callTutorialHook("setSidebarOpen", false);
    },
    track: "deep-dive",
    readingTimeSec: 5,
  },
]);

const AGENT_PANEL_STEP_IDS = new Set(["dd-agent-pill", "dd-agent-panel"]);

function getVisibleSteps(): TutorialStep[] {
  const generalist = isGeneralistAgentEnabled();
  return tutorialSteps.filter((s) => {
    if (!generalist && AGENT_PANEL_STEP_IDS.has(s.id)) return false;
    return true;
  });
}

export function getTrack(trackId: TutorialTrack): TutorialTrackDefinition | undefined {
  if (trackId !== "deep-dive") return undefined;
  const steps = getVisibleSteps();
  return {
    id: "deep-dive",
    name: msg("auto.features.tutorial.lib.steps.literal.34"),
    description: msg("auto.features.tutorial.lib.steps.literal.35"),
    icon: "deep-dive",
    stepCount: steps.length,
    steps,
  };
}
