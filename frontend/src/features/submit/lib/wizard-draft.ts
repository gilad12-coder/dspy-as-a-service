import type { ModelConfig, SplitFractions, WorkflowSpec } from "@/shared/types/api";
import type { ParsedDataset } from "@/shared/lib/parse-dataset";
import { LOCALE_RELOAD_EVENT } from "@/shared/lib/locale";
import type { ReactConfig, ColumnRole } from "../constants";

/**
 * In-memory draft of the new-optimization wizard.
 *
 * The wizard lives in the `/submit` route subtree, which Next unmounts the moment
 * the user navigates to another sidebar destination — losing every in-progress
 * field. This module parks the latest snapshot in a module-level singleton
 * (deliberately NOT localStorage) so it survives client-side navigation without
 * serializing the potentially multi-MB parsed dataset or risking a storage-quota
 * throw. The timestamp expires the draft after `DRAFT_TTL_MS`, so a long-abandoned
 * form starts fresh; it is intentionally not durable across a hard refresh. The
 * one exception is the locale-switch reload, which stashes the draft through
 * sessionStorage for that single hop (see LOCALE_RELOAD_EVENT below).
 */
export interface WizardDraftData {
  step: number;
  furthestReachedStep: number;
  summaryTab: number;
  summaryCodeTab: string;
  jobType: "run" | "grid_search";
  isPrivate: boolean;
  jobName: string;
  jobDescription: string;
  moduleName: string;
  moduleChosen: boolean;
  optimizerName: string;
  reactConfig: ReactConfig;
  workflowSpec?: WorkflowSpec | null;
  signatureCode: string;
  metricCode: string;
  signatureManuallyEdited: boolean;
  metricManuallyEdited: boolean;
  parsedDataset: ParsedDataset | null;
  datasetFileName: string | null;
  columnRoles: Record<string, ColumnRole>;
  columnKinds: Record<string, "text" | "image">;
  modelConfig: ModelConfig;
  secondModelConfig: ModelConfig | null;
  generationModels: ModelConfig[];
  reflectionModels: ModelConfig[];
  split: SplitFractions;
  seed: number | undefined;
  autoLevel: string;
  reflectionMinibatchSize: string;
  maxFullEvals: string;
  // Optional: drafts saved before the explicit metric-call budget existed.
  maxMetricCalls?: string;
  useMerge: boolean;
  targetScore: string;
  // Optional: drafts saved before PxN sampling existed carry neither field.
  pxnParents?: string;
  pxnProposals?: string;
  shuffle: boolean;
  maxCostCredits: number | null;
}

// A half-filled draft survives sidebar navigation for this long, then self-expires
// so a much-later return starts clean. 30 minutes: long enough to go check a
// dataset or another run and come back, short enough not to resurrect stale work.
const DRAFT_TTL_MS = 30 * 60 * 1000;

let draft: { savedAt: number; data: WizardDraftData } | null = null;

// A locale switch is implemented as a full page reload (see LocaleProvider),
// which would wipe the singleton even though the user is mid-form. Right
// before that reload the provider fires LOCALE_RELOAD_EVENT; the freshest
// draft is stashed in sessionStorage for that single hop and re-adopted (and
// cleared) on the next load. Ordinary hard refreshes stay non-durable.
const RELOAD_STASH_KEY = "skynet.wizard-draft.reload-stash";

/** Stash a draft in sessionStorage so it survives the locale-switch reload. */
export function stashWizardDraftForReload(data: WizardDraftData): void {
  try {
    window.sessionStorage.setItem(RELOAD_STASH_KEY, JSON.stringify({ savedAt: Date.now(), data }));
  } catch {
    // Best-effort: a multi-MB parsed dataset can exceed the quota, in which
    // case the switch loses the draft exactly as it did before this stash.
  }
}

if (typeof window !== "undefined") {
  // Covers the wizard-unmounted case (a draft parked by an earlier nav-away);
  // a mounted wizard stashes its own live state via the same helper, and its
  // later-registered listener overwrites this write with fresher data.
  window.addEventListener(LOCALE_RELOAD_EVENT, () => {
    if (draft) stashWizardDraftForReload(draft.data);
  });
  try {
    const raw = window.sessionStorage.getItem(RELOAD_STASH_KEY);
    if (raw) {
      window.sessionStorage.removeItem(RELOAD_STASH_KEY);
      const parsed = JSON.parse(raw) as { savedAt: number; data: WizardDraftData };
      if (parsed?.data && typeof parsed.savedAt === "number") draft = parsed;
    }
  } catch {
    // Corrupt or inaccessible stash — start clean.
  }
}

/** Park the latest wizard snapshot, stamping it so the TTL can expire it later. */
export function saveWizardDraft(data: WizardDraftData): void {
  draft = { savedAt: Date.now(), data };
}

/** Return the parked draft if still within the TTL window, else null (and evict). */
export function readWizardDraft(): WizardDraftData | null {
  if (!draft) return null;
  if (Date.now() - draft.savedAt > DRAFT_TTL_MS) {
    draft = null;
    return null;
  }
  return draft.data;
}

/** Drop any parked draft — after a successful submit, or an explicit reset. */
export function clearWizardDraft(): void {
  draft = null;
}
