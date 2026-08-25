import { readPref } from "@/features/settings";
import type { ModelConfig } from "@/shared/types/api";
import type {
  Annotation,
  AnnotationMode,
  AssistPrediction,
  AssistState,
  DataRow,
  ReviewRound,
  TaggerConfig,
} from "./types";

/** Rows per review round the human audits before the gate re-evaluates. */
export const REVIEW_BATCH_SIZE = 20;
/** Auto-tagged rows below this confidence are flagged for the optional pass. */
export const FLAG_CONFIDENCE = 0.75;
/** Freetext agreement threshold on the token-overlap similarity. */
const FREETEXT_MATCH = 0.85;

/** Agreement a closed round must reach before "tag the rest" unlocks. */
export function agreementGate(mode: AnnotationMode): number {
  return mode === "freetext" ? 0.85 : 0.92;
}

/**
 * Recommended calibration-set size (deliberately not user-overridable):
 * enough signal to compile a working tagger, small enough to stay a
 * ten-minute session.
 */
export function calibrationTarget(config: TaggerConfig): number {
  if (config.mode === "binary") return 24;
  if (config.mode === "multiclass") {
    return Math.min(60, Math.max(30, 6 * (config.categories?.length ?? 0)));
  }
  return 30;
}

/** Evenly-spread sample of ``count`` row ids, skipping ``exclude``. */
export function sampleRowIds(
  data: DataRow[],
  count: number,
  exclude?: ReadonlySet<string>,
): string[] {
  const pool = data.map((row) => String(row.id)).filter((id) => !exclude?.has(id));
  if (pool.length <= count) return pool;
  // Deterministic stride sampling: spreads picks across the dataset (upload
  // order often correlates with content) without a seeded RNG dependency.
  const step = pool.length / count;
  const picked: string[] = [];
  for (let i = 0; i < count; i++) picked.push(pool[Math.floor(i * step)]!);
  return picked;
}

function tokenSet(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .replace(/[^\p{L}\p{N}\s]/gu, " ")
      .split(/\s+/)
      .filter(Boolean),
  );
}

/** Dice coefficient over word tokens — cheap, language-agnostic similarity. */
function tokenSimilarity(a: string, b: string): number {
  const setA = tokenSet(a);
  const setB = tokenSet(b);
  if (setA.size === 0 && setB.size === 0) return 1;
  if (setA.size === 0 || setB.size === 0) return 0;
  let overlap = 0;
  for (const t of setA) if (setB.has(t)) overlap++;
  return (2 * overlap) / (setA.size + setB.size);
}

/**
 * Whether a final label agrees with the AI's prediction. Exact match for
 * binary, set equality for multiclass, fuzzy token overlap for freetext.
 */
export function labelsAgree(mode: AnnotationMode, final: Annotation, predicted: Annotation): boolean {
  if (final === undefined || predicted === undefined) return false;
  if (mode === "multiclass") {
    const a = Array.isArray(final) ? [...final].sort() : [];
    const b = Array.isArray(predicted) ? [...predicted].sort() : [];
    return a.length === b.length && a.every((v, i) => v === b[i]);
  }
  if (mode === "freetext") {
    return tokenSimilarity(String(final), String(predicted)) >= FREETEXT_MATCH;
  }
  // Binary: canonicalize so a legacy "yes"/"no" label agrees with a fresh
  // "1"/"0" prediction (sessions saved before the 1/0 vocabulary).
  return canonBinary(final) === canonBinary(predicted);
}

function canonBinary(value: Annotation): Annotation {
  if (value === "yes") return "1";
  if (value === "no") return "0";
  return value;
}

/**
 * Live agreement over the given row ids: rows that carry both a final label
 * and a prediction, scored with {@link labelsAgree}. Null until any row
 * qualifies.
 */
export function agreementOver(
  mode: AnnotationMode,
  rowIds: readonly string[],
  annotations: Record<string, Annotation>,
  predictions: Record<string, AssistPrediction>,
): number | null {
  let scored = 0;
  let agreed = 0;
  for (const id of rowIds) {
    const final = annotations[id];
    const predicted = predictions[id]?.value as Annotation;
    if (final === undefined || predicted === undefined) continue;
    scored++;
    if (labelsAgree(mode, final, predicted)) agreed++;
  }
  return scored === 0 ? null : agreed / scored;
}

/** Ids of auto-tagged rows the model was unsure about (the flagged pass). */
export function flaggedRowIds(assist: AssistState): string[] {
  return Object.entries(assist.provenance)
    .filter(
      ([id, source]) =>
        source === "ai_auto" && (assist.predictions[id]?.confidence ?? 1) < FLAG_CONFIDENCE,
    )
    .map(([id]) => id);
}

/** Closed rounds, most recent last (flagged passes excluded from gate math). */
function closedGateRounds(assist: AssistState): ReviewRound[] {
  return assist.rounds.filter((r) => !r.flaggedPass && r.agreement !== undefined);
}

/** Whether the agreement gate is open (last closed round passed it). */
export function gateUnlocked(config: TaggerConfig, assist: AssistState): boolean {
  const rounds = closedGateRounds(assist);
  const last = rounds[rounds.length - 1];
  return last !== undefined && (last.agreement ?? 0) >= agreementGate(config.mode);
}

/** Counts of final labels by provenance, for the completion summary. */
export function provenanceCounts(
  assist: AssistState,
  annotations: Record<string, Annotation>,
): { human: number; aiConfirmed: number; aiAuto: number } {
  const counts = { human: 0, aiConfirmed: 0, aiAuto: 0 };
  for (const [id, value] of Object.entries(annotations)) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value) && value.length === 0) continue;
    const source = assist.provenance[id] ?? "human";
    if (source === "ai_auto") counts.aiAuto++;
    else if (source === "ai_confirmed") counts.aiConfirmed++;
    else counts.human++;
  }
  return counts;
}

/** The assist's chosen tagging model, viewed as the shared ModelConfig shape. */
export function assistModelConfig(
  assist: Pick<AssistState, "model" | "modelParams">,
): ModelConfig {
  return { name: assist.model ?? "", ...assist.modelParams };
}

/**
 * Split a config saved by the shared model dialog back into the assist's
 * ``model`` + ``modelParams`` fields. An empty name clears both — removing
 * the chip resets to the server's default model with default parameters.
 */
export function assistModelPatch(
  config?: ModelConfig,
): Pick<AssistState, "model" | "modelParams"> {
  if (!config || !config.name.trim()) return { model: undefined, modelParams: undefined };
  const { name, ...params } = config;
  const hasParams = Object.values(params).some(
    (v) => v != null && (typeof v !== "object" || Object.keys(v).length > 0),
  );
  return { model: name.trim(), modelParams: hasParams ? params : undefined };
}

/**
 * The interview composer's effective model: an explicit per-session pick
 * (including ``null`` — the auto router) wins; a session that never picked
 * follows the app-wide composer default from settings.
 */
export function interviewComposerModel(assist: Pick<AssistState, "interviewModel">): string | null {
  return assist.interviewModel === undefined ? readPref("composerModel") : assist.interviewModel;
}

/** Effective reasoning effort for the interview composer; same fallback split. */
export function interviewComposerEffort(
  assist: Pick<AssistState, "interviewEffort">,
): string | null {
  return assist.interviewEffort === undefined ? readPref("composerEffort") : assist.interviewEffort;
}

/** A fresh assist state for a session starting in the interview phase. */
export function initialAssistState(
  mode: "copilot" | "autopilot",
  modelConfig?: ModelConfig,
): AssistState {
  return {
    mode,
    ...assistModelPatch(modelConfig),
    interview: { turns: [], done: false },
    rubric: [],
    predictions: {},
    provenance: {},
    rounds: [],
  };
}
