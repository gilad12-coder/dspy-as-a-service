import type { ModelConfig } from "@/shared/types/api";

export type AnnotationMode = "binary" | "multiclass" | "freetext";

export interface Category {
  id: string;
  label: string;
}

export interface TaggerConfig {
  mode: AnnotationMode;
  inputColumns: string[];
  question?: string;
  categories?: Category[];
  prompt?: string;
  /**
   * Assisted sessions are created before any answer style is chosen; the
   * interview infers ``mode`` and stores it in the task override, which the
   * effective-config merge (client and server alike) applies over this
   * placeholder value.
   */
  modeProvisional?: boolean;
  /** Assist level chosen at setup; the session list projects it onto cards. */
  assistMode?: TaggerAssistMode;
  /** Display name of the dataset the session was created from. */
  sourceName?: string;
}

export interface DataField {
  column: string;
  value: unknown;
}

export interface DataRow {
  id: string | number;
  text: string;
  // Multi-column annotation: per-column raw values so the UI can render
  // each field with type-aware formatting (lists, objects) instead of
  // collapsing everything into a single JSON-flavoured string.
  fields?: DataField[];
  [key: string]: unknown;
}

export type Annotation = string | string[] | undefined;

/** Stored binary labels — "1" (yes) / "0" (no). */
export type BinaryLabel = "1" | "0";
export const BINARY_YES: BinaryLabel = "1";
export const BINARY_NO: BinaryLabel = "0";

/**
 * Whether an annotation is the binary positive/negative label. Sessions saved
 * before the 1/0 vocabulary stored "yes"/"no", so reads accept both forms.
 */
export function isBinaryYes(ann: Annotation): boolean {
  return ann === BINARY_YES || ann === "yes";
}

export function isBinaryNo(ann: Annotation): boolean {
  return ann === BINARY_NO || ann === "no";
}

/** Assist level chosen at setup. Manual sessions carry no assist state at all. */
export type TaggerAssistMode = "manual" | "copilot" | "autopilot";

/** Session phases. Manual sessions only ever use "setup" | "annotating". */
export type TaggerPhase =
  | "setup"
  | "interview"
  | "calibration"
  | "review"
  | "autotagging"
  | "complete"
  | "annotating";

/** Who produced a row's final label. */
export type AnnotationProvenance = "human" | "ai_confirmed" | "ai_auto";

export interface AssistPrediction {
  value: string | string[];
  /** Model-reported confidence in [0, 1]. */
  confidence: number;
  /** One-sentence rationale, shown in disagreement/review moments. */
  reason?: string;
}

export interface InterviewTurn {
  role: "assistant" | "user";
  content: string;
  /** LiteLLM model id that produced an assistant turn (the reply's chip). */
  model?: string | null;
  /** Concrete model the Auto Router picked for the turn, when resolved. */
  servedModel?: string | null;
}

/** One AI-tags-human-audits batch (review rounds and the flagged pass alike). */
export interface ReviewRound {
  rowIds: string[];
  /** Per-row outcome; a row absent here is not yet audited. */
  decided: Record<string, "confirmed" | "corrected">;
  /** Fraction of audited rows confirmed, fixed when the round closes. */
  agreement?: number;
  /** Marks the optional post-autotag pass over low-confidence rows. */
  flaggedPass?: boolean;
}

export interface AutotagProgress {
  status: "running" | "done" | "failed" | "canceled";
  total: number;
  done: number;
  /** Credits the bulk job actually spent (server-written, snake_case). */
  credits_spent?: number;
}

/**
 * AI co-tagging state, persisted as the session's ``assist`` JSON. Final
 * labels always live in ``annotations``; this only carries how they came to be
 * (predictions, provenance) and the collaboration bookkeeping.
 */
export interface AssistState {
  mode: Exclude<TaggerAssistMode, "manual">;
  /**
   * LiteLLM id of the model that tags rows — predictions, estimates and the
   * bulk job alike. Absent means the server's default tagging model.
   */
  model?: string;
  /**
   * Sampling parameters saved with the chosen model from the shared model
   * config dialog (temperature, max_tokens, top_p, extra.reasoning_effort).
   * Only ever present alongside ``model``; the server merges them into the
   * tagging LM the same way optimizations do.
   */
  modelParams?: Omit<ModelConfig, "name">;
  /** LiteLLM id of the model conducting the interview (the composer's model
   * menu). ``null`` is an explicit auto-router pick; absent (never picked)
   * follows the app-wide composer default. Distinct from ``model``, which is
   * the model that tags rows. */
  interviewModel?: string | null;
  /** Reasoning-effort level for ``interviewModel``; same null/absent split. */
  interviewEffort?: string | null;
  /**
   * Only sessions saved before AI-first calibration carry this; it steers the
   * legacy human-first calibration phase, which new sessions never enter.
   */
  calibrationStyle?: "blind" | "assisted";
  interview: { turns: InterviewTurn[]; done: boolean };
  /** The labeling rubric distilled from the interview; grows with corrections. */
  rubric: string[];
  /** Row ids sampled by the legacy calibration phase (pre-AI-first sessions). */
  calibrationIds: string[];
  predictions: Record<string, AssistPrediction>;
  provenance: Record<string, AnnotationProvenance>;
  rounds: ReviewRound[];
  autotag?: AutotagProgress;
  /**
   * Interview-derived task definition; config itself stays immutable. On
   * provisional-mode sessions this also carries the inferred answer style.
   */
  taskOverride?: Partial<Pick<TaggerConfig, "mode" | "question" | "categories" | "prompt">>;
}
