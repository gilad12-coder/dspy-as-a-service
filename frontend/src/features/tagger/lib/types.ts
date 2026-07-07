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

/** A submitted deep-optimize GEPA run being tracked from the tagger. */
export interface DeepOptimizeState {
  jobId: string;
  status: "running" | "success" | "failed";
  /** Held-out scores from the finished run (the lift the user earned). */
  baseline?: number | null;
  optimized?: number | null;
}

/**
 * AI co-tagging state, persisted as the session's ``assist`` JSON. Final
 * labels always live in ``annotations``; this only carries how they came to be
 * (predictions, provenance) and the collaboration bookkeeping.
 */
export interface AssistState {
  mode: Exclude<TaggerAssistMode, "manual">;
  calibrationStyle: "blind" | "assisted";
  interview: { turns: InterviewTurn[]; done: boolean };
  /** The labeling rubric distilled from the interview; grows with corrections. */
  rubric: string[];
  /** Row ids (String(row.id)) sampled for the calibration set. */
  calibrationIds: string[];
  predictions: Record<string, AssistPrediction>;
  provenance: Record<string, AnnotationProvenance>;
  rounds: ReviewRound[];
  autotag?: AutotagProgress;
  deepOptimize?: DeepOptimizeState;
  /** Interview refinements to the task text; config itself stays immutable. */
  taskOverride?: { question?: string; prompt?: string };
}
