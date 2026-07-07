"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  createTaggerSession,
  updateTaggerSession,
  stashTaggerSession,
  getTaggerSession,
  taggerAssistInterview,
  taggerAssistPredict,
  taggerAssistOptimize,
  taggerAssistEstimate,
  taggerAssistAutotagStart,
  taggerAssistAutotagStatus,
  taggerAssistAutotagCancel,
  taggerAssistDeepOptimize,
  getOptimizationStatusLite,
  getOptimizationOptimizedPrompt,
  type TaggerSessionDetail,
} from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import { markRecentSession } from "@/shared/lib/recent-session";
import { getActiveLocale } from "@/shared/lib/runtime-locale";
import type {
  DataRow,
  Annotation,
  TaggerConfig,
  AnnotationMode,
  TaggerPhase,
  TaggerAssistMode,
  AssistState,
  AssistPrediction,
  ReviewRound,
} from "../lib/types";
import {
  REVIEW_BATCH_SIZE,
  agreementOver,
  calibrationTarget,
  flaggedRowIds,
  initialAssistState,
  labelsAgree,
  sampleRowIds,
} from "../lib/assist";

/** Window event the sidebar listens for to refresh its saved-session list. */
export const TAGGER_SESSIONS_CHANGED = "tagger-sessions-changed";

const AUTOSAVE_INTERVAL_MS = 60_000;
const AUTOTAG_POLL_MS = 2_500;
/** GEPA runs take minutes; a slow poll on the run's summary is plenty. */
const DEEP_OPTIMIZE_POLL_MS = 5_000;
const TERMINAL_RUN_STATUSES = new Set(["success", "failed", "cancelled", "paused"]);
/** Calibration predictions are prefetched this many rows ahead of the cursor. */
const PREDICT_AHEAD = 6;

function isTagged(ann: Annotation, mode: AnnotationMode): boolean {
  if (ann === undefined || ann === null) return false;
  if (mode === "multiclass") return Array.isArray(ann) && ann.length > 0;
  return typeof ann === "string" && ann !== "";
}

/** Default name for a freshly-created session, derived from its config. */
function deriveSessionName(config: TaggerConfig): string {
  if (config.mode === "binary" && config.question?.trim()) {
    return config.question.trim().slice(0, 80);
  }
  if (config.mode === "freetext" && config.prompt?.trim()) {
    return config.prompt.trim().slice(0, 80);
  }
  return msg("tagger.session.untitled");
}

/** The estimate returned before a bulk auto-tag run. */
export interface AutotagEstimate {
  rows: number;
  model: string;
  credits_low: number;
  credits_high: number;
}

/**
 * Tagger annotation state with server-side persistence.
 *
 * Pass ``initialSession`` to rehydrate a saved session (the ``/tagger/[id]``
 * restore route); omit it to start fresh at the setup wizard. Once a session
 * exists it is autosaved on a 60-second loop — plus an immediate flush
 * whenever you leave the page — so it survives reloads and follows the user
 * across devices, the way optimizations persist.
 *
 * Assist sessions (co-pilot / autopilot) extend the same machine with the
 * interview → calibration → review → autotag phases. All final labels live in
 * ``annotations`` regardless of who produced them; ``assist`` carries the AI
 * bookkeeping (rubric, predictions, provenance, rounds, bulk-job progress).
 * During the ``autotagging`` phase the server owns the row, so autosaving is
 * suspended and progress is polled instead.
 */
export function useTagger(initialSession?: TaggerSessionDetail | null) {
  const router = useRouter();
  const [phase, setPhase] = useState<TaggerPhase>(
    (initialSession?.phase as TaggerPhase | undefined) ?? "setup",
  );
  const [config, setConfig] = useState<TaggerConfig | null>(
    (initialSession?.config as TaggerConfig | undefined) ?? null,
  );
  const [data, setData] = useState<DataRow[]>(
    (initialSession?.data as DataRow[] | undefined) ?? [],
  );
  const [columns, setColumns] = useState<string[]>(initialSession?.columns ?? []);
  const [annotations, setAnnotations] = useState<Record<string, Annotation>>(
    (initialSession?.annotations as Record<string, Annotation> | undefined) ?? {},
  );
  const [assist, setAssist] = useState<AssistState | null>(
    (initialSession?.assist as AssistState | null | undefined) ?? null,
  );
  const [currentIndex, setCurrentIndex] = useState(initialSession?.current_index ?? 0);
  const [sessionId, setSessionId] = useState<string | null>(initialSession?.id ?? null);

  // Transient AI-call states — never persisted.
  const [interviewBusy, setInterviewBusy] = useState(false);
  const [assistError, setAssistError] = useState<string | null>(null);
  const [roundLoading, setRoundLoading] = useState(false);
  const [optimizeBusy, setOptimizeBusy] = useState(false);
  const [estimate, setEstimate] = useState<AutotagEstimate | null>(null);
  const [autotagStatus, setAutotagStatus] = useState<{
    status: string;
    total: number;
    done: number;
    credits_spent: number;
    live: boolean;
  } | null>(null);

  // Latest annotation progress not yet written to the server, or null when the
  // server is up to date. Buffered here so the 60s autosave loop (and the
  // leave-the-page flush) can batch many edits into one cheap PUT.
  const pendingRef = useRef<{
    annotations: Record<string, unknown>;
    assist?: Record<string, unknown>;
    current_index: number;
    phase: TaggerPhase;
  } | null>(null);

  // Always-current mirrors so async callbacks and the post-create handoff see
  // the freshest state without re-subscribing.
  const annotationsRef = useRef(annotations);
  const assistRef = useRef(assist);
  const currentIndexRef = useRef(currentIndex);
  const phaseRef = useRef(phase);
  useEffect(() => {
    annotationsRef.current = annotations;
    assistRef.current = assist;
    currentIndexRef.current = currentIndex;
    phaseRef.current = phase;
  }, [annotations, assist, currentIndex, phase]);

  const startAnnotating = useCallback(
    (
      cfg: TaggerConfig,
      rows: DataRow[],
      cols: string[],
      assistMode: TaggerAssistMode = "manual",
      calibrationStyle: "blind" | "assisted" = "blind",
    ) => {
      const startPhase: TaggerPhase = assistMode === "manual" ? "annotating" : "interview";
      const startAssist =
        assistMode === "manual" ? null : initialAssistState(assistMode, calibrationStyle);
      setConfig(cfg);
      setData(rows);
      setColumns(cols);
      setCurrentIndex(0);
      setAnnotations({});
      setAssist(startAssist);
      setAssistError(null);
      setPhase(startPhase);
      setSessionId(null);
      // Persist immediately so the session shows up in the sidebar and survives a
      // reload; the returned id then drives the progress autosaves. The dataset is
      // uploaded once here, not on every subsequent annotation.
      void createTaggerSession({
        name: deriveSessionName(cfg),
        phase: startPhase,
        config: cfg as unknown as Record<string, unknown>,
        columns: cols,
        data: rows as unknown as Array<Record<string, unknown>>,
        annotations: {},
        assist: startAssist as unknown as Record<string, unknown> | null,
        current_index: 0,
      })
        .then((detail) => {
          setSessionId(detail.id);
          window.dispatchEvent(new Event(TAGGER_SESSIONS_CHANGED));
          // Move to the session's own URL so leaving and returning (control panel,
          // browser back) rehydrates from the server instead of dropping back to
          // the wizard. Hand off the freshest local state — including any rows
          // tagged while the dataset uploaded — so the gate resumes without a
          // refetch. Skip the redirect if the user already navigated away.
          stashTaggerSession({
            ...detail,
            annotations: annotationsRef.current as Record<string, unknown>,
            assist: assistRef.current as unknown as Record<string, unknown> | null,
            phase: phaseRef.current,
            current_index: currentIndexRef.current,
          });
          const path = window.location.pathname;
          if (path === "/tagger" || path.startsWith("/tagger/")) {
            router.replace(`/tagger/${detail.id}`);
          }
        })
        .catch(() => {
          // Best-effort: a storage-quota 409 opens the shared modal centrally and
          // the session simply isn't persisted — manual annotation still works in
          // memory. Assist needs the server, so its calls surface their own error.
        });
    },
    [router],
  );

  const backToSetup = useCallback(() => {
    setConfig(null);
    setData([]);
    setColumns([]);
    setAnnotations({});
    setAssist(null);
    setAssistError(null);
    setCurrentIndex(0);
    setPhase("setup");
    // Detach from the saved session — it stays in the sidebar to resume later;
    // the next ``startAnnotating`` creates a new one.
    setSessionId(null);
  }, []);

  // Buffer each edit as pending rather than writing on every keystroke; the
  // autosave loop and the leave-the-page handler drain it. Only the mutable
  // fields are tracked (annotations / assist / cursor / phase) — never the
  // dataset. While the bulk job runs the server owns the row, so nothing is
  // buffered (the PUT would be rejected with 409 anyway).
  useEffect(() => {
    if (!sessionId || phase === "setup" || phase === "autotagging") return;
    pendingRef.current = {
      annotations: annotations as unknown as Record<string, unknown>,
      assist: (assist as unknown as Record<string, unknown>) ?? undefined,
      current_index: currentIndex,
      phase,
    };
  }, [sessionId, annotations, assist, currentIndex, phase]);

  // Write any buffered progress to the server. Best-effort: on failure the
  // payload is re-armed (unless a newer edit already replaced it) so the next
  // tick retries. Reads the ref, so it always sends the freshest state.
  const flush = useCallback(() => {
    if (!sessionId || !pendingRef.current) return;
    const payload = pendingRef.current;
    pendingRef.current = null;
    void updateTaggerSession(sessionId, payload).catch(() => {
      pendingRef.current = pendingRef.current ?? payload;
    });
  }, [sessionId]);

  // Awaited flush of the *current* state (not just the buffered delta) for the
  // moments the server must see fresh labels before an AI call: predictions
  // compile their few-shot examples server-side, and the bulk job snapshots
  // the row when it starts.
  const flushNow = useCallback(async () => {
    if (!sessionId) return;
    pendingRef.current = null;
    await updateTaggerSession(sessionId, {
      annotations: annotationsRef.current as unknown as Record<string, unknown>,
      assist: (assistRef.current as unknown as Record<string, unknown>) ?? undefined,
      current_index: currentIndexRef.current,
      phase: phaseRef.current,
    });
  }, [sessionId]);

  // Autosave progress every 60 seconds — and immediately whenever the user
  // leaves (tab hidden, page unload, or navigating back to the control panel,
  // which unmounts this hook) — so a returning user resumes where they left off
  // without a write on every annotation. At each of those points we also stamp
  // this session as recently visited, so the sidebar's Text-tagging button can
  // resume it when the user returns within the resume window.
  useEffect(() => {
    if (!sessionId) return;
    markRecentSession("tagger", sessionId);
    const interval = window.setInterval(() => {
      markRecentSession("tagger", sessionId);
      flush();
    }, AUTOSAVE_INTERVAL_MS);
    const onLeave = () => {
      markRecentSession("tagger", sessionId);
      flush();
    };
    const onHide = () => {
      if (document.visibilityState === "hidden") onLeave();
    };
    window.addEventListener("pagehide", onLeave);
    document.addEventListener("visibilitychange", onHide);
    return () => {
      window.clearInterval(interval);
      window.removeEventListener("pagehide", onLeave);
      document.removeEventListener("visibilitychange", onHide);
      onLeave();
    };
  }, [sessionId, flush]);

  /** Merge a partial update into the assist state (no-op without assist). */
  const patchAssist = useCallback((patch: Partial<AssistState>) => {
    setAssist((prev) => (prev ? { ...prev, ...patch } : prev));
  }, []);

  // ---------------------------------------------------------------- frames
  // In calibration/review the annotator surface works on a subset of rows (the
  // calibration set or the open round); everywhere else it sees the full
  // dataset. ``currentIndex`` is always an index into the active frame.

  const openRound: ReviewRound | null = useMemo(() => {
    const last = assist?.rounds[assist.rounds.length - 1];
    return last && last.agreement === undefined ? last : null;
  }, [assist]);

  const frameIds: string[] | null = useMemo(() => {
    if (phase === "calibration") return assist?.calibrationIds ?? null;
    if (phase === "review") return openRound?.rowIds ?? null;
    return null;
  }, [phase, assist, openRound]);

  const frameData: DataRow[] = useMemo(() => {
    if (!frameIds) return data;
    const byId = new Map(data.map((row) => [String(row.id), row]));
    return frameIds.map((id) => byId.get(id)).filter((row): row is DataRow => row !== undefined);
  }, [frameIds, data]);

  const navigate = useCallback(
    (dir: 1 | -1) => {
      (document.activeElement as HTMLElement)?.blur();
      setCurrentIndex((i) => {
        const next = i + dir;
        if (next < 0 || next >= frameData.length) return i;
        return next;
      });
    },
    [frameData.length],
  );

  const goTo = useCallback(
    (idx: number) => {
      (document.activeElement as HTMLElement)?.blur();
      if (idx >= 0 && idx < frameData.length) setCurrentIndex(idx);
    },
    [frameData.length],
  );

  const jumpToUntagged = useCallback(() => {
    if (!config) return;
    for (let i = 0; i < frameData.length; i++) {
      if (!isTagged(annotations[String(frameData[i]!.id)], config.mode)) {
        setCurrentIndex(i);
        return;
      }
    }
  }, [frameData, annotations, config]);

  const toggleBinary = useCallback((id: string, value: "yes" | "no") => {
    (document.activeElement as HTMLElement)?.blur();
    setAnnotations((prev) => {
      const next = { ...prev };
      if (next[id] === value) {
        delete next[id];
      } else {
        next[id] = value;
      }
      return next;
    });
  }, []);

  const toggleCategory = useCallback((id: string, catId: string) => {
    (document.activeElement as HTMLElement)?.blur();
    setAnnotations((prev) => {
      const next = { ...prev };
      const current = Array.isArray(next[id]) ? [...(next[id] as string[])] : [];
      const idx = current.indexOf(catId);
      if (idx >= 0) current.splice(idx, 1);
      else current.push(catId);
      if (current.length === 0) delete next[id];
      else next[id] = current;
      return next;
    });
  }, []);

  const setFreetext = useCallback((id: string, text: string) => {
    setAnnotations((prev) => {
      const next = { ...prev };
      if (text.trim()) next[id] = text.trim();
      else delete next[id];
      return next;
    });
  }, []);

  // In assist phases every human edit stamps provenance (and, during review,
  // the audited-row bookkeeping). Wrapped here — not in the base mutators — so
  // the manual tagger stays untouched.
  const stampHumanEdit = useCallback(
    (id: string) => {
      const state = assistRef.current;
      if (!state) return;
      // The mutators run first, so the ref already reflects the edit.
      const value = annotationsRef.current[id];
      setAssist((prev) => {
        if (!prev) return prev;
        const provenance = { ...prev.provenance };
        if (value === undefined) delete provenance[id];
        else provenance[id] = "human";
        let rounds = prev.rounds;
        const last = rounds[rounds.length - 1];
        if (
          phaseRef.current === "review" &&
          last &&
          last.agreement === undefined &&
          last.rowIds.includes(id)
        ) {
          const decided = { ...last.decided };
          if (value === undefined) {
            delete decided[id];
          } else {
            const predicted = prev.predictions[id]?.value as Annotation;
            decided[id] = labelsAgree(
              (config?.mode ?? "binary") as AnnotationMode,
              value,
              predicted,
            )
              ? "confirmed"
              : "corrected";
          }
          rounds = [...rounds.slice(0, -1), { ...last, decided }];
        }
        return { ...prev, provenance, rounds };
      });
    },
    [config],
  );

  const assistToggleBinary = useCallback(
    (id: string, value: "yes" | "no") => {
      toggleBinary(id, value);
      // annotationsRef updates in an effect; mirror the toggle's outcome now.
      annotationsRef.current = { ...annotationsRef.current };
      if (annotationsRef.current[id] === value) delete annotationsRef.current[id];
      else annotationsRef.current[id] = value;
      stampHumanEdit(id);
    },
    [toggleBinary, stampHumanEdit],
  );

  const assistToggleCategory = useCallback(
    (id: string, catId: string) => {
      toggleCategory(id, catId);
      const current = Array.isArray(annotationsRef.current[id])
        ? [...(annotationsRef.current[id] as string[])]
        : [];
      const idx = current.indexOf(catId);
      if (idx >= 0) current.splice(idx, 1);
      else current.push(catId);
      annotationsRef.current = { ...annotationsRef.current };
      if (current.length === 0) delete annotationsRef.current[id];
      else annotationsRef.current[id] = current;
      stampHumanEdit(id);
    },
    [toggleCategory, stampHumanEdit],
  );

  const assistSetFreetext = useCallback(
    (id: string, text: string) => {
      setFreetext(id, text);
      annotationsRef.current = { ...annotationsRef.current };
      if (text.trim()) annotationsRef.current[id] = text.trim();
      else delete annotationsRef.current[id];
      stampHumanEdit(id);
    },
    [setFreetext, stampHumanEdit],
  );

  /** Accept the AI's prediction as the row's final label (Enter / switch). */
  const acceptPrediction = useCallback(
    (id: string) => {
      const state = assistRef.current;
      const predicted = state?.predictions[id];
      if (!state || !predicted) return;
      setAnnotations((prev) => ({ ...prev, [id]: predicted.value as Annotation }));
      annotationsRef.current = { ...annotationsRef.current, [id]: predicted.value as Annotation };
      setAssist((prev) => {
        if (!prev) return prev;
        const provenance = { ...prev.provenance, [id]: "ai_confirmed" as const };
        let rounds = prev.rounds;
        const last = rounds[rounds.length - 1];
        if (
          phaseRef.current === "review" &&
          last &&
          last.agreement === undefined &&
          last.rowIds.includes(id)
        ) {
          rounds = [
            ...rounds.slice(0, -1),
            { ...last, decided: { ...last.decided, [id]: "confirmed" as const } },
          ];
        }
        return { ...prev, provenance, rounds };
      });
    },
    [],
  );

  // ------------------------------------------------------------- interview

  const [quickReplies, setQuickReplies] = useState<string[]>([]);

  const sendInterviewMessage = useCallback(
    async (content: string | null) => {
      const state = assistRef.current;
      if (!sessionId || !state || interviewBusy) return;
      const turns =
        content === null
          ? state.interview.turns
          : [...state.interview.turns, { role: "user" as const, content }];
      if (content !== null) {
        patchAssist({ interview: { ...state.interview, turns } });
      }
      setInterviewBusy(true);
      setAssistError(null);
      try {
        const reply = await taggerAssistInterview(sessionId, {
          turns,
          locale: getActiveLocale(),
        });
        const current = assistRef.current;
        if (!current) return;
        patchAssist({
          interview: {
            turns: [...turns, { role: "assistant" as const, content: reply.message }],
            done: reply.done,
          },
          ...(reply.done && reply.rubric.length > 0 ? { rubric: reply.rubric } : {}),
        });
        setQuickReplies(reply.done ? [] : reply.quick_replies);
      } catch {
        setAssistError("interview");
      } finally {
        setInterviewBusy(false);
      }
    },
    [sessionId, interviewBusy, patchAssist],
  );

  // Fire the opening interview question once the session exists server-side.
  useEffect(() => {
    if (phase !== "interview" || !sessionId) return;
    const state = assistRef.current;
    if (!state || state.interview.turns.length > 0 || state.interview.done) return;
    if (interviewBusy) return;
    void sendInterviewMessage(null);
  }, [phase, sessionId]);

  /** Confirm the (possibly edited) rubric and leave the interview. */
  const confirmRubric = useCallback(
    (rubric: string[]) => {
      const state = assistRef.current;
      if (!state || !config) return;
      if (state.mode === "copilot") {
        const ids = sampleRowIds(data, calibrationTarget(config));
        setAssist((prev) =>
          prev
            ? {
                ...prev,
                rubric,
                interview: { ...prev.interview, done: true },
                calibrationIds: ids,
              }
            : prev,
        );
        setCurrentIndex(0);
        setPhase("calibration");
      } else {
        setAssist((prev) =>
          prev ? { ...prev, rubric, interview: { ...prev.interview, done: true } } : prev,
        );
        setCurrentIndex(0);
        setPhase("review");
      }
    },
    [config, data],
  );

  /** Update the rubric in place (the rail's inline editor). */
  const setRubric = useCallback(
    (rubric: string[]) => patchAssist({ rubric }),
    [patchAssist],
  );

  // ------------------------------------------------------------ calibration

  // Silently prefetch predictions a few rows ahead of the cursor so the
  // post-commit reveal is instant. Human-first stays intact: predictions are
  // only revealed after the row is committed.
  const predictInFlight = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (phase !== "calibration" || !sessionId || !config) return;
    const state = assistRef.current;
    if (!state) return;
    const upcoming = (state.calibrationIds ?? [])
      .slice(Math.max(0, currentIndex), currentIndex + PREDICT_AHEAD)
      .filter((id) => !state.predictions[id] && !predictInFlight.current.has(id));
    if (upcoming.length === 0) return;
    for (const id of upcoming) predictInFlight.current.add(id);
    void (async () => {
      try {
        await flushNow();
        const res = await taggerAssistPredict(sessionId, upcoming);
        setAssist((prev) =>
          prev
            ? {
                ...prev,
                predictions: {
                  ...prev.predictions,
                  ...(res.predictions as Record<string, AssistPrediction>),
                },
              }
            : prev,
        );
        setAssistError(null);
      } catch {
        setAssistError("predict");
      } finally {
        for (const id of upcoming) predictInFlight.current.delete(id);
      }
    })();
  }, [phase, sessionId, config, currentIndex, annotations, flushNow]);

  const calibrationDone = useMemo(() => {
    if (phase !== "calibration" || !assist || !config) return false;
    return (
      assist.calibrationIds.length > 0 &&
      assist.calibrationIds.every((id) => isTagged(annotations[id], config.mode))
    );
  }, [phase, assist, config, annotations]);

  /** Leave calibration for the review stage (or straight to done when tiny). */
  const finishCalibration = useCallback(() => {
    if (!config) return;
    const untagged = data.filter((row) => !isTagged(annotations[String(row.id)], config.mode));
    setCurrentIndex(0);
    setPhase(untagged.length === 0 ? "complete" : "review");
  }, [config, data, annotations]);

  // ---------------------------------------------------------------- review

  /** Sample a fresh batch of untagged rows, predict them, and open a round. */
  const startReviewRound = useCallback(async () => {
    if (!sessionId || !config || roundLoading) return;
    const labeled = new Set(
      Object.entries(annotationsRef.current)
        .filter(([, v]) => isTagged(v, config.mode))
        .map(([id]) => id),
    );
    const ids = sampleRowIds(data, REVIEW_BATCH_SIZE, labeled);
    if (ids.length === 0) {
      setPhase("complete");
      return;
    }
    setRoundLoading(true);
    setAssistError(null);
    try {
      await flushNow();
      const res = await taggerAssistPredict(sessionId, ids);
      setAssist((prev) => {
        if (!prev) return prev;
        return {
          ...prev,
          predictions: {
            ...prev.predictions,
            ...(res.predictions as Record<string, AssistPrediction>),
          },
          rounds: [...prev.rounds, { rowIds: ids, decided: {} }],
        };
      });
      // Freetext audits by fixing the AI's extraction in place, so the round's
      // rows are prefilled; binary/multiclass confirm or override per keystroke.
      if (config.mode === "freetext") {
        setAnnotations((prev) => {
          const next = { ...prev };
          for (const id of ids) {
            const value = (res.predictions[id]?.value as Annotation) ?? undefined;
            if (value !== undefined && !isTagged(next[id], config.mode)) next[id] = value;
          }
          return next;
        });
      }
      setCurrentIndex(0);
    } catch {
      setAssistError("predict");
    } finally {
      setRoundLoading(false);
    }
  }, [sessionId, config, data, roundLoading, flushNow]);

  // Close the open round once every row is audited (binary/multiclass); the
  // freetext round closes through ``finishRound`` below.
  useEffect(() => {
    if (phase !== "review" || !assist || !config || config.mode === "freetext") return;
    const last = assist.rounds[assist.rounds.length - 1];
    if (!last || last.agreement !== undefined) return;
    if (!last.rowIds.every((id) => last.decided[id] !== undefined)) return;
    const agreement =
      agreementOver(config.mode, last.rowIds, annotations, assist.predictions) ?? 0;
    setAssist((prev) =>
      prev
        ? {
            ...prev,
            rounds: [...prev.rounds.slice(0, -1), { ...last, agreement }],
          }
        : prev,
    );
  }, [phase, assist, config, annotations]);

  /** Close the open round explicitly (freetext rounds and the flagged pass). */
  const finishRound = useCallback(() => {
    const state = assistRef.current;
    if (!state || !config) return;
    const last = state.rounds[state.rounds.length - 1];
    if (!last || last.agreement !== undefined) return;
    const agreement =
      agreementOver(config.mode, last.rowIds, annotationsRef.current, state.predictions) ?? 0;
    setAssist((prev) => {
      if (!prev) return prev;
      const current = prev.rounds[prev.rounds.length - 1]!;
      const decided = { ...current.decided };
      const provenance = { ...prev.provenance };
      for (const id of current.rowIds) {
        const final = annotationsRef.current[id];
        if (final === undefined) continue;
        const agrees = labelsAgree(config.mode, final, prev.predictions[id]?.value as Annotation);
        if (decided[id] === undefined) decided[id] = agrees ? "confirmed" : "corrected";
        // Untouched prefilled rows were approved by finishing the round.
        if (provenance[id] === undefined || provenance[id] === "ai_auto") {
          provenance[id] = agrees ? "ai_confirmed" : "human";
        }
      }
      return {
        ...prev,
        provenance,
        rounds: [...prev.rounds.slice(0, -1), { ...current, decided, agreement }],
      };
    });
    if (last.flaggedPass) setPhase("complete");
  }, [config]);

  /** Instant optimize: reflectively rewrite the rubric from all labels so far. */
  const runOptimize = useCallback(async () => {
    if (!sessionId || optimizeBusy) return;
    setOptimizeBusy(true);
    setAssistError(null);
    try {
      await flushNow();
      const res = await taggerAssistOptimize(sessionId, getActiveLocale());
      patchAssist({ rubric: res.rubric });
    } catch {
      setAssistError("optimize");
    } finally {
      setOptimizeBusy(false);
    }
  }, [sessionId, optimizeBusy, flushNow, patchAssist]);

  /** Deep optimize: submit a real GEPA run trained on the labels so far. */
  const startDeepOptimize = useCallback(async () => {
    const state = assistRef.current;
    if (!sessionId || state?.deepOptimize?.status === "running") return;
    setAssistError(null);
    try {
      await flushNow();
      const res = await taggerAssistDeepOptimize(sessionId);
      patchAssist({
        deepOptimize: { jobId: res.optimization_id, status: "running" },
      });
    } catch {
      setAssistError("deep_optimize");
    }
  }, [sessionId, flushNow, patchAssist]);

  // Track a running deep-optimize job; on success, pull the evolved
  // instructions out of the run artifact and make them the labeling guide.
  const deepOptimizeJobId = assist?.deepOptimize?.jobId;
  const deepOptimizeRunning = assist?.deepOptimize?.status === "running";
  useEffect(() => {
    if (!deepOptimizeRunning || !deepOptimizeJobId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const summary = await getOptimizationStatusLite(deepOptimizeJobId);
        if (cancelled || !TERMINAL_RUN_STATUSES.has(summary.status)) return;
        if (summary.status === "success") {
          const artifact = await getOptimizationOptimizedPrompt(deepOptimizeJobId);
          if (cancelled) return;
          const instructions =
            artifact.program_artifact?.optimized_prompt?.instructions?.trim();
          setAssist((prev) =>
            prev
              ? {
                  ...prev,
                  ...(instructions ? { rubric: [instructions] } : {}),
                  deepOptimize: {
                    jobId: deepOptimizeJobId,
                    status: "success",
                    baseline: summary.baseline_test_metric,
                    optimized: summary.optimized_test_metric,
                  },
                }
              : prev,
          );
        } else {
          patchAssist({ deepOptimize: { jobId: deepOptimizeJobId, status: "failed" } });
        }
      } catch {
        // Transient poll failures are retried on the next tick.
      }
    };
    void tick();
    const interval = window.setInterval(() => void tick(), DEEP_OPTIMIZE_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [deepOptimizeRunning, deepOptimizeJobId, patchAssist]);

  /** Fetch the credit estimate for tagging everything that is still unlabeled. */
  const fetchEstimate = useCallback(async () => {
    if (!sessionId) return;
    try {
      await flushNow();
      setEstimate(await taggerAssistEstimate(sessionId));
    } catch {
      setEstimate(null);
    }
  }, [sessionId, flushNow]);

  // ---------------------------------------------------------------- autotag

  /** Start the bulk job; the server owns the session row until it finishes. */
  const startAutotag = useCallback(async () => {
    if (!sessionId) return;
    setAssistError(null);
    try {
      await flushNow();
      await taggerAssistAutotagStart(sessionId);
      pendingRef.current = null;
      setPhase("autotagging");
    } catch {
      setAssistError("autotag");
    }
  }, [sessionId, flushNow]);

  const cancelAutotag = useCallback(async () => {
    if (!sessionId) return;
    try {
      await taggerAssistAutotagCancel(sessionId);
    } catch {
      // The poll below surfaces the terminal state either way.
    }
  }, [sessionId]);

  // Poll the bulk job while it runs; when it ends, re-pull the whole session —
  // the server wrote labels, provenance and the phase flip.
  useEffect(() => {
    if (phase !== "autotagging" || !sessionId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const status = await taggerAssistAutotagStatus(sessionId);
        if (cancelled) return;
        setAutotagStatus(status);
        if (status.status !== "running") {
          const detail = await getTaggerSession(sessionId);
          if (cancelled) return;
          setAnnotations((detail.annotations as Record<string, Annotation>) ?? {});
          setAssist((detail.assist as AssistState | null) ?? null);
          setPhase((detail.phase as TaggerPhase) ?? "complete");
          setCurrentIndex(0);
        }
      } catch {
        // Transient poll failures are retried on the next tick.
      }
    };
    void tick();
    const interval = window.setInterval(() => void tick(), AUTOTAG_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [phase, sessionId]);

  // ------------------------------------------------------------ flagged pass

  /** Open a review round over the auto-tagged rows the model was unsure about. */
  const startFlaggedPass = useCallback(() => {
    const state = assistRef.current;
    if (!state) return;
    const ids = flaggedRowIds(state);
    if (ids.length === 0) return;
    setAssist((prev) =>
      prev
        ? { ...prev, rounds: [...prev.rounds, { rowIds: ids, decided: {}, flaggedPass: true }] }
        : prev,
    );
    setCurrentIndex(0);
    setPhase("review");
  }, []);

  /** Leave the completion summary for the plain full-dataset annotator. */
  const browseAll = useCallback(() => {
    setCurrentIndex(0);
    setPhase("annotating");
  }, []);

  const taggedCount = useMemo(
    () =>
      config ? data.filter((d) => isTagged(annotations[String(d.id)], config.mode)).length : 0,
    [config, data, annotations],
  );

  const frameTaggedCount = useMemo(
    () =>
      config
        ? frameData.filter((d) => isTagged(annotations[String(d.id)], config.mode)).length
        : 0,
    [config, frameData, annotations],
  );

  return {
    phase,
    config,
    data,
    columns,
    annotations,
    assist,
    currentIndex,
    taggedCount,
    sessionId,
    // Frame-scoped views for the annotator surface.
    frameData,
    frameTaggedCount,
    openRound,
    calibrationDone,
    // Base flow.
    startAnnotating,
    backToSetup,
    navigate,
    goTo,
    jumpToUntagged,
    toggleBinary,
    toggleCategory,
    setFreetext,
    // Assist flow.
    interviewBusy,
    quickReplies,
    assistError,
    roundLoading,
    optimizeBusy,
    estimate,
    autotagStatus,
    sendInterviewMessage,
    confirmRubric,
    setRubric,
    assistToggleBinary,
    assistToggleCategory,
    assistSetFreetext,
    acceptPrediction,
    finishCalibration,
    startReviewRound,
    finishRound,
    runOptimize,
    startDeepOptimize,
    fetchEstimate,
    startAutotag,
    cancelAutotag,
    startFlaggedPass,
    browseAll,
  };
}
