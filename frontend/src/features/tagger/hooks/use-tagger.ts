"use client";

import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import {
  createTaggerSession,
  updateTaggerSession,
  renameTaggerSession,
  stashTaggerSession,
  getTaggerSession,
  taggerAssistEstimate,
  taggerAssistAutotagStart,
  taggerAssistAutotagStatus,
  taggerAssistAutotagCancel,
  type TaggerSessionDetail,
  type InterviewOption,
} from "@/shared/lib/api";
import { msg } from "@/shared/lib/messages";
import {
  clearRecentSession,
  markRecentSession,
  refreshRecentSession,
} from "@/shared/lib/recent-session";
import { LOCALE_RELOAD_EVENT } from "@/shared/lib/locale";
import { getActiveLocale } from "@/shared/lib/runtime-locale";
import type { ModelConfig } from "@/shared/types/api";
import type { AgentThinking } from "@/shared/ui/agent";
import { streamInterviewTurn, streamPredictions } from "../lib/assist-stream";
import type {
  DataRow,
  Annotation,
  BinaryLabel,
  Category,
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
  assistModelPatch,
  calibrationTarget,
  flaggedRowIds,
  initialAssistState,
  interviewComposerEffort,
  interviewComposerModel,
  labelsAgree,
  sampleRowIds,
} from "../lib/assist";
import { markTaggerInterviewForLocaleReset } from "../lib/interview-locale-reset";

/** Window event the sidebar listens for to refresh its saved-session list. */
export const TAGGER_SESSIONS_CHANGED = "tagger-sessions-changed";

const AUTOSAVE_INTERVAL_MS = 60_000;
const AUTOTAG_POLL_MS = 2_500;

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
  if (config.mode === "freetext" && !config.modeProvisional && config.prompt?.trim()) {
    return config.prompt.trim().slice(0, 80);
  }
  const source = config.sourceName?.trim();
  if (source) return source.slice(0, 80);
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
 * interview → review → autotag phases. All final labels live in
 * ``annotations`` regardless of who produced them; ``assist`` carries the AI
 * bookkeeping (rubric, predictions, provenance, rounds, bulk-job progress).
 * During the ``autotagging`` phase the server owns the row, so autosaving is
 * suspended and progress is polled instead.
 */
export function useTagger(initialSession?: TaggerSessionDetail | null) {
  const router = useRouter();
  const [phase, setPhase] = useState<TaggerPhase>(() => {
    const saved = initialSession?.phase;
    // Sessions saved before AI-first calibration persist the retired
    // human-first "calibration" phase; land them on the review gate, where
    // their partial labels simply count as human-tagged rows.
    if (saved === "calibration") return "review";
    return (saved as TaggerPhase | undefined) ?? "setup";
  });
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
  // A shared-in viewer's session is read-only server-side: progress is never
  // buffered for autosave (the PUT would be rejected below editor) and row
  // navigation spans the full dataset rather than any active round frame.
  const readOnly = initialSession?.role === "viewer";

  // Transient AI-call states — never persisted.
  const [interviewBusy, setInterviewBusy] = useState(false);
  const [interviewStreamText, setInterviewStreamText] = useState("");
  const [interviewThinking, setInterviewThinking] = useState<AgentThinking | null>(null);
  const interviewAbortRef = useRef<AbortController | null>(null);
  const localeReloadingRef = useRef(false);
  const [assistError, setAssistError] = useState<string | null>(null);
  const [roundLoading, setRoundLoading] = useState(false);
  // An open round's predictions are still streaming in chunk by chunk; the
  // annotator shows a per-row "tagging…" hint for rows not yet predicted.
  const [roundPredicting, setRoundPredicting] = useState(false);
  // Autopilot contract confirmed, bulk job not yet started: held true until
  // the starter resolves so the between-rounds gate never renders on the way
  // out of the interview.
  const [contractStarting, setContractStarting] = useState(false);
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

  // Every consumer sees the interview's task refinements — including the
  // inferred answer style on provisional-mode sessions — merged over the
  // immutable stored config (the server applies the same merge on its side).
  const effectiveConfig = useMemo(() => {
    if (!config) return null;
    const override = assist?.taskOverride ?? {};
    const merged: TaggerConfig = {
      ...config,
      ...(override.mode ? { mode: override.mode } : {}),
      ...(override.question?.trim() ? { question: override.question.trim() } : {}),
      ...(override.categories && override.categories.length > 0
        ? { categories: override.categories }
        : {}),
      ...(override.prompt?.trim() ? { prompt: override.prompt.trim() } : {}),
    };
    if (override.mode) delete merged.modeProvisional;
    return merged;
  }, [config, assist?.taskOverride]);

  const startAnnotating = useCallback(
    (
      cfg: TaggerConfig,
      rows: DataRow[],
      cols: string[],
      assistMode: TaggerAssistMode = "manual",
      assistModel?: ModelConfig,
    ) => {
      const startPhase: TaggerPhase = assistMode === "manual" ? "annotating" : "interview";
      const startAssist =
        assistMode === "manual" ? null : initialAssistState(assistMode, assistModel);
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
    // Forget the resume mark and drop the /tagger/[id] URL so neither the
    // sidebar's resume window nor a reload reopens the session we just left.
    clearRecentSession("tagger");
    router.replace("/tagger");
  }, [router]);

  // Buffer each edit as pending rather than writing on every keystroke; the
  // autosave loop and the leave-the-page handler drain it. Only the mutable
  // fields are tracked (annotations / assist / cursor / phase) — never the
  // dataset. While the bulk job runs the server owns the row, so nothing is
  // buffered (the PUT would be rejected with 409 anyway).
  useEffect(() => {
    if (!sessionId || readOnly || phase === "setup" || phase === "autotagging") return;
    pendingRef.current = {
      annotations: annotations as unknown as Record<string, unknown>,
      assist: (assist as unknown as Record<string, unknown>) ?? undefined,
      current_index: currentIndex,
      phase,
    };
  }, [sessionId, readOnly, annotations, assist, currentIndex, phase]);

  useEffect(() => {
    if (!sessionId || readOnly || phase !== "interview") return;
    const onLocaleReload = () => {
      localeReloadingRef.current = true;
      markTaggerInterviewForLocaleReset(sessionId);
      interviewAbortRef.current?.abort();
    };
    window.addEventListener(LOCALE_RELOAD_EVENT, onLocaleReload);
    return () => window.removeEventListener(LOCALE_RELOAD_EVENT, onLocaleReload);
  }, [sessionId, readOnly, phase]);

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
      if (localeReloadingRef.current) return;
      // Refresh rather than set: a deliberate exit (back / start over) just
      // cleared the mark, and re-stamping it here would hand the sidebar back
      // the session the user explicitly left.
      refreshRecentSession("tagger", sessionId);
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

  // The ref is mirrored synchronously (not just via the effect) so a
  // flush-then-call sequence fired in the same tick — fetching a fresh
  // estimate right after a pick — already sends the new model.
  const setAssistModel = useCallback((config: ModelConfig) => {
    const patch = assistModelPatch(config);
    setAssist((prev) => (prev ? { ...prev, ...patch } : prev));
    if (assistRef.current) assistRef.current = { ...assistRef.current, ...patch };
  }, []);

  // The interviewer's model (the composer menu) — same flush semantics as
  // ``setAssistModel`` so the very next turn already runs on the new choice.
  // ``null`` (an explicit auto-router pick) is stored as-is: only a session
  // that never picked follows the app-wide composer default.
  const setInterviewModel = useCallback((model: string | null) => {
    const patch = { interviewModel: model };
    setAssist((prev) => (prev ? { ...prev, ...patch } : prev));
    if (assistRef.current) assistRef.current = { ...assistRef.current, ...patch };
  }, []);

  const setInterviewEffort = useCallback((effort: string | null) => {
    const patch = { interviewEffort: effort };
    setAssist((prev) => (prev ? { ...prev, ...patch } : prev));
    if (assistRef.current) assistRef.current = { ...assistRef.current, ...patch };
  }, []);

  // ---------------------------------------------------------------- frames
  // In review the annotator surface works on the open round's subset of rows;
  // everywhere else it sees the full dataset. ``currentIndex`` is always an
  // index into the active frame.

  const openRound: ReviewRound | null = useMemo(() => {
    const last = assist?.rounds[assist.rounds.length - 1];
    return last && last.agreement === undefined ? last : null;
  }, [assist]);

  const frameIds: string[] | null = useMemo(() => {
    if (readOnly) return null;
    if (phase === "review") return openRound?.rowIds ?? null;
    return null;
  }, [readOnly, phase, openRound]);

  const frameData: DataRow[] = useMemo(() => {
    if (!frameIds) return data;
    const byId = new Map(data.map((row) => [String(row.id), row]));
    return frameIds.map((id) => byId.get(id)).filter((row): row is DataRow => row !== undefined);
  }, [frameIds, data]);

  const normalizedCurrentIndex =
    frameData.length === 0
      ? 0
      : Math.min(Math.max(Math.trunc(currentIndex), 0), frameData.length - 1);

  useEffect(() => {
    currentIndexRef.current = normalizedCurrentIndex;
    if (currentIndex !== normalizedCurrentIndex) setCurrentIndex(normalizedCurrentIndex);
  }, [currentIndex, normalizedCurrentIndex]);

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
    if (!effectiveConfig) return;
    for (let i = 0; i < frameData.length; i++) {
      if (!isTagged(annotations[String(frameData[i]!.id)], effectiveConfig.mode)) {
        setCurrentIndex(i);
        return;
      }
    }
  }, [frameData, annotations, effectiveConfig]);

  const toggleBinary = useCallback((id: string, value: BinaryLabel) => {
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
      // Store the raw text: the textarea is controlled, so trimming here would
      // delete the trailing space the user just typed and make multi-word tags
      // impossible. Blank-only input still counts as untagged.
      if (text.trim()) next[id] = text;
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
              (effectiveConfig?.mode ?? "binary") as AnnotationMode,
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
    [effectiveConfig],
  );

  const assistToggleBinary = useCallback(
    (id: string, value: BinaryLabel) => {
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
  const acceptPrediction = useCallback((id: string) => {
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
  }, []);

  // ------------------------------------------------------------- interview

  const [interviewOptions, setInterviewOptions] = useState<InterviewOption[]>([]);
  // What's still generating server-side between the reply finishing and the
  // parsed turn arriving: answer choices, or — when the streamed ``done``
  // field says the turn is final — the labeling-guide contract.
  const [interviewPending, setInterviewPending] = useState<"options" | "contract" | null>(null);

  // Streams the turn over SSE with the generalist agent's event shapes, so
  // the interview gets the same live reply + thinking treatment as the agent
  // panel. ``truncateTo`` supports edit-and-resend: the transcript is cut to
  // that many turns before the new user message is appended.
  const sendInterviewMessage = useCallback(
    async (content: string | null, truncateTo?: number) => {
      const state = assistRef.current;
      // The abort ref is the re-entry guard: `interviewBusy` is stale closure
      // state during React's synchronous double effect-invocation (StrictMode
      // mounts), which used to fire two parallel opening turns whose second
      // result replaced the first "in a flash".
      if (!sessionId || !state || interviewBusy || interviewAbortRef.current) return;
      const base =
        truncateTo === undefined
          ? state.interview.turns
          : state.interview.turns.slice(0, truncateTo);
      const turns = content === null ? base : [...base, { role: "user" as const, content }];
      if (content !== null || truncateTo !== undefined) {
        patchAssist({ interview: { turns, done: false } });
      }
      setInterviewBusy(true);
      setAssistError(null);
      setInterviewStreamText("");
      setInterviewThinking(null);
      setInterviewOptions([]);
      setInterviewPending(null);
      const controller = new AbortController();
      interviewAbortRef.current = controller;
      try {
        await streamInterviewTurn(
          sessionId,
          {
            // Stored assistant turns carry display bookkeeping (model,
            // servedModel — possibly null) the request schema rejects; only
            // role/content go over the wire.
            turns: turns.map(({ role, content }) => ({ role, content })),
            locale: getActiveLocale(),
            model: interviewComposerModel(state) ?? undefined,
            reasoning_effort: interviewComposerEffort(state) ?? undefined,
          },
          {
            signal: controller.signal,
            onReasoningPatch: (chunk) =>
              setInterviewThinking((prev) => ({
                reasoning: (prev?.reasoning ?? "") + chunk,
                startedAt: prev?.startedAt ?? Date.now(),
                endedAt: null,
                streaming: true,
              })),
            onMessagePatch: (chunk) => {
              setInterviewThinking((prev) =>
                prev && prev.streaming ? { ...prev, streaming: false, endedAt: Date.now() } : prev,
              );
              setInterviewStreamText((text) => text + chunk);
            },
            onMessageEnd: () =>
              setInterviewPending((prev) => (prev === "contract" ? prev : "options")),
            onTurnHint: (final) => setInterviewPending(final ? "contract" : "options"),
            onMessageReset: () => {
              setInterviewStreamText("");
              setInterviewThinking(null);
              setInterviewPending(null);
            },
            onDone: (turn) => {
              patchAssist({
                interview: {
                  turns: [
                    ...turns,
                    {
                      role: "assistant" as const,
                      content: turn.message,
                      model: turn.model ?? null,
                      servedModel: turn.served_model ?? null,
                    },
                  ],
                  done: turn.done,
                },
                ...(turn.done && turn.rubric.length > 0 ? { rubric: turn.rubric } : {}),
                ...(turn.done && Object.keys(turn.taskOverride).length > 0
                  ? { taskOverride: turn.taskOverride }
                  : {}),
              });
              setInterviewOptions(turn.done ? [] : turn.options);
              // The interview names the session on its final turn; the user
              // hasn't had a rename affordance yet (the session was created
              // seconds ago), so applying it unconditionally is safe.
              if (turn.done && turn.title) {
                void renameTaggerSession(sessionId, turn.title)
                  .then(() => window.dispatchEvent(new Event(TAGGER_SESSIONS_CHANGED)))
                  .catch(() => {});
              }
            },
            onError: () => setAssistError("interview"),
          },
        );
      } finally {
        interviewAbortRef.current = null;
        setInterviewBusy(false);
        setInterviewStreamText("");
        setInterviewPending(null);
        setInterviewThinking((prev) =>
          prev ? { ...prev, streaming: false, endedAt: prev.endedAt ?? Date.now() } : prev,
        );
      }
    },
    [sessionId, interviewBusy, patchAssist],
  );

  /** Abort the in-flight interview turn (the composer's stop button). */
  const stopInterview = useCallback(() => {
    interviewAbortRef.current?.abort();
  }, []);

  // Fire the opening interview question once the session exists server-side.
  useEffect(() => {
    if (phase !== "interview" || !sessionId) return;
    const state = assistRef.current;
    if (!state || state.interview.turns.length > 0 || state.interview.done) return;
    if (interviewBusy) return;
    void sendInterviewMessage(null);
  }, [phase, sessionId]);

  /**
   * Confirm the task contract (answer style, artifacts, rubric) and leave the
   * interview. Both modes launch directly (once the confirmed state has
   * committed): copilot opens the first review batch — the AI tags it and the
   * human keeps or corrects each label — and autopilot starts the bulk job.
   * No interstitial screen re-asks what the launch button already promised.
   */
  const confirmRubric = useCallback(
    (
      rubric: string[],
      task?: { mode: AnnotationMode; question?: string; categories?: Category[] },
    ) => {
      const state = assistRef.current;
      if (!state || !effectiveConfig) return;
      // The confirmed contract replaces the interview's override wholesale so
      // artifacts from a discarded answer style don't linger.
      const override = task
        ? {
            mode: task.mode,
            ...(task.mode === "binary" && task.question?.trim()
              ? { question: task.question.trim() }
              : {}),
            ...(task.mode === "multiclass" && task.categories?.length
              ? { categories: task.categories }
              : {}),
            ...(task.mode === "freetext" && state.taskOverride?.prompt
              ? { prompt: state.taskOverride.prompt }
              : {}),
          }
        : state.taskOverride;
      const patch = task && override ? { taskOverride: override } : {};
      setAssist((prev) =>
        prev ? { ...prev, rubric, ...patch, interview: { ...prev.interview, done: true } } : prev,
      );
      setCurrentIndex(0);
      setPhase("review");
      setContractStarting(true);
    },
    [effectiveConfig],
  );

  /** Ask the interviewer to wrap up now and finish with its best-guess contract. */
  const skipInterview = useCallback(() => {
    void sendInterviewMessage(msg("tagger.assist.interview.skip_message"));
  }, [sendInterviewMessage]);

  /** Discard the transcript and restart the interview from its opening question. */
  const restartInterview = useCallback(() => {
    void sendInterviewMessage(null, 0);
  }, [sendInterviewMessage]);

  // ---------------------------------------------------------------- review

  /**
   * Sample a fresh batch of untagged rows, open the round immediately, and
   * stream its predictions row by row over SSE. Each suggestion appears the
   * moment the model writes it — the human starts auditing on the first row
   * while the rest are still generating; unpredicted rows show a "tagging…"
   * hint and fill in as their event lands.
   */
  const startReviewRound = useCallback(
    async (count: number = REVIEW_BATCH_SIZE) => {
      if (!sessionId || !effectiveConfig || roundLoading) return;
      const labeled = new Set(
        Object.entries(annotationsRef.current)
          .filter(([, v]) => isTagged(v, effectiveConfig.mode))
          .map(([id]) => id),
      );
      const ids = sampleRowIds(data, count, labeled);
      if (ids.length === 0) {
        setPhase("complete");
        return;
      }
      setRoundLoading(true);
      setAssistError(null);
      try {
        // Predictions compile their few-shot examples server-side, so the
        // server must see the freshest labels before the first chunk runs.
        await flushNow();
      } catch {
        setAssistError("predict");
        setRoundLoading(false);
        return;
      }
      const round: ReviewRound = { rowIds: ids, decided: {} };
      setAssist((prev) => (prev ? { ...prev, rounds: [...prev.rounds, round] } : prev));
      // Mirrored synchronously (like setAssistModel) so the streaming loop's
      // round-closed check below reads this round, not the previous one.
      if (assistRef.current) {
        assistRef.current = { ...assistRef.current, rounds: [...assistRef.current.rounds, round] };
      }
      setCurrentIndex(0);
      setRoundLoading(false);
      setRoundPredicting(true);
      const controller = new AbortController();
      const applyPrediction = (id: string, pred: AssistPrediction) => {
        // The round can close mid-stream (every row hand-tagged first);
        // stop spending on suggestions nobody will see.
        const rounds = assistRef.current?.rounds;
        if (!rounds || rounds[rounds.length - 1]?.agreement !== undefined) {
          controller.abort();
          return;
        }
        setAssist((prev) =>
          prev ? { ...prev, predictions: { ...prev.predictions, [id]: pred } } : prev,
        );
        // Freetext audits by fixing the AI's extraction in place — each row
        // is prefilled as its prediction arrives, never over a human edit;
        // binary/multiclass confirm or override per keystroke.
        if (effectiveConfig.mode === "freetext") {
          setAnnotations((prev) => {
            const value = pred.value as Annotation;
            if (value === undefined || isTagged(prev[id], effectiveConfig.mode)) return prev;
            return { ...prev, [id]: value };
          });
        }
      };
      try {
        await streamPredictions(sessionId, ids, {
          onPrediction: applyPrediction,
          onDone: (predictions) => {
            // The terminal map is authoritative — re-applying is a no-op for
            // rows already streamed and fills in any missed events.
            for (const [id, pred] of Object.entries(predictions)) applyPrediction(id, pred);
          },
          onError: () => setAssistError("predict"),
          signal: controller.signal,
        });
      } finally {
        setRoundPredicting(false);
      }
    },
    [sessionId, effectiveConfig, data, roundLoading, flushNow],
  );

  // Close the open round once every row is audited (binary/multiclass); the
  // freetext round closes through ``finishRound`` below.
  useEffect(() => {
    if (phase !== "review" || !assist || !effectiveConfig || effectiveConfig.mode === "freetext")
      return;
    const last = assist.rounds[assist.rounds.length - 1];
    if (!last || last.agreement !== undefined) return;
    if (!last.rowIds.every((id) => last.decided[id] !== undefined)) return;
    const agreement =
      agreementOver(effectiveConfig.mode, last.rowIds, annotations, assist.predictions) ?? 0;
    setAssist((prev) =>
      prev
        ? {
            ...prev,
            rounds: [...prev.rounds.slice(0, -1), { ...last, agreement }],
          }
        : prev,
    );
  }, [phase, assist, effectiveConfig, annotations]);

  /** Close the open round explicitly (freetext rounds and the flagged pass). */
  const finishRound = useCallback(() => {
    const state = assistRef.current;
    if (!state || !effectiveConfig) return;
    const last = state.rounds[state.rounds.length - 1];
    if (!last) return;
    if (last.agreement === undefined) {
      // Every row needs an explicit decision (confirm keystroke or edit) —
      // finishing must never silently approve rows the human hasn't audited.
      if (!last.rowIds.every((id) => last.decided[id] !== undefined)) return;
      const agreement =
        agreementOver(
          effectiveConfig.mode,
          last.rowIds,
          annotationsRef.current,
          state.predictions,
        ) ?? 0;
      setAssist((prev) =>
        prev
          ? {
              ...prev,
              rounds: [
                ...prev.rounds.slice(0, -1),
                { ...prev.rounds[prev.rounds.length - 1]!, agreement },
              ],
            }
          : prev,
      );
    }
    if (last.flaggedPass) setPhase("complete");
  }, [effectiveConfig]);

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

  // Start what the contract card promised: copilot's first review batch (the
  // AI tags it, the human audits) or autopilot's bulk job. Runs as an effect
  // rather than inside ``confirmRubric`` so the confirmed contract has
  // committed (and the state mirrors above are fresh) before the starter
  // flushes it to the server. On failure the flag clears and the
  // between-rounds gate takes over as the recovery surface, error included.
  const contractStartFired = useRef(false);
  useEffect(() => {
    if (!contractStarting || phase !== "review" || contractStartFired.current) return;
    contractStartFired.current = true;
    void (async () => {
      if (assistRef.current?.mode === "copilot") {
        await startReviewRound(effectiveConfig ? calibrationTarget(effectiveConfig) : undefined);
      } else {
        await startAutotag();
      }
      contractStartFired.current = false;
      setContractStarting(false);
    })();
  }, [contractStarting, phase]);

  const cancelAutotag = useCallback(async () => {
    if (!sessionId) return;
    try {
      await taggerAssistAutotagCancel(sessionId);
    } catch {
      // The poll below surfaces the terminal state either way.
    }
  }, [sessionId]);

  // Poll the bulk job while it runs; when it ends, re-pull the whole session —
  // the server wrote labels, provenance and the phase flip. While it runs,
  // fresh labels are pulled down whenever the done-count moves (the worker
  // persists each batch into the session row), so the live walkthrough shows
  // rows being tagged as it happens. Local autosave is suspended in this
  // phase, so adopting the server's annotations can't echo back a stale PUT.
  useEffect(() => {
    if (phase !== "autotagging" || !sessionId) return;
    let cancelled = false;
    let syncedDone = -1;
    let syncing = false;
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
        } else if (status.done !== syncedDone && !syncing) {
          syncing = true;
          try {
            const detail = await getTaggerSession(sessionId);
            if (!cancelled) {
              syncedDone = status.done;
              setAnnotations((detail.annotations as Record<string, Annotation>) ?? {});
            }
          } finally {
            syncing = false;
          }
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

  /** Drop an interrupted autotag job into the plain full-dataset annotator. */
  const browseAll = useCallback(() => {
    setCurrentIndex(0);
    setPhase("annotating");
  }, []);

  const taggedCount = useMemo(
    () =>
      effectiveConfig
        ? data.filter((d) => isTagged(annotations[String(d.id)], effectiveConfig.mode)).length
        : 0,
    [effectiveConfig, data, annotations],
  );

  const frameTaggedCount = useMemo(
    () =>
      effectiveConfig
        ? frameData.filter((d) => isTagged(annotations[String(d.id)], effectiveConfig.mode)).length
        : 0,
    [effectiveConfig, frameData, annotations],
  );

  return {
    phase,
    config: effectiveConfig,
    data,
    columns,
    annotations,
    assist,
    currentIndex: normalizedCurrentIndex,
    taggedCount,
    sessionId,
    // Frame-scoped views for the annotator surface.
    frameData,
    frameTaggedCount,
    openRound,
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
    interviewStreamText,
    interviewThinking,
    interviewOptions,
    interviewPending,
    assistError,
    roundLoading,
    roundPredicting,
    contractStarting,
    estimate,
    autotagStatus,
    sendInterviewMessage,
    stopInterview,
    skipInterview,
    restartInterview,
    confirmRubric,
    setAssistModel,
    setInterviewModel,
    setInterviewEffort,
    assistToggleBinary,
    assistToggleCategory,
    assistSetFreetext,
    acceptPrediction,
    startReviewRound,
    finishRound,
    fetchEstimate,
    startAutotag,
    cancelAutotag,
    startFlaggedPass,
    browseAll,
  };
}
