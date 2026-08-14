"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useSession } from "next-auth/react";
import {
  XCircle,
  Clock,
  Code,
  Terminal,
  TrendUp,
  Timer,
  PaperPlaneTilt,
  Copy,
  Database,
  Gear,
  Pulse,
  Eye,
  PencilSimple,
  ArrowCounterClockwise,
  Play,
  Pause,
  HardDrive,
  RocketLaunch,
  GridFour,
  Package,
} from "@/shared/ui/icons";
import { toast } from "react-toastify";

import { Button } from "@/shared/ui/primitives/button";
import { Badge } from "@/shared/ui/primitives/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/shared/ui/primitives/tabs";
import { PingDot } from "@/shared/ui/ping-dot";
import { markRecentSession } from "@/shared/lib/recent-session";
import { FadeIn } from "@/shared/ui/motion";
import { TooltipButton } from "@/shared/ui/tooltip-button";
import { CopyGlyph, useCopyToClipboard } from "@/shared/ui/copy-button";
import {
  getJob,
  cancelJob,
  pauseJob,
  restartJob,
  resumeJob,
  getOptimizationPayload,
  getServeInfo,
  getPairServeInfo,
  serveProgram,
  serveProgramStream,
  servePairProgramStream,
  serveSharedOptimization,
} from "@/shared/lib/api";
import type {
  LMActivity,
  ServeInfoResponse,
  WorkflowNodeTrace,
  WorkflowSpec,
} from "@/shared/types/api";
// Leaf import on purpose — the tutorial barrel deliberately does not re-export
// the demo fixtures (see features/tutorial/index.ts).
// eslint-disable-next-line no-restricted-imports -- deliberate leaf import; see above
import {
  DEMO_OPTIMIZATION_ID,
  DEMO_GRID_OPTIMIZATION_ID,
  DEMO_TRAJECTORY_PREVIEW_LAYOUT,
  buildDemoOptimizationPayload,
  buildGridDemoJob,
  resetDemoSimulation,
  startDemoSimulation,
} from "@/features/tutorial/lib/demo-data";
import { OptimizationDetailSkeleton } from "./OptimizationDetailSkeleton";
import { formatMsg, msg } from "@/shared/lib/messages";
import { formatBytes } from "@/shared/lib/formatters";
import { TERMS } from "@/shared/lib/terms";
import { getRuntimeEnv } from "@/shared/lib/runtime-env";
import { getActiveDir } from "@/shared/lib/runtime-locale";
import { ACTIVE_STATUSES, TERMINAL_STATUSES } from "@/shared/constants/job-status";
import { registerTutorialHook } from "@/features/tutorial";
import type { OptimizationStatusResponse, OptimizationPayloadResponse } from "@/shared/types/api";
import type { SharedOptimizationData } from "@/shared/lib/api";
import type { PipelineStage } from "../constants";
import { extractScoresFromLogs } from "../lib/extract-scores";
import { isReactModuleName } from "../lib/is-react-module";
import { reconstructGridResult } from "../lib/reconstruct-grid";
import { DataTab } from "./DataTab";
import { LogsTab } from "./LogsTab";
import { DeleteJobDialog } from "./DeleteJobDialog";
import { ShareDialog } from "./ShareDialog";
import { StatusBadge } from "@/shared/ui/status-badge";
import { ConfigTab } from "./ConfigTab";
import { CodeTab } from "./CodeTab";
import { ArtifactTab } from "./ArtifactTab";
import { StageInfoModal } from "./StageInfoModal";
import { PairSelectionStrip } from "./PairSelectionStrip";
import { OverviewTab } from "./OverviewTab";
import { RunCreditsChip } from "./RunCreditsChip";
import { GridServeTab } from "./GridServeTab";
import { LMActivityTab } from "./LMActivityTab";
import { ReactServeChat } from "./ReactServeChat";
import { ReactServeApi } from "./ReactServeApi";
import { RunPlayground } from "./RunPlayground";
import { linkifyMessage } from "@/shared/lib/linkify";
import { useStreamWithPollFallback } from "@/shared/hooks/use-stream-with-poll-fallback";

// Treat naive ISO timestamps (no trailing tz marker) as UTC — that matches the
// backend, which stores UTC datetimes that Pydantic emits without a suffix.
// Without this, browsers in a positive UTC offset parse them as local time
// and the elapsed clock either jumps backward or stays at 00:00:00.
function parseTimestampMs(value: string | null | undefined): number | null {
  if (!value) return null;
  const hasTz = /[zZ]|[+-]\d{2}:?\d{2}$/.test(value);
  const ms = Date.parse(hasTz ? value : `${value}Z`);
  return Number.isFinite(ms) ? ms : null;
}

function formatElapsed(seconds: number): string {
  const h = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const s = String(seconds % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

// Fold a (possibly delta) detail response into the held job. Every non-stream
// field on `next` is the full current value; only progress_events / logs may be
// tails — a 0 offset (or no prior buffer) replaces, a positive offset splices
// the tail onto rows already held. Slicing at the server-echoed offset
// reconstructs the complete array even if the request cursor lagged a tick,
// which keeps grid reconstruction (driven off the full progress_events) sound.
function mergeJobDelta(
  prev: OptimizationStatusResponse | null,
  next: OptimizationStatusResponse,
): OptimizationStatusResponse {
  const po = next.progress_offset ?? 0;
  const lo = next.logs_offset ?? 0;
  const progress_events =
    po > 0 && prev?.progress_events
      ? [...prev.progress_events.slice(0, po), ...next.progress_events]
      : next.progress_events;
  const logs = lo > 0 && prev?.logs ? [...prev.logs.slice(0, lo), ...next.logs] : next.logs;
  return { ...next, progress_events, logs };
}

// Owns the 1Hz `now` tick + elapsed-time derivation in a leaf so that, while a
// job is active, the per-second re-render is confined to this clock badge rather
// than the 1200-line detail root (which would otherwise re-render the whole
// active-tab subtree — chart/grid/trajectory — every second). Mirrors the
// dashboard's LiveElapsed.tsx isolation.
function LiveElapsedBadge({
  isActive,
  startedAt,
  createdAt,
  completedAt,
  elapsedSeconds,
}: {
  isActive: boolean;
  startedAt: string | null;
  createdAt: string | null;
  completedAt: string | null;
  elapsedSeconds: number | null;
}) {
  // `now` ticks once per second only while the job is active, driving the
  // elapsed-time derivation in `liveElapsed`. Decoupling the tick from the
  // timestamp deps keeps the interval alive across polling refreshes that
  // replace identical timestamp strings.
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!isActive) return;
    setNow(Date.now());
    const handle = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(handle);
  }, [isActive]);
  // Anchor on the server-computed elapsed_seconds so the live counter is
  // immune to client/server clock skew — wall-clock derivation alone can go
  // negative and stall the badge at 00:00:00.
  const [elapsedAnchor, setElapsedAnchor] = useState<{ ms: number; sec: number } | null>(null);
  useEffect(() => {
    if (elapsedSeconds != null && Number.isFinite(elapsedSeconds) && elapsedSeconds >= 0) {
      setElapsedAnchor({ ms: Date.now(), sec: elapsedSeconds });
    }
  }, [elapsedSeconds]);
  const liveElapsed = useMemo(() => {
    if (!isActive) {
      if (elapsedSeconds != null && elapsedSeconds > 0) {
        return formatElapsed(Math.floor(elapsedSeconds));
      }
      const start = parseTimestampMs(startedAt ?? createdAt);
      const end = parseTimestampMs(completedAt);
      if (start !== null && end !== null) {
        return formatElapsed(Math.max(0, Math.floor((end - start) / 1000)));
      }
      return "00:00:00";
    }
    // Take the larger of wallclock (now - started_at) and the server anchor so
    // a stale upstream `elapsed_seconds` (e.g. cached 0) doesn't reset the
    // header counter back to zero on every page reload. Anchor the wallclock on
    // started_at ONLY — never fall back to created_at here. A resumed run is
    // re-queued with started_at cleared, so between the resume click and the
    // worker re-stamping started_at the job is "active" with started_at null;
    // measuring from created_at there would fold the entire paused gap (which
    // can be hours) into the timer. With no started_at we lean on the server
    // anchor, which already measures only the live leg.
    const start = parseTimestampMs(startedAt);
    const wall = start !== null ? Math.max(0, (now - start) / 1000) : 0;
    const anchored = elapsedAnchor
      ? elapsedAnchor.sec + Math.max(0, (now - elapsedAnchor.ms) / 1000)
      : 0;
    return formatElapsed(Math.floor(Math.max(wall, anchored)));
  }, [now, isActive, startedAt, createdAt, completedAt, elapsedSeconds, elapsedAnchor]);

  return (
    <span className="flex items-center gap-1.5 tabular-nums" dir="ltr">
      <Clock className="size-3.5" />
      {liveElapsed}
    </span>
  );
}

// Copy control for the failure card. Mirrors the app-wide animated copy
// pattern (Copy morphs into a check, tooltip flips to "copied"), tinted to
// the warm error palette so it reads as part of the card rather than a
// generic action.
function FailureCopyButton({ text }: { text: string }) {
  const { copied, copy } = useCopyToClipboard();
  const label = msg(copied ? "shared.code_editor.copied" : "shared.code_editor.copy");
  return (
    <TooltipButton tooltip={label}>
      <button
        type="button"
        aria-label={label}
        onClick={() => void copy(text)}
        className="-me-1 flex size-7 shrink-0 cursor-pointer items-center justify-center rounded-md text-[#B04030]/70 transition-colors hover:bg-[#B04030]/10 hover:text-[#B04030] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#B04030]/30"
      >
        <CopyGlyph copied={copied} className="size-3.5" checkClassName="text-[#B04030]" />
      </button>
    </TooltipButton>
  );
}

export function OptimizationDetailView({ shareData }: { shareData?: SharedOptimizationData } = {}) {
  const params = useParams<{ id?: string; token?: string }>();
  const isShare = !!shareData;
  const id = shareData?.optimization_id ?? (params.id as string);
  // The /share/[token] route carries the capability token in the URL; the
  // shared serve + clone-with-token paths need it. shareData itself omits it.
  const shareToken = params.token ?? null;
  const shareRole = shareData?.role ?? null;
  // Cloning is viewer+ (it makes the caller's own copy, no spend). Serving/chat
  // spends the owner's key, so it's editor+ only (see shareCanServe).
  const shareCanInteract = isShare && shareRole != null;
  const shareCanServe = isShare && (shareRole === "editor" || shareRole === "owner");
  const router = useRouter();
  const searchParams = useSearchParams();
  const initialTab = searchParams.get("tab") ?? "overview";
  const [detailTab, setDetailTab] = useState(initialTab);
  // Expose for tutorial via the typed bridge (features/tutorial/lib/bridge.ts).
  useEffect(() => registerTutorialHook("setDetailTab", setDetailTab), []);

  const isDemoMode = id === DEMO_OPTIMIZATION_ID;
  const isGridDemoMode = id === DEMO_GRID_OPTIMIZATION_ID;
  const isAnyDemoMode = isDemoMode || isGridDemoMode;
  // Public read-only share view: seed from props; never call authed endpoints.
  const skipNetwork = isAnyDemoMode || isShare;

  // The NextAuth session resolves asynchronously on the client. Without this
  // gate the first getJob() fires before setApiAuthToken() runs, returns 401,
  // and flashes the "not found" UI until SSE/poll refetches a few seconds
  // later with the token now in place.
  const { status: sessionStatus } = useSession();
  const authReady = isAnyDemoMode || isShare || sessionStatus === "authenticated";

  const [job, setJob] = useState<OptimizationStatusResponse | null>(null);
  const [payload, setPayload] = useState<OptimizationPayloadResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Stamp this run as recent only while it's actually in progress, so the
  // sidebar's Optimize button resumes ongoing work — never a finished run the
  // user merely opened to look at. Share/demo views aren't the caller's to resume.
  useEffect(() => {
    if (skipNetwork || !id || !job || !ACTIVE_STATUSES.has(job.status)) return;
    markRecentSession("optimization", id);
    return () => markRecentSession("optimization", id);
  }, [id, skipNetwork, job?.status]);
  // Re-run mints a brand-new run per call, so guard against a double-click
  // firing two retries (and creating two duplicate runs).
  const [retrying, setRetrying] = useState(false);
  const [resuming, setResuming] = useState(false);
  const [pausing, setPausing] = useState(false);

  // Bump to replay the demo optimization simulation. The deep-dive tour
  // triggers this when reaching the trajectory step so users watch the tree
  // grow live instead of seeing the already-completed final state.
  const [demoReplayKey, setDemoReplayKey] = useState(0);

  useEffect(
    () =>
      registerTutorialHook("replayDemoSimulation", () => {
        resetDemoSimulation();
        setDemoReplayKey((k) => k + 1);
      }),
    [],
  );

  useEffect(() => {
    if (!isDemoMode) return;
    setPayload(buildDemoOptimizationPayload());
    const fromTrajectory = demoReplayKey > 0;
    return startDemoSimulation({ setJob: (fn) => setJob(fn), setLoading }, { fromTrajectory });
  }, [isDemoMode, demoReplayKey]);

  useEffect(() => {
    if (!isGridDemoMode) return;
    setJob(buildGridDemoJob());
    setLoading(false);
  }, [isGridDemoMode]);

  // Seed the read-only share view from the public composite; no fetching.
  useEffect(() => {
    if (!shareData) return;
    setJob(shareData.status);
    setPayload({
      optimization_type: shareData.status.optimization_type,
      payload: shareData.payload,
    } as unknown as OptimizationPayloadResponse);
    setLoading(false);
  }, [shareData]);

  const [serveInfo, setServeInfo] = useState<ServeInfoResponse | null>(null);
  const [serveInfoError, setServeInfoError] = useState<string | null>(null);
  const [serveLoading, setServeLoading] = useState(false);
  const [runHistory, setRunHistory] = useState<
    Array<{
      inputs: Record<string, string>;
      outputs: Record<string, unknown>;
      model: string;
      ts: number;
      nodeTraces?: WorkflowNodeTrace[] | null;
    }>
  >([]);
  const [streamingRun, setStreamingRun] = useState<{
    inputs: Record<string, string>;
    partial: Record<string, string>;
  } | null>(null);
  const streamReqIdRef = useRef(0);
  const streamAbortRef = useRef<AbortController | null>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);
  const textareaRefs = useRef<Record<string, HTMLTextAreaElement | null>>({});
  const [serveError, setServeError] = useState<string | null>(null);
  const [stageModal, setStageModal] = useState<PipelineStage | null>(null);

  const activePairIndex =
    searchParams.get("pair") != null ? parseInt(searchParams.get("pair")!, 10) : null;

  /* Cancelled/failed grid jobs have no persisted grid_result — rebuild from
     progress_events so overview + per-pair views still render. Memoized on
     [job] so the reconstruction + fresh object reference rebuild only on a real
     refetch, not on every render — a stable ref is what lets the memoized
     OverviewTab/GridOverview skip re-rendering between unrelated state changes. */
  const effectiveJob: OptimizationStatusResponse | null = useMemo(() => {
    if (!job) return null;
    if (job.grid_result || job.optimization_type !== "grid_search") return job;
    const rebuilt = reconstructGridResult(job);
    return rebuilt ? { ...job, grid_result: rebuilt } : job;
  }, [job]);

  const activePair =
    activePairIndex === null || !effectiveJob?.grid_result
      ? null
      : (effectiveJob.grid_result.pair_results.find((p) => p.pair_index === activePairIndex) ??
        null);
  const isPairContext = activePair != null;

  // Hoisted so the memos below depend on a plain local — the React Compiler lint
  // can't equate an inferred `job.logs` path with a `job?.logs` dependency
  // literal, which would skip optimizing the whole component.
  const jobLogs = job?.logs;

  const pairScorePoints = useMemo(() => {
    if (activePairIndex === null || !jobLogs) return [];
    const pairLogs = jobLogs.filter((l) => l.pair_index === activePairIndex);
    return extractScoresFromLogs(pairLogs);
  }, [jobLogs, activePairIndex]);

  const pairFilteredLogs = useMemo(
    () =>
      activePairIndex === null || !jobLogs
        ? (jobLogs ?? [])
        : jobLogs.filter(
            (l) =>
              l.pair_index === activePairIndex ||
              l.pair_index === null ||
              l.pair_index === undefined,
          ),
    [jobLogs, activePairIndex],
  );

  const jobRef = useRef(job);
  // Serialize refetches: delta tails must be applied in arrival order, so an
  // overlapping fetch (an SSE tick racing the 5s poll) is skipped rather than
  // risk interleaving. The continuous SSE + poll re-trigger within a tick, so
  // dropping a concurrent call costs nothing.
  const fetchingRef = useRef(false);
  const fetchJob = useCallback(async () => {
    if (fetchingRef.current) return;
    fetchingRef.current = true;
    try {
      // Send what we already hold so the server can return just the newer tail
      // of progress_events / logs instead of the whole (growing) history. First
      // load has no prior buffer → full fetch.
      const prev = jobRef.current;
      const cursor = prev
        ? {
            sinceProgress: prev.progress_events?.length ?? 0,
            sinceLog: prev.logs?.length ?? 0,
          }
        : undefined;
      const data = await getJob(id, cursor);
      setJob((cur) => mergeJobDelta(cur, data));
      setError(null);
    } catch (err) {
      console.warn("OptimizationDetailView: getJob failed", err);
      // A transient failure (timeout/blip while the backend is busiest —
      // e.g. spawning validation subprocesses right after submit) must not
      // replace an already-loaded run with the "wasn't found" page; the
      // SSE/poll loop retries within seconds and self-heals. Only a first
      // load with nothing to show gets the error copy.
      if (!jobRef.current) {
        setError(
          formatMsg("auto.app.optimizations.id.page.template.1", { p1: TERMS.optimization }),
        );
      }
    } finally {
      setLoading(false);
      fetchingRef.current = false;
    }
  }, [id]);

  useEffect(() => {
    if (skipNetwork) return;
    if (!authReady) return;
    getOptimizationPayload(id)
      .then(setPayload)
      .catch(() => {});
  }, [id, isAnyDemoMode, authReady]);

  useEffect(() => {
    jobRef.current = job;
  }, [job]);
  const lastCountsRef = useRef({ logs: 0, progress: 0 });

  useEffect(() => {
    if (skipNetwork) return;
    if (!authReady) return;
    void fetchJob();
  }, [id, isAnyDemoMode, authReady, fetchJob]);

  const API = getRuntimeEnv().apiUrl;
  useStreamWithPollFallback({
    url: skipNetwork ? "" : `${API}/optimizations/${encodeURIComponent(id)}/stream`,
    enabled: !skipNetwork && authReady,
    onMessage: (event) => {
      try {
        const sseData = JSON.parse(event.data);
        const logCount = sseData.log_count ?? 0;
        const progressCount = sseData.progress_count ?? 0;
        const prev = lastCountsRef.current;
        const progressed = progressCount > prev.progress;
        const statusChanged = sseData.status !== jobRef.current?.status;
        lastCountsRef.current = { logs: logCount, progress: progressCount };
        // During an active run DSPy/GEPA emit log lines continuously, so a gate
        // that re-fetches on every `log_count` bump re-downloads the full
        // (growing) progress_events+logs payload ~every tick — and since the
        // backend ETag is keyed on those counts, the 304 fast-path can never
        // fire. Re-fetch only when progress advances (drives the trajectory /
        // grid / score views, emitted throughout the run via capture_tqdm) or
        // the status changes; for logs-only bumps, patch the lightweight live
        // fields in place. The Logs tab catches up on the next progress tick and
        // a terminal status always triggers a final full fetch.
        if (progressed || statusChanged) {
          void fetchJob();
        } else {
          setJob((p) =>
            p
              ? {
                  ...p,
                  status: sseData.status ?? p.status,
                  message: sseData.message ?? p.message,
                  latest_metrics: sseData.latest_metrics ?? p.latest_metrics,
                }
              : p,
          );
        }
      } catch {
        void fetchJob();
      }
    },
    events: { done: () => void fetchJob() },
    closeOnEvents: ["done"],
    poll: () => void fetchJob(),
    pollIntervalMs: 5000,
    shouldStopPolling: () => !!jobRef.current && TERMINAL_STATUSES.has(jobRef.current.status),
    // Stream auth failed even after a token refresh — re-fetch through the
    // self-healing request() path so a still-bad token surfaces the existing
    // error banner instead of the page silently freezing on stale data.
    onAuthError: () => void fetchJob(),
  });

  useEffect(() => {
    if (skipNetwork) return;
    const onRenamed = (e: Event) => {
      const { optimizationId, name } = (e as CustomEvent).detail;
      if (optimizationId === id) setJob((prev) => (prev ? { ...prev, name } : prev));
    };
    const onUpdated = (e: Event) => {
      const { optimizationId } = (e as CustomEvent).detail;
      if (optimizationId === id) void fetchJob();
    };
    window.addEventListener("optimization-renamed", onRenamed);
    window.addEventListener("optimization-updated", onUpdated);
    return () => {
      window.removeEventListener("optimization-renamed", onRenamed);
      window.removeEventListener("optimization-updated", onUpdated);
    };
  }, [id, fetchJob, isAnyDemoMode]);

  const handleCancel = async () => {
    if (skipNetwork) return;
    try {
      await cancelJob(id);
      toast.success(msg("optimization.cancel.sent"));
      void fetchJob();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("optimization.cancel.failed"));
    }
  };

  const handleRetry = async () => {
    if (skipNetwork || retrying) return;
    setRetrying(true);
    try {
      await restartJob(id);
      toast.success(msg("optimization.rerun.success"));
      window.dispatchEvent(new Event("optimizations-changed"));
      // Restart re-runs the same id in place, so refresh this view rather than
      // navigating to a new run.
      void fetchJob();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("optimization.rerun.failed"));
    } finally {
      setRetrying(false);
    }
  };

  const handleResume = async () => {
    if (skipNetwork || resuming) return;
    setResuming(true);
    try {
      await resumeJob(id);
      toast.success(msg("optimization.resume.success"));
      window.dispatchEvent(new Event("optimizations-changed"));
      // Resume continues the same run in place — refresh this view rather than
      // navigating to a new id.
      void fetchJob();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("optimization.resume.failed"));
    } finally {
      setResuming(false);
    }
  };

  const handlePause = async () => {
    if (skipNetwork || pausing) return;
    setPausing(true);
    try {
      await pauseJob(id);
      toast.success(msg("optimization.pause.success"));
      window.dispatchEvent(new Event("optimizations-changed"));
      // Pause suspends the same run in place — refresh this view rather than navigating.
      void fetchJob();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : msg("optimization.pause.failed"));
    } finally {
      setPausing(false);
    }
  };

  useEffect(() => {
    if (isDemoMode && job?.status === "success") {
      setServeInfo({
        optimization_id: id,
        module_name: "Predict",
        optimizer_name: "GEPA",
        model_name: "gpt-4o-mini",
        input_fields: ["email_text"],
        output_fields: ["category"],
        instructions: "Classify an email into a category: spam, important, or promotional.",
        demo_count: 3,
      });
    }
    if (isGridDemoMode && job?.status === "success") {
      const best = job.grid_result?.best_pair;
      setServeInfo({
        optimization_id: id,
        module_name: "ChainOfThought",
        optimizer_name: "GEPA",
        model_name: best?.generation_model ?? "openai/gpt-4o-mini",
        input_fields: ["article"],
        output_fields: ["summary"],
        instructions: best?.program_artifact?.optimized_prompt?.instructions ?? "",
        demo_count: 1,
      });
    }
  }, [isDemoMode, isGridDemoMode, id, job?.status, job?.grid_result]);

  // Share view: seed the playground serve info from the public composite.
  // It is non-null only for viewer+ (the backend nulls it for anonymous view).
  useEffect(() => {
    if (!shareData) return;
    setServeInfo(shareData.serve_info ?? null);
    setServeInfoError(null);
  }, [shareData]);

  useEffect(() => {
    if (skipNetwork) return;
    if (job?.status !== "success") return;
    // A failure here used to be swallowed as `setServeInfo(null)`, which
    // silently blanked the Usage tab (e.g. when an expired bearer 401'd).
    // request() now self-heals a stale token; a persistent failure surfaces.
    const onFail = (err: unknown) => {
      setServeInfo(null);
      setServeInfoError(
        err instanceof Error
          ? err.message
          : formatMsg("auto.app.optimizations.id.page.template.1", { p1: TERMS.optimization }),
      );
    };
    const isGrid = job.optimization_type === "grid_search";
    const loader =
      isGrid && activePairIndex != null ? getPairServeInfo(id, activePairIndex) : getServeInfo(id);
    loader
      .then((info) => {
        setServeInfo(info);
        setServeInfoError(null);
      })
      .catch(onFail);
  }, [id, job?.status, job?.optimization_type, activePairIndex, isAnyDemoMode]);

  useEffect(() => {
    if (chatScrollRef.current) {
      const el = chatScrollRef.current;
      if (el.scrollHeight - el.scrollTop - el.clientHeight < 300) {
        el.scrollTop = el.scrollHeight;
      }
    }
  }, [runHistory, streamingRun]);

  useEffect(() => {
    return () => {
      streamAbortRef.current?.abort();
    };
  }, []);

  useEffect(() => {
    // Share view has no pair switching; skip so the seeded share serveInfo
    // (set from the public composite above) is not clobbered to null on mount.
    if (isShare) return;
    streamAbortRef.current?.abort();
    setRunHistory([]);
    setStreamingRun(null);
    setServeLoading(false);
    setServeError(null);
    setServeInfo(null);
    setServeInfoError(null);
  }, [activePairIndex, isShare]);

  const readServeInputs = () => {
    const vals: Record<string, string> = {};
    for (const f of serveInfo?.input_fields ?? []) vals[f] = textareaRefs.current[f]?.value ?? "";
    return vals;
  };

  const handleServe = async (overrideInputs?: Record<string, string>) => {
    if (!serveInfo) return;
    const inputs = overrideInputs ?? readServeInputs();
    const missing = serveInfo.input_fields.filter((f) => !inputs[f]?.trim());
    if (missing.length > 0) {
      toast.error(
        <div>
          {msg("auto.app.optimizations.id.page.1")}
          <br />
          {missing.join(", ")}
        </div>,
      );
      return;
    }
    // Abort any in-flight stream, then start a new one tagged with a fresh id
    streamAbortRef.current?.abort();
    const reqId = ++streamReqIdRef.current;
    const controller = new AbortController();
    streamAbortRef.current = controller;
    setServeLoading(true);
    setServeError(null);
    setStreamingRun({ inputs: { ...inputs }, partial: {} });
    if (!overrideInputs) {
      Object.values(textareaRefs.current).forEach((el) => {
        if (el) {
          el.value = "";
          el.style.height = "auto";
        }
      });
    }
    const isStale = () => reqId !== streamReqIdRef.current;

    // Share view runs inference through the token-gated, non-streaming
    // /share/{token}/serve endpoint (owner key applied server-side).
    if (isShare) {
      if (!shareToken) {
        setStreamingRun(null);
        setServeLoading(false);
        return;
      }
      try {
        const res = await serveSharedOptimization(shareToken, inputs);
        if (isStale()) return;
        setRunHistory((prev) => {
          const next = [
            { inputs: { ...inputs }, outputs: res.outputs, model: res.model_used, ts: Date.now() },
            ...prev,
          ];
          return next.length > 50 ? next.slice(0, 50) : next;
        });
        setStreamingRun(null);
      } catch (err) {
        if (isStale()) return;
        setServeError(err instanceof Error ? err.message : msg("share.inference_failed"));
        setStreamingRun(null);
      } finally {
        if (!isStale()) setServeLoading(false);
      }
      return;
    }
    // Workflow runs serve through the blocking endpoint: the response carries
    // the per-node trace the playground replays, which the SSE final event
    // does not.
    if ((job?.module_name ?? "").toLowerCase() === "workflow" && activePairIndex == null) {
      try {
        const res = await serveProgram(id, inputs);
        if (isStale()) return;
        setRunHistory((prev) => {
          const next = [
            {
              inputs: { ...inputs },
              outputs: res.outputs,
              model: res.model_used,
              ts: Date.now(),
              nodeTraces: res.node_traces ?? null,
            },
            ...prev,
          ];
          return next.length > 50 ? next.slice(0, 50) : next;
        });
        setStreamingRun(null);
      } catch (err) {
        if (isStale()) return;
        setServeError(err instanceof Error ? err.message : msg("share.inference_failed"));
        setStreamingRun(null);
      } finally {
        if (!isStale()) setServeLoading(false);
      }
      return;
    }
    const streamFn =
      job?.optimization_type === "grid_search" && activePairIndex != null
        ? (i: Record<string, string>, h: Parameters<typeof serveProgramStream>[2]) =>
            servePairProgramStream(id, activePairIndex, i, h)
        : (i: Record<string, string>, h: Parameters<typeof serveProgramStream>[2]) =>
            serveProgramStream(id, i, h);
    await streamFn(inputs, {
      signal: controller.signal,
      onToken: (field, chunk) => {
        if (isStale()) return;
        setStreamingRun((prev) =>
          prev
            ? {
                ...prev,
                partial: { ...prev.partial, [field]: (prev.partial[field] ?? "") + chunk },
              }
            : prev,
        );
      },
      onFinal: (res) => {
        if (isStale()) return;
        setRunHistory((prev) => {
          // Cap history so the chat panel stays responsive across long sessions.
          const next = [
            { inputs: { ...inputs }, outputs: res.outputs, model: res.model_used, ts: Date.now() },
            ...prev,
          ];
          return next.length > 50 ? next.slice(0, 50) : next;
        });
        setStreamingRun(null);
      },
      onError: (errorMsg) => {
        if (isStale()) return;
        setServeError(errorMsg);
        setStreamingRun(null);
      },
    });
    if (!isStale()) setServeLoading(false);
  };

  const handleStopServe = () => {
    streamReqIdRef.current += 1;
    streamAbortRef.current?.abort();
    streamAbortRef.current = null;
    setServeLoading(false);
    setStreamingRun(null);
  };

  const handleClearHistory = () => {
    setRunHistory([]);
    setServeError(null);
  };

  const metrics = job?.latest_metrics ?? {};

  const signatureCode = (payload?.payload?.signature_code as string) ?? null;
  const metricCode = (payload?.payload?.metric_code as string) ?? null;
  const workflowSpec = (payload?.payload?.workflow as WorkflowSpec | undefined) ?? null;

  const scorePoints = useMemo(
    () => (jobLogs?.length ? extractScoresFromLogs(jobLogs) : []),
    [jobLogs],
  );

  // Optimized prompt picks the pair's artifact in pair view, otherwise falls
  // back to the run's artifact, otherwise to the grid's best pair (for the
  // grid-root overview banner/code tab).
  const optimizedPrompt = isPairContext
    ? (activePair.program_artifact?.optimized_prompt ?? null)
    : (job?.result?.program_artifact?.optimized_prompt ??
      job?.grid_result?.best_pair?.program_artifact?.optimized_prompt ??
      null);

  // The real optimized artifact for a react run lives in react_overlay (tuned
  // tool descriptions / display names), not the reasoning-predictor prompt.
  const reactOverlay = isPairContext
    ? (activePair.program_artifact?.react_overlay ?? null)
    : (job?.result?.program_artifact?.react_overlay ?? null);

  // Flex optimizes the program's code alongside its prompt; the rewritten
  // module source is the headline artifact for those runs.
  const optimizedModuleSrc = isPairContext
    ? (activePair.program_artifact?.optimized_module_src ?? null)
    : (job?.result?.program_artifact?.optimized_module_src ??
      job?.grid_result?.best_pair?.program_artifact?.optimized_module_src ??
      null);
  // A workflow is not itself a Flex, so its flex nodes arrive keyed by node.
  const optimizedComponentSrcs = isPairContext
    ? activePair.program_artifact?.optimized_component_srcs
    : (job?.result?.program_artifact?.optimized_component_srcs ??
      job?.grid_result?.best_pair?.program_artifact?.optimized_component_srcs);

  // The unified per-node view of a workflow's optimized surface — each node's
  // prompt, react overlay, or rewritten code, keyed "n_<node_id>". Present only
  // for workflow runs, where it supersedes the scalar prompt/flex/react
  // projections above in the artifact tab.
  const optimizedNodes = isPairContext
    ? activePair.program_artifact?.optimized_nodes
    : (job?.result?.program_artifact?.optimized_nodes ??
      job?.grid_result?.best_pair?.program_artifact?.optimized_nodes);

  // Artifact tab appears once the run has produced something durable — an
  // optimized prompt, tuned react tools, or downloadable program files.
  const artifactFiles = isPairContext
    ? (activePair.program_artifact ?? null)
    : (job?.result?.program_artifact ?? job?.grid_result?.best_pair?.program_artifact ?? null);
  const showArtifactTab = !!(optimizedPrompt || reactOverlay || artifactFiles);

  // Demos passed to the serve playground — pair-scoped in pair view.
  const playgroundDemos = isPairContext
    ? (activePair.program_artifact?.optimized_prompt?.demos ?? [])
    : (job?.result?.program_artifact?.optimized_prompt?.demos ??
      job?.grid_result?.best_pair?.program_artifact?.optimized_prompt?.demos ??
      []);

  // LM activity is pair-scoped in pair view, otherwise the run's.
  const viewLmActivity: LMActivity | null = isPairContext
    ? ((activePair.lm_activity as LMActivity | undefined) ?? null)
    : ((job?.result?.lm_activity as LMActivity | undefined) ?? null);

  const isActive = job ? ACTIVE_STATUSES.has(job.status) : false;
  const startedAt = job?.started_at ?? null;
  const createdAt = job?.created_at ?? null;
  const completedAt = job?.completed_at ?? null;
  const elapsedSeconds = job?.elapsed_seconds ?? null;

  // Stable identities so the memoized OverviewTab/GridOverview don't re-render
  // on every parent state tick (live elapsed, SSE patches) just because these
  // handlers were freshly allocated.
  const handlePairSelect = useCallback(
    (pi: number) => router.push(`/optimizations/${id}?pair=${pi}`),
    [id, router],
  );
  const handlePairDeleted = useCallback(
    (pi: number) => {
      try {
        window.localStorage.removeItem(`grid-serve:pair:${id}`);
      } catch {
        /* localStorage unavailable — nothing to clean up */
      }
      if (activePairIndex === pi) {
        router.replace(`/optimizations/${id}`);
      }
      void fetchJob();
    },
    [id, activePairIndex, router, fetchJob],
  );

  if (loading || !authReady) {
    return <OptimizationDetailSkeleton />;
  }

  if (error || !job) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <XCircle className="size-12 text-destructive" />
        <p className="text-lg text-muted-foreground">
          {error ??
            formatMsg("auto.app.optimizations.id.page.template.2", { p1: TERMS.optimization })}
        </p>
      </div>
    );
  }

  const isTerminal = TERMINAL_STATUSES.has(job.status);
  // Effective role on the normal (non-share) route. null = the caller's own run
  // or an owner-tier grant (full control); "editor"/"viewer" = a member reached
  // it via sharing. Mutations gate on this so a viewer can't cancel/delete/manage.
  const effectiveRole = job.effective_role ?? null;
  const canEditRun = effectiveRole == null || effectiveRole === "editor";
  const canDeleteRun = effectiveRole == null;
  // Sharing is owner-only: members (even editors) can't invite, change roles,
  // or set general access. null = the owner's own run / admin / co-owner.
  // Never on the public /share/<token> route: there the job is seeded from
  // shareData and carries no effective_role, so it would read as owner-level
  // and leak the management controls to viewers (even anonymous visitors).
  const canManageShare = !isShare && effectiveRole == null;
  // Share banner: surface the caller's granted tier whenever they're NOT the
  // owner — both the public /share link and an explicit member grant. "viewer"
  // maps to the viewer tier, "editor" to the editor tier. null / "owner" is the
  // owner's own view (or admin/co-owner) and gets no banner.
  const callerRole = isShare ? shareRole : effectiveRole;
  const sharedTier = callerRole === "editor" ? "editor" : callerRole === "viewer" ? "viewer" : null;
  const sharedByOwner = isShare ? shareData?.owner : job.username;
  // Split "מאת {name}" so the emphasis (semibold/foreground) lands only on the
  // owner name; the "by" prefix stays muted meta text.
  const [sharedByPrefix, sharedBySuffix] = msg("optimization.readonly_by").split("{name}");
  // Pair-aware terminal: a pair is considered "available for data export" once
  // it has either a program artifact or an error recorded. This mirrors the
  // standalone job's "isTerminal" gate so the data/export surfaces appear
  // identically in both contexts.
  const isPairTerminal = isPairContext
    ? !!(
        activePair.error ||
        activePair.program_artifact ||
        activePair.optimized_test_metric != null
      )
    : false;

  // Tab gating uniform across run and pair contexts.
  // Share view: only editor+ may run inference (it spends the owner's key),
  // and only through the single non-streaming /share serve endpoint, so it
  // needs a seeded serveInfo (the backend nulls serve_info below editor).
  const showPlaygroundTab = isShare
    ? shareCanServe && job.status === "success" && !!serveInfo
    : job.status === "success" && (job.optimization_type === "grid_search" || !!serveInfo);
  const showDataTab = isShare
    ? !!shareData?.dataset
    : isPairContext
      ? isPairTerminal
      : isTerminal && job.optimization_type !== "grid_search";
  const showLogsTab = job.optimization_type !== "grid_search" || isPairContext;
  const showLmActivityTab = viewLmActivity != null;

  const pairCount = effectiveJob?.grid_result?.pair_results.length ?? 0;
  const isBestPair =
    isPairContext && effectiveJob?.grid_result?.best_pair?.pair_index === activePair.pair_index;

  return (
    <div className="space-y-6 pb-12">
      {sharedTier && (
        <div
          role="status"
          className="flex w-full items-center gap-3 rounded-xl border border-border/60 bg-gradient-to-br from-muted/60 to-muted/25 px-4 py-2.5 shadow-sm"
        >
          <span className="grid size-7 shrink-0 place-items-center rounded-lg bg-primary/5 text-primary/80">
            {sharedTier === "editor" ? (
              <PencilSimple className="size-4" aria-hidden="true" />
            ) : (
              <Eye className="size-4" aria-hidden="true" />
            )}
          </span>
          <span className="whitespace-nowrap text-sm font-medium text-foreground/90">
            {msg(
              sharedTier === "editor"
                ? "optimization.access_banner.editor"
                : "optimization.access_banner.viewer",
            )}
          </span>
          {sharedByOwner && (
            <span className="ms-auto flex min-w-0 items-center gap-1.5 text-xs text-muted-foreground">
              <span dir="auto" className="min-w-0 truncate">
                {sharedByPrefix}
                <span dir="auto" className="font-semibold text-foreground">
                  {sharedByOwner}
                </span>
                {sharedBySuffix}
              </span>
              <span
                aria-hidden="true"
                className="grid size-4 shrink-0 place-items-center rounded-full bg-primary/10 text-[0.5625rem] font-semibold uppercase text-primary"
              >
                {sharedByOwner.trim().charAt(0)}
              </span>
            </span>
          )}
        </div>
      )}
      <FadeIn delay={0.1}>
        <div
          className=" rounded-xl border border-border/40 bg-gradient-to-br from-card to-card/80 p-5"
          data-tutorial="detail-header"
        >
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="space-y-2 min-w-0">
              <div className="flex flex-col items-start gap-1.5">
                <StatusBadge status={job.status} />
                {job.name && (
                  <h2 className="text-lg sm:text-xl font-bold tracking-tight" dir="auto">
                    {job.name}
                  </h2>
                )}
              </div>
              {job.description && (
                <p className="text-sm text-muted-foreground/70 leading-relaxed">
                  {job.description}
                </p>
              )}
              <code
                className="text-xs font-mono text-muted-foreground/60 cursor-pointer hover:text-primary transition-colors break-all"
                title={msg("auto.app.optimizations.id.page.literal.1")}
                aria-label={formatMsg("auto.app.optimizations.id.page.template.3", {
                  p1: TERMS.optimization,
                })}
                role="button"
                tabIndex={0}
                onClick={() => {
                  void navigator.clipboard.writeText(job.optimization_id);
                  toast.success(msg("clipboard.copied_short"), { autoClose: 1000 });
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    void navigator.clipboard.writeText(job.optimization_id);
                    toast.success(msg("clipboard.copied_short"), { autoClose: 1000 });
                  }
                }}
              >
                {job.optimization_id}
              </code>
              <div className="flex items-center gap-3 flex-wrap text-sm text-muted-foreground">
                <Badge
                  variant="outline"
                  className="gap-1.5 border-[#C8A882]/45 bg-[#C8A882]/15 font-semibold text-[0.6875rem] text-[#3D2E22] [&>svg]:text-[#8a6d44]"
                >
                  {job.optimization_type === "grid_search" ? (
                    <>
                      <GridFour />
                      {msg("auto.app.optimizations.id.page.literal.2")}
                    </>
                  ) : (
                    <>
                      <RocketLaunch />
                      {msg("auto.app.optimizations.id.page.literal.3")}
                    </>
                  )}
                </Badge>
                <LiveElapsedBadge
                  isActive={isActive}
                  startedAt={startedAt}
                  createdAt={createdAt}
                  completedAt={completedAt}
                  elapsedSeconds={elapsedSeconds}
                />
                {isActive && job.estimated_remaining && (
                  <span className="flex items-center gap-1.5">
                    <Timer className="size-3.5" />
                    {msg("auto.app.optimizations.id.page.3")}
                    {job.estimated_remaining}
                  </span>
                )}
                {!isPairContext && (job.stored_bytes ?? 0) > 0 && (
                  <Link
                    href="/storage"
                    title={msg("optimization.storage_label")}
                    dir="ltr"
                    className="flex items-center gap-1.5 tabular-nums transition-colors hover:text-foreground"
                  >
                    <HardDrive className="size-3.5" />
                    {formatBytes(job.stored_bytes ?? 0)}
                  </Link>
                )}
                {!isPairContext && <RunCreditsChip details={job.result?.details} />}
              </div>
            </div>
            {!isShare && (
              <div className="flex items-center gap-2" data-tutorial="result-actions">
                {canManageShare && <ShareDialog optimizationId={job.optimization_id} />}
                <TooltipButton tooltip={msg("auto.app.optimizations.id.page.4")}>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-8"
                    onClick={() => router.push(`/submit?clone=${job.optimization_id}`)}
                    aria-label={msg("auto.app.optimizations.id.page.literal.4")}
                  >
                    <Copy className="size-4" />
                  </Button>
                </TooltipButton>
                {canEditRun &&
                  (job.status === "failed" ||
                    job.status === "cancelled" ||
                    job.status === "paused") &&
                  (job.resumable ? (
                    <TooltipButton tooltip={msg("optimization.resume_tooltip")}>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="size-8"
                        onClick={handleResume}
                        disabled={resuming}
                        aria-label={msg("optimization.resume")}
                      >
                        {/* Mirror the triangle to point in the reading direction — left in RTL, right in LTR. */}
                        <Play
                          className={`size-4${getActiveDir() === "rtl" ? " -scale-x-100" : ""}${
                            resuming ? " animate-spin" : ""
                          }`}
                        />
                      </Button>
                    </TooltipButton>
                  ) : (
                    <TooltipButton tooltip={msg("optimization.rerun_tooltip")}>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="size-8"
                        onClick={handleRetry}
                        disabled={retrying}
                        aria-label={msg("optimization.rerun")}
                      >
                        <ArrowCounterClockwise
                          className={`size-4${retrying ? " animate-spin" : ""}`}
                        />
                      </Button>
                    </TooltipButton>
                  ))}
                {canEditRun && isActive && job.pausable && (
                  <TooltipButton tooltip={msg("optimization.pause_tooltip")}>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-8"
                      onClick={handlePause}
                      disabled={pausing}
                      aria-label={msg("optimization.pause")}
                    >
                      <Pause className={`size-4${pausing ? " animate-pulse" : ""}`} />
                    </Button>
                  </TooltipButton>
                )}
                {canEditRun && isActive && (
                  <TooltipButton tooltip={msg("auto.app.optimizations.id.page.5")}>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-8 text-destructive hover:bg-destructive/10 hover:text-destructive focus-visible:ring-0 focus-visible:border-0"
                      onClick={handleCancel}
                      aria-label={msg("auto.app.optimizations.id.page.literal.5")}
                    >
                      <XCircle className="size-4" />
                    </Button>
                  </TooltipButton>
                )}
                {canDeleteRun && isTerminal && (
                  <DeleteJobDialog
                    optimizationId={job.optimization_id}
                    onDeleted={() => router.push("/")}
                  />
                )}
              </div>
            )}
            {shareCanInteract && (
              <div className="flex items-center gap-2">
                <TooltipButton tooltip={msg("share.clone_tooltip")}>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="size-8"
                    onClick={() =>
                      router.push(
                        shareToken
                          ? `/submit?clone=${job.optimization_id}&shareToken=${encodeURIComponent(shareToken)}`
                          : `/submit?clone=${job.optimization_id}&public=1`,
                      )
                    }
                    aria-label={msg("share.clone")}
                  >
                    <Copy className="size-4" />
                  </Button>
                </TooltipButton>
              </div>
            )}
          </div>
        </div>
      </FadeIn>

      {isPairContext && effectiveJob?.grid_result && (
        <PairSelectionStrip
          job={effectiveJob}
          activePair={activePair}
          activePairIndex={activePair.pair_index}
          pairCount={pairCount}
          isBest={isBestPair}
          jobActive={isActive}
          jobTerminal={isTerminal}
          onBack={() => router.push(`/optimizations/${id}`)}
          onPrev={() => router.push(`/optimizations/${id}?pair=${activePair.pair_index - 1}`)}
          onNext={() => router.push(`/optimizations/${id}?pair=${activePair.pair_index + 1}`)}
          onClone={() =>
            router.push(
              `/submit?clone=${effectiveJob.optimization_id}&pair=${activePair.pair_index}`,
            )
          }
          onCancel={handleCancel}
          onDeleted={() => router.push(`/optimizations/${id}`)}
        />
      )}

      {job.status === "failed" && !isPairContext && (job.message || (metrics.error as string)) && (
        <FadeIn delay={0.15}>
          <div className="rounded-xl border border-[#B04030]/20 bg-gradient-to-br from-[#B04030]/[0.05] to-[#B04030]/[0.015] p-5 shadow-[0_1px_3px_rgba(176,64,48,0.07)]">
            <div className="flex items-center gap-3">
              <span className="flex size-9 shrink-0 items-center justify-center rounded-full bg-[#B04030]/10 text-[#B04030]">
                <XCircle className="size-5" />
              </span>
              <p className="min-w-0 flex-1 text-sm font-semibold text-[#7A2A1E]">
                {msg("auto.app.optimizations.id.page.6")}
              </p>
            </div>
            <div className="mt-3 rounded-lg border border-[#B04030]/15 bg-[#B04030]/[0.035] px-3.5 py-2.5">
              <div className="flex items-start gap-2" dir="ltr">
                <div className="min-w-0 flex-1">
                  <pre
                    className="text-xs text-[#8A3326] whitespace-pre-wrap break-words font-mono leading-relaxed"
                    dir="ltr"
                  >
                    {linkifyMessage(
                      job.message ?? "",
                      "underline decoration-[#B04030]/40 underline-offset-2 hover:text-[#7A2A1E] transition-colors",
                    )}
                  </pre>
                  {typeof metrics.error === "string" && !job.message?.includes(metrics.error) && (
                    <pre
                      className="text-xs text-[#8A3326] whitespace-pre-wrap break-words font-mono leading-relaxed mt-2 border-t border-[#B04030]/15 pt-2"
                      dir="ltr"
                    >
                      {linkifyMessage(
                        String(metrics.error),
                        "underline decoration-[#B04030]/40 underline-offset-2 hover:text-[#7A2A1E] transition-colors",
                      )}
                    </pre>
                  )}
                </div>
                <FailureCopyButton
                  text={[
                    job.message ?? "",
                    typeof metrics.error === "string" && !job.message?.includes(metrics.error)
                      ? String(metrics.error)
                      : "",
                  ]
                    .filter(Boolean)
                    .join("\n")}
                />
              </div>
            </div>
          </div>
        </FadeIn>
      )}

      {isPairContext && activePair.error && (
        <FadeIn delay={0.05}>
          <div className="rounded-xl border border-[#B04030]/30 bg-[#B04030]/5 p-4">
            <div className="text-sm font-medium text-[#B04030] mb-1">
              {msg("auto.features.optimizations.components.pairdetailview.3")}
            </div>
            <pre className="text-xs font-mono text-[#B04030]/80 whitespace-pre-wrap" dir="ltr">
              {activePair.error}
            </pre>
          </div>
        </FadeIn>
      )}

      {job.status === "cancelled" && !isPairContext && (
        <FadeIn>
          <div className="flex items-center gap-3 p-4 rounded-xl border border-stone-300 bg-stone-50 text-stone-700">
            <XCircle className="size-5 shrink-0" />
            <div>
              <p className="text-sm font-semibold">
                {msg("auto.app.optimizations.id.page.7")}
                {TERMS.optimization}
                {msg("auto.app.optimizations.id.page.8")}
              </p>
              <p className="text-xs text-stone-500 mt-0.5">
                {job.message ||
                  formatMsg("auto.app.optimizations.id.page.template.4", {
                    p1: TERMS.optimization,
                  })}
              </p>
            </div>
          </div>
        </FadeIn>
      )}

      {job.optimization_type === "grid_search" &&
        activePairIndex !== null &&
        (!activePair || !effectiveJob?.grid_result) && (
          <FadeIn>
            <div className="rounded-xl border border-border/50 bg-card/80 p-8 text-center space-y-3">
              <p className="text-sm font-medium">{msg("auto.app.optimizations.id.page.10")}</p>
              <p className="text-xs text-muted-foreground">
                {msg("auto.app.optimizations.id.page.11")}
                {TERMS.optimization}
                {msg("auto.app.optimizations.id.page.12")}
              </p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => router.push(`/optimizations/${id}`)}
              >
                {msg("auto.app.optimizations.id.page.13")}
              </Button>
            </div>
          </FadeIn>
        )}

      {(() => {
        const tabCls =
          "relative shrink-0 flex-none px-2.5 sm:px-4 py-2.5 rounded-none border-b-2 border-transparent data-[state=active]:border-transparent data-[state=active]:border-b-primary data-[state=active]:text-foreground data-[state=active]:bg-transparent data-[state=active]:shadow-none transition-all duration-200 text-xs sm:text-sm";
        // Pair view pings the trajectory/logs tabs only while the pair itself
        // is still running, matching the standalone run's behaviour exactly.
        const pingActive = isPairContext
          ? isActive &&
            !activePair.error &&
            !(activePair.program_artifact || activePair.optimized_test_metric != null)
          : isActive;
        return (
          <Tabs value={detailTab} onValueChange={setDetailTab}>
            <TabsList
              variant="line"
              className="border-b border-border/50 pb-0 gap-0 overflow-x-auto no-scrollbar"
              data-tutorial="detail-tabs"
            >
              <TabsTrigger value="overview" className={tabCls}>
                <TrendUp className="size-3.5" />
                {msg("auto.app.optimizations.id.page.14")}
                {pingActive && <PingDot className="ms-1" />}
              </TabsTrigger>
              {showPlaygroundTab && (
                <TabsTrigger value="playground" className={tabCls}>
                  <PaperPlaneTilt className="size-3.5" />
                  {msg("auto.app.optimizations.id.page.15")}
                </TabsTrigger>
              )}
              {showDataTab && (
                <TabsTrigger value="data" className={tabCls}>
                  <Database className="size-3.5" />
                  {msg("auto.app.optimizations.id.page.16")}
                </TabsTrigger>
              )}
              <TabsTrigger value="code" className={tabCls}>
                <Code className="size-3.5" />
                {msg("auto.app.optimizations.id.page.17")}
              </TabsTrigger>
              {showArtifactTab && (
                <TabsTrigger value="artifact" className={tabCls}>
                  <Package className="size-3.5" />
                  {msg("optimization.artifact.tab")}
                </TabsTrigger>
              )}
              {showLogsTab && (
                <TabsTrigger value="logs" className={tabCls}>
                  <Terminal className="size-3.5" />
                  {msg("auto.app.optimizations.id.page.18")}
                  {pingActive && <PingDot className="ms-1" />}
                </TabsTrigger>
              )}
              {showLmActivityTab && (
                <TabsTrigger value="lm-activity" className={tabCls}>
                  <Pulse className="size-3.5" />
                  {msg("auto.app.optimizations.id.page.lm_activity")}
                </TabsTrigger>
              )}
              <TabsTrigger value="config" className={tabCls}>
                <Gear className="size-3.5" />
                {msg("auto.app.optimizations.id.page.19")}
              </TabsTrigger>
            </TabsList>

            <TabsContent
              value="overview"
              className="space-y-6 mt-4"
              data-tutorial={isPairContext ? "pair-detail" : "overview-tab"}
            >
              <OverviewTab
                job={effectiveJob ?? job}
                isActive={isActive}
                scorePoints={isPairContext ? pairScorePoints : scorePoints}
                activePairIndex={activePairIndex}
                activePair={activePair}
                onStageClick={setStageModal}
                onPairSelect={handlePairSelect}
                onPairDeleted={handlePairDeleted}
                trajectoryPreviewLayout={
                  isDemoMode && !isPairContext ? DEMO_TRAJECTORY_PREVIEW_LAYOUT : undefined
                }
              />
            </TabsContent>

            {showPlaygroundTab && (
              <TabsContent value="playground" className="space-y-4 mt-4">
                {!isShare && job.optimization_type === "grid_search" && !isPairContext ? (
                  <GridServeTab job={job} />
                ) : !isShare &&
                  !isPairContext &&
                  (isReactModuleName(serveInfo?.module_name) ||
                    isReactModuleName(job.module_name)) ? (
                  <>
                    <ReactServeChat optimizationId={job.optimization_id} />
                    <ReactServeApi optimizationId={job.optimization_id} />
                  </>
                ) : serveInfo ? (
                  <RunPlayground
                    serveInfo={serveInfo}
                    runHistory={runHistory}
                    setRunHistory={setRunHistory}
                    streamingRun={streamingRun}
                    serveLoading={serveLoading}
                    serveError={serveError}
                    setServeError={setServeError}
                    textareaRefs={textareaRefs}
                    chatScrollRef={chatScrollRef}
                    handleServe={handleServe}
                    handleStopServe={handleStopServe}
                    demos={playgroundDemos}
                    optimizationId={job.optimization_id}
                    pairIndex={isPairContext ? activePair.pair_index : undefined}
                    onClearHistory={handleClearHistory}
                    isShare={isShare}
                  />
                ) : null}
              </TabsContent>
            )}

            {showDataTab && (
              <TabsContent value="data">
                <DataTab
                  job={job}
                  pairIndex={isPairContext ? activePair.pair_index : undefined}
                  sharedDataset={shareData?.dataset ?? undefined}
                  sharedTestResults={shareData?.test_results ?? undefined}
                />
              </TabsContent>
            )}

            <TabsContent value="code" className="space-y-6 mt-4">
              <CodeTab
                signatureCode={signatureCode ?? ""}
                metricCode={metricCode ?? ""}
                workflowSpec={workflowSpec}
              />
            </TabsContent>

            {showArtifactTab && (
              <TabsContent value="artifact" className="space-y-6 mt-4">
                <ArtifactTab
                  job={job}
                  activePair={activePair}
                  optimizedPrompt={optimizedPrompt}
                  reactOverlay={reactOverlay}
                  optimizedModuleSrc={optimizedModuleSrc}
                  optimizedComponentSrcs={optimizedComponentSrcs}
                  optimizedNodes={optimizedNodes}
                  workflowSpec={workflowSpec}
                  isShare={isShare}
                />
              </TabsContent>
            )}

            {showLogsTab && (
              <TabsContent value="logs">
                <LogsTab logs={isPairContext ? pairFilteredLogs : (job.logs ?? [])} />
              </TabsContent>
            )}

            {showLmActivityTab && viewLmActivity && (
              <TabsContent value="lm-activity" className="mt-4">
                <LMActivityTab lmActivity={viewLmActivity} />
              </TabsContent>
            )}

            <TabsContent value="config" className="mt-4" data-tutorial="config-section">
              <ConfigTab job={job} payload={payload} activePair={activePair ?? undefined} />
            </TabsContent>
          </Tabs>
        );
      })()}

      <StageInfoModal stage={stageModal} job={job} onClose={() => setStageModal(null)} />
    </div>
  );
}
