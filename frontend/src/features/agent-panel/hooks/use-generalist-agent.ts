"use client";

import * as React from "react";
import { toast } from "react-toastify";
import { readPref } from "@/features/settings";
import { formatMsg, msg } from "@/shared/lib/messages";
import { getActiveLocale } from "@/shared/lib/runtime-locale";

import type { AgentMessage, AgentStatus, AgentToolCall } from "@/shared/ui/agent/types";

import { confirmGeneralistApproval, streamGeneralistAgent } from "../lib/stream";
import type {
  ChatTurn,
  PendingApprovalPayload,
  ToolEndPayload,
  ToolStartPayload,
  TrustMode,
  WizardState,
} from "../lib/types";

/** Where a stream event came from, relative to what the panel is showing. */
export interface SessionEventContext {
  /** True when the event's session is the one currently displayed. */
  isActive: boolean;
  /** The session's conversation id at event time (null before meta arrives). */
  conversationId: string | null;
}

export interface GeneralistAgentState {
  status: AgentStatus;
  statusLabel: string;
  messages: AgentMessage[];
  reasoning: string;
  reasoningStartedAt: number | null;
  reasoningEndedAt: number | null;
  error: string | null;
  pendingApproval: PendingApprovalPayload | null;
  conversationId: string | null;
  /** LiteLLM id of the composer menu's chosen model; null runs the default. */
  model: string | null;
  setModel: (model: string | null) => void;
  /** Reasoning-effort level for the chosen model; null runs its default. */
  reasoningEffort: string | null;
  setReasoningEffort: (effort: string | null) => void;
  send: (message: string, wizardStateOverride?: WizardState) => void;
  editAndResend: (messageIndex: number, content: string) => void;
  retry: () => void;
  stop: () => void;
  confirmApproval: (approved: boolean) => Promise<void>;
  /** Start a fresh empty chat; every other chat keeps streaming untouched. */
  newSession: () => void;
  /**
   * Switch to the live session already holding this conversation, if one
   * exists (streaming, queued, or finished-but-kept). Returns false when the
   * conversation has no in-memory session and must be loaded from the server.
   */
  activateConversation: (id: string) => boolean;
  /** Open a server-loaded conversation as the displayed session. */
  openConversation: (id: string, messages: AgentMessage[]) => void;
  /** Drop a conversation's session (aborting its stream) after deletion. */
  discardConversation: (id: string) => void;
  /** Conversation ids with a stream running or waiting in the queue. */
  busyConversationIds: ReadonlySet<string>;
  /** Busy sessions other than the displayed one (badge / pill signal). */
  backgroundBusyCount: number;
}

export interface UseGeneralistAgentArgs {
  wizardState: WizardState;
  trustMode: TrustMode;
  onToolStart?: (ev: ToolStartPayload, ctx: SessionEventContext) => void;
  onToolEnd?: (ev: ToolEndPayload, ctx: SessionEventContext) => void;
  onConversationMeta?: (id: string, title: string, ctx: SessionEventContext) => void;
}

// MCP tool names that mutate the user's optimization set. When one of these
// completes successfully inside the generalist agent, dispatch the same
// window event that manual UI flows (DeleteJobDialog, bulk delete, etc.) fire,
// so the sidebar and dashboard refresh without a page reload.
const OPTIMIZATION_MUTATING_TOOLS: ReadonlySet<string> = new Set([
  "delete_job_optimizations",
  "bulk_delete_jobs_optimizations_bulk_delete_post",
  "cancel_job_optimizations",
  "bulk_cancel_jobs_optimizations_bulk_cancel_post",
  "submit_job_run_post",
  "submit_grid_search_grid_search_post",
  "rename_job_optimizations",
  "toggle_pin_job_optimizations",
  "clone_job_optimizations",
  "retry_job_optimizations",
  "bulk_pin_jobs_optimizations_bulk_pin_post",
]);

// Submit tools whose success consumes the staged dataset + readiness. After
// one fires we drop the sticky wizard overlay so the next turn starts clean
// rather than re-carrying the just-submitted run's dataset id and flags.
const SUBMIT_TOOLS: ReadonlySet<string> = new Set([
  "submit_job_run_post",
  "submit_grid_search_grid_search_post",
]);

// SSE streams hold one HTTP connection each for their whole lifetime. Over
// HTTP/1.1 the browser allows six connections per origin, so an uncapped
// fan-out would starve every other API call to the backend. Runs beyond the
// cap wait in a FIFO queue and start automatically as slots free up.
const MAX_PARALLEL_STREAMS = 4;

// The first session's key; ``keyCounterRef`` therefore starts at 1.
const INITIAL_SESSION_KEY = "session-0";

/** Per-session render state — everything the panel needs to draw one chat. */
interface SessionView {
  key: string;
  conversationId: string | null;
  status: AgentStatus;
  statusLabel: string;
  messages: AgentMessage[];
  reasoning: string;
  reasoningStartedAt: number | null;
  reasoningEndedAt: number | null;
  error: string | null;
  pendingApproval: PendingApprovalPayload | null;
}

/** A turn captured at send time, replayed verbatim when a slot frees up. */
interface PendingRun {
  message: string;
  chatHistory: ChatTurn[];
  wizardState: WizardState;
  trustMode: TrustMode;
  regenerate: boolean;
}

/** Per-session mutable machinery that must never trigger a render. */
interface SessionRuntime {
  controller: AbortController | null;
  reasoningBuf: string;
  replyBuf: string;
  // Sticky overlay of wizard fields derived synchronously this session
  // (e.g. ``staged_dataset_id`` from an in-panel upload). The panel-only
  // flow has no ``wizardCtx`` to write into, so without this every turn
  // after the upload would lose the staged dataset and the agent would
  // ask the user to re-upload.
  persistentExtras: Partial<WizardState>;
  pendingRun: PendingRun | null;
}

function blankSession(key: string): SessionView {
  return {
    key,
    conversationId: null,
    status: "idle",
    statusLabel: "",
    messages: [],
    reasoning: "",
    reasoningStartedAt: null,
    reasoningEndedAt: null,
    error: null,
    pendingApproval: null,
  };
}

function blankRuntime(): SessionRuntime {
  return {
    controller: null,
    reasoningBuf: "",
    replyBuf: "",
    persistentExtras: {},
    pendingRun: null,
  };
}

function isBusy(session: SessionView): boolean {
  return session.status === "streaming" || session.status === "queued";
}

/**
 * Multi-session generalist-agent manager. Every chat owns its own stream,
 * buffers, and abort controller, so switching threads or opening a new chat
 * never cancels work in progress — streams keep running in the background
 * and their state is re-attached when the user returns. Up to
 * ``MAX_PARALLEL_STREAMS`` turns run concurrently; further sends wait in a
 * FIFO queue. The returned state mirrors the previously single-session shape,
 * projected from whichever session is currently displayed.
 */
export function useGeneralistAgent(args: UseGeneralistAgentArgs): GeneralistAgentState {
  const { wizardState, trustMode } = args;

  const keyCounterRef = React.useRef(1);
  const nextKey = React.useCallback(() => `session-${keyCounterRef.current++}`, []);

  const [sessions, setSessions] = React.useState<ReadonlyMap<string, SessionView>>(
    () => new Map([[INITIAL_SESSION_KEY, blankSession(INITIAL_SESSION_KEY)]]),
  );
  const sessionsRef = React.useRef(sessions);
  const [activeKey, setActiveKey] = React.useState(INITIAL_SESSION_KEY);
  const activeKeyRef = React.useRef(activeKey);

  const runtimesRef = React.useRef(new Map<string, SessionRuntime>());
  const streamingKeysRef = React.useRef(new Set<string>());
  const queueRef = React.useRef<string[]>([]);

  // Seeded from the settings-modal default; the panel is client-only
  // (ssr:false), so the localStorage read is hydration-safe.
  const [model, setModelState] = React.useState<string | null>(() => readPref("composerModel"));
  // Mirrored in a ref so the streaming closures (retry, regenerate) always
  // send the current choice without re-memoizing the run machinery.
  const modelRef = React.useRef<string | null>(model);
  const setModel = React.useCallback((next: string | null) => {
    modelRef.current = next;
    setModelState(next);
  }, []);
  const [reasoningEffort, setReasoningEffortState] = React.useState<string | null>(() =>
    readPref("composerEffort"),
  );
  const effortRef = React.useRef<string | null>(reasoningEffort);
  const setReasoningEffort = React.useCallback((next: string | null) => {
    effortRef.current = next;
    setReasoningEffortState(next);
  }, []);

  // Effect ordering guarantees this snapshot is current at send time: this
  // child hook's effects run before the parent panel's, so any same-commit
  // programmatic send (e.g. the code-authoring handoff) sees the latest props.
  // Same-tick sends that must carry not-yet-committed values pass them via the
  // ``wizardStateOverride`` argument instead (see the dataset/code handoffs).
  const snapshotRef = React.useRef({ wizardState, trustMode });
  React.useEffect(() => {
    snapshotRef.current = { wizardState, trustMode };
  }, [wizardState, trustMode]);

  const callbacksRef = React.useRef<{
    onToolStart: UseGeneralistAgentArgs["onToolStart"];
    onToolEnd: UseGeneralistAgentArgs["onToolEnd"];
    onConversationMeta: UseGeneralistAgentArgs["onConversationMeta"];
  }>({
    onToolStart: args.onToolStart,
    onToolEnd: args.onToolEnd,
    onConversationMeta: args.onConversationMeta,
  });
  React.useEffect(() => {
    callbacksRef.current = {
      onToolStart: args.onToolStart,
      onToolEnd: args.onToolEnd,
      onConversationMeta: args.onConversationMeta,
    };
  }, [args.onToolStart, args.onToolEnd, args.onConversationMeta]);

  // The ref is updated synchronously with every commit so stream callbacks and
  // same-tick sends always read post-mutation state, never a stale render.
  const commit = React.useCallback(
    (mutate: (draft: Map<string, SessionView>) => void) => {
      const next = new Map(sessionsRef.current);
      mutate(next);
      sessionsRef.current = next;
      setSessions(next);
    },
    [],
  );

  const patchSession = React.useCallback(
    (key: string, patch: Partial<SessionView>) => {
      commit((draft) => {
        const cur = draft.get(key);
        if (cur) draft.set(key, { ...cur, ...patch });
      });
    },
    [commit],
  );

  const patchMessages = React.useCallback(
    (key: string, updater: (prev: AgentMessage[]) => AgentMessage[]) => {
      commit((draft) => {
        const cur = draft.get(key);
        if (!cur) return;
        const nextMessages = updater(cur.messages);
        if (nextMessages !== cur.messages) draft.set(key, { ...cur, messages: nextMessages });
      });
    },
    [commit],
  );

  const getRuntime = React.useCallback((key: string): SessionRuntime => {
    let rt = runtimesRef.current.get(key);
    if (!rt) {
      rt = blankRuntime();
      runtimesRef.current.set(key, rt);
    }
    return rt;
  }, []);

  const setActive = React.useCallback((key: string) => {
    activeKeyRef.current = key;
    setActiveKey(key);
  }, []);

  const appendReply = React.useCallback(
    (key: string, chunk: string) => {
      patchMessages(key, (prev) => {
        const last = prev[prev.length - 1];
        if (!last || last.role !== "assistant") return prev;
        const next = prev.slice();
        next[next.length - 1] = { ...last, content: last.content + chunk };
        return next;
      });
    },
    [patchMessages],
  );

  const pushToolCall = React.useCallback(
    (key: string, call: AgentToolCall) => {
      patchMessages(key, (prev) => {
        const last = prev[prev.length - 1];
        if (!last || last.role !== "assistant") return prev;
        const next = prev.slice();
        next[next.length - 1] = { ...last, toolCalls: [...(last.toolCalls ?? []), call] };
        return next;
      });
    },
    [patchMessages],
  );

  const finishToolCall = React.useCallback(
    (key: string, id: string, nextStatus: "done" | "error", result?: unknown) => {
      patchMessages(key, (prev) => {
        const last = prev[prev.length - 1];
        if (!last || last.role !== "assistant" || !last.toolCalls?.length) return prev;
        const next = prev.slice();
        next[next.length - 1] = {
          ...last,
          toolCalls: last.toolCalls.map((t) =>
            t.id === id
              ? {
                  ...t,
                  status: nextStatus,
                  endedAt: Date.now(),
                  payload: { ...(t.payload ?? {}), result },
                }
              : t,
          ),
        };
        return next;
      });
    },
    [patchMessages],
  );

  const eventContext = React.useCallback((key: string): SessionEventContext => {
    return {
      isActive: activeKeyRef.current === key,
      conversationId: sessionsRef.current.get(key)?.conversationId ?? null,
    };
  }, []);

  // ``pumpQueue`` (below) and ``startStream`` are mutually recursive: a
  // finished stream frees a slot, which starts the next queued stream. The
  // back-edge goes through a state tick — releasing a slot bumps the tick and
  // the pump effect runs on the next commit, so neither callback needs the
  // other in scope at declaration time.
  const [pumpTick, setPumpTick] = React.useState(0);
  const requestPump = React.useCallback(() => setPumpTick((t) => t + 1), []);

  const startStream = React.useCallback(
    (key: string, run: PendingRun) => {
      const rt = getRuntime(key);
      const controller = new AbortController();
      rt.controller = controller;
      rt.reasoningBuf = "";
      rt.replyBuf = "";
      streamingKeysRef.current.add(key);

      patchSession(key, {
        status: "streaming",
        statusLabel: msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.1"),
        reasoning: "",
        reasoningStartedAt: Date.now(),
        reasoningEndedAt: null,
        error: null,
        pendingApproval: null,
      });

      const releaseSlot = () => {
        rt.controller = null;
        if (streamingKeysRef.current.delete(key)) requestPump();
      };

      // Every stream callback short-circuits on ``controller.signal.aborted``
      // so a slow stream replaced by a fresh ``send`` (or by the unmount-time
      // abort) cannot leak state writes or window events into the live view.
      void streamGeneralistAgent(
        {
          user_message: run.message,
          chat_history: run.chatHistory,
          wizard_state: run.wizardState,
          trust_mode: run.trustMode,
          conversation_id: sessionsRef.current.get(key)?.conversationId ?? null,
          regenerate: run.regenerate,
          locale: getActiveLocale(),
          model: modelRef.current ?? undefined,
          reasoning_effort: effortRef.current ?? undefined,
        },
        {
          signal: controller.signal,
          onConversationMeta: (ev) => {
            if (controller.signal.aborted) return;
            if (!ev.conversation_id) return;
            patchSession(key, { conversationId: ev.conversation_id });
            callbacksRef.current.onConversationMeta?.(ev.conversation_id, ev.title, eventContext(key));
          },
          onReasoningPatch: (chunk) => {
            if (controller.signal.aborted) return;
            if (rt.reasoningBuf === "") {
              patchSession(key, { reasoningStartedAt: Date.now() });
            }
            rt.reasoningBuf += chunk;
            patchSession(key, { reasoning: rt.reasoningBuf });
          },
          onStatusPatch: (label) => {
            if (controller.signal.aborted) return;
            if (label) patchSession(key, { statusLabel: label });
          },
          onToolStart: (ev) => {
            if (controller.signal.aborted) return;
            patchSession(key, {
              statusLabel: formatMsg(
                "auto.features.agent.panel.hooks.use.generalist.agent.template.1",
                { p1: ev.tool },
              ),
            });
            pushToolCall(key, {
              id: ev.id,
              tool: ev.tool,
              reason: ev.reason,
              status: "running",
              startedAt: Date.now(),
              endedAt: null,
              payload: { arguments: ev.arguments },
            });
            callbacksRef.current.onToolStart?.(ev, eventContext(key));
          },
          onToolEnd: (ev) => {
            if (controller.signal.aborted) return;
            finishToolCall(key, ev.id, ev.status === "ok" ? "done" : "error", ev.result);
            if (ev.status === "ok") {
              if (OPTIMIZATION_MUTATING_TOOLS.has(ev.tool)) {
                window.dispatchEvent(new Event("optimizations-changed"));
              }
              if (SUBMIT_TOOLS.has(ev.tool)) {
                rt.persistentExtras = {};
              }
            }
            callbacksRef.current.onToolEnd?.(ev, eventContext(key));
          },
          onPendingApproval: (ev) => {
            if (controller.signal.aborted) return;
            patchSession(key, {
              pendingApproval: ev,
              statusLabel: msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.2"),
            });
          },
          onApprovalResolved: () => {
            if (controller.signal.aborted) return;
            patchSession(key, {
              pendingApproval: null,
              statusLabel: msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.3"),
            });
          },
          onMessagePatch: (chunk) => {
            if (controller.signal.aborted) return;
            if (rt.replyBuf === "") {
              patchSession(key, {
                statusLabel: msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.4"),
                ...(rt.reasoningBuf ? { reasoningEndedAt: Date.now() } : {}),
              });
            }
            rt.replyBuf += chunk;
            appendReply(key, chunk);
          },
          onDone: (result) => {
            if (controller.signal.aborted) return;
            patchSession(key, {
              status: "done",
              statusLabel: "",
              ...(rt.reasoningBuf ? { reasoningEndedAt: Date.now() } : {}),
            });
            patchMessages(key, (prev) => {
              const last = prev[prev.length - 1];
              if (!last || last.role !== "assistant") return prev;
              const fallback =
                last.content ||
                (last.toolCalls?.length
                  ? ""
                  : msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.5"));
              const next = prev.slice();
              next[next.length - 1] = {
                ...last,
                content: result.assistant_message || fallback,
                model: result.model,
                servedModel: result.served_model,
              };
              return next;
            });
            releaseSlot();
          },
          onError: (message, code) => {
            if (controller.signal.aborted) return;
            patchSession(key, {
              status: "error",
              statusLabel: msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.6"),
              // Known machine codes get a localized message; anything else
              // shows the backend's text as-is.
              error:
                code === "context_too_long" ? msg("agent.error.context_too_long") : message,
            });
            patchMessages(key, (prev) => {
              const last = prev[prev.length - 1];
              if (!last || last.role !== "assistant") return prev;
              if (!last.content && !last.toolCalls?.length) {
                // Nothing rendered yet — drop the empty assistant placeholder.
                return prev.slice(0, -1);
              }
              // Mark any in-flight tool pills as errored so they stop
              // spinning forever when the SSE socket dies mid-tool (e.g.
              // backend restart, network drop). Without this the chip
              // sits at "פועל כעת" indefinitely with no recovery path.
              const hasRunning = last.toolCalls?.some((t) => t.status === "running");
              if (!hasRunning) return prev;
              const next = prev.slice();
              next[next.length - 1] = {
                ...last,
                toolCalls: last.toolCalls?.map((t) =>
                  t.status === "running"
                    ? { ...t, status: "error", endedAt: Date.now() }
                    : t,
                ),
              };
              return next;
            });
            releaseSlot();
          },
        },
      );
    },
    [appendReply, eventContext, finishToolCall, getRuntime, patchMessages, patchSession, pushToolCall, requestPump],
  );

  // The pump is idempotent, so the mount-time effect run and
  // identity-change re-runs are harmless no-ops.
  const pumpQueue = React.useCallback(() => {
    while (streamingKeysRef.current.size < MAX_PARALLEL_STREAMS && queueRef.current.length > 0) {
      const key = queueRef.current.shift();
      if (!key) continue;
      const rt = runtimesRef.current.get(key);
      const run = rt?.pendingRun;
      if (!rt || !run || !sessionsRef.current.has(key)) continue;
      rt.pendingRun = null;
      startStream(key, run);
    }
  }, [startStream]);
  React.useEffect(() => {
    pumpQueue();
  }, [pumpTick, pumpQueue]);

  /** Cancel a session's stream or queued turn and free its slot. */
  const abortSession = React.useCallback(
    (key: string) => {
      const rt = runtimesRef.current.get(key);
      if (rt) {
        rt.controller?.abort();
        rt.controller = null;
        rt.pendingRun = null;
      }
      queueRef.current = queueRef.current.filter((k) => k !== key);
      if (streamingKeysRef.current.delete(key)) requestPump();
    },
    [requestPump],
  );

  // Abort every in-flight stream when the hook unmounts so callbacks can't
  // resume firing into a torn-down React tree (setState-on-unmounted warnings,
  // window-event dispatch from a stale tool-end, etc.).
  React.useEffect(
    () => () => {
      for (const rt of runtimesRef.current.values()) {
        rt.controller?.abort();
        rt.controller = null;
        rt.pendingRun = null;
      }
      queueRef.current = [];
      streamingKeysRef.current.clear();
    },
    [],
  );

  const requestRun = React.useCallback(
    (
      key: string,
      userMessage: string,
      history: AgentMessage[],
      wizardStateOverride?: WizardState,
      regenerate = false,
    ) => {
      // A resend within the same session replaces its own in-flight or queued
      // turn — other sessions' streams are untouched.
      abortSession(key);
      const rt = getRuntime(key);
      rt.reasoningBuf = "";
      rt.replyBuf = "";

      const chatHistory: ChatTurn[] = history
        .filter((m) => m.content.trim().length > 0)
        .map((m) => ({ role: m.role, content: m.content }));
      // Merge order: snapshot (from wizardCtx if mounted) <- sticky extras
      // (accumulated across turns from panel-side derivations like
      // ``staged_dataset_id``) <- this-turn override. The sticky layer
      // is what survives between turns when there's no wizardCtx writer.
      const { wizardState: snapshotWs, trustMode: tm } = snapshotRef.current;
      const ws = {
        ...snapshotWs,
        ...rt.persistentExtras,
        ...(wizardStateOverride ?? {}),
      };
      const run: PendingRun = {
        message: userMessage,
        chatHistory,
        wizardState: ws,
        trustMode: tm,
        regenerate,
      };

      commit((draft) => {
        const cur = draft.get(key);
        if (!cur) return;
        draft.set(key, {
          ...cur,
          messages: [
            ...cur.messages,
            { role: "user", content: userMessage },
            { role: "assistant", content: "", toolCalls: [] },
          ],
          reasoning: "",
          reasoningStartedAt: null,
          reasoningEndedAt: null,
          error: null,
          pendingApproval: null,
        });
      });

      if (streamingKeysRef.current.size < MAX_PARALLEL_STREAMS) {
        startStream(key, run);
      } else {
        rt.pendingRun = run;
        queueRef.current.push(key);
        patchSession(key, {
          status: "queued",
          statusLabel: msg("agent.parallel.queued"),
        });
      }
    },
    [abortSession, commit, getRuntime, patchSession, startStream],
  );

  const send = React.useCallback(
    (message: string, wizardStateOverride?: WizardState) => {
      const trimmed = message.trim();
      if (!trimmed) return;
      const key = activeKeyRef.current;
      if (wizardStateOverride) {
        const rt = getRuntime(key);
        rt.persistentExtras = {
          ...rt.persistentExtras,
          ...wizardStateOverride,
        };
      }
      const history = sessionsRef.current.get(key)?.messages ?? [];
      requestRun(key, trimmed, history, wizardStateOverride);
    },
    [getRuntime, requestRun],
  );

  const editAndResend = React.useCallback(
    (messageIndex: number, content: string) => {
      const trimmed = content.trim();
      if (!trimmed) return;
      const key = activeKeyRef.current;
      const truncated = (sessionsRef.current.get(key)?.messages ?? []).slice(0, messageIndex);
      requestRun(key, trimmed, truncated);
    },
    [requestRun],
  );

  // Re-run the most recent user turn (used by the error-banner retry button
  // and by the end-of-conversation regenerate action). Truncates back to the
  // user message so we don't re-feed a failed assistant turn into history.
  const retry = React.useCallback(() => {
    const key = activeKeyRef.current;
    const current = sessionsRef.current.get(key)?.messages ?? [];
    let lastUserIndex = -1;
    for (let i = current.length - 1; i >= 0; i--) {
      if (current[i]?.role === "user") {
        lastUserIndex = i;
        break;
      }
    }
    if (lastUserIndex === -1) return;
    const lastUser = current[lastUserIndex];
    if (!lastUser) return;
    const truncated = current.slice(0, lastUserIndex);
    requestRun(key, lastUser.content, truncated, undefined, true);
  }, [requestRun]);

  const stop = React.useCallback(() => {
    const key = activeKeyRef.current;
    const wasQueued = sessionsRef.current.get(key)?.status === "queued";
    abortSession(key);
    patchSession(key, { status: "idle", statusLabel: "" });
    if (wasQueued) {
      // Nothing streamed yet — drop the empty assistant placeholder so the
      // transcript doesn't keep a hollow bubble for a turn that never ran.
      patchMessages(key, (prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant" && !last.content && !last.toolCalls?.length) {
          return prev.slice(0, -1);
        }
        return prev;
      });
    }
  }, [abortSession, patchMessages, patchSession]);

  /**
   * Drop background sessions that are safely re-openable: idle/done ones whose
   * transcript lives on the server (or that are empty). Busy and errored
   * sessions are kept so running work and unseen failures survive switching.
   */
  const pruneInactive = React.useCallback(
    (draft: Map<string, SessionView>, keepKey: string) => {
      for (const [key, session] of draft) {
        if (key === keepKey) continue;
        if (isBusy(session) || session.status === "error") continue;
        if (session.conversationId === null && session.messages.length > 0) continue;
        draft.delete(key);
        runtimesRef.current.delete(key);
      }
    },
    [],
  );

  const newSession = React.useCallback(() => {
    const cur = sessionsRef.current.get(activeKeyRef.current);
    if (cur && cur.status === "idle" && cur.messages.length === 0 && cur.conversationId === null) {
      return;
    }
    const key = nextKey();
    commit((draft) => {
      draft.set(key, blankSession(key));
      pruneInactive(draft, key);
    });
    setActive(key);
  }, [commit, nextKey, pruneInactive, setActive]);

  const activateConversation = React.useCallback(
    (id: string): boolean => {
      let found: string | null = null;
      for (const [key, session] of sessionsRef.current) {
        if (session.conversationId === id) {
          found = key;
          break;
        }
      }
      if (found === null) return false;
      const key = found;
      commit((draft) => pruneInactive(draft, key));
      setActive(key);
      return true;
    },
    [commit, pruneInactive, setActive],
  );

  const openConversation = React.useCallback(
    (id: string, loaded: AgentMessage[]) => {
      const key = nextKey();
      commit((draft) => {
        draft.set(key, { ...blankSession(key), conversationId: id, messages: loaded });
        pruneInactive(draft, key);
      });
      setActive(key);
    },
    [commit, nextKey, pruneInactive, setActive],
  );

  const discardConversation = React.useCallback(
    (id: string) => {
      let found: string | null = null;
      for (const [key, session] of sessionsRef.current) {
        if (session.conversationId === id) {
          found = key;
          break;
        }
      }
      if (found === null) return;
      const key = found;
      abortSession(key);
      runtimesRef.current.delete(key);
      const wasActive = activeKeyRef.current === key;
      if (wasActive) {
        const freshKey = nextKey();
        commit((draft) => {
          draft.delete(key);
          draft.set(freshKey, blankSession(freshKey));
        });
        setActive(freshKey);
      } else {
        commit((draft) => {
          draft.delete(key);
        });
      }
    },
    [abortSession, commit, nextKey, setActive],
  );

  const active = sessions.get(activeKey) ?? blankSession(activeKey);

  const confirmApproval = React.useCallback(
    async (approved: boolean) => {
      const key = activeKeyRef.current;
      const pa = sessionsRef.current.get(key)?.pendingApproval;
      if (!pa) return;
      patchSession(key, { pendingApproval: null });
      const resolved = await confirmGeneralistApproval(pa.id, approved);
      if (!resolved) {
        // The confirm never reached the stream's process (network blip or a
        // different replica) — silently dropping it leaves the tool hanging
        // with no feedback. Restore the card so the user can retry.
        patchSession(key, { pendingApproval: pa });
        toast.error(msg("agent.approval.confirm_failed"));
      }
    },
    [patchSession],
  );

  const busyConversationIds = React.useMemo(() => {
    const out = new Set<string>();
    for (const session of sessions.values()) {
      if (isBusy(session) && session.conversationId) out.add(session.conversationId);
    }
    return out;
  }, [sessions]);

  const backgroundBusyCount = React.useMemo(() => {
    let count = 0;
    for (const [key, session] of sessions) {
      if (key !== activeKey && isBusy(session)) count++;
    }
    return count;
  }, [sessions, activeKey]);

  return {
    status: active.status,
    statusLabel: active.statusLabel,
    messages: active.messages,
    reasoning: active.reasoning,
    reasoningStartedAt: active.reasoningStartedAt,
    reasoningEndedAt: active.reasoningEndedAt,
    error: active.error,
    pendingApproval: active.pendingApproval,
    conversationId: active.conversationId,
    model,
    setModel,
    reasoningEffort,
    setReasoningEffort,
    send,
    editAndResend,
    retry,
    stop,
    confirmApproval,
    newSession,
    activateConversation,
    openConversation,
    discardConversation,
    busyConversationIds,
    backgroundBusyCount,
  };
}
