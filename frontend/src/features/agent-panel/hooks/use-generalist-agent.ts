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
  reset: () => void;
  confirmApproval: (approved: boolean) => Promise<void>;
  loadConversation: (id: string, messages: AgentMessage[]) => void;
}

export interface UseGeneralistAgentArgs {
  wizardState: WizardState;
  trustMode: TrustMode;
  onToolStart?: (ev: ToolStartPayload) => void;
  onToolEnd?: (ev: ToolEndPayload) => void;
  onConversationMeta?: (id: string, title: string) => void;
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

export function useGeneralistAgent(args: UseGeneralistAgentArgs): GeneralistAgentState {
  const { wizardState, trustMode } = args;

  const [status, setStatus] = React.useState<AgentStatus>("idle");
  const [statusLabel, setStatusLabel] = React.useState("");
  const [messages, setMessages] = React.useState<AgentMessage[]>([]);
  const [reasoning, setReasoning] = React.useState("");
  const [reasoningStartedAt, setReasoningStartedAt] = React.useState<number | null>(null);
  const [reasoningEndedAt, setReasoningEndedAt] = React.useState<number | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [pendingApproval, setPendingApproval] = React.useState<PendingApprovalPayload | null>(null);
  const [conversationId, setConversationId] = React.useState<string | null>(null);
  // Seeded from the settings-modal default; the panel is client-only
  // (ssr:false), so the localStorage read is hydration-safe.
  const [model, setModelState] = React.useState<string | null>(() => readPref("composerModel"));
  // Mirrored in a ref so the streaming closures (retry, regenerate) always
  // send the current choice without re-memoizing runAgent.
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

  const abortRef = React.useRef<AbortController | null>(null);
  const reasoningBufRef = React.useRef("");
  const replyBufRef = React.useRef("");
  const messagesRef = React.useRef<AgentMessage[]>(messages);
  React.useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  const conversationIdRef = React.useRef<string | null>(null);
  React.useEffect(() => {
    conversationIdRef.current = conversationId;
  }, [conversationId]);

  // Effect ordering guarantees this snapshot is current at send time: this
  // child hook's effects run before the parent panel's, so any same-commit
  // programmatic send (e.g. the code-authoring handoff) sees the latest props.
  // Same-tick sends that must carry not-yet-committed values pass them via the
  // ``wizardStateOverride`` argument instead (see the dataset/code handoffs).
  const snapshotRef = React.useRef({ wizardState, trustMode });
  React.useEffect(() => {
    snapshotRef.current = { wizardState, trustMode };
  }, [wizardState, trustMode]);

  // Sticky overlay of wizard fields derived synchronously this session
  // (e.g. ``staged_dataset_id`` from an in-panel upload). The panel-only
  // flow has no ``wizardCtx`` to write into, so without this every turn
  // after the upload would lose the staged dataset and the agent would
  // ask the user to re-upload. Cleared on ``reset`` so a fresh thread
  // starts clean.
  const persistentExtrasRef = React.useRef<Partial<WizardState>>({});

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

  // Abort any in-flight stream when the hook unmounts so callbacks can't
  // resume firing into a torn-down React tree (setState-on-unmounted warnings,
  // window-event dispatch from a stale tool-end, etc.).
  React.useEffect(
    () => () => {
      abortRef.current?.abort();
      abortRef.current = null;
    },
    [],
  );

  const appendReply = React.useCallback((chunk: string) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (!last || last.role !== "assistant") return prev;
      const next = prev.slice();
      next[next.length - 1] = { ...last, content: last.content + chunk };
      return next;
    });
  }, []);

  const pushToolCall = React.useCallback((call: AgentToolCall) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (!last || last.role !== "assistant") return prev;
      const next = prev.slice();
      next[next.length - 1] = { ...last, toolCalls: [...(last.toolCalls ?? []), call] };
      return next;
    });
  }, []);

  const finishToolCall = React.useCallback(
    (id: string, nextStatus: "done" | "error", result?: unknown) => {
      setMessages((prev) => {
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
    [],
  );

  const runAgent = React.useCallback(
    (
      userMessage: string,
      history: AgentMessage[],
      wizardStateOverride?: WizardState,
      regenerate = false,
    ) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      reasoningBufRef.current = "";
      replyBufRef.current = "";

      setStatus("streaming");
      setStatusLabel(msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.1"));
      setReasoning("");
      setReasoningStartedAt(Date.now());
      setReasoningEndedAt(null);
      setError(null);
      setPendingApproval(null);

      setMessages((m) => [
        ...m,
        { role: "user", content: userMessage },
        { role: "assistant", content: "", toolCalls: [] },
      ]);

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
        ...persistentExtrasRef.current,
        ...(wizardStateOverride ?? {}),
      };

      // Every stream callback short-circuits on ``controller.signal.aborted``
      // so a slow stream replaced by a fresh ``send`` (or by the unmount-time
      // abort) cannot leak state writes or window events into the live view.
      void streamGeneralistAgent(
        {
          user_message: userMessage,
          chat_history: chatHistory,
          wizard_state: ws,
          trust_mode: tm,
          conversation_id: conversationIdRef.current,
          regenerate,
          locale: getActiveLocale(),
          model: modelRef.current ?? undefined,
          reasoning_effort: effortRef.current ?? undefined,
        },
        {
          signal: controller.signal,
          onConversationMeta: (ev) => {
            if (controller.signal.aborted) return;
            if (!ev.conversation_id) return;
            setConversationId(ev.conversation_id);
            callbacksRef.current.onConversationMeta?.(ev.conversation_id, ev.title);
          },
          onReasoningPatch: (chunk) => {
            if (controller.signal.aborted) return;
            if (reasoningBufRef.current === "") {
              setReasoningStartedAt(Date.now());
            }
            reasoningBufRef.current += chunk;
            setReasoning(reasoningBufRef.current);
          },
          onStatusPatch: (label) => {
            if (controller.signal.aborted) return;
            if (label) setStatusLabel(label);
          },
          onToolStart: (ev) => {
            if (controller.signal.aborted) return;
            setStatusLabel(
              formatMsg("auto.features.agent.panel.hooks.use.generalist.agent.template.1", {
                p1: ev.tool,
              }),
            );
            pushToolCall({
              id: ev.id,
              tool: ev.tool,
              reason: ev.reason,
              status: "running",
              startedAt: Date.now(),
              endedAt: null,
              payload: { arguments: ev.arguments },
            });
            callbacksRef.current.onToolStart?.(ev);
          },
          onToolEnd: (ev) => {
            if (controller.signal.aborted) return;
            finishToolCall(ev.id, ev.status === "ok" ? "done" : "error", ev.result);
            if (ev.status === "ok") {
              if (OPTIMIZATION_MUTATING_TOOLS.has(ev.tool)) {
                window.dispatchEvent(new Event("optimizations-changed"));
              }
              if (SUBMIT_TOOLS.has(ev.tool)) {
                persistentExtrasRef.current = {};
              }
            }
            callbacksRef.current.onToolEnd?.(ev);
          },
          onPendingApproval: (ev) => {
            if (controller.signal.aborted) return;
            setPendingApproval(ev);
            setStatusLabel(msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.2"));
          },
          onApprovalResolved: () => {
            if (controller.signal.aborted) return;
            setPendingApproval(null);
            setStatusLabel(msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.3"));
          },
          onMessagePatch: (chunk) => {
            if (controller.signal.aborted) return;
            if (replyBufRef.current === "") {
              setStatusLabel(msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.4"));
              if (reasoningBufRef.current) setReasoningEndedAt(Date.now());
            }
            replyBufRef.current += chunk;
            appendReply(chunk);
          },
          onDone: (result) => {
            if (controller.signal.aborted) return;
            setStatus("done");
            setStatusLabel("");
            if (reasoningBufRef.current) setReasoningEndedAt(Date.now());
            setMessages((prev) => {
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
          },
          onError: (message) => {
            if (controller.signal.aborted) return;
            setStatus("error");
            setStatusLabel(msg("auto.features.agent.panel.hooks.use.generalist.agent.literal.6"));
            setError(message);
            setMessages((prev) => {
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
          },
        },
      );
    },
    [appendReply, pushToolCall, finishToolCall],
  );

  const send = React.useCallback(
    (message: string, wizardStateOverride?: WizardState) => {
      const trimmed = message.trim();
      if (!trimmed) return;
      if (wizardStateOverride) {
        persistentExtrasRef.current = {
          ...persistentExtrasRef.current,
          ...wizardStateOverride,
        };
      }
      runAgent(trimmed, messagesRef.current, wizardStateOverride);
    },
    [runAgent],
  );

  const editAndResend = React.useCallback(
    (messageIndex: number, content: string) => {
      const trimmed = content.trim();
      if (!trimmed) return;
      abortRef.current?.abort();
      abortRef.current = null;
      const truncated = messagesRef.current.slice(0, messageIndex);
      setMessages(truncated);
      messagesRef.current = truncated;
      runAgent(trimmed, truncated);
    },
    [runAgent],
  );

  // Re-run the most recent user turn (used by the error-banner retry button
  // and by the end-of-conversation regenerate action). Truncates back to the
  // user message so we don't re-feed a failed assistant turn into history.
  const retry = React.useCallback(() => {
    const current = messagesRef.current;
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
    abortRef.current?.abort();
    abortRef.current = null;
    const truncated = current.slice(0, lastUserIndex);
    setMessages(truncated);
    messagesRef.current = truncated;
    runAgent(lastUser.content, truncated, undefined, true);
  }, [runAgent]);

  const stop = React.useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus("idle");
    setStatusLabel("");
  }, []);

  const reset = React.useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    reasoningBufRef.current = "";
    replyBufRef.current = "";
    persistentExtrasRef.current = {};
    setMessages([]);
    setStatus("idle");
    setStatusLabel("");
    setReasoning("");
    setReasoningStartedAt(null);
    setReasoningEndedAt(null);
    setError(null);
    setPendingApproval(null);
    setConversationId(null);
  }, []);

  // Replace the in-memory state with a persisted conversation's id +
  // messages, so the panel can switch threads without losing rehydrated
  // tool-call rendering. The next ``send`` will continue this conversation
  // by passing the id back to the server.
  const loadConversation = React.useCallback(
    (id: string, loaded: AgentMessage[]) => {
      abortRef.current?.abort();
      abortRef.current = null;
      reasoningBufRef.current = "";
      replyBufRef.current = "";
      persistentExtrasRef.current = {};
      setMessages(loaded);
      setStatus("idle");
      setStatusLabel("");
      setReasoning("");
      setReasoningStartedAt(null);
      setReasoningEndedAt(null);
      setError(null);
      setPendingApproval(null);
      setConversationId(id);
    },
    [],
  );

  const confirmApproval = React.useCallback(
    async (approved: boolean) => {
      const pa = pendingApproval;
      if (!pa) return;
      setPendingApproval(null);
      const resolved = await confirmGeneralistApproval(pa.id, approved);
      if (!resolved) {
        // The confirm never reached the stream's process (network blip or a
        // different replica) — silently dropping it leaves the tool hanging
        // with no feedback. Restore the card so the user can retry.
        setPendingApproval(pa);
        toast.error(msg("agent.approval.confirm_failed"));
      }
    },
    [pendingApproval],
  );

  return {
    status,
    statusLabel,
    messages,
    reasoning,
    reasoningStartedAt,
    reasoningEndedAt,
    error,
    pendingApproval,
    conversationId,
    model,
    setModel,
    reasoningEffort,
    setReasoningEffort,
    send,
    editAndResend,
    retry,
    stop,
    reset,
    confirmApproval,
    loadConversation,
  };
}
