"use client";

import * as React from "react";
import { toast } from "react-toastify";
import { formatMsg, msg } from "@/shared/lib/messages";

import { streamCodeAgent, type CodeAgentToolName } from "@/shared/lib/api";
import { getActiveLocale } from "@/shared/lib/runtime-locale";
import { LOCALE_RELOAD_EVENT } from "@/shared/lib/locale";
import { TERMS } from "@/shared/lib/terms";
import type { ParsedDataset } from "@/shared/lib/parse-dataset";
import type { ValidateCodeResponse, WorkflowSpec } from "@/shared/types/api";

type AgentStatus = "idle" | "streaming" | "done" | "error";
type AgentMode = "seed" | "chat";
export type ArtifactStatus = "idle" | "waiting" | "writing" | "done";

type AgentToolName = CodeAgentToolName;
type AgentToolStatus = "running" | "done" | "error";

export interface AgentToolCall {
  id: string;
  tool: AgentToolName;
  reason: string;
  status: AgentToolStatus;
  startedAt: number;
  endedAt: number | null;
  prevCode?: string;
  newCode?: string;
}

// Per-index line diff — returns 1-based line numbers in `next` that differ
// from `prev`. Good enough for flashing small refinements; if the agent
// restructures heavily, most lines will flash (acceptable).
function diffChangedLines(prev: string, next: string): number[] {
  if (!prev.trim()) return [];
  const prevLines = prev.split("\n");
  const nextLines = next.split("\n");
  const changed: number[] = [];
  for (let i = 0; i < nextLines.length; i++) {
    if (nextLines[i] !== prevLines[i]) changed.push(i + 1);
  }
  return changed;
}

// Compact the backend ValidateCodeResponse into a short string the agent can
// read on the next turn — gives it concrete "what's broken" context so the
// retry is targeted, not another blind rewrite.
function summarizeValidation(v: ValidateCodeResponse | null): string {
  if (!v) return "";
  if (v.valid && v.warnings.length === 0) return "OK";
  const parts: string[] = [];
  if (v.errors.length > 0) parts.push(`errors: ${v.errors.join(" | ")}`);
  if (v.warnings.length > 0) parts.push(`warnings: ${v.warnings.join(" | ")}`);
  return parts.join("; ");
}

const MAX_AUTO_FIX = 2;

// Synthetic opener shown as a user bubble before the seed reply. It gives
// the conversation a natural starting point — the AI's first message reads
// as a response to a real request, not a monologue — and lets the user
// revise it via the edit pencil to steer the initial generation. Resolved on
// call (not at module scope) so it can't capture a raw key before the message
// shim has loaded.
function seedUserMessage(): string {
  return formatMsg("auto.features.submit.hooks.use.code.agent.template.1", {
    p1: TERMS.dataset,
  });
}

// Seed mode runs two authors (signature+metric, or graph+metric) in parallel
// over one multiplexed SSE stream. Keyed buffers keep each stream's reasoning
// contiguous; multi-stream runs render as labeled sections instead of
// token-interleaved gibberish.
interface ReasoningSections {
  order: string[];
  bufs: Record<string, string>;
}

function reasoningSourceLabel(source: string): string {
  if (source === "signature") return TERMS.signature;
  if (source === "metric") return TERMS.metric;
  // Module names are technical terms kept in English throughout the UI.
  if (source === "workflow") return "Workflow";
  return source;
}

function composeReasoning(sections: ReasoningSections): string {
  const first = sections.order[0];
  if (first === undefined) return "";
  if (sections.order.length === 1) return sections.bufs[first] ?? "";
  return sections.order
    .map((key) => `[${reasoningSourceLabel(key)}]\n${sections.bufs[key] ?? ""}`)
    .join("\n\n");
}

interface AgentMessage {
  role: "assistant" | "user";
  content: string;
  toolCalls?: AgentToolCall[];
  model?: string | null;
}

interface ArtifactVersion {
  code: string;
  ts: number;
}

// A locale switch reloads the page (see LocaleProvider), which would drop the
// conversation. Consumers that pass `reloadPersistKey` get their transcript
// stashed in sessionStorage for that single hop and restored (then cleared) on
// the next mount — the same pattern as the submit wizard's draft stash.
interface AgentReloadStash {
  messages: AgentMessage[];
  mode: AgentMode;
  signatureVersions: ArtifactVersion[];
  metricVersions: ArtifactVersion[];
  signatureVersionIndex: number;
  metricVersionIndex: number;
  reasoning: string;
  reasoningStartedAt: number | null;
  reasoningEndedAt: number | null;
}

function reloadStashKey(persistKey: string): string {
  return `skynet.code-agent.reload-stash:${persistKey}`;
}

function readAgentReloadStash(persistKey: string): AgentReloadStash | null {
  try {
    const raw = window.sessionStorage.getItem(reloadStashKey(persistKey));
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AgentReloadStash;
    return Array.isArray(parsed?.messages) && parsed.messages.length > 0 ? parsed : null;
  } catch {
    return null;
  }
}

export interface CodeAgentState {
  status: AgentStatus;
  mode: AgentMode;
  statusLabel: string;
  signatureStatus: ArtifactStatus;
  metricStatus: ArtifactStatus;
  messages: AgentMessage[];
  error: string | null;
  canSend: boolean;
  signatureVersions: ArtifactVersion[];
  metricVersions: ArtifactVersion[];
  signatureVersionIndex: number;
  metricVersionIndex: number;
  signatureFlashLines: number[];
  metricFlashLines: number[];
  reasoning: string;
  reasoningStartedAt: number | null;
  reasoningEndedAt: number | null;
  goToSignatureVersion: (index: number) => void;
  goToMetricVersion: (index: number) => void;
  send: (message: string) => void;
  editAndResend: (messageIndex: number, content: string) => void;
  retry: () => void;
  fallbackToManual: () => void;
  stop: () => void;
  reset: () => void;
}

export interface UseCodeAgentArgs {
  codeAssistMode: "auto" | "manual";
  setCodeAssistMode: (m: "auto" | "manual") => void;
  // Plain string roles (input/output/ignore); this hook only reads the
  // "input"/"output" roles.
  columnRoles: Record<string, string>;
  columnKinds: Record<string, "text" | "image">;
  parsedDataset: ParsedDataset | null;
  moduleName: string;
  signatureCode: string;
  metricCode: string;
  setSignatureCode: (v: string) => void;
  setMetricCode: (v: string) => void;
  signatureManuallyEdited: boolean;
  metricManuallyEdited: boolean;
  setSignatureManuallyEdited: (v: boolean) => void;
  setMetricManuallyEdited: (v: boolean) => void;
  setSignatureValidation: (v: ValidateCodeResponse | null) => void;
  setMetricValidation: (v: ValidateCodeResponse | null) => void;
  signatureValidation: ValidateCodeResponse | null;
  metricValidation: ValidateCodeResponse | null;
  runSignatureValidation: (overrideCode?: string) => Promise<unknown>;
  runMetricValidation: (overrideCode?: string) => Promise<unknown>;
  // Workflow mode (module_name === "workflow"): the agent authors the graph
  // instead of a single signature. `workflowTouched` guards the auto-seed
  // like the manual-edit flags do for code; `applyAgentWorkflow` lands a
  // graph snapshot on the canvas (with the changed node pulsed).
  isWorkflow?: boolean;
  workflowSpec?: WorkflowSpec | null;
  workflowTouched?: boolean;
  applyAgentWorkflow?: (spec: WorkflowSpec, changedNodeId: string | null) => void;
  // Gates the auto-seed while the wizard's module picker is still open; the
  // seed fires as soon as this flips true (i.e. the user picked a module).
  seedEnabled?: boolean;
  // When set, the conversation survives the locale-switch reload under this
  // stash key. Leave unset for surfaces that shouldn't persist.
  reloadPersistKey?: string;
}

export function useCodeAgent(args: UseCodeAgentArgs): CodeAgentState {
  const {
    codeAssistMode,
    setCodeAssistMode,
    columnRoles,
    columnKinds,
    parsedDataset,
    moduleName,
    signatureCode,
    metricCode,
    setSignatureCode,
    setMetricCode,
    signatureManuallyEdited,
    metricManuallyEdited,
    setSignatureManuallyEdited,
    setMetricManuallyEdited,
    setSignatureValidation,
    setMetricValidation,
    signatureValidation,
    metricValidation,
    runSignatureValidation,
    runMetricValidation,
    isWorkflow = false,
    workflowSpec = null,
    workflowTouched = false,
    applyAgentWorkflow,
    seedEnabled = true,
    reloadPersistKey,
  } = args;

  // Read the reload stash once at mount, before the initializers below
  // consume it. Reading is idempotent — the stash is only cleared in the
  // mount effect further down — so StrictMode's double-invoked initializer
  // can't read-then-lose it.
  const [restored] = React.useState<AgentReloadStash | null>(() =>
    reloadPersistKey && typeof window !== "undefined"
      ? readAgentReloadStash(reloadPersistKey)
      : null,
  );

  const [status, setStatus] = React.useState<AgentStatus>("idle");
  const [mode, setMode] = React.useState<AgentMode>(restored?.mode ?? "seed");
  const [statusLabel, setStatusLabel] = React.useState("");
  const [signatureStatus, setSignatureStatus] = React.useState<ArtifactStatus>(
    restored && restored.signatureVersions.length > 0 ? "done" : "idle",
  );
  const [metricStatus, setMetricStatus] = React.useState<ArtifactStatus>(
    restored && restored.metricVersions.length > 0 ? "done" : "idle",
  );
  const [messages, setMessages] = React.useState<AgentMessage[]>(restored?.messages ?? []);
  const [error, setError] = React.useState<string | null>(null);
  const [signatureVersions, setSignatureVersions] = React.useState<ArtifactVersion[]>(
    restored?.signatureVersions ?? [],
  );
  const [metricVersions, setMetricVersions] = React.useState<ArtifactVersion[]>(
    restored?.metricVersions ?? [],
  );
  const [signatureVersionIndex, setSignatureVersionIndex] = React.useState(
    restored?.signatureVersionIndex ?? -1,
  );
  const [metricVersionIndex, setMetricVersionIndex] = React.useState(
    restored?.metricVersionIndex ?? -1,
  );
  const [signatureFlashLines, setSignatureFlashLines] = React.useState<number[]>([]);
  const [metricFlashLines, setMetricFlashLines] = React.useState<number[]>([]);
  const [reasoning, setReasoning] = React.useState(restored?.reasoning ?? "");
  const [reasoningStartedAt, setReasoningStartedAt] = React.useState<number | null>(
    restored?.reasoningStartedAt ?? null,
  );
  const [reasoningEndedAt, setReasoningEndedAt] = React.useState<number | null>(
    restored?.reasoningEndedAt ?? null,
  );

  const abortRef = React.useRef<AbortController | null>(null);
  const sigBufRef = React.useRef("");
  const metricBufRef = React.useRef("");
  const replyBufRef = React.useRef("");
  const reasoningBufRef = React.useRef("");
  const reasoningSectionsRef = React.useRef<ReasoningSections>({ order: [], bufs: {} });
  // Latches once per run: the timer stops the moment the model moves from
  // reasoning to producing artifacts, even if a parallel stream reasons on.
  const reasoningEndedRef = React.useRef(false);
  // A restored transcript already contains the seed exchange — never re-seed
  // over it after the locale-switch reload.
  const autoRanRef = React.useRef(restored != null);
  const [sessionKey, setSessionKey] = React.useState(0);
  const pendingValidationsRef = React.useRef<
    Array<{ kind: "signature" | "metric"; promise: Promise<unknown> }>
  >([]);
  const autoFixAttemptsRef = React.useRef(0);
  const runAgentRef = React.useRef<((msg: string, hist: AgentMessage[]) => void) | null>(null);
  const messagesRef = React.useRef<AgentMessage[]>([]);
  const flashClearRef = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // Latest values for the stream callbacks — kept in a ref so closures
  // created inside `runAgent` stay in sync without re-creating the callback.
  const snapshotRef = React.useRef({
    signatureCode,
    metricCode,
    signatureValidation,
    metricValidation,
    workflowSpec,
  });
  React.useEffect(() => {
    snapshotRef.current = {
      signatureCode,
      metricCode,
      signatureValidation,
      metricValidation,
      workflowSpec,
    };
  }, [signatureCode, metricCode, signatureValidation, metricValidation, workflowSpec]);

  // Revert support across the conversation (first graph ever seen).
  const initialWorkflowRef = React.useRef<WorkflowSpec | null>(null);
  const applyAgentWorkflowRef = React.useRef(applyAgentWorkflow);
  React.useEffect(() => {
    applyAgentWorkflowRef.current = applyAgentWorkflow;
  }, [applyAgentWorkflow]);

  const runnersRef = React.useRef({ runSignatureValidation, runMetricValidation });
  React.useEffect(() => {
    runnersRef.current = { runSignatureValidation, runMetricValidation };
  }, [runSignatureValidation, runMetricValidation]);

  const versionsRef = React.useRef({ signatureVersions, metricVersions });
  React.useEffect(() => {
    versionsRef.current = { signatureVersions, metricVersions };
  }, [signatureVersions, metricVersions]);

  React.useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  // Mirror the stashable transcript every render so the locale-reload
  // listener parks the freshest state, then stash it when the reload event
  // fires. The mount-time removeItem is the "restored, now consume it" half
  // of the handshake with readAgentReloadStash above.
  const reloadSnapshotRef = React.useRef<AgentReloadStash | null>(null);
  React.useEffect(() => {
    reloadSnapshotRef.current = {
      messages,
      mode,
      signatureVersions,
      metricVersions,
      signatureVersionIndex,
      metricVersionIndex,
      reasoning,
      reasoningStartedAt,
      reasoningEndedAt,
    };
  });
  React.useEffect(() => {
    if (!reloadPersistKey) return;
    window.sessionStorage.removeItem(reloadStashKey(reloadPersistKey));
    const stash = () => {
      const snap = reloadSnapshotRef.current;
      if (!snap || snap.messages.length === 0) return;
      try {
        window.sessionStorage.setItem(reloadStashKey(reloadPersistKey), JSON.stringify(snap));
      } catch {
        // Best-effort — a blown quota loses the transcript, exactly as the
        // reload did before this stash existed.
      }
    };
    window.addEventListener(LOCALE_RELOAD_EVENT, stash);
    return () => window.removeEventListener(LOCALE_RELOAD_EVENT, stash);
  }, [reloadPersistKey]);

  const hasRequiredContext = React.useMemo(() => {
    if (!parsedDataset || parsedDataset.rowCount === 0) return false;
    const hasInput = Object.values(columnRoles).some((r) => r === "input");
    const hasOutput = Object.values(columnRoles).some((r) => r === "output");
    return hasInput && hasOutput;
  }, [parsedDataset, columnRoles]);

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
      next[next.length - 1] = {
        ...last,
        toolCalls: [...(last.toolCalls ?? []), call],
      };
      return next;
    });
  }, []);

  const finishToolCall = React.useCallback((id: string, status: AgentToolStatus) => {
    setMessages((prev) => {
      const last = prev[prev.length - 1];
      if (!last || last.role !== "assistant" || !last.toolCalls?.length) return prev;
      const next = prev.slice();
      next[next.length - 1] = {
        ...last,
        toolCalls: last.toolCalls.map((t) =>
          t.id === id ? { ...t, status, endedAt: Date.now() } : t,
        ),
      };
      return next;
    });
  }, []);

  const attachCodeToLatestRunningToolCall = React.useCallback(
    (tool: AgentToolName, prevCode: string, newCode: string) => {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (!last || last.role !== "assistant" || !last.toolCalls?.length) return prev;
        const calls = last.toolCalls;
        let targetIdx = -1;
        for (let i = calls.length - 1; i >= 0; i--) {
          const c = calls[i];
          if (c && c.tool === tool && c.status === "running") {
            targetIdx = i;
            break;
          }
        }
        if (targetIdx < 0) return prev;
        const updatedCalls = calls.slice();
        const target = updatedCalls[targetIdx];
        if (!target) return prev;
        updatedCalls[targetIdx] = { ...target, prevCode, newCode };
        const next = prev.slice();
        next[next.length - 1] = { ...last, toolCalls: updatedCalls };
        return next;
      });
    },
    [],
  );

  const runAgent = React.useCallback(
    (userMessage: string, history: AgentMessage[]) => {
      if (!parsedDataset || !hasRequiredContext) return;

      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      sigBufRef.current = "";
      metricBufRef.current = "";
      replyBufRef.current = "";
      reasoningBufRef.current = "";
      reasoningSectionsRef.current = { order: [], bufs: {} };
      reasoningEndedRef.current = false;
      pendingValidationsRef.current = [];

      const isChat = userMessage.length > 0;
      setMode(isChat ? "chat" : "seed");
      setStatus("streaming");
      setStatusLabel(
        isChat
          ? msg("auto.features.submit.hooks.use.code.agent.literal.1")
          : formatMsg("auto.features.submit.hooks.use.code.agent.template.2", {
              p1: TERMS.dataset,
            }),
      );
      setSignatureStatus(isChat ? "idle" : "waiting");
      setMetricStatus(isChat ? "idle" : "waiting");
      setReasoning("");
      setReasoningStartedAt(Date.now());
      setReasoningEndedAt(null);
      setError(null);

      if (isChat) {
        setMessages((m) => [
          ...m,
          { role: "user", content: userMessage },
          { role: "assistant", content: "", toolCalls: [] },
        ]);
      } else {
        setMessages((m) => [
          ...m,
          { role: "user", content: seedUserMessage() },
          { role: "assistant", content: "", toolCalls: [] },
        ]);
      }

      const sampleRows = parsedDataset.rows.slice(0, 5) as Array<Record<string, unknown>>;
      const snapshot = snapshotRef.current;
      const { signatureVersions: sigVers, metricVersions: metVers } = versionsRef.current;
      const initialSignature = sigVers[0]?.code ?? snapshot.signatureCode;
      const initialMetric = metVers[0]?.code ?? snapshot.metricCode;
      // Track the prevCode for replace callbacks within this run. The outer
      // `snapshot` is captured at run-start, so reading it on the second
      // replace would diff against the run's starting code instead of the
      // previous version — the tool-call card would show a stale diff.
      let lastSignatureCode = snapshot.signatureCode;
      let lastMetricCode = snapshot.metricCode;
      const chatHistory = history
        .filter((m) => m.content.trim().length > 0)
        .map((m) => ({ role: m.role, content: m.content }));

      // Send only image-typed entries — the backend defaults the rest to
      // text. Keeps the payload lean and makes the wire shape symmetric
      // with the wizard's mental model: text is the default.
      const imageColumnKinds: Record<string, "image"> = {};
      for (const [col, kind] of Object.entries(columnKinds)) {
        if (kind === "image") imageColumnKinds[col] = "image";
      }

      const priorWorkflow = isWorkflow ? (snapshot.workflowSpec ?? null) : null;
      if (isWorkflow && !initialWorkflowRef.current && priorWorkflow) {
        initialWorkflowRef.current = priorWorkflow;
      }

      // Stop the thinking timer as soon as artifact output starts flowing —
      // otherwise "Thinking" keeps pulsing through the whole code-writing
      // stretch (which renders in the editors, not the chat) and the stream
      // reads as stuck.
      const markReasoningDone = () => {
        if (reasoningBufRef.current && !reasoningEndedRef.current) {
          reasoningEndedRef.current = true;
          setReasoningEndedAt(Date.now());
        }
      };

      void streamCodeAgent(
        {
          dataset_columns: parsedDataset.columns,
          column_roles: columnRoles,
          column_kinds: imageColumnKinds,
          sample_rows: sampleRows,
          user_message: userMessage,
          chat_history: chatHistory,
          prior_signature: snapshot.signatureCode,
          prior_metric: snapshot.metricCode,
          prior_signature_validation: summarizeValidation(snapshot.signatureValidation),
          prior_metric_validation: summarizeValidation(snapshot.metricValidation),
          initial_signature: initialSignature,
          initial_metric: initialMetric,
          locale: getActiveLocale(),
          ...(isWorkflow
            ? {
                prior_workflow: priorWorkflow,
                initial_workflow: initialWorkflowRef.current ?? priorWorkflow,
              }
            : {}),
        },
        {
          signal: controller.signal,
          onReasoningPatch: (chunk, source) => {
            if (reasoningBufRef.current === "") {
              setReasoningStartedAt(Date.now());
            }
            const sections = reasoningSectionsRef.current;
            if (!(source in sections.bufs)) {
              sections.order.push(source);
              sections.bufs[source] = "";
            }
            sections.bufs[source] += chunk;
            reasoningBufRef.current = composeReasoning(sections);
            setReasoning(reasoningBufRef.current);
          },
          onSignaturePatch: (chunk) => {
            if (sigBufRef.current === "") {
              setStatusLabel(msg("auto.features.submit.hooks.use.code.agent.literal.2"));
              setSignatureStatus("writing");
              markReasoningDone();
            }
            sigBufRef.current += chunk;
            setSignatureCode(sigBufRef.current);
            setSignatureValidation(null);
          },
          onMetricPatch: (chunk) => {
            if (metricBufRef.current === "") {
              setStatusLabel(msg("auto.features.submit.hooks.use.code.agent.literal.3"));
              setSignatureStatus("done");
              setMetricStatus("writing");
              markReasoningDone();
            }
            metricBufRef.current += chunk;
            setMetricCode(metricBufRef.current);
            setMetricValidation(null);
          },
          onMessagePatch: (chunk) => {
            if (replyBufRef.current === "") {
              setStatusLabel(msg("auto.features.submit.hooks.use.code.agent.literal.4"));
              markReasoningDone();
            }
            replyBufRef.current += chunk;
            appendReply(chunk);
          },
          onToolStart: (ev) => {
            markReasoningDone();
            setStatusLabel(
              ev.tool === "edit_signature"
                ? msg("auto.features.submit.hooks.use.code.agent.literal.5")
                : ev.tool === "edit_metric"
                  ? msg("auto.features.submit.hooks.use.code.agent.literal.6")
                  : msg("workflow.agent.editing_graph"),
            );
            if (ev.tool === "edit_signature") setSignatureStatus("writing");
            else if (ev.tool === "edit_metric") setMetricStatus("writing");
            pushToolCall({
              id: ev.id,
              tool: ev.tool,
              reason: ev.reason,
              status: "running",
              startedAt: Date.now(),
              endedAt: null,
            });
          },
          onToolEnd: (ev) => {
            finishToolCall(ev.id, ev.status === "ok" ? "done" : "error");
            if (ev.tool === "edit_signature") setSignatureStatus("done");
            else if (ev.tool === "edit_metric") setMetricStatus("done");
          },
          onWorkflowReplace: (workflow, changedNodeId) => {
            markReasoningDone();
            applyAgentWorkflowRef.current?.(workflow, changedNodeId);
          },
          onSignatureReplace: (code) => {
            const prevCode = lastSignatureCode;
            lastSignatureCode = code;
            setSignatureCode(code);
            setSignatureManuallyEdited(false);
            attachCodeToLatestRunningToolCall("edit_signature", prevCode, code);
            const changed = diffChangedLines(prevCode, code);
            setSignatureFlashLines(changed);
            if (flashClearRef.current) clearTimeout(flashClearRef.current);
            flashClearRef.current = setTimeout(() => {
              setSignatureFlashLines([]);
            }, 1200);
            setSignatureVersions((prev) => {
              const next = [...prev, { code, ts: Date.now() }];
              setSignatureVersionIndex(next.length - 1);
              return next;
            });
            pendingValidationsRef.current.push({
              kind: "signature",
              promise: runnersRef.current.runSignatureValidation(code),
            });
          },
          onMetricReplace: (code) => {
            const prevCode = lastMetricCode;
            lastMetricCode = code;
            setMetricCode(code);
            setMetricManuallyEdited(false);
            attachCodeToLatestRunningToolCall("edit_metric", prevCode, code);
            const changed = diffChangedLines(prevCode, code);
            setMetricFlashLines(changed);
            if (flashClearRef.current) clearTimeout(flashClearRef.current);
            flashClearRef.current = setTimeout(() => {
              setMetricFlashLines([]);
            }, 1200);
            setMetricVersions((prev) => {
              const next = [...prev, { code, ts: Date.now() }];
              setMetricVersionIndex(next.length - 1);
              return next;
            });
            pendingValidationsRef.current.push({
              kind: "metric",
              promise: runnersRef.current.runMetricValidation(code),
            });
          },
          onDone: async (result) => {
            if (!isChat && isWorkflow) {
              // Workflow seed: the graph landed via workflow_replace (or
              // rides the done payload after a repair); only the metric
              // flows through the code editor.
              if (result.workflow) {
                applyAgentWorkflowRef.current?.(result.workflow, null);
              }
              setMetricCode(result.metric_code);
              setMetricManuallyEdited(false);
              setMetricVersions((prev) => {
                const next = [...prev, { code: result.metric_code, ts: Date.now() }];
                setMetricVersionIndex(next.length - 1);
                return next;
              });
              pendingValidationsRef.current.push({
                kind: "metric",
                promise: runnersRef.current.runMetricValidation(result.metric_code),
              });
            } else if (!isChat) {
              setSignatureCode(result.signature_code);
              setMetricCode(result.metric_code);
              setSignatureManuallyEdited(false);
              setMetricManuallyEdited(false);
              const sigChanged = diffChangedLines(snapshot.signatureCode, result.signature_code);
              const metChanged = diffChangedLines(snapshot.metricCode, result.metric_code);
              setSignatureFlashLines(sigChanged);
              setMetricFlashLines(metChanged);
              if (flashClearRef.current) clearTimeout(flashClearRef.current);
              flashClearRef.current = setTimeout(() => {
                setSignatureFlashLines([]);
                setMetricFlashLines([]);
              }, 800);
              setSignatureVersions((prev) => {
                const next = [...prev, { code: result.signature_code, ts: Date.now() }];
                setSignatureVersionIndex(next.length - 1);
                return next;
              });
              setMetricVersions((prev) => {
                const next = [...prev, { code: result.metric_code, ts: Date.now() }];
                setMetricVersionIndex(next.length - 1);
                return next;
              });
              pendingValidationsRef.current.push({
                kind: "signature",
                promise: runnersRef.current.runSignatureValidation(result.signature_code),
              });
              pendingValidationsRef.current.push({
                kind: "metric",
                promise: runnersRef.current.runMetricValidation(result.metric_code),
              });
            }
            setSignatureStatus("done");
            setMetricStatus("done");
            setStatus("done");
            setStatusLabel(msg("auto.features.submit.hooks.use.code.agent.literal.7"));
            if (reasoningBufRef.current) setReasoningEndedAt(Date.now());

            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (!last || last.role !== "assistant") return prev;
              const fallback = isChat
                ? "Done."
                : "I wrote a Signature and Metric based on your data.";
              const finalContent = result.assistant_message || last.content || fallback;
              const next = prev.slice();
              next[next.length - 1] = { ...last, content: finalContent, model: result.model };
              return next;
            });

            // Auto-fix: wait for any validations kicked off during this run,
            // then if the validator flagged errors, fire a follow-up chat
            // turn so the agent can see the errors (via prior_*_validation)
            // and patch them. Bounded by MAX_AUTO_FIX to prevent loops.
            const pending = pendingValidationsRef.current;
            pendingValidationsRef.current = [];
            if (pending.length === 0) return;
            // allSettled so a single failed validation doesn't suppress
            // auto-fix for the other kind. A rejection here means the
            // validation request itself errored (network/server) — there
            // is nothing concrete for the agent to patch in that case, so
            // we skip just that entry rather than aborting the whole run.
            const settled = await Promise.allSettled(pending.map((e) => e.promise));
            const responses: Array<ValidateCodeResponse | null> = settled.map((r) =>
              r.status === "fulfilled" ? ((r.value as ValidateCodeResponse | null) ?? null) : null,
            );
            for (let i = 0; i < pending.length; i++) {
              const entry = pending[i];
              if (!entry) continue;
              const resp = responses[i] ?? null;
              if (entry.kind === "signature") {
                snapshotRef.current = {
                  ...snapshotRef.current,
                  signatureValidation: resp,
                };
              } else {
                snapshotRef.current = {
                  ...snapshotRef.current,
                  metricValidation: resp,
                };
              }
            }
            let signatureHasError = false;
            let metricHasError = false;
            for (let i = 0; i < pending.length; i++) {
              const entry = pending[i];
              const resp = responses[i];
              const broken = !!resp && !resp.valid && resp.errors.length > 0;
              if (!entry || !broken) continue;
              if (entry.kind === "signature") signatureHasError = true;
              else metricHasError = true;
            }
            if (!signatureHasError && !metricHasError) return;
            if (controller.signal.aborted) return;
            if (autoFixAttemptsRef.current >= MAX_AUTO_FIX) return;
            autoFixAttemptsRef.current += 1;
            // Name only the artifact(s) that actually failed so the retry
            // message (the chat bubble the user reads) points at the real
            // problem instead of blaming both. The per-artifact validation
            // summaries in the request body already carry the concrete errors.
            const fixKey =
              signatureHasError && metricHasError
                ? "auto.features.submit.hooks.use.code.agent.template.3"
                : metricHasError
                  ? "auto.features.submit.hooks.use.code.agent.template.5"
                  : "auto.features.submit.hooks.use.code.agent.template.4";
            const fixMessage = formatMsg(fixKey, {
              p1: TERMS.dataset,
            });
            queueMicrotask(() => {
              runAgentRef.current?.(fixMessage, messagesRef.current);
            });
          },
          onError: (message) => {
            if (controller.signal.aborted) return;
            setStatus("error");
            setStatusLabel(msg("auto.features.submit.hooks.use.code.agent.literal.8"));
            setSignatureStatus("idle");
            setMetricStatus("idle");
            setError(message);
            // Drop the empty placeholder agent bubble so the error UI
            // isn't followed by a blank message.
            setMessages((prev) => {
              const last = prev[prev.length - 1];
              if (last && last.role === "assistant" && !last.content && !last.toolCalls?.length) {
                return prev.slice(0, -1);
              }
              return prev;
            });
          },
        },
      );
    },
    [
      parsedDataset,
      hasRequiredContext,
      columnRoles,
      columnKinds,
      isWorkflow,
      setSignatureCode,
      setMetricCode,
      setSignatureValidation,
      setMetricValidation,
      setSignatureManuallyEdited,
      setMetricManuallyEdited,
      appendReply,
      pushToolCall,
      finishToolCall,
      attachCodeToLatestRunningToolCall,
    ],
  );

  React.useEffect(() => {
    runAgentRef.current = runAgent;
  }, [runAgent]);

  const send = React.useCallback(
    (message: string) => {
      const trimmed = message.trim();
      if (!trimmed) return;
      autoFixAttemptsRef.current = 0;
      runAgent(trimmed, messages);
    },
    [runAgent, messages],
  );

  const editAndResend = React.useCallback(
    (messageIndex: number, content: string) => {
      const trimmed = content.trim();
      if (!trimmed) return;
      const truncated = messages.slice(0, messageIndex);
      setMessages(truncated);
      autoFixAttemptsRef.current = 0;
      runAgent(trimmed, truncated);
    },
    [runAgent, messages],
  );

  const retry = React.useCallback(() => {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    autoFixAttemptsRef.current = 0;
    runAgent(lastUser?.content ?? "", messages);
  }, [messages, runAgent]);

  const goToSignatureVersion = React.useCallback(
    (index: number) => {
      const v = signatureVersions[index];
      if (!v) return;
      setSignatureCode(v.code);
      setSignatureManuallyEdited(false);
      setSignatureVersionIndex(index);
      void runnersRef.current.runSignatureValidation(v.code);
    },
    [signatureVersions, setSignatureCode, setSignatureManuallyEdited],
  );

  const goToMetricVersion = React.useCallback(
    (index: number) => {
      const v = metricVersions[index];
      if (!v) return;
      setMetricCode(v.code);
      setMetricManuallyEdited(false);
      setMetricVersionIndex(index);
      void runnersRef.current.runMetricValidation(v.code);
    },
    [metricVersions, setMetricCode, setMetricManuallyEdited],
  );

  const fallbackToManual = React.useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setCodeAssistMode("manual");
    toast.info(msg("auto.features.submit.hooks.use.code.agent.literal.9"));
  }, [setCodeAssistMode]);

  const stop = React.useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStatus("idle");
    setStatusLabel("");
    setSignatureStatus("idle");
    setMetricStatus("idle");
  }, []);

  const reset = React.useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    sigBufRef.current = "";
    metricBufRef.current = "";
    replyBufRef.current = "";
    reasoningBufRef.current = "";
    reasoningSectionsRef.current = { order: [], bufs: {} };
    reasoningEndedRef.current = false;
    pendingValidationsRef.current = [];
    autoFixAttemptsRef.current = 0;
    autoRanRef.current = false;
    if (flashClearRef.current) {
      clearTimeout(flashClearRef.current);
      flashClearRef.current = null;
    }
    setMessages([]);
    setStatus("idle");
    setMode("seed");
    setStatusLabel("");
    setSignatureStatus("idle");
    setMetricStatus("idle");
    setError(null);
    setReasoning("");
    setReasoningStartedAt(null);
    setReasoningEndedAt(null);
    setSignatureVersions([]);
    setMetricVersions([]);
    setSignatureVersionIndex(-1);
    setMetricVersionIndex(-1);
    setSignatureFlashLines([]);
    setMetricFlashLines([]);
    initialWorkflowRef.current = null;
    // Clear the manual-edit gate so the auto-seed effect re-fires on
    // sessionKey bump — otherwise "new chat" would wipe the messages
    // but leave the old Signature/Metric untouched.
    setSignatureManuallyEdited(false);
    setMetricManuallyEdited(false);
    setSessionKey((k) => k + 1);
  }, [setSignatureManuallyEdited, setMetricManuallyEdited]);

  // Swapping the dataset invalidates the entire conversation: messages,
  // version history, and any in-flight stream all refer to schemas that no
  // longer apply. Identity change is a reliable signal — `parseDatasetFile`
  // returns a fresh object per upload. Skipping the initial null→value
  // transition keeps the first mount a no-op.
  const prevDatasetRef = React.useRef(parsedDataset);
  React.useEffect(() => {
    if (prevDatasetRef.current !== parsedDataset) {
      if (prevDatasetRef.current !== null) {
        reset();
      } else {
        autoRanRef.current = false;
      }
      prevDatasetRef.current = parsedDataset;
    }
  }, [parsedDataset, reset]);

  // Re-arm when the user changes their DSPy module (predict ↔ chain_of_thought
  // ↔ react, …). The manual-edit gate still protects user-authored edits;
  // fresh seed output just flips to the new module's expected shape. After a
  // locale-reload restore, module-name churn is hydration rather than a user
  // switch — re-arming then would seed over the restored transcript, and real
  // switches reset the conversation via the wizard's chooseModule() anyway.
  React.useEffect(() => {
    if (restored != null) return;
    autoRanRef.current = false;
  }, [moduleName, restored]);

  // Kick off the seed run as soon as the user has a dataset + I/O roles.
  // This hook lives at the wizard level, so the seed fires even when the
  // user hasn't arrived at the code step yet — by the time they do, the
  // editors are already filled (or actively filling). We skip the seed
  // if either artifact has been manually authored (including clone-pre-fill,
  // which marks both flags true) — a fresh dataset upload clears the flags.
  React.useEffect(() => {
    if (codeAssistMode !== "auto") return;
    if (!seedEnabled) return;
    if (autoRanRef.current) return;
    if (!hasRequiredContext) return;
    // Workflow mode waits for the wizard's starter graph (the request needs
    // a prior_workflow to route the graph-aware path) and respects canvas
    // edits the way the code path respects manual code edits.
    if (isWorkflow && (!workflowSpec || workflowTouched || metricManuallyEdited)) return;
    if (!isWorkflow && (signatureManuallyEdited || metricManuallyEdited)) return;
    autoRanRef.current = true;
    autoFixAttemptsRef.current = 0;
    runAgent("", []);
  }, [
    codeAssistMode,
    seedEnabled,
    hasRequiredContext,
    signatureManuallyEdited,
    metricManuallyEdited,
    isWorkflow,
    workflowSpec,
    workflowTouched,
    runAgent,
    sessionKey,
  ]);

  // If the user switches to manual mid-stream, abort and reset.
  React.useEffect(() => {
    if (codeAssistMode === "auto") return;
    abortRef.current?.abort();
    abortRef.current = null;
  }, [codeAssistMode]);

  return {
    status,
    mode,
    statusLabel,
    signatureStatus,
    metricStatus,
    messages,
    error,
    canSend: hasRequiredContext && status !== "streaming",
    signatureVersions,
    metricVersions,
    signatureVersionIndex,
    metricVersionIndex,
    signatureFlashLines,
    metricFlashLines,
    reasoning,
    reasoningStartedAt,
    reasoningEndedAt,
    goToSignatureVersion,
    goToMetricVersion,
    send,
    editAndResend,
    retry,
    fallbackToManual,
    stop,
    reset,
  };
}
