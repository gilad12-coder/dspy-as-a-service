"use client";

import * as React from "react";

import { streamCodeInterviewTurn, type CodeAgentChatTurn } from "@/shared/lib/api";
import type { ParsedDataset } from "@/shared/lib/parse-dataset";
import { getActiveLocale } from "@/shared/lib/runtime-locale";
import type { AgentMessage, AgentThinking } from "@/shared/ui/agent";

interface InterviewTurn extends CodeAgentChatTurn {
  model?: string | null;
}

export interface CodeInterviewState {
  /** Interview transcript mapped for the shared chat primitives. */
  messages: AgentMessage[];
  busy: boolean;
  streamText: string;
  thinking: AgentThinking | null;
  quickReplies: string[];
  error: boolean;
  /** The model finished asking; the brief card is showing. */
  done: boolean;
  /** The model-proposed brief (editable in the card until confirmed). */
  brief: string[];
  /** Confirmed or skipped — the seed generation may run. */
  resolved: boolean;
  /** The user-confirmed directives; empty when the interview was skipped. */
  confirmedBrief: string[];
  send: (content: string) => void;
  editAndResend: (messageIndex: number, content: string) => void;
  stop: () => void;
  retry: () => void;
  skip: () => void;
  confirm: (brief: string[]) => void;
}

export interface UseCodeInterviewArgs {
  /**
   * Master gate: true only while the interview should run (auto assist mode,
   * dataset with input+output roles, module picked, user on the code step,
   * no pre-existing code work). The opening question fires when this flips
   * true with an empty transcript.
   */
  enabled: boolean;
  parsedDataset: ParsedDataset | null;
  columnRoles: Record<string, string>;
  columnKinds: Record<string, "text" | "image">;
  /** LiteLLM id of the job's target model; empty when not chosen yet. */
  jobModel: string;
}

/**
 * The Signature & Metric interview: the same live chat experience as the
 * generalist agent (streamed reply tokens, collapsible thinking, stop,
 * edit-and-resend), driving a purpose-built interviewer that distills the
 * user's answers into an authoring brief for the seed generation. Mirrors
 * the tagger's dataset interview; the transcript is client-owned.
 */
export function useCodeInterview(args: UseCodeInterviewArgs): CodeInterviewState {
  const { enabled, parsedDataset, columnRoles, columnKinds, jobModel } = args;

  const [turns, setTurns] = React.useState<InterviewTurn[]>([]);
  const [busy, setBusy] = React.useState(false);
  const [streamText, setStreamText] = React.useState("");
  const [thinking, setThinking] = React.useState<AgentThinking | null>(null);
  const [quickReplies, setQuickReplies] = React.useState<string[]>([]);
  const [error, setError] = React.useState(false);
  const [done, setDone] = React.useState(false);
  const [brief, setBrief] = React.useState<string[]>([]);
  const [resolved, setResolved] = React.useState(false);
  const [confirmedBrief, setConfirmedBrief] = React.useState<string[]>([]);

  const abortRef = React.useRef<AbortController | null>(null);

  const runTurn = React.useCallback(
    (next: InterviewTurn[]) => {
      if (!parsedDataset || busy) return;
      setTurns(next);
      setDone(false);
      setBusy(true);
      setError(false);
      setStreamText("");
      setThinking(null);
      setQuickReplies([]);
      const controller = new AbortController();
      abortRef.current = controller;

      // Only image-typed entries go over the wire — text is the default,
      // matching the code agent's request shape.
      const imageColumnKinds: Record<string, "image"> = {};
      for (const [col, kind] of Object.entries(columnKinds)) {
        if (kind === "image") imageColumnKinds[col] = "image";
      }

      void streamCodeInterviewTurn(
        {
          dataset_columns: parsedDataset.columns,
          column_roles: columnRoles,
          column_kinds: imageColumnKinds,
          sample_rows: parsedDataset.rows.slice(0, 5) as Array<Record<string, unknown>>,
          turns: next.map((t) => ({ role: t.role, content: t.content })),
          job_model: jobModel,
          locale: getActiveLocale(),
        },
        {
          signal: controller.signal,
          onReasoningPatch: (chunk) =>
            setThinking((prev) => ({
              reasoning: (prev?.reasoning ?? "") + chunk,
              startedAt: prev?.startedAt ?? Date.now(),
              endedAt: null,
              streaming: true,
            })),
          onMessagePatch: (chunk) => {
            setThinking((prev) =>
              prev && prev.streaming ? { ...prev, streaming: false, endedAt: Date.now() } : prev,
            );
            setStreamText((text) => text + chunk);
          },
          onDone: (turn) => {
            setTurns((prev) => [
              ...prev,
              { role: "assistant", content: turn.message, model: turn.model ?? null },
            ]);
            setQuickReplies(turn.done ? [] : turn.quick_replies);
            if (turn.done) {
              setDone(true);
              setBrief(turn.brief);
            }
            setBusy(false);
            setStreamText("");
            setThinking(null);
          },
          onError: () => {
            setError(true);
            setBusy(false);
            setStreamText("");
            setThinking((prev) =>
              prev ? { ...prev, streaming: false, endedAt: prev.endedAt ?? Date.now() } : prev,
            );
          },
        },
      ).finally(() => {
        if (abortRef.current === controller) abortRef.current = null;
      });
    },
    [parsedDataset, busy, columnRoles, columnKinds, jobModel],
  );

  // Fire the opening question exactly once per eligible interview.
  React.useEffect(() => {
    if (!enabled || resolved || done || busy || error) return;
    if (turns.length > 0) return;
    runTurn([]);
  }, [enabled, resolved, done, busy, error, turns, runTurn]);

  const send = React.useCallback(
    (content: string) => runTurn([...turns, { role: "user", content }]),
    [runTurn, turns],
  );

  const editAndResend = React.useCallback(
    (messageIndex: number, content: string) =>
      runTurn([...turns.slice(0, messageIndex), { role: "user", content }]),
    [runTurn, turns],
  );

  const stop = React.useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
    setStreamText("");
    setThinking((prev) =>
      prev ? { ...prev, streaming: false, endedAt: prev.endedAt ?? Date.now() } : prev,
    );
  }, []);

  const retry = React.useCallback(() => runTurn(turns), [runTurn, turns]);

  const skip = React.useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setBusy(false);
    setConfirmedBrief([]);
    setResolved(true);
  }, []);

  const confirm = React.useCallback((confirmed: string[]) => {
    const cleaned = confirmed.map((d) => d.trim()).filter(Boolean);
    setBrief(cleaned);
    setConfirmedBrief(cleaned);
    setResolved(true);
  }, []);

  const messages: AgentMessage[] = React.useMemo(() => {
    const mapped: AgentMessage[] = turns.map((t) => ({
      role: t.role,
      content: t.content,
      model: t.model ?? null,
    }));
    if (busy && (streamText || thinking)) {
      mapped.push({ role: "assistant", content: streamText });
    }
    return mapped;
  }, [turns, busy, streamText, thinking]);

  return {
    messages,
    busy,
    streamText,
    thinking,
    quickReplies,
    error,
    done,
    brief,
    resolved,
    confirmedBrief,
    send,
    editAndResend,
    stop,
    retry,
    skip,
    confirm,
  };
}
